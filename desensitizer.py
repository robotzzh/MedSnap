# -*- coding: utf-8 -*-
"""
MedSnap 敏感信息检测与脱敏模块
纯本地正则处理，不依赖任何网络调用。
"""

import re
import copy

# ========== 常量定义 ==========

COMPOUND_SURNAMES = [
    '欧阳', '太史', '端木', '上官', '司马', '东方', '独孤', '南宫', '万俟', '闻人',
    '夏侯', '诸葛', '尉迟', '公羊', '赫连', '澹台', '皇甫', '宗政', '濮阳', '公冶',
    '太叔', '申屠', '公孙', '慕容', '仲孙', '钟离', '长孙', '宇文', '司徒', '鲜于',
    '司空', '令狐', '百里', '呼延',
]

NAME_CONTEXT_LABELS = (
    r'(?:姓名|患者|病人|家属|联系人|医师?|主治|主管|护士|责任护士|'
    r'评估人|送检人|接诊|报告人|签名|记录人|审核人|术者|助手|麻醉)'
)

# 身份证号正则 (18位新版 + 15位旧版)
RE_ID_CARD_18 = re.compile(
    r'(?<!\d)([1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx])(?!\d)'
)
RE_ID_CARD_15 = re.compile(
    r'(?<!\d)([1-9]\d{5}\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3})(?!\d)'
)

# 手机号正则
RE_PHONE = re.compile(r'(?<!\d)(1[3-9]\d{9})(?!\d)')

# 银行卡号正则 (上下文辅助)
RE_BANK_CONTEXT = re.compile(
    r'((?:银行|账号|卡号|账户)\s*[:：]\s*)(\d{16,19})',
    re.UNICODE
)
RE_BANK_STANDALONE = re.compile(r'(?<!\d)(\d{16,19})(?!\d)')

# 医保/社保号正则
RE_INSURANCE = re.compile(
    r'((?:医保卡?号?|社保号?|社保卡?号?|医疗保险号|社会保险号)\s*[:：]\s*)(\w{6,20})',
    re.UNICODE
)

# 中文姓名 (上下文标签触发)
RE_NAME_WITH_CONTEXT = re.compile(
    r'(' + NAME_CONTEXT_LABELS + r'\s*[:：]\s*)([\u4e00-\u9fa5]{2,4})',
    re.UNICODE
)

# 地址 (上下文标签触发)
RE_ADDRESS_WITH_CONTEXT = re.compile(
    r'((?:地址|住址|户籍|籍贯|居住地|家庭住址|工作单位|单位地址)\s*[:：]\s*)'
    r'([\u4e00-\u9fa5]{2,8}(?:省|自治区|市)[\u4e00-\u9fa5]{0,10}(?:市|区|县|旗|盟)?)'
    r'([\u4e00-\u9fa5\d\-\.]{2,})',
    re.UNICODE
)

# 独立地址模式 (省市区+详细)
RE_ADDRESS_STANDALONE = re.compile(
    r'([\u4e00-\u9fa5]{2,6}(?:省|自治区)[\u4e00-\u9fa5]{2,10}(?:市|区|县))'
    r'([\u4e00-\u9fa5]{2,6}(?:区|县|镇|乡))'
    r'([\u4e00-\u9fa5\d]{2,30}(?:路|街|道|巷|弄|号|栋|楼|室|单元|幢)[\u4e00-\u9fa5\d]*)',
    re.UNICODE
)

# 结构化数据字段名 → 脱敏类型映射
SENSITIVE_FIELD_PATTERNS = {
    'name': ['姓名', '患者姓名', 'name', 'patient_name', '病人姓名', '联系人'],
    'id_card': ['身份证', 'id_card', 'id_number', '证件号'],
    'phone': ['手机', '电话', 'phone', 'tel', 'mobile', '联系方式', '联系电话'],
    'address': ['地址', '住址', 'address', '籍贯', '居住地', '工作单位', '单位'],
    'insurance': ['医保', '社保', '保险号', 'insurance'],
    'bank': ['银行', '账号', '卡号', 'bank', 'account'],
}


# ========== 单项脱敏函数 ==========

def _mask_id_card(id_str):
    """身份证号脱敏：保留前6后4"""
    s = id_str.strip()
    if len(s) == 18:
        return s[:6] + '********' + s[-4:]
    elif len(s) == 15:
        return s[:6] + '*****' + s[-4:]
    return s


def _mask_phone(phone_str):
    """手机号脱敏：保留前3后4"""
    s = phone_str.strip()
    if len(s) == 11:
        return s[:3] + '****' + s[-4:]
    return s


def _mask_name(name_str):
    """中文姓名脱敏：保留姓氏，名用*"""
    s = name_str.strip()
    if not s:
        return s
    # 检查复姓
    for cs in COMPOUND_SURNAMES:
        if s.startswith(cs):
            return cs + '*' * (len(s) - len(cs)) if len(s) > len(cs) else s
    # 单姓
    if len(s) >= 2:
        return s[0] + '*' * (len(s) - 1)
    return s


def _mask_address_detail(detail_str):
    """地址详细部分脱敏"""
    return '***'


def _mask_generic_number(num_str):
    """通用长号码脱敏：保留前4后4"""
    s = num_str.strip()
    if len(s) <= 8:
        return s[:2] + '*' * (len(s) - 4) + s[-2:] if len(s) > 4 else s
    return s[:4] + '*' * (len(s) - 8) + s[-4:]


# ========== 核心函数 ==========

def desensitize_text(text):
    """
    对原始文本进行敏感信息检测和脱敏。
    按优先级：身份证 → 银行卡 → 手机号 → 医保/社保 → 姓名 → 地址

    Args:
        text: 原始文本字符串

    Returns:
        tuple: (脱敏后文本, 检测报告dict)
    """
    if not text or not isinstance(text, str):
        return text, {"detected_count": 0, "items": []}

    result = text
    report_items = []

    # 1. 身份证号 (18位)
    matches_18 = RE_ID_CARD_18.findall(result)
    if matches_18:
        def _replace_id18(m):
            return _mask_id_card(m.group(1))
        result = RE_ID_CARD_18.sub(lambda m: _mask_id_card(m.group(1)), result)
        report_items.append({"type": "身份证号", "count": len(matches_18)})

    # 身份证号 (15位旧版)
    matches_15 = RE_ID_CARD_15.findall(result)
    if matches_15:
        result = RE_ID_CARD_15.sub(lambda m: _mask_id_card(m.group(1)), result)
        report_items.append({"type": "身份证号(旧版)", "count": len(matches_15)})

    # 2. 银行卡号 (上下文模式)
    matches_bank = RE_BANK_CONTEXT.findall(result)
    if matches_bank:
        result = RE_BANK_CONTEXT.sub(
            lambda m: m.group(1) + _mask_generic_number(m.group(2)), result
        )
        report_items.append({"type": "银行卡号", "count": len(matches_bank)})

    # 3. 手机号
    matches_phone = RE_PHONE.findall(result)
    if matches_phone:
        result = RE_PHONE.sub(lambda m: _mask_phone(m.group(1)), result)
        report_items.append({"type": "手机号", "count": len(matches_phone)})

    # 4. 医保/社保号
    matches_ins = RE_INSURANCE.findall(result)
    if matches_ins:
        result = RE_INSURANCE.sub(
            lambda m: m.group(1) + _mask_generic_number(m.group(2)), result
        )
        report_items.append({"type": "医保/社保号", "count": len(matches_ins)})

    # 5. 中文姓名 (上下文触发)
    matches_name = RE_NAME_WITH_CONTEXT.findall(result)
    if matches_name:
        result = RE_NAME_WITH_CONTEXT.sub(
            lambda m: m.group(1) + _mask_name(m.group(2)), result
        )
        report_items.append({"type": "姓名", "count": len(matches_name)})

    # 6. 地址 (上下文触发)
    matches_addr_ctx = RE_ADDRESS_WITH_CONTEXT.findall(result)
    if matches_addr_ctx:
        result = RE_ADDRESS_WITH_CONTEXT.sub(
            lambda m: m.group(1) + m.group(2) + '***', result
        )
        report_items.append({"type": "地址", "count": len(matches_addr_ctx)})

    # 地址 (独立模式)
    matches_addr_std = RE_ADDRESS_STANDALONE.findall(result)
    if matches_addr_std:
        result = RE_ADDRESS_STANDALONE.sub(
            lambda m: m.group(1) + m.group(2) + '***', result
        )
        if not matches_addr_ctx:
            report_items.append({"type": "地址", "count": len(matches_addr_std)})
        else:
            for item in report_items:
                if item["type"] == "地址":
                    item["count"] += len(matches_addr_std)

    detected_count = sum(item["count"] for item in report_items)
    report = {"detected_count": detected_count, "items": report_items}

    if detected_count > 0:
        print(f"[PRIVACY] 文本脱敏完成: 检测到 {detected_count} 处敏感信息 {report_items}")

    return result, report


def _detect_field_type(key):
    """根据字段名判断敏感信息类型"""
    if not isinstance(key, str):
        return None
    key_lower = key.lower()
    for sens_type, patterns in SENSITIVE_FIELD_PATTERNS.items():
        for pattern in patterns:
            if pattern.lower() in key_lower:
                return sens_type
    return None


def _mask_value_by_type(value, sens_type):
    """根据敏感类型对值进行脱敏"""
    if not isinstance(value, str) or not value.strip():
        return value

    s = value.strip()
    if sens_type == 'name':
        return _mask_name(s)
    elif sens_type == 'id_card':
        return _mask_id_card(s)
    elif sens_type == 'phone':
        return _mask_phone(s)
    elif sens_type == 'address':
        # 尝试保留省市级别
        m = re.match(
            r'([\u4e00-\u9fa5]{2,8}(?:省|自治区|市)[\u4e00-\u9fa5]{0,10}(?:市|区|县|旗)?)(.*)',
            s, re.UNICODE
        )
        if m and m.group(2):
            return m.group(1) + '***'
        return s
    elif sens_type == 'insurance':
        return _mask_generic_number(s)
    elif sens_type == 'bank':
        return _mask_generic_number(s)
    return value


def _apply_generic_scan(value):
    """对字符串值进行通用正则兜底扫描"""
    if not isinstance(value, str):
        return value

    result = value
    # 身份证
    result = RE_ID_CARD_18.sub(lambda m: _mask_id_card(m.group(1)), result)
    # 手机号
    result = RE_PHONE.sub(lambda m: _mask_phone(m.group(1)), result)
    return result


def desensitize_structured_data(data, _skip_keys=None):
    """
    对结构化字典数据进行敏感信息脱敏。
    递归遍历所有 key-value，按字段名判断类型并脱敏。

    Args:
        data: 解析后的 dict 数据
        _skip_keys: 要跳过的 key 集合

    Returns:
        dict: 脱敏后的新字典（深拷贝，不修改原始数据）
    """
    if _skip_keys is None:
        _skip_keys = {'confidence', 'confidence_data'}

    if not isinstance(data, dict):
        return data

    result = copy.deepcopy(data)
    return _desensitize_recursive(result, _skip_keys)


def _desensitize_recursive(obj, skip_keys, parent_key=None):
    """递归脱敏处理"""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in skip_keys:
                continue
            sens_type = _detect_field_type(key)
            if sens_type and isinstance(value, str):
                obj[key] = _mask_value_by_type(value, sens_type)
            elif isinstance(value, (dict, list)):
                obj[key] = _desensitize_recursive(value, skip_keys, parent_key=key)
            elif isinstance(value, str):
                obj[key] = _apply_generic_scan(value)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            if isinstance(item, (dict, list)):
                obj[i] = _desensitize_recursive(item, skip_keys, parent_key=parent_key)
            elif isinstance(item, str):
                if parent_key:
                    sens_type = _detect_field_type(parent_key)
                    if sens_type:
                        obj[i] = _mask_value_by_type(item, sens_type)
                    else:
                        obj[i] = _apply_generic_scan(item)
                else:
                    obj[i] = _apply_generic_scan(item)
    return obj
