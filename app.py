# -*- coding: utf-8 -*-
"""
临床科研病历AI识别与结构化提取工具 - 主程序（多角色版）
Web服务监听: 0.0.0.0:7860
支持角色: 临床医生 / 护士 / 临床科研人员
"""

import os
import json
import uuid
import base64
import tempfile
import sqlite3
import shutil
import re
from datetime import datetime
from flask import Flask, render_template, request, send_file, jsonify
from openai import OpenAI
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import numpy as np
import pandas as pd

HAS_PYMUPDF = False
try:
    import fitz
    HAS_PYMUPDF = True
except ImportError:
    pass

# ========== Flask 应用初始化 ==========
app = Flask(__name__)
app.secret_key = uuid.uuid4().hex

UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "medical_ocr_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_DIR
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'local_medical_data.db')

# ========== 多模态大模型配置 ==========
API_KEY = os.environ.get("MODELSCOPE_API_KEY", "")
if not API_KEY:
    print("[WARN] 环境变量 MODELSCOPE_API_KEY 未设置，AI识别功能将不可用")
client = OpenAI(
    base_url='https://api-inference.modelscope.cn/v1',
    api_key=API_KEY
)
MODEL_NAME = "Qwen/Qwen3-VL-235B-A22B-Instruct"

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ========== 角色与模板配置 ==========
ROLE_CONFIGS = {
    'doctor': {'name': '临床医生', 'color': '#2563eb', 'icon': 'doctor'},
    'nurse':  {'name': '护士',     'color': '#059669', 'icon': 'nurse'},
    'researcher': {'name': '临床科研', 'color': '#d97706', 'icon': 'researcher'},
}

# ========== 内置模板Prompt定义 ==========

PROMPT_DOCTOR_MEDICAL_RECORD = """你是临床医生数据提取专家。请仔细识别该病历图片中的所有文本内容，并严格按照以下JSON格式输出结构化数据。

## 输出格式要求
请直接输出合法的JSON对象，不要包含任何markdown标记、代码块标记或多余文字。JSON结构如下：

{
  "patient_info": {
    "姓名": "", "性别": "", "年龄": null, "科室": "", "床号": "", "住院号": ""
  },
  "chief_complaint": "",
  "present_illness": "",
  "past_history": "",
  "personal_history": "",
  "family_history": "",
  "physical_exam": {
    "体温": null, "脉搏": null, "呼吸": null, "血压": "",
    "一般情况": "", "专科检查": ""
  },
  "diagnosis": [
    {"诊断名称": "", "ICD10编码": ""}
  ],
  "treatment_plan": {
    "药物治疗": "", "手术治疗": "", "其他治疗": ""
  },
  "surgery_record": "",
  "discharge_summary": "",
  "confidence": {}
}

## 提取规则
1. 年龄提取纯数字（如"56岁"→56）。
2. 诊断需尽量识别ICD-10编码（如 E11.9 2型糖尿病、I10 高血压病）。
3. 治疗方案区分药物治疗、手术治疗、其他治疗。
4. confidence字段：对每个已提取字段给出0-1之间的置信度。未识别到的字段置信度为0。
5. 手写体请尽力识别，无法确认的字用?标记。

只输出JSON，不要输出任何其他内容。"""

PROMPT_DOCTOR_LAB_RESULTS = """你是临床检验数据提取专家。请仔细识别该检查检验结果图片中的所有文本内容，并严格按照以下JSON格式输出结构化数据。

## 输出格式要求
请直接输出合法的JSON对象，不要包含任何markdown标记、代码块标记或多余文字。JSON结构如下：

{
  "patient_info": {
    "姓名": "", "性别": "", "年龄": null, "科室": "", "住院号": ""
  },
  "report_info": {
    "报告类型": "", "检查日期": "", "报告日期": ""
  },
  "lab_tests": [
    {
      "项目名称": "", "英文缩写": "", "数值": null,
      "单位": "", "参考范围": "", "异常标注": ""
    }
  ],
  "confidence": {}
}

## 提取规则
1. 识别常见医学缩写：WBC、RBC、PLT、Hb、ALT、AST、Cr、BUN、GLU、TC、TG、HDL-C、LDL-C、UA、CRP、ESR、HbA1c、TSH、FT3、FT4、Na、K、Ca、Cl、PT、APTT、INR、D-Dimer、AFP、CEA等。
2. 异常标注用"↑"(偏高)/"↓"(偏低)/"正常"。
3. 数值标准化为数字格式。
4. confidence字段：对每个已提取字段给出0-1之间的置信度。

## 医疗专业词汇参考
- 血常规：WBC、RBC、PLT、Hb、HCT、MCV、MCH、MCHC、RDW
- 肝功能：ALT、AST、GGT、ALP、TBIL、DBIL、TP、ALB
- 肾功能：Cr、BUN、UA、Cys-C、eGFR
- 血脂：TC、TG、HDL-C、LDL-C、ApoA1、ApoB
- 血糖：GLU、FPG、2hPG、HbA1c、OGTT
- 凝血：PT、APTT、TT、FIB、INR、D-Dimer
- 电解质：Na、K、Ca、Cl、Mg、P
- 炎症指标：CRP、PCT、ESR、IL-6
- 肿瘤标志物：AFP、CEA、CA125、CA199、CA153、PSA

只输出JSON，不要输出任何其他内容。"""

PROMPT_NURSE_ADMISSION = """你是护理评估数据提取专家。请仔细识别该入院护理评估表图片中的所有文本内容，并严格按照以下JSON格式输出结构化数据。

## 输出格式要求
请直接输出合法的JSON对象。JSON结构如下：

{
  "patient_info": {
    "姓名": "", "性别": "", "年龄": null, "科室": "", "床号": "", "住院号": "", "入院日期": ""
  },
  "vital_signs": {
    "体温": null, "脉搏": null, "呼吸": null, "血压_收缩压": null, "血压_舒张压": null,
    "血氧饱和度": null, "身高_cm": null, "体重_kg": null
  },
  "assessment": {
    "意识状态": "", "精神状态": "", "皮肤完整性": "", "皮肤异常描述": "",
    "营养状况": "", "饮食类型": "", "排便情况": "", "排尿情况": "",
    "睡眠情况": "", "活动能力": "", "自理能力初筛": "",
    "跌倒风险初筛": "", "压疮风险初筛": "", "疼痛评分": null,
    "过敏史": "", "特殊用药": ""
  },
  "nursing_diagnosis": [],
  "nursing_plan": "",
  "confidence": {}
}

## 提取规则
1. 生命体征数值标准化为纯数字。血压格式拆分为收缩压和舒张压。
2. 意识状态：清醒/嗜睡/昏睡/浅昏迷/深昏迷。
3. 自理能力/跌倒/压疮风险初筛：识别勾选框或评分。
4. confidence字段：对每个已提取字段给出0-1之间的置信度。

只输出JSON，不要输出任何其他内容。"""

PROMPT_NURSE_BARTHEL = """你是护理评估数据提取专家。请仔细识别该Barthel自理能力指数量表图片中的所有内容，并严格按照以下JSON格式输出结构化数据。

## 输出格式要求
请直接输出合法的JSON对象。JSON结构如下：

{
  "patient_info": {
    "姓名": "", "科室": "", "床号": "", "评估日期": "", "评估人": ""
  },
  "barthel_items": [
    {"项目": "进食", "评分": null, "评分标准": "10=自理 5=需部分帮助 0=完全依赖", "备注": ""},
    {"项目": "洗澡", "评分": null, "评分标准": "5=自理 0=需帮助", "备注": ""},
    {"项目": "修饰", "评分": null, "评分标准": "5=自理 0=需帮助", "备注": ""},
    {"项目": "穿衣", "评分": null, "评分标准": "10=自理 5=需部分帮助 0=完全依赖", "备注": ""},
    {"项目": "控制大便", "评分": null, "评分标准": "10=可控 5=偶有失禁 0=失禁", "备注": ""},
    {"项目": "控制小便", "评分": null, "评分标准": "10=可控 5=偶有失禁 0=失禁", "备注": ""},
    {"项目": "如厕", "评分": null, "评分标准": "10=自理 5=需部分帮助 0=完全依赖", "备注": ""},
    {"项目": "床椅转移", "评分": null, "评分标准": "15=自理 10=少量帮助 5=较大帮助 0=完全依赖", "备注": ""},
    {"项目": "平地行走", "评分": null, "评分标准": "15=自行45m 10=在帮助下45m 5=轮椅45m 0=不能", "备注": ""},
    {"项目": "上下楼梯", "评分": null, "评分标准": "10=自理 5=需帮助 0=不能", "备注": ""}
  ],
  "total_score": null,
  "dependency_level": "",
  "confidence": {}
}

## 提取规则
1. 评分只提取数字。
2. 总分范围0-100。依赖等级判定：100=自理、61-99=轻度依赖、41-60=中度依赖、≤40=重度依赖。
3. 如图中有勾选标记，按勾选对应的分值提取。
4. confidence字段：对每个项目给出0-1之间的置信度。

只输出JSON，不要输出任何其他内容。"""

PROMPT_NURSE_MORSE = """你是护理评估数据提取专家。请仔细识别该Morse跌倒风险评估量表图片中的所有内容，并严格按照以下JSON格式输出结构化数据。

## 输出格式要求
请直接输出合法的JSON对象。JSON结构如下：

{
  "patient_info": {
    "姓名": "", "科室": "", "床号": "", "评估日期": "", "评估人": ""
  },
  "morse_items": [
    {"项目": "跌倒史", "评分": null, "评分标准": "是=25 否=0", "选项": ""},
    {"项目": "继发诊断", "评分": null, "评分标准": "是=15 否=0", "选项": ""},
    {"项目": "步行辅助", "评分": null, "评分标准": "卧床/护士协助=0 拐杖/助行器/轮椅=15 扶家具行走=30", "选项": ""},
    {"项目": "静脉输液/肝素锁", "评分": null, "评分标准": "是=20 否=0", "选项": ""},
    {"项目": "步态", "评分": null, "评分标准": "正常/卧床/不能活动=0 虚弱=10 损伤=20", "选项": ""},
    {"项目": "认知状态", "评分": null, "评分标准": "能正确认识自身活动能力=0 高估/忘记限制=15", "选项": ""}
  ],
  "total_score": null,
  "risk_level": "",
  "confidence": {}
}

## 提取规则
1. 评分只提取数字。
2. 风险判定：0-24=低风险、25-44=中风险、≥45=高风险。
3. confidence字段：对每个项目给出0-1之间的置信度。

只输出JSON，不要输出任何其他内容。"""

PROMPT_NURSE_BRADEN = """你是护理评估数据提取专家。请仔细识别该Braden压疮风险评估量表图片中的所有内容，并严格按照以下JSON格式输出结构化数据。

## 输出格式要求
请直接输出合法的JSON对象。JSON结构如下：

{
  "patient_info": {
    "姓名": "", "科室": "", "床号": "", "评估日期": "", "评估人": ""
  },
  "braden_items": [
    {"项目": "感知能力", "评分": null, "评分标准": "1=完全受限 2=非常受限 3=轻度受限 4=未受损", "备注": ""},
    {"项目": "潮湿程度", "评分": null, "评分标准": "1=持续潮湿 2=非常潮湿 3=偶尔潮湿 4=很少潮湿", "备注": ""},
    {"项目": "活动能力", "评分": null, "评分标准": "1=卧床 2=坐椅 3=偶尔步行 4=经常步行", "备注": ""},
    {"项目": "移动能力", "评分": null, "评分标准": "1=完全不能 2=严重受限 3=轻度受限 4=不受限", "备注": ""},
    {"项目": "营养摄取", "评分": null, "评分标准": "1=非常差 2=可能不足 3=足够 4=良好", "备注": ""},
    {"项目": "摩擦力和剪切力", "评分": null, "评分标准": "1=存在问题 2=潜在问题 3=不存在问题", "备注": ""}
  ],
  "total_score": null,
  "risk_level": "",
  "confidence": {}
}

## 提取规则
1. 评分只提取数字（1-4分，摩擦力1-3分）。
2. 总分6-23分。风险判定：≤9=极高风险、10-12=高风险、13-14=中度风险、15-18=低风险、≥19=无风险。
3. confidence字段：对每个项目给出0-1之间的置信度。

只输出JSON，不要输出任何其他内容。"""

PROMPT_NURSE_PAIN = """你是护理评估数据提取专家。请仔细识别该疼痛评估记录图片中的所有内容，并严格按照以下JSON格式输出结构化数据。

## 输出格式要求
请直接输出合法的JSON对象。JSON结构如下：

{
  "patient_info": {
    "姓名": "", "科室": "", "床号": "", "评估日期": "", "评估人": ""
  },
  "pain_assessment": {
    "疼痛部位": "",
    "疼痛性质": "",
    "NRS评分": null,
    "VAS评分": null,
    "疼痛频率": "",
    "持续时间": "",
    "加重因素": "",
    "缓解因素": "",
    "对睡眠影响": "",
    "对日常活动影响": "",
    "当前镇痛措施": "",
    "镇痛效果": ""
  },
  "confidence": {}
}

## 提取规则
1. NRS评分0-10（0=无痛，10=最剧烈疼痛）。VAS评分0-10。
2. 疼痛性质：锐痛/钝痛/刺痛/胀痛/灼痛/绞痛等。
3. confidence字段：对每个项目给出0-1之间的置信度。

只输出JSON，不要输出任何其他内容。"""

PROMPT_NURSE_RECORD = """你是护理记录数据提取专家。请仔细识别该护理记录单图片中的所有文本内容，并严格按照以下JSON格式输出结构化数据。

## 输出格式要求
请直接输出合法的JSON对象。JSON结构如下：

{
  "patient_info": {
    "姓名": "", "科室": "", "床号": "", "记录日期": ""
  },
  "vital_signs_records": [
    {"时间": "", "体温": null, "脉搏": null, "呼吸": null, "血压": "", "血氧": null, "备注": ""}
  ],
  "medication_execution": [
    {"时间": "", "医嘱内容": "", "执行情况": "", "执行人": ""}
  ],
  "nursing_measures": [
    {"时间": "", "护理措施": "", "患者反应": "", "记录人": ""}
  ],
  "handover_notes": "",
  "confidence": {}
}

## 提取规则
1. 时间格式统一为HH:MM。
2. 生命体征数值标准化为数字。
3. confidence字段：对每个记录项给出0-1之间的置信度。

只输出JSON，不要输出任何其他内容。"""

PROMPT_RESEARCHER = """你是临床科研数据提取专家。请仔细识别该病历图片中的所有文本内容，并严格按照以下JSON格式输出结构化数据。

## 输出格式要求
请直接输出合法的JSON对象，不要包含任何markdown标记、代码块标记或多余文字。JSON结构如下：

{
  "demographics": {
    "姓名": "", "性别": "", "年龄": null, "身高_cm": null, "体重_kg": null,
    "BMI": null, "婚姻状况": "", "职业": "", "民族": "",
    "吸烟史": "", "饮酒史": "", "过敏史": ""
  },
  "lab_tests": [
    {"项目名称": "", "英文缩写": "", "数值": null, "单位": "", "参考范围": "", "异常标注": ""}
  ],
  "treatment": {
    "入院时间": "", "出院时间": "", "住院天数": null, "科室": "",
    "主诊断": "", "主诊断ICD10编码": "", "其他诊断": [],
    "手术操作": "", "治疗方案": "", "出院医嘱": ""
  },
  "confidence": {
    "demographics": {}, "lab_tests": {}, "treatment": {}
  }
}

## 提取规则
1. 年龄提取纯数字（如"56岁"→56）；BMI如未给出尝试计算。
2. 实验室检查识别常见缩写：WBC、RBC、PLT、Hb、ALT、AST、Cr、BUN、GLU、TC、TG、HDL-C、LDL-C、UA、CRP、ESR、HbA1c、TSH、FT3、FT4、Na、K、Ca、Cl。异常标注"↑"/"↓"/"正常"。
3. 诊疗资料尽量识别ICD-10编码；治疗方案区分药物/手术/其他。
4. 置信度0-1，未识别到的字段为0。

## 医疗专业词汇参考
- 血常规：WBC、RBC、PLT、Hb、HCT、MCV、MCH、MCHC、RDW
- 肝功能：ALT、AST、GGT、ALP、TBIL、DBIL、TP、ALB
- 肾功能：Cr、BUN、UA、Cys-C、eGFR
- 血脂：TC、TG、HDL-C、LDL-C、ApoA1、ApoB
- 血糖：GLU、FPG、2hPG、HbA1c、OGTT
- 凝血：PT、APTT、TT、FIB、INR、D-Dimer
- 甲功：TSH、FT3、FT4、T3、T4
- 电解质：Na、K、Ca、Cl、Mg、P
- 炎症指标：CRP、PCT、ESR、IL-6
- 肿瘤标志物：AFP、CEA、CA125、CA199、CA153、PSA

只输出JSON，不要输出任何其他内容。"""

# 护士自定义模板的Prompt生成框架
NURSE_CUSTOM_PROMPT_TEMPLATE = """你是护理评估数据提取专家。请仔细识别该护理文档图片中的所有文本内容，并严格按照以下JSON格式输出结构化数据。

## 输出格式要求
请直接输出合法的JSON对象。JSON结构如下：

{{
  "patient_info": {{
    "姓名": "", "科室": "", "床号": "", "日期": ""
  }},
  "custom_fields": {{
    {field_schema}
  }},
  "confidence": {{}}
}}

## 提取规则
1. 识别图片中与以下字段相关的所有信息：{field_names}。
2. 数值型数据提取为数字，文本型数据提取为字符串。
{score_rule}
3. confidence字段：对每个已提取字段给出0-1之间的置信度。

只输出JSON，不要输出任何其他内容。"""

# 医生自定义模板的Prompt生成框架
DOCTOR_CUSTOM_PROMPT_TEMPLATE = """你是临床病历数据提取专家。请仔细识别该医疗文档图片中的所有文本内容，并严格按照以下JSON格式输出结构化数据。

## 输出格式要求
请直接输出合法的JSON对象。JSON结构如下：

{{
  "patient_info": {{
    "姓名": "", "性别": "", "年龄": null, "科室": "", "床号": "", "住院号": ""
  }},
  "custom_fields": {{
    {field_schema}
  }},
  "confidence": {{}}
}}

## 提取规则
1. 识别图片中与以下字段相关的所有信息：{field_names}。
2. 年龄提取纯数字，诊断需尽量识别ICD-10编码。
3. 手写体请尽力识别，无法确认的字用?标记。
4. confidence字段：对每个已提取字段给出0-1之间的置信度。

只输出JSON，不要输出任何其他内容。"""

# 科研自定义模板的Prompt生成框架
RESEARCHER_CUSTOM_PROMPT_TEMPLATE = """你是临床科研数据提取专家。请仔细识别该医疗文档图片中的所有文本内容，并严格按照以下JSON格式输出结构化数据。

## 输出格式要求
请直接输出合法的JSON对象。JSON结构如下：

{{
  "demographics": {{
    "姓名": "", "性别": "", "年龄": null
  }},
  "custom_fields": {{
    {field_schema}
  }},
  "confidence": {{}}
}}

## 提取规则
1. 识别图片中与以下字段相关的所有信息：{field_names}。
2. 数值型数据提取为数字（如年龄、体重、血压数值等），文本型数据提取为字符串。
3. 日期格式统一为YYYY-MM-DD。
4. confidence字段：对每个已提取字段给出0-1之间的置信度。

只输出JSON，不要输出任何其他内容。"""


# ========== 数据库初始化 ==========
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    c = conn.cursor()

    # 模板表
    c.execute('''CREATE TABLE IF NOT EXISTS extraction_templates (
        template_id TEXT PRIMARY KEY,
        role_id TEXT,
        template_name TEXT,
        template_type TEXT,
        ai_prompt TEXT,
        output_schema TEXT,
        display_layout TEXT,
        is_active INTEGER DEFAULT 1,
        create_time TEXT
    )''')

    # 记录表（新版）
    c.execute('''CREATE TABLE IF NOT EXISTS medical_records (
        id TEXT PRIMARY KEY,
        case_number TEXT UNIQUE,
        original_filename TEXT,
        role_id TEXT,
        template_id TEXT,
        extracted_data TEXT,
        confidence_data TEXT,
        demographics TEXT,
        lab_tests TEXT,
        treatment TEXT,
        confidence TEXT,
        raw_text TEXT,
        create_time TEXT
    )''')

    # 检查是否需要加新列（兼容旧数据库）
    existing_cols = {row[1] for row in c.execute("PRAGMA table_info(medical_records)").fetchall()}
    for col in ['role_id', 'template_id', 'extracted_data', 'confidence_data']:
        if col not in existing_cols:
            c.execute(f"ALTER TABLE medical_records ADD COLUMN {col} TEXT")

    conn.commit()
    conn.close()

    # 初始化内置模板
    _init_builtin_templates()


def _init_builtin_templates():
    """插入系统内置模板（如果尚未存在）"""
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) as cnt FROM extraction_templates WHERE template_type='fixed'")
    if c.fetchone()['cnt'] > 0:
        conn.close()
        return

    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    templates = [
        # 临床医生模板
        ('tpl_doctor_medical', 'doctor', '门诊/住院病历', 'fixed',
         PROMPT_DOCTOR_MEDICAL_RECORD, 'table', now),
        ('tpl_doctor_lab', 'doctor', '检查检验结果', 'fixed',
         PROMPT_DOCTOR_LAB_RESULTS, 'table', now),
        # 护士模板
        ('tpl_nurse_admission', 'nurse', '入院护理评估表', 'fixed',
         PROMPT_NURSE_ADMISSION, 'card', now),
        ('tpl_nurse_barthel', 'nurse', 'Barthel自理能力量表', 'fixed',
         PROMPT_NURSE_BARTHEL, 'scale', now),
        ('tpl_nurse_morse', 'nurse', 'Morse跌倒风险量表', 'fixed',
         PROMPT_NURSE_MORSE, 'scale', now),
        ('tpl_nurse_braden', 'nurse', 'Braden压疮风险量表', 'fixed',
         PROMPT_NURSE_BRADEN, 'scale', now),
        ('tpl_nurse_pain', 'nurse', 'NRS/VAS疼痛评估', 'fixed',
         PROMPT_NURSE_PAIN, 'card', now),
        ('tpl_nurse_record', 'nurse', '护理记录单', 'fixed',
         PROMPT_NURSE_RECORD, 'table', now),
        # 科研模板
        ('tpl_researcher_default', 'researcher', '综合科研数据提取', 'fixed',
         PROMPT_RESEARCHER, 'table', now),
    ]

    for t in templates:
        c.execute('''INSERT OR IGNORE INTO extraction_templates
            (template_id, role_id, template_name, template_type, ai_prompt, display_layout, create_time)
            VALUES (?, ?, ?, ?, ?, ?, ?)''', t)

    # 迁移旧数据：标记为科研角色
    c.execute('''UPDATE medical_records
        SET role_id='researcher', template_id='tpl_researcher_default'
        WHERE role_id IS NULL''')

    conn.commit()
    conn.close()


# ========== 图片处理 ==========
def preprocess_image(image_path):
    try:
        img = Image.open(image_path)
        if img.mode not in ('RGB', 'L'):
            img = img.convert('RGB')
        gray = img.convert('L')
        gray = ImageOps.autocontrast(gray, cutoff=1)
        gray = gray.filter(ImageFilter.SHARPEN)
        enhancer = ImageEnhance.Contrast(gray)
        gray = enhancer.enhance(1.5)
        preprocessed_path = os.path.join(
            app.config['UPLOAD_FOLDER'],
            f"pre_{uuid.uuid4().hex[:8]}_{os.path.basename(image_path)}"
        )
        gray.save(preprocessed_path)
        return preprocessed_path
    except Exception:
        return image_path


def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


def pdf_to_images(pdf_path):
    if not HAS_PYMUPDF:
        raise RuntimeError("PDF功能需要PyMuPDF库，请运行: pip install PyMuPDF")
    doc = fitz.open(pdf_path)
    image_paths = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)
        img_path = os.path.join(
            app.config['UPLOAD_FOLDER'],
            f"pdf_page_{uuid.uuid4().hex[:8]}_{page_num}.png"
        )
        pix.save(img_path)
        image_paths.append(img_path)
    doc.close()
    return image_paths


# ========== AI 识别核心 ==========
def extract_medical_data(image_path, ai_prompt):
    """调用多模态模型识别，使用传入的Prompt"""
    preprocessed_path = preprocess_image(image_path)
    b64_image = image_to_base64(preprocessed_path)

    ext = os.path.splitext(image_path)[1].lower()
    mime_map = {
        '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
        '.png': 'image/png', '.bmp': 'image/bmp',
        '.tiff': 'image/tiff'
    }
    mime_type = mime_map.get(ext, 'image/jpeg')

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{
            'role': 'user',
            'content': [
                {'type': 'text', 'text': ai_prompt},
                {
                    'type': 'image_url',
                    'image_url': {
                        'url': f"data:{mime_type};base64,{b64_image}"
                    }
                }
            ]
        }],
        temperature=0.1,
        max_tokens=4096
    )

    raw_text = response.choices[0].message.content

    try:
        if preprocessed_path != image_path and os.path.exists(preprocessed_path):
            os.remove(preprocessed_path)
    except Exception:
        pass

    parsed = parse_ai_response(raw_text)
    return parsed, raw_text


def parse_ai_response(raw_text):
    text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL)
    text = text.strip()
    if '```' in text:
        match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
        if match:
            text = match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    return {"error": "AI返回结果解析失败，请重试或检查图片质量", "raw_response": raw_text[:500]}


# ========== Excel 导出 ==========
def generate_excel(data_list):
    """多角色Excel导出，不同角色放不同Sheet"""
    from openpyxl.styles import Font, PatternFill
    red_fill = PatternFill(start_color='FFCCCC', end_color='FFCCCC', fill_type='solid')
    red_font = Font(color='CC0000', bold=True)

    # 按角色分组
    grouped = {}
    for item in data_list:
        role = item.get('role_id', 'researcher')
        role_name = ROLE_CONFIGS.get(role, {}).get('name', role)
        grouped.setdefault(role_name, []).append(item)

    excel_path = os.path.join(
        app.config['UPLOAD_FOLDER'],
        f"临床数据_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    )

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        for sheet_name, items in grouped.items():
            rows = []
            low_conf_fields = set()

            for item in items:
                row = {
                    '病历编号': item.get('case_number', ''),
                    '模板': item.get('template_name', ''),
                    '录入时间': item.get('create_time', ''),
                }
                data = item.get('extracted_data', {})
                conf = item.get('confidence_data', {})

                # 递归展平JSON数据为Excel列
                _flatten_to_row(data, row, '', conf, low_conf_fields)
                rows.append(row)

            if not rows:
                continue

            df = pd.DataFrame(rows)
            safe_name = sheet_name[:31]  # Excel sheet名最长31字符
            df.to_excel(writer, index=False, sheet_name=safe_name)

            # 标红低置信度
            if low_conf_fields:
                ws = writer.sheets[safe_name]
                for col_idx, col_name in enumerate(df.columns, 1):
                    for field_name in low_conf_fields:
                        if field_name in col_name:
                            for row_idx in range(2, len(df) + 2):
                                cell = ws.cell(row=row_idx, column=col_idx)
                                cell.fill = red_fill
                                cell.font = red_font
                            ws.cell(row=1, column=col_idx).fill = red_fill
                            ws.cell(row=1, column=col_idx).font = red_font
                            break

    return excel_path


def _flatten_to_row(data, row, prefix, confidence, low_conf_fields):
    """递归展平嵌套JSON为Excel行的列"""
    if isinstance(data, dict):
        for k, v in data.items():
            if k == 'confidence':
                # 收集低置信度字段
                _collect_low_conf(v, low_conf_fields)
                continue
            full_key = f"{prefix}_{k}" if prefix else k
            if isinstance(v, dict):
                _flatten_to_row(v, row, full_key, confidence, low_conf_fields)
            elif isinstance(v, list):
                _flatten_list_to_row(v, row, full_key, confidence, low_conf_fields)
            else:
                row[full_key] = v
    elif isinstance(data, list):
        _flatten_list_to_row(data, row, prefix, confidence, low_conf_fields)


def _flatten_list_to_row(data_list, row, prefix, confidence, low_conf_fields):
    """展平列表数据"""
    for i, item in enumerate(data_list):
        if isinstance(item, dict):
            # 尝试用名称作为键
            name_key = item.get('项目名称') or item.get('英文缩写') or item.get('项目') or item.get('诊断名称') or str(i + 1)
            item_prefix = f"{prefix}_{name_key}" if prefix else name_key
            for k, v in item.items():
                if k in ('项目名称', '英文缩写', '项目', '诊断名称'):
                    continue
                if isinstance(v, (dict, list)):
                    continue
                col = f"{item_prefix}_{k}" if k != '数值' and k != '评分' else item_prefix
                row[col] = v
        elif isinstance(item, str):
            row[f"{prefix}_{i+1}"] = item


def _collect_low_conf(conf, low_conf_fields):
    """收集置信度<0.9的字段"""
    if isinstance(conf, dict):
        for k, v in conf.items():
            if isinstance(v, dict):
                _collect_low_conf(v, low_conf_fields)
            else:
                try:
                    if float(v) < 0.9:
                        low_conf_fields.add(k)
                except (ValueError, TypeError):
                    pass


# ========== 路由: 角色与模板 API ==========

@app.route('/api/roles', methods=['GET'])
def api_get_roles():
    """获取角色列表"""
    conn = get_db()
    c = conn.cursor()
    roles = []
    for role_id, cfg in ROLE_CONFIGS.items():
        c.execute("SELECT COUNT(*) as cnt FROM extraction_templates WHERE role_id=? AND is_active=1",
                  (role_id,))
        count = c.fetchone()['cnt']
        roles.append({
            'role_id': role_id,
            'name': cfg['name'],
            'color': cfg['color'],
            'icon': cfg['icon'],
            'template_count': count
        })
    conn.close()
    return jsonify({"status": "success", "roles": roles})


@app.route('/api/templates/<role_id>', methods=['GET'])
def api_get_templates(role_id):
    """获取某角色下的模板列表"""
    conn = get_db()
    c = conn.cursor()
    c.execute('''SELECT template_id, template_name, template_type, display_layout, create_time
        FROM extraction_templates WHERE role_id=? AND is_active=1 ORDER BY template_type, create_time''',
              (role_id,))
    rows = c.fetchall()
    conn.close()
    templates = []
    for row in rows:
        templates.append({
            'template_id': row['template_id'],
            'template_name': row['template_name'],
            'template_type': row['template_type'],
            'display_layout': row['display_layout'],
        })
    return jsonify({"status": "success", "templates": templates})


@app.route('/api/templates', methods=['POST'])
def api_create_template():
    """创建自定义模板（护士角色）"""
    data = request.get_json()
    if not data:
        return jsonify({"status": "error", "msg": "无数据"})

    role_id = data.get('role_id', 'nurse')
    template_name = data.get('template_name', '').strip()
    fields = data.get('fields', [])
    include_score = data.get('include_score', False)

    if not template_name or not fields:
        return jsonify({"status": "error", "msg": "请填写模板名称和提取字段"})

    # 生成输出JSON模板
    field_schema_parts = []
    for f in fields:
        f = f.strip()
        if f:
            field_schema_parts.append(f'    "{f}": null')
    field_schema = ',\n'.join(field_schema_parts)
    field_names = '、'.join([f.strip() for f in fields if f.strip()])

    score_rule = ""
    if include_score:
        score_rule = "3. 如果字段是评分项，提取纯数字评分。如有总分，一并计算。\n"

    # 根据角色选择Prompt模板
    if role_id == 'doctor':
        ai_prompt = DOCTOR_CUSTOM_PROMPT_TEMPLATE.format(
            field_schema=field_schema,
            field_names=field_names
        )
        display_layout = 'table'
    elif role_id == 'researcher':
        ai_prompt = RESEARCHER_CUSTOM_PROMPT_TEMPLATE.format(
            field_schema=field_schema,
            field_names=field_names
        )
        display_layout = 'table'
    else:  # nurse
        ai_prompt = NURSE_CUSTOM_PROMPT_TEMPLATE.format(
            field_schema=field_schema,
            field_names=field_names,
            score_rule=score_rule
        )
        display_layout = 'scale' if include_score else 'card'

    template_id = f"tpl_custom_{uuid.uuid4().hex[:8]}"
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    conn = get_db()
    c = conn.cursor()
    c.execute('''INSERT INTO extraction_templates
        (template_id, role_id, template_name, template_type, ai_prompt, display_layout, is_active, create_time)
        VALUES (?, ?, ?, 'custom', ?, ?, 1, ?)''',
              (template_id, role_id, template_name, ai_prompt, display_layout, now))
    conn.commit()
    conn.close()

    return jsonify({"status": "success", "template_id": template_id, "msg": "模板创建成功"})


@app.route('/api/templates/<template_id>', methods=['DELETE'])
def api_delete_template(template_id):
    """删除自定义模板"""
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT template_type FROM extraction_templates WHERE template_id=?", (template_id,))
    row = c.fetchone()
    if not row:
        conn.close()
        return jsonify({"status": "error", "msg": "模板不存在"})
    if row['template_type'] == 'fixed':
        conn.close()
        return jsonify({"status": "error", "msg": "系统内置模板不可删除"})

    c.execute("DELETE FROM extraction_templates WHERE template_id=?", (template_id,))
    conn.commit()
    conn.close()
    return jsonify({"status": "success", "msg": "模板已删除"})


# ========== 路由: 核心功能 ==========

@app.route('/')
def index():
    init_db()
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_and_recognize():
    """上传文件并AI识别（支持角色/模板选择）"""
    if 'files' not in request.files:
        return jsonify({"status": "error", "msg": "未选择文件"})

    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        return jsonify({"status": "error", "msg": "未选择有效文件"})

    role_id = request.form.get('role_id', 'researcher')
    template_id = request.form.get('template_id', 'tpl_researcher_default')

    # 查询模板Prompt
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT ai_prompt, template_name, display_layout FROM extraction_templates WHERE template_id=?",
              (template_id,))
    tpl_row = c.fetchone()
    conn.close()

    if not tpl_row:
        return jsonify({"status": "error", "msg": "模板不存在"})

    ai_prompt = tpl_row['ai_prompt']
    template_name = tpl_row['template_name']
    display_layout = tpl_row['display_layout']

    results = []
    errors = []

    for file in files:
        if not allowed_file(file.filename):
            errors.append(f"不支持的格式: {file.filename}")
            continue

        file_ext = os.path.splitext(file.filename)[1].lower()
        temp_name = f"{uuid.uuid4().hex}{file_ext}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_name)
        file.save(file_path)

        try:
            if file_ext == '.pdf':
                image_paths = pdf_to_images(file_path)
            else:
                image_paths = [file_path]

            for img_path in image_paths:
                data, raw_text = extract_medical_data(img_path, ai_prompt)

                if "error" in data:
                    errors.append(f"{file.filename}: {data['error']}")
                    continue

                case_number = f"CASE_{datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:6].upper()}"
                record_id = str(uuid.uuid4())
                create_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                # 分离置信度数据
                confidence_data = data.pop('confidence', {})

                # 存储
                conn = get_db()
                c = conn.cursor()
                c.execute('''INSERT INTO medical_records
                    (id, case_number, original_filename, role_id, template_id,
                     extracted_data, confidence_data, raw_text, create_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                    (record_id, case_number, file.filename, role_id, template_id,
                     json.dumps(data, ensure_ascii=False),
                     json.dumps(confidence_data, ensure_ascii=False),
                     raw_text, create_time))
                conn.commit()
                conn.close()

                results.append({
                    "id": record_id,
                    "case_number": case_number,
                    "filename": file.filename,
                    "role_id": role_id,
                    "template_name": template_name,
                    "display_layout": display_layout,
                    "data": data,
                    "confidence": confidence_data,
                    "create_time": create_time
                })

        except Exception as e:
            errors.append(f"{file.filename}: 识别失败 - {str(e)}")
        finally:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception:
                pass

    return jsonify({
        "status": "success" if results else "error",
        "results": results,
        "errors": errors,
        "msg": f"成功识别 {len(results)} 份" + (f"，{len(errors)} 份失败" if errors else "")
    })


@app.route('/records', methods=['GET'])
def get_records():
    role_id = request.args.get('role_id', None)
    conn = get_db()
    c = conn.cursor()
    if role_id:
        c.execute('''SELECT id, case_number, original_filename, role_id, template_id, create_time
            FROM medical_records WHERE role_id=? ORDER BY create_time DESC''', (role_id,))
    else:
        c.execute('''SELECT id, case_number, original_filename, role_id, template_id, create_time
            FROM medical_records ORDER BY create_time DESC''')
    rows = c.fetchall()
    conn.close()

    records = []
    for row in rows:
        records.append({
            "id": row['id'],
            "case_number": row['case_number'],
            "filename": row['original_filename'],
            "role_id": row['role_id'] or 'researcher',
            "template_id": row['template_id'] or '',
            "create_time": row['create_time']
        })
    return jsonify({"status": "success", "records": records})


@app.route('/record/<record_id>', methods=['GET'])
def get_record_detail(record_id):
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT * FROM medical_records WHERE id = ?', (record_id,))
    row = c.fetchone()

    # 查模板信息
    template_name = ''
    display_layout = 'table'
    if row and row['template_id']:
        c.execute("SELECT template_name, display_layout FROM extraction_templates WHERE template_id=?",
                  (row['template_id'],))
        tpl = c.fetchone()
        if tpl:
            template_name = tpl['template_name']
            display_layout = tpl['display_layout']
    conn.close()

    if not row:
        return jsonify({"status": "error", "msg": "记录不存在"})

    role_id = row['role_id'] or 'researcher'

    # 兼容旧数据
    if row['extracted_data']:
        extracted_data = json.loads(row['extracted_data'])
        confidence_data = json.loads(row['confidence_data']) if row['confidence_data'] else {}
    else:
        # 旧格式兼容
        extracted_data = {
            'demographics': json.loads(row['demographics']) if row['demographics'] else {},
            'lab_tests': json.loads(row['lab_tests']) if row['lab_tests'] else [],
            'treatment': json.loads(row['treatment']) if row['treatment'] else {},
        }
        confidence_data = json.loads(row['confidence']) if row['confidence'] else {}
        display_layout = 'table'
        template_name = '综合科研数据提取'

    return jsonify({
        "status": "success",
        "data": {
            "id": row['id'],
            "case_number": row['case_number'],
            "filename": row['original_filename'],
            "role_id": role_id,
            "template_name": template_name,
            "display_layout": display_layout,
            "extracted_data": extracted_data,
            "confidence": confidence_data,
            "create_time": row['create_time']
        }
    })


@app.route('/record/<record_id>', methods=['PUT'])
def update_record(record_id):
    update_data = request.get_json()
    if not update_data:
        return jsonify({"status": "error", "msg": "无更新数据"})

    conn = get_db()
    c = conn.cursor()
    if 'extracted_data' in update_data:
        c.execute("UPDATE medical_records SET extracted_data=? WHERE id=?",
                  (json.dumps(update_data['extracted_data'], ensure_ascii=False), record_id))
    if 'confidence' in update_data:
        c.execute("UPDATE medical_records SET confidence_data=? WHERE id=?",
                  (json.dumps(update_data['confidence'], ensure_ascii=False), record_id))
    conn.commit()
    conn.close()
    return jsonify({"status": "success", "msg": "数据已更新"})


@app.route('/record/<record_id>', methods=['DELETE'])
def delete_record(record_id):
    conn = get_db()
    c = conn.cursor()
    c.execute('DELETE FROM medical_records WHERE id = ?', (record_id,))
    conn.commit()
    conn.close()
    return jsonify({"status": "success", "msg": "记录已删除"})


@app.route('/export', methods=['POST'])
def export_excel():
    req_data = request.get_json() or {}
    record_ids = req_data.get('record_ids', [])
    role_filter = req_data.get('role_id', None)

    conn = get_db()
    c = conn.cursor()

    if record_ids:
        placeholders = ','.join(['?'] * len(record_ids))
        c.execute(f'SELECT * FROM medical_records WHERE id IN ({placeholders})', record_ids)
    elif role_filter:
        c.execute('SELECT * FROM medical_records WHERE role_id=? ORDER BY create_time', (role_filter,))
    else:
        c.execute('SELECT * FROM medical_records ORDER BY create_time')

    rows = c.fetchall()
    conn.close()

    if not rows:
        return jsonify({"status": "error", "msg": "暂无可导出的数据"})

    data_list = _rows_to_export_list(rows)
    excel_path = generate_excel(data_list)
    return send_file(excel_path, as_attachment=True,
                     download_name=f"临床数据_{datetime.now().strftime('%Y%m%d')}.xlsx")


@app.route('/export_all', methods=['GET'])
def export_all_excel():
    role_filter = request.args.get('role_id', None)
    conn = get_db()
    c = conn.cursor()
    if role_filter:
        c.execute('SELECT * FROM medical_records WHERE role_id=? ORDER BY create_time', (role_filter,))
    else:
        c.execute('SELECT * FROM medical_records ORDER BY create_time')
    rows = c.fetchall()
    conn.close()

    if not rows:
        return jsonify({"status": "error", "msg": "暂无可导出的数据"})

    data_list = _rows_to_export_list(rows)
    excel_path = generate_excel(data_list)
    return send_file(excel_path, as_attachment=True,
                     download_name=f"临床数据_{datetime.now().strftime('%Y%m%d')}.xlsx")


def _rows_to_export_list(rows):
    """将数据库行转为导出数据列表"""
    conn = get_db()
    c = conn.cursor()
    data_list = []
    for row in rows:
        role_id = row['role_id'] or 'researcher'
        template_name = ''
        if row['template_id']:
            c.execute("SELECT template_name FROM extraction_templates WHERE template_id=?",
                      (row['template_id'],))
            tpl = c.fetchone()
            if tpl:
                template_name = tpl['template_name']

        if row['extracted_data']:
            extracted = json.loads(row['extracted_data'])
            conf = json.loads(row['confidence_data']) if row['confidence_data'] else {}
        else:
            extracted = {
                'demographics': json.loads(row['demographics']) if row['demographics'] else {},
                'lab_tests': json.loads(row['lab_tests']) if row['lab_tests'] else [],
                'treatment': json.loads(row['treatment']) if row['treatment'] else {},
            }
            conf = json.loads(row['confidence']) if row['confidence'] else {}

        data_list.append({
            'case_number': row['case_number'],
            'create_time': row['create_time'],
            'role_id': role_id,
            'template_name': template_name,
            'extracted_data': extracted,
            'confidence_data': conf,
        })
    conn.close()
    return data_list


@app.route('/clean', methods=['POST'])
def clean_all():
    try:
        shutil.rmtree(app.config['UPLOAD_FOLDER'])
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    except Exception:
        pass
    conn = get_db()
    c = conn.cursor()
    c.execute('DELETE FROM medical_records')
    conn.commit()
    conn.close()
    return jsonify({"status": "success", "msg": "所有数据已清理"})


@app.route('/stats', methods=['GET'])
def get_stats():
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT COUNT(*) as cnt FROM medical_records')
    total = c.fetchone()['cnt']
    # 按角色统计
    c.execute("SELECT role_id, COUNT(*) as cnt FROM medical_records GROUP BY role_id")
    by_role = {}
    for row in c.fetchall():
        by_role[row['role_id'] or 'researcher'] = row['cnt']
    conn.close()
    return jsonify({"status": "success", "total_records": total, "by_role": by_role})


# ========== 启动入口 ==========
if __name__ == '__main__':
    print("=" * 50)
    print("  临床科研病历AI识别与结构化提取工具（多角色版）")
    print("  访问地址: http://localhost:7860")
    print("  按 Ctrl+C 停止服务")
    print("=" * 50)
    init_db()
    app.run(host='0.0.0.0', port=7860, debug=False)
