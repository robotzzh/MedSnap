# -*- coding: utf-8 -*-
"""
MedSnap 模板管理模块
包含所有 AI Prompt 模板定义、角色配置、模板生成与字段解析工具。
"""

import re

# ========== 角色与分类配置 ==========

CATEGORY_CONFIGS = {
    'diagnosis': {'name': '诊疗', 'color': '#2563eb'},
    'nursing':   {'name': '护理', 'color': '#059669'},
    'other':     {'name': '其他', 'color': '#d97706'},
}

# ========== 内置模板 Prompt 定义 ==========

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

# ========== 自定义模板 Prompt 生成框架 ==========

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

# ========== 字段预览 Prompt ==========

PROMPT_FIELD_PREVIEW = """你是医疗数据分析专家。请仔细分析以下医疗文档，识别所有可以提取的数据字段。

输出JSON格式:
{{
  "available_fields": [
    {{
      "field_name": "字段名称",
      "field_type": "text或number或date",
      "example_value": "从文档中提取的示例值",
      "confidence": 0.95,
      "category": "类别"
    }}
  ]
}}

要求:
1. 字段名称使用中文，简洁明确
2. 尽可能识别所有有意义的数据项（患者信息、检验指标、诊断、治疗、评估等）
3. category取值范围：基本信息、检验结果、诊疗记录、护理评估、其他
4. example_value必须是从文档中实际提取的真实值
5. confidence表示该字段在文档中的识别置信度(0-1)
6. 按category分组，同类别字段放在一起

只输出JSON，不要输出任何其他内容。"""

PROMPT_FIELD_PREVIEW_TEXT = """你是医疗数据分析专家。请仔细分析以下医疗文本，识别所有可以提取的数据字段。

输出JSON格式:
{{
  "available_fields": [
    {{
      "field_name": "字段名称",
      "field_type": "text或number或date",
      "example_value": "从文本中提取的示例值",
      "confidence": 0.95,
      "category": "类别"
    }}
  ]
}}

要求:
1. 字段名称使用中文，简洁明确
2. 尽可能识别所有有意义的数据项
3. category取值范围：基本信息、检验结果、诊疗记录、护理评估、其他
4. example_value必须是从文本中实际提取的真实值
5. confidence表示该字段的识别置信度(0-1)

文本内容:
{text_content}

只输出JSON，不要输出任何其他内容。"""

PROMPT_EXTRACT_FIELD_NAMES = """你是医疗模板设计专家，当前视角为{role_hint}。请仔细分析以下医疗相关文本，从中识别所有可以作为"数据提取模板字段名"的医学术语或概念。

注意：你要提取的是**字段名称**（如"血压"、"诊断"、"护理措施"），而不是具体的数据值。

输出JSON格式:
{{
  "fields": [
    {{
      "name": "字段名称",
      "category": "类别",
      "confidence": 0.95
    }}
  ]
}}

要求:
1. 字段名称使用中文，简洁明确（2-8个字为宜）
2. 只提取与医疗场景相关的字段，忽略无关词汇
3. category取值范围：基本信息、检验结果、诊疗记录、护理评估、科研数据、其他
4. confidence表示该词作为模板字段的合理程度(0-1)
5. 不要重复提取含义相同的字段
6. 根据{role_hint}的视角，优先识别该角色关注的字段
7. 不要将患者的具体姓名、具体数值等作为字段名

文本内容:
{text_content}

只输出JSON，不要输出任何其他内容。"""

# ========== 音频专用 Prompt 模板 ==========

PROMPT_AUDIO_DOCTOR = """你是临床医生数据提取专家。以下是医患对话的语音转录文本，请从中提取结构化病历信息。

## 输出格式(JSON):
{
  "patient_info": {"姓名": "", "性别": "", "年龄": null},
  "chief_complaint": "",
  "present_illness": "",
  "past_history": "",
  "physical_exam": {},
  "diagnosis": [{"诊断名称": "", "ICD10编码": ""}],
  "treatment_plan": {"药物治疗": "", "医嘱": "", "其他": ""},
  "conversation_notes": "",
  "confidence": {}
}

## 提取规则:
1. 从对话中识别患者自述的症状和病史
2. 提取医生口述的诊断和治疗建议
3. 主诉通常是患者开场描述的主要不适
4. 注意区分医生询问和患者回答
5. 如信息不完整或无法识别，对应字段留空字符串
6. confidence字段：对每个已提取字段给出0-1之间的置信度

只输出JSON，不要输出任何其他内容。"""

PROMPT_AUDIO_NURSE = """你是护理评估专家。以下是护理交班或患者访谈的语音转录文本，请提取护理相关信息。

## 输出格式(JSON):
{
  "patient_info": {"姓名": "", "床号": "", "科室": ""},
  "vital_signs_verbal": {},
  "nursing_observations": "",
  "patient_complaints": "",
  "nursing_actions": "",
  "handover_notes": "",
  "risk_alerts": "",
  "confidence": {}
}

## 提取规则:
1. 识别口述的生命体征数值（体温、血压、脉搏、呼吸、血氧等）
2. 提取护理观察内容（皮肤、伤口、活动能力、意识状态）
3. 记录患者主观感受和主诉
4. 提取交班时的重点提醒事项
5. 识别提及的护理风险（跌倒、压疮、管路等）
6. confidence字段：对每个已提取字段给出0-1之间的置信度

只输出JSON，不要输出任何其他内容。"""

PROMPT_AUDIO_RESEARCHER = """你是临床科研数据提取专家。以下是研究访谈或病历口述的语音转录文本，请提取科研相关数据。

## 输出格式(JSON):
{
  "demographics": {"姓名": "", "性别": "", "年龄": null, "职业": "", "教育程度": ""},
  "medical_history": "",
  "intervention_details": "",
  "outcome_measures": "",
  "patient_experience": "",
  "adherence_notes": "",
  "adverse_events": "",
  "research_notes": "",
  "confidence": {}
}

## 提取规则:
1. 提取人口学特征
2. 识别干预措施的描述
3. 提取患者自我报告的结局（症状改善、生活质量变化）
4. 注意提及的依从性和不良反应
5. 日期格式统一为YYYY-MM-DD
6. confidence字段：对每个已提取字段给出0-1之间的置信度

只输出JSON，不要输出任何其他内容。"""

# ========== 系统模板字段映射 ==========

TEMPLATE_FIELDS = {
    'tpl_doctor_medical': ['主诉', '现病史', '既往史', '个人史', '家族史', '过敏史', '体格检查', '专科检查', '辅助检查', '诊断', '诊疗计划', '处理意见'],
    'tpl_doctor_lab': ['项目名称', '结果', '参考值', '单位', '异常提示', '标本类型', '采集时间', '报告时间'],
    'tpl_nurse_admission': ['一般资料', '过敏史', '既往史', '用药史', '生命体征', '意识状态', '皮肤黏膜', '营养状况', '排泄', '活动能力', '跌倒风险', '压疮风险', '疼痛评分', '吞咽功能', '心理状态', '睡眠', '饮食', '专科情况', '护理问题', '护理措施'],
    'tpl_nurse_barthel': ['进食', '洗澡', '修饰', '穿衣', '控制大便', '控制小便', '如厕', '床椅转移', '平地行走', '上下楼梯'],
    'tpl_nurse_morse': ['跌倒史', '继发诊断', '步行辅助', '静脉输液/肝素锁', '步态', '认知状态'],
    'tpl_nurse_braden': ['感知能力', '潮湿程度', '活动能力', '移动能力', '营养摄取', '摩擦力和剪切力'],
    'tpl_nurse_pain': ['疼痛部位', '疼痛性质', '疼痛强度', '诱发因素', '缓解因素', '伴随症状', '疼痛持续时间'],
    'tpl_nurse_record': ['生命体征', '意识状态', '皮肤完整性', '跌倒风险', '压疮风险', '护理措施'],
    'tpl_researcher_default': ['人口学特征', '实验室检查', '主要终点事件', '随访日期', '血压', '血脂', '用药情况', '治疗结局'],
    'tpl_audio_doctor': ['主诉', '现病史', '诊断', '治疗方案'],
    'tpl_audio_nurse': ['生命体征', '护理观察', '风险提醒'],
    'tpl_audio_researcher': ['人口学特征', '病史', '干预措施', '结局指标'],
}

# ========== 模板工具函数 ==========

def generate_template_prompt(role_id, fields, include_score=False):
    """
    根据角色和字段列表生成 AI 提取 Prompt 和 display_layout。

    Args:
        role_id: 角色标识（diagnosis / nursing / other）
        fields: 字段名称列表
        include_score: 是否包含评分规则（护理量表专用）

    Returns:
        tuple: (ai_prompt, display_layout)
    """
    field_schema_parts = []
    for field in fields:
        field = field.strip()
        if field:
            field_schema_parts.append(f'    "{field}": null')
    field_schema = ',\n'.join(field_schema_parts)
    field_names = '、'.join([f.strip() for f in fields if f.strip()])

    score_rule = ""
    if include_score:
        score_rule = "3. 如果字段是评分项，提取纯数字评分。如有总分，一并计算。\n"

    if role_id == 'diagnosis':
        ai_prompt = DOCTOR_CUSTOM_PROMPT_TEMPLATE.format(
            field_schema=field_schema, field_names=field_names)
        display_layout = 'table'
    elif role_id == 'other':
        ai_prompt = RESEARCHER_CUSTOM_PROMPT_TEMPLATE.format(
            field_schema=field_schema, field_names=field_names)
        display_layout = 'table'
    else:
        ai_prompt = NURSE_CUSTOM_PROMPT_TEMPLATE.format(
            field_schema=field_schema, field_names=field_names, score_rule=score_rule)
        display_layout = 'scale' if include_score else 'card'

    return ai_prompt, display_layout


def extract_fields_from_prompt(ai_prompt):
    """
    从 ai_prompt 中反向提取自定义字段列表。

    Args:
        ai_prompt: 模板的 AI Prompt 字符串

    Returns:
        list: 字段名称列表
    """
    fields = []
    matches = re.findall(r'"custom_fields"\s*:\s*\{([^}]+)\}', ai_prompt, re.DOTALL)
    if matches:
        field_matches = re.findall(r'"([^"]+)"\s*:\s*null', matches[0])
        fields = [f.strip() for f in field_matches if f.strip()]
    return fields
