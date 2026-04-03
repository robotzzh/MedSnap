# -*- coding: utf-8 -*-
"""
临床科研病历AI识别与结构化提取工具 - 主程序（多角色版）
Web服务监听: 0.0.0.0:7860
支持角色: 临床医生 / 护士 / 临床科研人员

架构说明:
  model.py              - AI 模型客户端封装
  template.py           - Prompt 模板定义与模板管理工具
  image_util.py         - 图片/PDF 处理与 OCR 识别
  audio_util.py         - 音频转写与质性研究分析
  build_conversation.py - 对话构建、统计分析、Excel 导出
  memory.py             - 数据库连接与初始化
  desensitizer.py       - 敏感信息脱敏
  statistics_routes.py  - 统计分析 Blueprint
"""

import os
import json
import uuid
import shutil
from datetime import datetime
from flask import Flask, render_template, request, send_file, jsonify

# ========== 各功能模块导入 ==========
from model import client, MODEL_NAME, parse_ai_response, get_model_config, set_model_config
from template import (
    CATEGORY_CONFIGS,
    TEMPLATE_FIELDS,
    PROMPT_FIELD_PREVIEW,
    PROMPT_FIELD_PREVIEW_TEXT,
    PROMPT_EXTRACT_FIELD_NAMES,
    generate_template_prompt,
    extract_fields_from_prompt,
)
from image_util import (
    extract_medical_data,
    extract_medical_data_multimodal,
    extract_pdf_embedded_text,
    local_ocr_pdf,
    pdf_to_images,
)
from audio_util import (
    transcribe_audio,
    extract_from_transcript,
    qualitative_analysis,
    qualitative_analysis_enhanced,
    parse_text_file,
    preprocess_text,
)
from build_conversation import (
    extract_from_ocr_text,
    analyze_structured_data,
    generate_excel,
    collect_field_paths,
    extract_nested_field,
    is_numeric,
)
from memory import (
    UPLOAD_DIR,
    get_db,
    init_db,
    allowed_file,
    is_audio_file,
    is_text_file,
)
from desensitizer import desensitize_text

# ========== Flask 应用初始化 ==========

app = Flask(__name__)
app.secret_key = uuid.uuid4().hex
app.config['JSON_AS_ASCII'] = False
try:
    app.json.ensure_ascii = False
except (AttributeError, TypeError):
    pass

app.config['UPLOAD_FOLDER'] = UPLOAD_DIR
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

from statistics_routes import stats_bp
app.register_blueprint(stats_bp)

# ========== 路由: 模型配置 API ==========

@app.route('/api/model/config', methods=['GET'])
def api_get_model_config():
    """获取当前 AI 模型配置。"""
    return jsonify({"status": "success", "config": get_model_config()})

@app.route('/api/model/config', methods=['POST'])
def api_set_model_config():
    """
    运行时更新 AI 模型配置，只需传入要修改的字段。
    请求体示例:
      { "model_name": "qwen-vl-max", "model_name_omni": "qwen2.5-omni-7b" }
      { "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "api_key": "sk-xxx" }
    """
    data = request.get_json(silent=True) or {}
    allowed_fields = {"model_name", "model_name_omni", "base_url", "api_key"}
    unknown_fields = set(data.keys()) - allowed_fields
    if unknown_fields:
        return jsonify({"status": "error", "message": f"不支持的字段: {unknown_fields}"}), 400

    set_model_config(
        model_name=data.get("model_name"),
        model_name_omni=data.get("model_name_omni"),
        base_url=data.get("base_url"),
        api_key=data.get("api_key"),
    )
    return jsonify({"status": "success", "message": "模型配置已更新", "config": get_model_config()})

# ========== 路由: 角色与模板 API ==========

@app.route('/api/roles', methods=['GET'])
def api_get_roles():
    """获取分类列表及各分类下的模板数量。"""
    conn = get_db()
    cursor = conn.cursor()
    roles = []
    for role_id, cfg in CATEGORY_CONFIGS.items():
        cursor.execute(
            "SELECT COUNT(*) as cnt FROM extraction_templates WHERE role_id=? AND is_active=1",
            (role_id,)
        )
        count = cursor.fetchone()['cnt']
        roles.append({
            'role_id': role_id,
            'name': cfg['name'],
            'color': cfg['color'],
            'template_count': count
        })
    conn.close()
    return jsonify({"status": "success", "roles": roles})


@app.route('/api/templates/<role_id>', methods=['GET'])
def api_get_templates(role_id):
    """获取某角色下的模板列表。"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        '''SELECT template_id, template_name, template_type, display_layout, ai_prompt, create_time
           FROM extraction_templates WHERE role_id=? AND is_active=1
           ORDER BY template_type, create_time''',
        (role_id,)
    )
    rows = cursor.fetchall()
    conn.close()

    templates = []
    for row in rows:
        fields = extract_fields_from_prompt(row['ai_prompt']) if row['ai_prompt'] else []
        templates.append({
            'template_id': row['template_id'],
            'template_name': row['template_name'],
            'template_type': row['template_type'],
            'display_layout': row['display_layout'],
            'field_count': len(fields),
        })
    return jsonify({"status": "success", "templates": templates})


@app.route('/api/templates/<template_id>/detail', methods=['GET'])
def api_get_template_detail(template_id):
    """获取模板完整信息用于编辑。"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        '''SELECT template_id, role_id, template_name, template_type,
                  ai_prompt, display_layout, create_time
           FROM extraction_templates WHERE template_id=?''',
        (template_id,)
    )
    row = cursor.fetchone()
    conn.close()

    if not row:
        return jsonify({"status": "error", "msg": "模板不存在"})

    fields = []
    if row['ai_prompt']:
        fields = extract_fields_from_prompt(row['ai_prompt'])
    if not fields:
        fields = TEMPLATE_FIELDS.get(row['template_id'], [])
    include_score = row['display_layout'] == 'scale'

    return jsonify({
        "status": "success",
        "template": {
            "template_id": row['template_id'],
            "role_id": row['role_id'],
            "template_name": row['template_name'],
            "template_type": row['template_type'],
            "display_layout": row['display_layout'],
            "fields": fields,
            "include_score": include_score,
            "create_time": row['create_time']
        }
    })


@app.route('/api/templates', methods=['POST'])
def api_create_template():
    """创建自定义模板。"""
    data = request.get_json()
    if not data:
        return jsonify({"status": "error", "msg": "无数据"})

    role_id = data.get('role_id', 'nurse')
    template_name = data.get('template_name', '').strip()
    fields = data.get('fields', [])
    include_score = data.get('include_score', False)

    if not template_name or not fields:
        return jsonify({"status": "error", "msg": "请填写模板名称和提取字段"})

    ai_prompt, display_layout = generate_template_prompt(role_id, fields, include_score)
    template_id = f"tpl_custom_{uuid.uuid4().hex[:8]}"
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        '''INSERT INTO extraction_templates
           (template_id, role_id, template_name, template_type, ai_prompt, display_layout, is_active, create_time)
           VALUES (?, ?, ?, 'custom', ?, ?, 1, ?)''',
        (template_id, role_id, template_name, ai_prompt, display_layout, now)
    )
    conn.commit()
    conn.close()
    return jsonify({"status": "success", "template_id": template_id, "msg": "模板创建成功"})


@app.route('/api/templates/<template_id>', methods=['DELETE'])
def api_delete_template(template_id):
    """删除自定义模板（内置模板不可删除）。"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT template_type FROM extraction_templates WHERE template_id=?", (template_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return jsonify({"status": "error", "msg": "模板不存在"})
    if row['template_type'] == 'fixed':
        conn.close()
        return jsonify({"status": "error", "msg": "系统内置模板不可删除"})

    cursor.execute("DELETE FROM extraction_templates WHERE template_id=?", (template_id,))
    conn.commit()
    conn.close()
    return jsonify({"status": "success", "msg": "模板已删除"})


@app.route('/api/templates/<template_id>', methods=['PUT'])
def api_update_template(template_id):
    """编辑模板字段（内置模板也支持字段调整）。"""
    data = request.get_json()
    if not data:
        return jsonify({"status": "error", "msg": "无数据"})

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT template_type, role_id, template_name FROM extraction_templates WHERE template_id=?",
        (template_id,)
    )
    row = cursor.fetchone()
    if not row:
        conn.close()
        return jsonify({"status": "error", "msg": "模板不存在"})

    role_id = row['role_id']
    template_name = data.get('template_name', '').strip() or row['template_name']
    fields = data.get('fields', [])
    include_score = data.get('include_score', False)

    if not template_name or not fields:
        conn.close()
        return jsonify({"status": "error", "msg": "请填写模板名称和提取字段"})

    ai_prompt, display_layout = generate_template_prompt(role_id, fields, include_score)
    cursor.execute(
        '''UPDATE extraction_templates
           SET template_name=?, ai_prompt=?, display_layout=?
           WHERE template_id=?''',
        (template_name, ai_prompt, display_layout, template_id)
    )
    conn.commit()
    conn.close()

    TEMPLATE_FIELDS[template_id] = fields
    return jsonify({"status": "success", "msg": "模板已更新"})

# ========== 路由: 字段预览与智能提取 ==========

@app.route('/api/extract_fields_from_text', methods=['POST'])
def api_extract_fields_from_text():
    """从用户输入的描述性文本中智能提取可用作模板字段的名称。"""
    try:
        data = request.get_json()
        text = (data.get('text', '') or '').strip()
        role_id = data.get('role_id', 'other')

        if len(text) < 5:
            return jsonify({"status": "error", "msg": "请输入更多文本内容（至少5个字符）"})

        if role_id not in ('diagnosis', 'nursing', 'other'):
            role_id = 'other'

        role_hints = {
            'diagnosis': '诊疗数据提取',
            'nursing': '护理评估数据提取',
            'other': '综合数据提取'
        }
        masked_text, _report = desensitize_text(text)
        prompt = PROMPT_EXTRACT_FIELD_NAMES.format(
            role_hint=role_hints[role_id],
            text_content=masked_text
        )
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.1,
            max_tokens=2048
        )
        parsed = parse_ai_response(response.choices[0].message.content)
        fields = parsed.get('fields', [])
        if not isinstance(fields, list):
            fields = []

        seen = set()
        filtered = []
        category_order = {'基本信息': 0, '检验结果': 1, '诊疗记录': 2, '护理评估': 3, '科研数据': 4, '其他': 5}
        for field in fields:
            if not isinstance(field, dict):
                continue
            name = (field.get('name', '') or '').strip()
            confidence = field.get('confidence', 0)
            if not name or name in seen:
                continue
            if isinstance(confidence, (int, float)) and confidence < 0.5:
                continue
            seen.add(name)
            filtered.append({
                'name': name,
                'category': field.get('category', '其他'),
                'confidence': round(confidence, 2) if isinstance(confidence, (int, float)) else 0.8
            })
        filtered.sort(key=lambda x: category_order.get(x['category'], 5))
        return jsonify({"status": "success", "fields": filtered})
    except Exception as e:
        return jsonify({"status": "error", "msg": f"分析失败: {str(e)}"})


@app.route('/api/preview_fields', methods=['POST'])
def api_preview_fields():
    """文档字段预览 - 分析文档并返回可提取的字段列表。"""
    text_content = request.form.get('text_content', '').strip()
    uploaded_files = request.files.getlist('files')
    raw_data = None

    try:
        if text_content:
            prompt = PROMPT_FIELD_PREVIEW_TEXT.format(text_content=text_content)
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0.1,
                max_tokens=4096
            )
            raw_data = parse_ai_response(response.choices[0].message.content)

        elif uploaded_files and uploaded_files[0].filename:
            file = uploaded_files[0]
            file_ext = os.path.splitext(file.filename)[1].lower()
            temp_name = f"{uuid.uuid4().hex}{file_ext}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_name)
            file.save(file_path)

            try:
                if is_audio_file(file.filename):
                    transcript_result = transcribe_audio(file_path)
                    prompt = PROMPT_FIELD_PREVIEW_TEXT.format(text_content=transcript_result['text'])
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[{'role': 'user', 'content': prompt}],
                        temperature=0.1,
                        max_tokens=4096
                    )
                    raw_data = parse_ai_response(response.choices[0].message.content)
                elif is_text_file(file.filename):
                    raw_file_text = parse_text_file(file_path)
                    processed_text = preprocess_text(raw_file_text)
                    prompt = PROMPT_FIELD_PREVIEW_TEXT.format(text_content=processed_text)
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[{'role': 'user', 'content': prompt}],
                        temperature=0.1,
                        max_tokens=4096
                    )
                    raw_data = parse_ai_response(response.choices[0].message.content)
                elif file_ext == '.pdf':
                    image_paths = pdf_to_images(file_path, app.config['UPLOAD_FOLDER'])
                    if image_paths:
                        raw_data, _ = extract_medical_data(
                            image_paths[0], PROMPT_FIELD_PREVIEW, app.config['UPLOAD_FOLDER']
                        )
                        for img_path in image_paths:
                            try:
                                os.remove(img_path)
                            except Exception:
                                pass
                else:
                    raw_data, _ = extract_medical_data(
                        file_path, PROMPT_FIELD_PREVIEW, app.config['UPLOAD_FOLDER']
                    )
            finally:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                except Exception:
                    pass
        else:
            return jsonify({"status": "error", "msg": "请提供文件或文本内容"})

        if not raw_data or 'error' in raw_data:
            return jsonify({
                "status": "error",
                "msg": raw_data.get('error', '分析失败') if raw_data else '分析失败'
            })

        fields = raw_data.get('available_fields', [])
        fields = [f for f in fields if f.get('confidence', 0) >= 0.5]
        category_order = {'基本信息': 0, '检验结果': 1, '诊疗记录': 2, '护理评估': 3, '其他': 4}
        fields.sort(key=lambda x: category_order.get(x.get('category', '其他'), 4))
        return jsonify({"status": "success", "fields": fields, "raw_data": raw_data})
    except Exception as e:
        return jsonify({"status": "error", "msg": f"字段预览失败: {str(e)}"})


@app.route('/api/extract_selected', methods=['POST'])
def api_extract_selected_fields():
    """根据用户选择的字段执行提取并存储。"""
    data = request.get_json()
    if not data:
        return jsonify({"status": "error", "msg": "无数据"})

    selected_fields = data.get('selected_fields', [])
    role_id = data.get('role_id', 'other')
    cached_raw_data = data.get('raw_data')
    text_content = data.get('text_content', '').strip()

    if not selected_fields:
        return jsonify({"status": "error", "msg": "请至少选择一个字段"})

    results = []
    errors = []

    try:
        extracted_data = {}

        if cached_raw_data:
            all_fields = cached_raw_data.get('available_fields', [])
            for field in all_fields:
                if field.get('field_name') in selected_fields:
                    extracted_data[field['field_name']] = field.get('example_value')
        elif text_content:
            masked_content, _privacy_report = desensitize_text(text_content)
            ai_prompt, _ = generate_template_prompt(role_id, selected_fields)
            parsed, raw_text = extract_from_transcript(masked_content, ai_prompt)
            if 'error' not in parsed:
                extracted_data = parsed.get('custom_fields', parsed)
            else:
                return jsonify({"status": "error", "msg": parsed.get('error', '提取失败')})
        else:
            return jsonify({"status": "error", "msg": "缺少数据来源"})

        case_number = f"DYN_{datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:6].upper()}"
        record_id = str(uuid.uuid4())
        create_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        conn = get_db()
        cursor = conn.cursor()
        cursor.execute(
            '''INSERT INTO medical_records
               (id, case_number, original_filename, role_id, template_id,
                extracted_data, confidence_data, raw_text, create_time,
                source_type, module_type)
               VALUES (?, ?, ?, ?, 'dynamic_extract', ?, NULL, NULL, ?, 'text', 'dynamic_extract')''',
            (record_id, case_number, '自定义字段提取', role_id,
             json.dumps(extracted_data, ensure_ascii=False), create_time)
        )
        conn.commit()
        conn.close()

        results.append({
            "id": record_id,
            "case_number": case_number,
            "filename": "自定义字段提取",
            "role_id": role_id,
            "template_name": "自定义字段",
            "display_layout": "table",
            "source_type": "text",
            "module_type": "dynamic_extract",
            "data": extracted_data,
            "create_time": create_time
        })
    except Exception as e:
        errors.append(f"提取失败: {str(e)}")

    return jsonify({
        "status": "success" if results else "error",
        "results": results,
        "errors": errors,
        "msg": f"成功提取 {len(results)} 份" if results else "提取失败"
    })

# ========== 路由: 核心功能 ==========

@app.route('/')
def index():
    init_db()
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_and_recognize():
    """上传文件并 AI 识别（支持角色/模板选择，支持图片和音频）。"""
    if 'files' not in request.files:
        return jsonify({"status": "error", "msg": "未选择文件"})

    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        return jsonify({"status": "error", "msg": "未选择有效文件"})

    role_id = request.form.get('role_id', 'other')
    template_id = request.form.get('template_id', 'tpl_researcher_default')
    module_type = request.form.get('module_type', '')

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT ai_prompt, template_name, display_layout FROM extraction_templates WHERE template_id=?",
        (template_id,)
    )
    tpl_row = cursor.fetchone()
    conn.close()

    if not tpl_row:
        return jsonify({"status": "error", "msg": "模板不存在"})

    ai_prompt = tpl_row['ai_prompt']
    template_name = tpl_row['template_name']
    display_layout = tpl_row['display_layout']

    results = []
    errors = []

    for file in files:
        is_audio = is_audio_file(file.filename)
        is_image = allowed_file(file.filename)

        if not is_audio and not is_image:
            errors.append(f"不支持的格式: {file.filename}")
            continue

        file_ext = os.path.splitext(file.filename)[1].lower()
        temp_name = f"{uuid.uuid4().hex}{file_ext}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_name)
        file.save(file_path)

        try:
            if is_audio:
                result_data = _process_audio_file(
                    file_path, file.filename, role_id, template_id,
                    ai_prompt, template_name, display_layout
                )
                if result_data.get('error'):
                    errors.append(f"{file.filename}: {result_data['error']}")
                else:
                    results.append(result_data)
            elif file_ext == '.pdf':
                _process_pdf_file(
                    file_path, file.filename, role_id, template_id,
                    ai_prompt, template_name, display_layout,
                    module_type, results, errors
                )
            else:
                data, raw_text = extract_medical_data(file_path, ai_prompt, app.config['UPLOAD_FOLDER'])
                if "error" in data:
                    errors.append(f"{file.filename}: {data['error']}")
                    continue
                _save_image_record(
                    data, raw_text, file.filename, role_id, template_id,
                    template_name, display_layout, module_type or 'image_ocr', results
                )
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


def _process_pdf_file(pdf_path, filename, role_id, template_id,
                      ai_prompt, template_name, display_layout,
                      module_type, results, errors):
    """PDF 三层处理策略：嵌入文本 → 本地OCR → 多模态逐页识别。"""
    pdf_data = None
    pdf_raw = None

    # 第1层: 提取 PDF 嵌入文本（数字 PDF）
    embedded_text = extract_pdf_embedded_text(pdf_path)
    if embedded_text and len(embedded_text) >= 20:
        print(f"[PDF] 提取到嵌入文本({len(embedded_text)}字符)，使用LLM结构化")
        pdf_data, pdf_raw = extract_from_ocr_text(embedded_text, ai_prompt)
        if 'error' in pdf_data:
            print("[PDF] 嵌入文本结构化失败，尝试OCR")
            pdf_data = None

    # 第2层: 本地 OCR（扫描件 PDF）
    if pdf_data is None:
        try:
            from image_util import HAS_TESSERACT
            if HAS_TESSERACT:
                ocr_text = local_ocr_pdf(pdf_path, app.config['UPLOAD_FOLDER'])
                if ocr_text and len(ocr_text) >= 10:
                    print(f"[PDF] OCR识别成功({len(ocr_text)}字符)，使用LLM结构化")
                    pdf_data, pdf_raw = extract_from_ocr_text(ocr_text, ai_prompt)
                    if 'error' in pdf_data:
                        print("[PDF] OCR文本结构化失败，回退到多模态识别")
                        pdf_data = None
        except Exception as ocr_error:
            print(f"[PDF] OCR处理失败: {ocr_error}")

    # 第3层: 多模态逐页识别
    if pdf_data is None:
        print("[PDF] 使用多模态模型逐页识别")
        image_paths = pdf_to_images(pdf_path, app.config['UPLOAD_FOLDER'])
        for img_path in image_paths:
            try:
                data, raw_text = extract_medical_data_multimodal(
                    img_path, ai_prompt, app.config['UPLOAD_FOLDER']
                )
                if "error" in data:
                    errors.append(f"{filename}: {data['error']}")
                    continue
                _save_image_record(
                    data, raw_text, filename, role_id, template_id,
                    template_name, display_layout, module_type or 'image_ocr', results
                )
            finally:
                try:
                    if os.path.exists(img_path):
                        os.remove(img_path)
                except Exception:
                    pass
        return  # 逐页处理完毕，直接返回

    # 前两层成功，保存单条记录
    if "error" in pdf_data:
        errors.append(f"{filename}: {pdf_data['error']}")
        return
    _save_image_record(
        pdf_data, pdf_raw, filename, role_id, template_id,
        template_name, display_layout, module_type or 'image_ocr', results
    )


def _save_image_record(data, raw_text, filename, role_id, template_id,
                       template_name, display_layout, module_type, results):
    """将图片/PDF 识别结果保存到数据库并追加到 results 列表。"""
    case_number = f"CASE_{datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:6].upper()}"
    record_id = str(uuid.uuid4())
    create_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    confidence_data = data.pop('confidence', {})

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        '''INSERT INTO medical_records
           (id, case_number, original_filename, role_id, template_id,
            extracted_data, confidence_data, raw_text, create_time, source_type, module_type)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'image', ?)''',
        (record_id, case_number, filename, role_id, template_id,
         json.dumps(data, ensure_ascii=False),
         json.dumps(confidence_data, ensure_ascii=False),
         raw_text, create_time, module_type)
    )
    conn.commit()
    conn.close()

    results.append({
        "id": record_id,
        "case_number": case_number,
        "filename": filename,
        "role_id": role_id,
        "template_name": template_name,
        "display_layout": display_layout,
        "source_type": "image",
        "module_type": module_type,
        "data": data,
        "confidence": confidence_data,
        "create_time": create_time
    })


def _process_audio_file(audio_path, filename, role_id, template_id,
                        ai_prompt, template_name, display_layout):
    """处理单个音频文件：语音转写 → 结构化提取 → 可选质性分析。"""
    try:
        transcript_result = transcribe_audio(audio_path)
        transcript_text = transcript_result['text']
        transcript_text, _privacy_report = desensitize_text(transcript_text)

        data, raw_text = extract_from_transcript(transcript_text, ai_prompt)
        if "error" in data:
            return {"error": data.get('error', '文本提取失败')}

        qual_result = None
        if role_id == 'other':
            try:
                qual_result = qualitative_analysis(transcript_text)
            except Exception as qual_error:
                print(f"[WARN] 质性分析失败: {qual_error}")

        case_number = f"AUDIO_{datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:6].upper()}"
        record_id = str(uuid.uuid4())
        create_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        confidence_data = data.pop('confidence', {})

        conn = get_db()
        cursor = conn.cursor()
        cursor.execute(
            '''INSERT INTO medical_records
               (id, case_number, original_filename, role_id, template_id,
                extracted_data, confidence_data, raw_text, create_time,
                source_type, audio_transcript, qualitative_data, module_type)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'audio', ?, ?, 'voice_input')''',
            (record_id, case_number, filename, role_id, template_id,
             json.dumps(data, ensure_ascii=False),
             json.dumps(confidence_data, ensure_ascii=False),
             raw_text, create_time,
             transcript_text,
             json.dumps(qual_result, ensure_ascii=False) if qual_result else None)
        )
        conn.commit()
        conn.close()

        return {
            "id": record_id,
            "case_number": case_number,
            "filename": filename,
            "role_id": role_id,
            "template_name": template_name,
            "display_layout": display_layout,
            "source_type": "audio",
            "transcript": transcript_text,
            "qualitative_analysis": qual_result,
            "data": data,
            "confidence": confidence_data,
            "create_time": create_time
        }
    except Exception as e:
        return {"error": f"音频处理失败: {str(e)}"}


@app.route('/upload_text', methods=['POST'])
def upload_text():
    """文本输入模块：处理粘贴文本或 txt/docx 文件上传。"""
    role_id = request.form.get('role_id', 'other')
    template_id = request.form.get('template_id', 'tpl_researcher_default')
    text_content = request.form.get('text_content', '').strip()

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT ai_prompt, template_name, display_layout FROM extraction_templates WHERE template_id=?",
        (template_id,)
    )
    tpl_row = cursor.fetchone()
    conn.close()

    if not tpl_row:
        return jsonify({"status": "error", "msg": "模板不存在"})

    ai_prompt = tpl_row['ai_prompt']
    template_name = tpl_row['template_name']
    display_layout = tpl_row['display_layout']

    results = []
    errors = []

    # 模式1：直接粘贴文本
    if text_content:
        try:
            processed_text = preprocess_text(text_content)
            if len(processed_text) < 5:
                return jsonify({"status": "error", "msg": "文本内容过短，请输入更多内容"})
            processed_text, _privacy_report = desensitize_text(processed_text)
            data, raw_text = extract_from_transcript(processed_text, ai_prompt)
            if "error" in data:
                return jsonify({"status": "error", "msg": data.get('error', '文本提取失败')})

            case_number = f"TEXT_{datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:6].upper()}"
            record_id = str(uuid.uuid4())
            create_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            confidence_data = data.pop('confidence', {})

            conn = get_db()
            cursor = conn.cursor()
            cursor.execute(
                '''INSERT INTO medical_records
                   (id, case_number, original_filename, role_id, template_id,
                    extracted_data, confidence_data, raw_text, create_time,
                    source_type, module_type, text_source)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'text', 'text_input', ?)''',
                (record_id, case_number, '粘贴文本', role_id, template_id,
                 json.dumps(data, ensure_ascii=False),
                 json.dumps(confidence_data, ensure_ascii=False),
                 raw_text, create_time, processed_text)
            )
            conn.commit()
            conn.close()

            results.append({
                "id": record_id,
                "case_number": case_number,
                "filename": "粘贴文本",
                "role_id": role_id,
                "template_name": template_name,
                "display_layout": display_layout,
                "source_type": "text",
                "module_type": "text_input",
                "text_source": processed_text,
                "data": data,
                "confidence": confidence_data,
                "create_time": create_time
            })
        except Exception as e:
            errors.append(f"文本处理失败: {str(e)}")

    # 模式2：文件上传
    files = request.files.getlist('files')
    for file in files:
        if not file or file.filename == '':
            continue
        if not is_text_file(file.filename):
            errors.append(f"不支持的格式: {file.filename}")
            continue

        file_ext = os.path.splitext(file.filename)[1].lower()
        temp_name = f"{uuid.uuid4().hex}{file_ext}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_name)
        file.save(file_path)

        try:
            raw_file_text = parse_text_file(file_path)
            processed_text = preprocess_text(raw_file_text)
            if len(processed_text) < 5:
                errors.append(f"{file.filename}: 文件内容过短或为空")
                continue
            processed_text, _privacy_report = desensitize_text(processed_text)
            data, raw_text = extract_from_transcript(processed_text, ai_prompt)
            if "error" in data:
                errors.append(f"{file.filename}: {data.get('error', '提取失败')}")
                continue

            case_number = f"TEXT_{datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:6].upper()}"
            record_id = str(uuid.uuid4())
            create_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            confidence_data = data.pop('confidence', {})

            conn = get_db()
            cursor = conn.cursor()
            cursor.execute(
                '''INSERT INTO medical_records
                   (id, case_number, original_filename, role_id, template_id,
                    extracted_data, confidence_data, raw_text, create_time,
                    source_type, module_type, text_source)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'text', 'text_input', ?)''',
                (record_id, case_number, file.filename, role_id, template_id,
                 json.dumps(data, ensure_ascii=False),
                 json.dumps(confidence_data, ensure_ascii=False),
                 raw_text, create_time, processed_text)
            )
            conn.commit()
            conn.close()

            results.append({
                "id": record_id,
                "case_number": case_number,
                "filename": file.filename,
                "role_id": role_id,
                "template_name": template_name,
                "display_layout": display_layout,
                "source_type": "text",
                "module_type": "text_input",
                "text_source": processed_text,
                "data": data,
                "confidence": confidence_data,
                "create_time": create_time
            })
        except Exception as e:
            errors.append(f"{file.filename}: {str(e)}")
        finally:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception:
                pass

    if not results and not errors:
        return jsonify({"status": "error", "msg": "请输入文本或上传文件"})

    return jsonify({
        "status": "success" if results else "error",
        "results": results,
        "errors": errors,
        "msg": f"成功处理 {len(results)} 份" + (f"，{len(errors)} 份失败" if errors else "")
    })


@app.route('/qualitative_analyze', methods=['POST'])
def qualitative_analyze():
    """质性研究模块：独立的质性分析入口（支持音频/文本文件及粘贴文本）。"""
    analysis_type = request.form.get('analysis_type', 'interview')
    text_content = request.form.get('text_content', '').strip()

    results = []
    errors = []

    # 模式1：音频/文本文件
    files = request.files.getlist('files')
    for file in files:
        if not file or file.filename == '':
            continue

        file_ext = os.path.splitext(file.filename)[1].lower()
        temp_name = f"{uuid.uuid4().hex}{file_ext}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_name)
        file.save(file_path)

        try:
            if is_audio_file(file.filename):
                transcript_result = transcribe_audio(file_path)
                transcript_text = transcript_result['text']
                source_type = 'audio'
            elif is_text_file(file.filename):
                raw_text = parse_text_file(file_path)
                transcript_text = preprocess_text(raw_text)
                source_type = 'text'
            else:
                errors.append(f"不支持的格式: {file.filename}，请上传音频或文本文件")
                continue

            if len(transcript_text) < 10:
                errors.append(f"{file.filename}: 内容过短，无法进行质性分析")
                continue

            transcript_text, _privacy_report = desensitize_text(transcript_text)
            qual_result = qualitative_analysis_enhanced(transcript_text, analysis_type)

            case_number = f"QUAL_{datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:6].upper()}"
            record_id = str(uuid.uuid4())
            create_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            conn = get_db()
            cursor = conn.cursor()
            cursor.execute(
                '''INSERT INTO medical_records
                   (id, case_number, original_filename, role_id, template_id,
                    extracted_data, raw_text, create_time,
                    source_type, module_type, audio_transcript, qualitative_data, analysis_type)
                   VALUES (?, ?, ?, NULL, NULL, NULL, NULL, ?, ?, 'qualitative', ?, ?, ?)''',
                (record_id, case_number, file.filename, create_time,
                 source_type, transcript_text,
                 json.dumps(qual_result, ensure_ascii=False), analysis_type)
            )
            conn.commit()
            conn.close()

            results.append({
                "id": record_id,
                "case_number": case_number,
                "filename": file.filename,
                "source_type": source_type,
                "module_type": "qualitative",
                "analysis_type": analysis_type,
                "transcript": transcript_text,
                "qualitative_analysis": qual_result,
                "create_time": create_time
            })
        except Exception as e:
            errors.append(f"{file.filename}: {str(e)}")
        finally:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception:
                pass

    # 模式2：直接粘贴文本
    if text_content and not results:
        try:
            processed_text = preprocess_text(text_content)
            if len(processed_text) < 10:
                return jsonify({"status": "error", "msg": "文本内容过短，无法进行质性分析"})

            processed_text, _privacy_report = desensitize_text(processed_text)
            qual_result = qualitative_analysis_enhanced(processed_text, analysis_type)

            case_number = f"QUAL_{datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:6].upper()}"
            record_id = str(uuid.uuid4())
            create_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            conn = get_db()
            cursor = conn.cursor()
            cursor.execute(
                '''INSERT INTO medical_records
                   (id, case_number, original_filename, role_id, template_id,
                    extracted_data, raw_text, create_time,
                    source_type, module_type, audio_transcript, qualitative_data, analysis_type)
                   VALUES (?, ?, ?, NULL, NULL, NULL, NULL, ?, 'text', 'qualitative', ?, ?, ?)''',
                (record_id, case_number, '粘贴文本', create_time,
                 processed_text,
                 json.dumps(qual_result, ensure_ascii=False), analysis_type)
            )
            conn.commit()
            conn.close()

            results.append({
                "id": record_id,
                "case_number": case_number,
                "filename": "粘贴文本",
                "source_type": "text",
                "module_type": "qualitative",
                "analysis_type": analysis_type,
                "transcript": processed_text,
                "qualitative_analysis": qual_result,
                "create_time": create_time
            })
        except Exception as e:
            errors.append(f"质性分析失败: {str(e)}")

    if not results and not errors:
        return jsonify({"status": "error", "msg": "请上传文件或输入文本"})

    return jsonify({
        "status": "success" if results else "error",
        "results": results,
        "errors": errors,
        "msg": f"成功分析 {len(results)} 份" + (f"，{len(errors)} 份失败" if errors else "")
    })

# ========== 路由: 记录管理 ==========

@app.route('/records', methods=['GET'])
def get_records():
    """获取病历记录列表，支持按角色和模块类型过滤。"""
    role_id = request.args.get('role_id', None)
    module_type = request.args.get('module_type', None)

    conn = get_db()
    cursor = conn.cursor()
    conditions = []
    params = []
    if role_id:
        conditions.append("role_id=?")
        params.append(role_id)
    if module_type:
        conditions.append("module_type=?")
        params.append(module_type)

    where_clause = (" WHERE " + " AND ".join(conditions)) if conditions else ""
    cursor.execute(
        f'''SELECT id, case_number, original_filename, role_id, template_id,
                   create_time, source_type, module_type
            FROM medical_records{where_clause} ORDER BY create_time DESC''',
        params
    )
    rows = cursor.fetchall()
    conn.close()

    records = [
        {
            "id": row['id'],
            "case_number": row['case_number'],
            "filename": row['original_filename'],
            "role_id": row['role_id'] or 'other',
            "template_id": row['template_id'] or '',
            "create_time": row['create_time'],
            "source_type": row['source_type'] or 'image',
            "module_type": row['module_type'] or 'image_ocr'
        }
        for row in rows
    ]
    return jsonify({"status": "success", "records": records})


@app.route('/record/<record_id>', methods=['GET'])
def get_record_detail(record_id):
    """获取单条病历记录的完整详情。"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM medical_records WHERE id = ?', (record_id,))
    row = cursor.fetchone()

    template_name = ''
    display_layout = 'table'
    if row and row['template_id']:
        cursor.execute(
            "SELECT template_name, display_layout FROM extraction_templates WHERE template_id=?",
            (row['template_id'],)
        )
        tpl = cursor.fetchone()
        if tpl:
            template_name = tpl['template_name']
            display_layout = tpl['display_layout']
    conn.close()

    if not row:
        return jsonify({"status": "error", "msg": "记录不存在"})

    role_id = row['role_id'] or 'other'

    if row['extracted_data']:
        extracted_data = json.loads(row['extracted_data'])
        confidence_data = json.loads(row['confidence_data']) if row['confidence_data'] else {}
    else:
        # 兼容旧格式
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
            "create_time": row['create_time'],
            "source_type": row['source_type'] or 'image',
            "module_type": row['module_type'] or 'image_ocr',
            "audio_transcript": row['audio_transcript'] if row['source_type'] == 'audio' else None,
            "qualitative_data": json.loads(row['qualitative_data']) if row['qualitative_data'] else None,
            "text_source": row['text_source'] if row['source_type'] == 'text' else None,
            "analysis_type": row['analysis_type'] if row['module_type'] == 'qualitative' else None
        }
    })


@app.route('/record/<record_id>', methods=['PUT'])
def update_record(record_id):
    """更新病历记录的提取数据或置信度。"""
    update_data = request.get_json()
    if not update_data:
        return jsonify({"status": "error", "msg": "无更新数据"})

    conn = get_db()
    cursor = conn.cursor()
    if 'extracted_data' in update_data:
        cursor.execute(
            "UPDATE medical_records SET extracted_data=? WHERE id=?",
            (json.dumps(update_data['extracted_data'], ensure_ascii=False), record_id)
        )
    if 'confidence' in update_data:
        cursor.execute(
            "UPDATE medical_records SET confidence_data=? WHERE id=?",
            (json.dumps(update_data['confidence'], ensure_ascii=False), record_id)
        )
    conn.commit()
    conn.close()
    return jsonify({"status": "success", "msg": "数据已更新"})


@app.route('/record/<record_id>', methods=['DELETE'])
def delete_record(record_id):
    """删除单条病历记录。"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM medical_records WHERE id = ?', (record_id,))
    conn.commit()
    conn.close()
    return jsonify({"status": "success", "msg": "记录已删除"})

# ========== 路由: 导出 ==========

@app.route('/export', methods=['POST'])
def export_excel():
    """按选中记录或角色导出 Excel。"""
    req_data = request.get_json() or {}
    record_ids = req_data.get('record_ids', [])
    role_filter = req_data.get('role_id', None)

    conn = get_db()
    cursor = conn.cursor()
    if record_ids:
        placeholders = ','.join(['?'] * len(record_ids))
        cursor.execute(f'SELECT * FROM medical_records WHERE id IN ({placeholders})', record_ids)
    elif role_filter:
        cursor.execute('SELECT * FROM medical_records WHERE role_id=? ORDER BY create_time', (role_filter,))
    else:
        cursor.execute('SELECT * FROM medical_records ORDER BY create_time')
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return jsonify({"status": "error", "msg": "暂无可导出的数据"})

    data_list = _rows_to_export_list(rows)
    excel_path = generate_excel(data_list, app.config['UPLOAD_FOLDER'])
    return send_file(
        excel_path, as_attachment=True,
        download_name=f"临床数据_{datetime.now().strftime('%Y%m%d')}.xlsx"
    )


@app.route('/export_all', methods=['GET'])
def export_all_excel():
    """导出全部或指定角色的病历记录为 Excel。"""
    role_filter = request.args.get('role_id', None)
    conn = get_db()
    cursor = conn.cursor()
    if role_filter:
        cursor.execute('SELECT * FROM medical_records WHERE role_id=? ORDER BY create_time', (role_filter,))
    else:
        cursor.execute('SELECT * FROM medical_records ORDER BY create_time')
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return jsonify({"status": "error", "msg": "暂无可导出的数据"})

    data_list = _rows_to_export_list(rows)
    excel_path = generate_excel(data_list, app.config['UPLOAD_FOLDER'])
    return send_file(
        excel_path, as_attachment=True,
        download_name=f"临床数据_{datetime.now().strftime('%Y%m%d')}.xlsx"
    )


def _rows_to_export_list(rows):
    """将数据库行转为导出数据列表（兼容新旧数据格式）。"""
    conn = get_db()
    cursor = conn.cursor()
    data_list = []
    for row in rows:
        role_id = row['role_id'] or 'other'
        template_name = ''
        if row['template_id']:
            cursor.execute(
                "SELECT template_name FROM extraction_templates WHERE template_id=?",
                (row['template_id'],)
            )
            tpl = cursor.fetchone()
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
            'source_type': row['source_type'] or 'image',
            'audio_transcript': row['audio_transcript'] if row['source_type'] == 'audio' else None,
            'qualitative_data': json.loads(row['qualitative_data']) if row['qualitative_data'] else None,
        })
    conn.close()
    return data_list

# ========== 路由: 统计与数据分析 ==========

@app.route('/clean', methods=['POST'])
def clean_all():
    """清理所有上传文件和数据库记录。"""
    try:
        shutil.rmtree(app.config['UPLOAD_FOLDER'])
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    except Exception:
        pass
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM medical_records')
    conn.commit()
    conn.close()
    return jsonify({"status": "success", "msg": "所有数据已清理"})


@app.route('/stats', methods=['GET'])
def get_stats():
    """获取记录总数及按角色分布统计。"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) as cnt FROM medical_records')
    total = cursor.fetchone()['cnt']
    cursor.execute("SELECT role_id, COUNT(*) as cnt FROM medical_records GROUP BY role_id")
    by_role = {(row['role_id'] or 'other'): row['cnt'] for row in cursor.fetchall()}
    conn.close()
    return jsonify({"status": "success", "total_records": total, "by_role": by_role})


@app.route('/data_analysis/fields', methods=['GET'])
def data_analysis_fields():
    """获取选中记录的所有可用数值字段路径。"""
    record_ids = request.args.getlist('ids')
    if not record_ids:
        return jsonify({"status": "error", "msg": "请选择记录"})

    conn = get_db()
    cursor = conn.cursor()
    placeholders = ','.join(['?'] * len(record_ids))
    cursor.execute(
        f'SELECT extracted_data FROM medical_records WHERE id IN ({placeholders})',
        record_ids
    )
    rows = cursor.fetchall()
    conn.close()

    all_paths = set()
    for row in rows:
        if row['extracted_data']:
            data = json.loads(row['extracted_data'])
            all_paths.update(collect_field_paths(data))

    numeric_fields = []
    text_fields = []
    for path in sorted(all_paths):
        has_numeric = any(
            is_numeric(extract_nested_field(json.loads(row['extracted_data']), path))
            for row in rows if row['extracted_data']
        )
        if has_numeric:
            numeric_fields.append(path)
        else:
            text_fields.append(path)

    return jsonify({
        "status": "success",
        "numeric_fields": numeric_fields,
        "text_fields": text_fields
    })


@app.route('/data_analysis/analyze', methods=['POST'])
def data_analysis_analyze():
    """数据分析模块：对选中记录进行统计分析。"""
    req_data = request.get_json()
    if not req_data:
        return jsonify({"status": "error", "msg": "无请求数据"})

    record_ids = req_data.get('record_ids', [])
    fields = req_data.get('fields', [])
    analysis_type = req_data.get('analysis_type', 'descriptive')

    if not record_ids:
        return jsonify({"status": "error", "msg": "请选择至少1条记录"})
    if not fields:
        return jsonify({"status": "error", "msg": "请选择至少1个分析字段"})

    try:
        result = analyze_structured_data(record_ids, fields, analysis_type)
        return jsonify({"status": "success", **result})
    except Exception as e:
        return jsonify({"status": "error", "msg": f"分析失败: {str(e)}"})

# ========== 启动入口 ==========

if __name__ == '__main__':
    print("=" * 50)
    print("  临床科研病历AI识别与结构化提取工具（多角色版）")
    print("  访问地址: http://localhost:7860")
    print("  按 Ctrl+C 停止服务")
    print("=" * 50)
    init_db()
    app.run(host='0.0.0.0', port=7860, debug=False)
