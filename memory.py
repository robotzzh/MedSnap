# -*- coding: utf-8 -*-
"""
MedSnap 数据库与记忆层模块
提供数据库连接、初始化、内置模板写入，以及文件类型校验工具函数。
"""

import os
import sqlite3
import tempfile
import uuid
from datetime import datetime

from template import (
    PROMPT_DOCTOR_MEDICAL_RECORD, PROMPT_DOCTOR_LAB_RESULTS,
    PROMPT_NURSE_ADMISSION, PROMPT_NURSE_BARTHEL, PROMPT_NURSE_MORSE,
    PROMPT_NURSE_BRADEN, PROMPT_NURSE_PAIN, PROMPT_NURSE_RECORD,
    PROMPT_RESEARCHER,
    PROMPT_AUDIO_DOCTOR, PROMPT_AUDIO_NURSE, PROMPT_AUDIO_RESEARCHER,
)

# ========== 文件上传配置 ==========

UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "medical_ocr_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'pdf'}
ALLOWED_AUDIO_EXTENSIONS = {'wav', 'mp3', 'aac', 'amr', 'opus', 'm4a', 'flac'}
ALLOWED_TEXT_EXTENSIONS = {'txt', 'doc', 'docx'}

# ========== 数据库路径 ==========

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'local_medical_data.db')

# ========== 文件类型校验 ==========

def allowed_file(filename):
    """判断文件是否为允许的图片/PDF 格式。"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def is_audio_file(filename):
    """判断文件是否为允许的音频格式。"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_AUDIO_EXTENSIONS


def is_text_file(filename):
    """判断文件是否为允许的文本格式。"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_TEXT_EXTENSIONS

# ========== 数据库连接 ==========

def get_db():
    """获取 SQLite 数据库连接，Row 工厂模式返回字典式行。"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# ========== 数据库初始化 ==========

def init_db():
    """
    初始化数据库表结构，兼容旧版数据库（自动补列），并写入内置模板。
    幂等操作，可重复调用。
    """
    conn = get_db()
    cursor = conn.cursor()

    # 模板表
    cursor.execute('''CREATE TABLE IF NOT EXISTS extraction_templates (
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

    # 病历记录表
    cursor.execute('''CREATE TABLE IF NOT EXISTS medical_records (
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

    # 兼容旧数据库：自动补充缺失列
    existing_columns = {
        row[1] for row in cursor.execute("PRAGMA table_info(medical_records)").fetchall()
    }
    new_columns = [
        'role_id', 'template_id', 'extracted_data', 'confidence_data',
        'source_type', 'audio_transcript', 'qualitative_data',
        'module_type', 'text_source', 'analysis_type'
    ]
    for column in new_columns:
        if column not in existing_columns:
            cursor.execute(f"ALTER TABLE medical_records ADD COLUMN {column} TEXT")

    # 迁移旧记录的 module_type
    cursor.execute(
        "UPDATE medical_records SET module_type='image_ocr' "
        "WHERE module_type IS NULL AND (source_type='image' OR source_type IS NULL)"
    )
    cursor.execute(
        "UPDATE medical_records SET module_type='voice_input' "
        "WHERE module_type IS NULL AND source_type='audio'"
    )

    conn.commit()
    conn.close()

    _init_builtin_templates()


def _init_builtin_templates():
    """
    插入系统内置模板（INSERT OR IGNORE，幂等操作）。
    同时执行旧 role_id 数据迁移（doctor→diagnosis, nurse→nursing, researcher→other）。
    """
    conn = get_db()
    cursor = conn.cursor()
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # 数据迁移：旧 role_id 映射到新分类 ID
    cursor.execute("UPDATE extraction_templates SET role_id='diagnosis' WHERE role_id='doctor'")
    cursor.execute("UPDATE extraction_templates SET role_id='nursing'   WHERE role_id='nurse'")
    cursor.execute("UPDATE extraction_templates SET role_id='other'     WHERE role_id='researcher'")
    cursor.execute("UPDATE medical_records SET role_id='diagnosis' WHERE role_id='doctor'")
    cursor.execute("UPDATE medical_records SET role_id='nursing'   WHERE role_id='nurse'")
    cursor.execute("UPDATE medical_records SET role_id='other'     WHERE role_id='researcher'")

    builtin_templates = [
        # 诊疗模板
        ('tpl_doctor_medical', 'diagnosis', '门诊/住院病历', 'fixed',
         PROMPT_DOCTOR_MEDICAL_RECORD, 'table', now),
        ('tpl_doctor_lab', 'diagnosis', '检查检验结果', 'fixed',
         PROMPT_DOCTOR_LAB_RESULTS, 'table', now),
        # 护理模板
        ('tpl_nurse_admission', 'nursing', '入院护理评估表', 'fixed',
         PROMPT_NURSE_ADMISSION, 'card', now),
        ('tpl_nurse_barthel', 'nursing', 'Barthel自理能力量表', 'fixed',
         PROMPT_NURSE_BARTHEL, 'scale', now),
        ('tpl_nurse_morse', 'nursing', 'Morse跌倒风险量表', 'fixed',
         PROMPT_NURSE_MORSE, 'scale', now),
        ('tpl_nurse_braden', 'nursing', 'Braden压疮风险量表', 'fixed',
         PROMPT_NURSE_BRADEN, 'scale', now),
        ('tpl_nurse_pain', 'nursing', 'NRS/VAS疼痛评估', 'fixed',
         PROMPT_NURSE_PAIN, 'card', now),
        ('tpl_nurse_record', 'nursing', '护理记录单', 'fixed',
         PROMPT_NURSE_RECORD, 'table', now),
        # 其他/科研模板
        ('tpl_researcher_default', 'other', '综合科研数据提取', 'fixed',
         PROMPT_RESEARCHER, 'table', now),
        # 音频模板
        ('tpl_audio_doctor', 'diagnosis', '医患对话录音', 'fixed',
         PROMPT_AUDIO_DOCTOR, 'table', now),
        ('tpl_audio_nurse', 'nursing', '护理交班录音', 'fixed',
         PROMPT_AUDIO_NURSE, 'card', now),
        ('tpl_audio_researcher', 'other', '研究访谈录音', 'fixed',
         PROMPT_AUDIO_RESEARCHER, 'table', now),
    ]

    for template in builtin_templates:
        cursor.execute(
            '''INSERT OR IGNORE INTO extraction_templates
               (template_id, role_id, template_name, template_type, ai_prompt, display_layout, create_time)
               VALUES (?, ?, ?, ?, ?, ?, ?)''',
            template
        )

    # 迁移旧数据：无角色记录归入科研
    cursor.execute(
        '''UPDATE medical_records
           SET role_id='other', template_id='tpl_researcher_default'
           WHERE role_id IS NULL'''
    )

    conn.commit()
    conn.close()
