# -*- coding: utf-8 -*-
"""
MedSnap 图片/PDF 处理工具模块
提供图片预处理、Base64 编码、PDF 转图片及 Qwen2.5-Omni 多模态 OCR 识别功能。

OCR 策略（优先级从高到低）：
  1. 数字 PDF 嵌入文本直接提取（PyMuPDF，无需 AI）
  2. Qwen2.5-Omni 多模态直接识别图片/扫描件（主力 OCR）
"""

import os
import base64
from PIL import Image, ImageFilter, ImageEnhance, ImageOps

from model import client, MODEL_NAME, MODEL_NAME_OMNI, parse_ai_response
from desensitizer import desensitize_structured_data

# ========== 可选依赖检测 ==========

HAS_PYMUPDF = False
try:
    import fitz
    HAS_PYMUPDF = True
except ImportError:
    pass

# Tesseract 保留检测标志以兼容旧调用，但不再作为主力 OCR
HAS_TESSERACT = False

# ========== 图片预处理 ==========

def preprocess_image(image_path, upload_folder):
    """
    对图片进行灰度化、自动对比度、锐化、对比度增强预处理，
    以提升 OCR 和多模态识别的准确率。

    Args:
        image_path: 原始图片路径
        upload_folder: 临时文件存放目录

    Returns:
        str: 预处理后的图片路径（失败时返回原路径）
    """
    try:
        import uuid
        img = Image.open(image_path)
        if img.mode not in ('RGB', 'L'):
            img = img.convert('RGB')
        gray = img.convert('L')
        gray = ImageOps.autocontrast(gray, cutoff=1)
        gray = gray.filter(ImageFilter.SHARPEN)
        enhancer = ImageEnhance.Contrast(gray)
        gray = enhancer.enhance(1.5)
        preprocessed_path = os.path.join(
            upload_folder,
            f"pre_{uuid.uuid4().hex[:8]}_{os.path.basename(image_path)}"
        )
        gray.save(preprocessed_path)
        return preprocessed_path
    except Exception:
        return image_path


def image_to_base64(image_path):
    """将图片文件读取并编码为 Base64 字符串。"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# ========== PDF 处理 ==========

def pdf_to_images(pdf_path, upload_folder):
    """
    使用 PyMuPDF 将 PDF 每页渲染为 PNG 图片。

    Args:
        pdf_path: PDF 文件路径
        upload_folder: 图片输出目录

    Returns:
        list: 各页图片路径列表

    Raises:
        RuntimeError: PyMuPDF 未安装时抛出
    """
    if not HAS_PYMUPDF:
        raise RuntimeError("PDF功能需要PyMuPDF库，请运行: pip install PyMuPDF")
    import uuid
    doc = fitz.open(pdf_path)
    image_paths = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)
        img_path = os.path.join(
            upload_folder,
            f"pdf_page_{uuid.uuid4().hex[:8]}_{page_num}.png"
        )
        pix.save(img_path)
        image_paths.append(img_path)
    doc.close()
    return image_paths


def extract_pdf_embedded_text(pdf_path):
    """
    使用 PyMuPDF 提取数字 PDF 中的嵌入文本（无需 OCR）。

    Args:
        pdf_path: PDF 文件路径

    Returns:
        str: 提取到的文本，失败时返回空字符串
    """
    if not HAS_PYMUPDF:
        return ''
    try:
        doc = fitz.open(pdf_path)
        text_parts = []
        for page in doc:
            text = page.get_text()
            if text and text.strip():
                text_parts.append(text.strip())
        doc.close()
        return '\n\n'.join(text_parts)
    except Exception:
        return ''

# ========== 本地 OCR ==========

def local_ocr(image_path, upload_folder):
    """
    使用 Tesseract 对单张图片进行本地 OCR 识别（中英文混合）。

    Args:
        image_path: 图片路径
        upload_folder: 临时文件目录

    Returns:
        str: 识别出的文本

    Raises:
        RuntimeError: pytesseract 未安装时抛出
    """
    if not HAS_TESSERACT:
        raise RuntimeError(
            "pytesseract 未安装，请运行: pip install pytesseract，并确保系统已安装 Tesseract-OCR"
        )
    preprocessed_path = preprocess_image(image_path, upload_folder)
    try:
        img = Image.open(preprocessed_path)
        text = pytesseract.image_to_string(img, lang='chi_sim+eng', config='--psm 6')
        return text.strip()
    finally:
        if preprocessed_path != image_path and os.path.exists(preprocessed_path):
            try:
                os.remove(preprocessed_path)
            except Exception:
                pass


def local_ocr_pdf(pdf_path, upload_folder):
    """
    对 PDF 每页进行本地 OCR 识别，合并全部文本。

    Args:
        pdf_path: PDF 文件路径
        upload_folder: 临时文件目录

    Returns:
        str: 所有页面 OCR 文本合并结果
    """
    image_paths = pdf_to_images(pdf_path, upload_folder)
    all_text = []
    for img_path in image_paths:
        try:
            page_text = local_ocr(img_path, upload_folder)
            if page_text:
                all_text.append(page_text)
        finally:
            try:
                if os.path.exists(img_path):
                    os.remove(img_path)
            except Exception:
                pass
    return '\n\n'.join(all_text)

# ========== 多模态 AI 识别 ==========

def extract_medical_data_multimodal(image_path, ai_prompt, upload_folder):
    """
    调用 Qwen2.5-Omni 全模态模型直接识别图片，完成 OCR + 结构化提取一步到位。
    返回数据在本地进行脱敏处理。

    Args:
        image_path: 图片路径
        ai_prompt: 提取用的 Prompt
        upload_folder: 临时文件目录

    Returns:
        tuple: (parsed_dict, raw_text)
    """
    print("[OCR] Qwen2.5-Omni 多模态识别：图片将发送至AI服务")
    preprocessed_path = preprocess_image(image_path, upload_folder)
    b64_image = image_to_base64(preprocessed_path)

    ext = os.path.splitext(image_path)[1].lower()
    mime_map = {
        '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
        '.png': 'image/png', '.bmp': 'image/bmp',
        '.tiff': 'image/tiff'
    }
    mime_type = mime_map.get(ext, 'image/jpeg')

    response = client.chat.completions.create(
        model=MODEL_NAME_OMNI,
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
    # 对返回的结构化数据进行本地脱敏
    parsed = desensitize_structured_data(parsed)
    return parsed, raw_text


def extract_medical_data(image_path, ai_prompt, upload_folder):
    """
    图片 OCR 识别入口：直接使用 Qwen2.5-Omni 全模态模型完成识别与结构化提取。

    策略（优先级从高到低）：
      1. 数字 PDF 嵌入文本已在上层提取，此处只处理图片
      2. Qwen2.5-Omni 多模态直接识别（主力 OCR）

    Args:
        image_path: 图片路径
        ai_prompt: 提取用的 Prompt
        upload_folder: 临时文件目录

    Returns:
        tuple: (parsed_dict, raw_text)
    """
    return extract_medical_data_multimodal(image_path, ai_prompt, upload_folder)
