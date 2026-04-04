# -*- coding: utf-8 -*-
"""
MedSnap AI 模型客户端模块
封装 OpenAI 兼容客户端初始化、模型配置和 AI 响应解析。

模型说明:
  MODEL_NAME       - 主力文本/结构化提取模型（Qwen3-VL）
  MODEL_NAME_OMNI  - 全模态模型，负责 OCR（图片）和 ASR（音频）（Qwen2.5-Omni）
"""

import os
import re
import json
import base64
from openai import OpenAI

# ========== 模型配置 ==========

API_KEY = os.environ.get("MODELSCOPE_API_KEY", "")
if not API_KEY:
    print("[WARN] 环境变量 MODELSCOPE_API_KEY 未设置，AI识别功能将不可用")

client = OpenAI(
    base_url='https://api-inference.modelscope.cn/v1',
    api_key=API_KEY
)

# 主力文本结构化提取模型
MODEL_NAME = "Qwen/Qwen3-VL-235B-A22B-Instruct"

# 全模态模型：同时处理图片 OCR 和音频 ASR
MODEL_NAME_OMNI = "Qwen/Qwen2.5-Omni-7B"

# ========== 运行时模型切换 ==========

def get_model_config():
    """获取当前模型配置。"""
    return {
        "model_name": MODEL_NAME,
        "model_name_omni": MODEL_NAME_OMNI,
        "base_url": client.base_url,
    }

def set_model_config(model_name=None, model_name_omni=None, base_url=None, api_key=None):
    """
    运行时更新模型配置。
    只传入需要修改的字段，未传入的字段保持不变。
    """
    global MODEL_NAME, MODEL_NAME_OMNI, client

    if model_name:
        MODEL_NAME = model_name
    if model_name_omni:
        MODEL_NAME_OMNI = model_name_omni

    # 如果 base_url 或 api_key 有变化，重新初始化 client
    if base_url or api_key:
        new_base_url = base_url or str(client.base_url)
        new_api_key = api_key or client.api_key
        client = OpenAI(base_url=new_base_url, api_key=new_api_key)


# ========== 媒体文件工具 ==========

def audio_to_base64(audio_path):
    """将音频文件读取并编码为 Base64 字符串。"""
    with open(audio_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode('utf-8')


# ========== AI 响应解析 ==========

def parse_ai_response(raw_text):
    """
    解析 AI 返回的原始文本，提取合法 JSON 对象。
    支持去除  推理过程标签、markdown 代码块包装。
    """
    text = re.sub(r'', '', raw_text, flags=re.DOTALL)
    text = text.strip()

    # 去除 markdown 代码块包装
    if '```' in text:
        match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
        if match:
            text = match.group(1).strip()

    # 直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 提取最外层 JSON 对象
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    return {
        "error": "AI返回结果解析失败，请重试或检查图片质量",
        "raw_response": raw_text[:500]
    }



