"""
配置模块
从环境变量读取配置，便于部署和切换环境
"""
import os

from dotenv import load_dotenv

load_dotenv()


# OSS 配置（阿里云凭证通过 OSS_ACCESS_KEY_ID / OSS_ACCESS_KEY_SECRET 加载）
OSS_REGION = os.getenv("OSS_REGION", "oss-cn-hangzhou")
OSS_ENDPOINT = os.getenv("OSS_ENDPOINT", "https://oss-cn-hangzhou.aliyuncs.com")  # 华东1
OSS_BUCKET = os.getenv("OSS_BUCKET", "your-bucket-name")

# API Keys
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")  # 千问 OCR
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")    # DeepSeek 翻译

# PDF 渲染
PDF_DPI = int(os.getenv("PDF_DPI", "200"))

# 并发限制（OCR/翻译同时请求数，避免 API 限流）
CONCURRENCY_LIMIT = int(os.getenv("CONCURRENCY_LIMIT", "5"))

# Paddle 版面 → 插图裁剪（transAgent 可选，需 PADDLE_OCR_TOKEN）
PADDLE_OCR_TOKEN = os.getenv("PADDLE_OCR_TOKEN", "").strip()
FIGURE_EXTRACT_DPI = int(os.getenv("FIGURE_EXTRACT_DPI", "300"))
PADDLE_LAYOUT_MODEL = os.getenv("PADDLE_LAYOUT_MODEL", "PaddleOCR-VL-1.5")
PADDLE_LAYOUT_USE_CHART = os.getenv("PADDLE_LAYOUT_USE_CHART", "true").lower() in (
    "1",
    "true",
    "yes",
)
FIGURE_CROP_PADDING = int(os.getenv("FIGURE_CROP_PADDING", "12"))
FIGURE_CAPTION_GAP_PADDLE = float(os.getenv("FIGURE_CAPTION_GAP_PADDLE", "18"))
