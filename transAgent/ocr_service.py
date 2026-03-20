"""
OCR 服务模块
调用 qwen-vl-ocr 将图片中的内容识别为 LaTeX 格式
"""
import os
from openai import AsyncOpenAI, OpenAI


def create_ocr_client() -> OpenAI:
    """创建 DashScope 客户端（兼容 OpenAI 接口）"""
    return OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


def image_to_latex(
    client: OpenAI,
    image_url: str,
    model: str = "qwen-vl-ocr-2025-11-20",
) -> str:
    """
    将图片中的内容识别为 LaTeX 格式。

    Args:
        client: OpenAI 兼容客户端
        image_url: 图片 URL（需公网可访问）
        model: 使用的 OCR 模型

    Returns:
        识别出的 LaTeX 文本
    """
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": "请将图像中的数学公式和文本内容转换为 LaTeX 格式输出，仅输出 LaTeX 内容。"},
                ],
            },
        ],
    )
    return completion.choices[0].message.content or ""


def create_async_ocr_client() -> AsyncOpenAI:
    """创建异步 DashScope 客户端"""
    return AsyncOpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


async def image_to_latex_async(
    client: AsyncOpenAI,
    image_url: str,
    model: str = "qwen-vl-ocr-2025-11-20",
) -> str:
    """异步：将图片中的内容识别为 LaTeX 格式"""
    completion = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": "请将图像中的数学公式和文本内容转换为 LaTeX 格式输出，仅输出 LaTeX 内容。"},
                ],
            },
        ],
    )
    return completion.choices[0].message.content or ""
