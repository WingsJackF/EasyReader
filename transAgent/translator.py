"""
翻译服务模块
调用 DeepSeek 将 LaTeX 内容翻译为中文 Markdown
"""
import os
from openai import AsyncOpenAI, OpenAI

# DeepSeek 翻译用的 system prompt（同步/异步共用）
_TRANSLATOR_SYSTEM = (
    "你是一位专业的学术文献翻译助手。请将用户提供的 LaTeX 格式的学术内容翻译成中文，并以 Markdown 格式输出。\n\n"
    "重要：\n"
    "1. 直接输出 Markdown 正文，不要用 ```markdown 或任何代码块包裹。\n"
    "2. 数学公式：行内公式用 $...$，块级公式用 $$...$$ 包裹（不要用 \\begin{equation}）。\n"
    "3. 表格：必须用 Markdown 表格语法（| 列1 | 列2 | 和 --- 分隔行），不要用 LaTeX 的 \\begin{tabular} 或 \\begin{table}。\n"
    "4. 章节用 ## 或 ###，不要用 \\section、\\subsection。\n"
    "5. 保持段落结构和逻辑清晰。\n"
    "6. 若原文有 Markdown 图片行 ![说明](image.png) 等占位链接，请保留整行图片语法：可将方括号内说明译成中文，但必须保留 [](image.png) 形式，不要将图片改成纯文字「**图 N.** …」以免丢失插图位置。"
)


def create_translator_client() -> OpenAI:
    """创建 DeepSeek 客户端（兼容 OpenAI 接口）"""
    return OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",
    )


def latex_to_chinese_markdown(
    client: OpenAI,
    latex_content: str,
    model: str = "deepseek-chat",
) -> str:
    """
    将 LaTeX 格式的学术内容翻译为中文 Markdown。

    Args:
        client: OpenAI 兼容客户端
        latex_content: LaTeX 格式的原文
        model: 使用的 DeepSeek 模型

    Returns:
        翻译后的中文 Markdown 文本
    """
    if not latex_content.strip():
        return ""

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _TRANSLATOR_SYSTEM},
            {
                "role": "user",
                "content": latex_content,
            },
        ],
    )
    return completion.choices[0].message.content or ""


def create_async_translator_client() -> AsyncOpenAI:
    """创建异步 DeepSeek 客户端"""
    return AsyncOpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",
    )


async def latex_to_chinese_markdown_async(
    client: AsyncOpenAI,
    latex_content: str,
    model: str = "deepseek-chat",
) -> str:
    """异步：将 LaTeX 翻译为中文 Markdown"""
    if not latex_content.strip():
        return ""

    completion = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _TRANSLATOR_SYSTEM},
            {
                "role": "user",
                "content": latex_content,
            },
        ],
    )
    return completion.choices[0].message.content or ""
