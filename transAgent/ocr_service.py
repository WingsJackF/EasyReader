"""
OCR 服务模块
调用 qwen-vl-ocr 将图片中的内容识别为 LaTeX 格式
"""
import os
import re
from openai import AsyncOpenAI, OpenAI

# 千问 VL OCR 用户指令（同步/异步共用）
_OCR_USER_PROMPT = """请识别本页图像中的学术版面内容，并输出为 **LaTeX 源码**（与后续翻译流水线衔接）。

【输出格式】
- 只输出 LaTeX 正文，不要用 ``` 或 ```latex 代码块包裹。
- 数学公式：行内用 $...$，独立公式可用 $$...$$ 或标准 amsmath 环境（如 equation、align），避免混用 Markdown。

【正文与结构】
- 按阅读顺序输出：标题层级可用 \\section、\\subsection、\\subsubsection。
- 列表、引用、加粗等与版面一致即可；英文原文保持，勿翻译。

【双栏/多栏期刊页（极其重要）】
- 若本页为左右（或更多）分栏，**禁止**按「从左到右横穿整页」扫描，也禁止先读右栏再读左栏。
- 正确顺序：**先完整输出左栏**（从上到下：该栏内的图/表/图注/正文依次写出），**再输出右栏**（同样从上到下）。
- 跨栏句子：若一句话在左栏末断开、在右栏首接续，请把**左栏末片段与右栏首片段**在输出中**接成连续的一句**（可放在左栏段末或单独一行），不要拆成无关的两段。
- 大图、通栏图：若图占据某一栏或通栏，按其在**左栏阅读流**中的位置插入（一般在左栏顶部则优先输出在左栏正文之前）。

【插图（重要，请统一风格）】
- 凡占据明显版心的图、多子图、带图注框的内容，一律用标准浮动体：
  \\begin{figure}[htbp]
    \\centering
    \\includegraphics[width=\\textwidth]{image.png}
    \\caption{图注全文，与图中可见英文一致}
  \\end{figure}
- \\includegraphics 的路径统一使用占位符 **image.png**（不要填真实 URL）。
- **\\caption{...}** 必须写出图中可见的完整图题说明（含 (a)(b) 等小标题若在图注内）。
- **\\label** 规则（二选一，整页保持一致）：
  - 若图题或图旁**明确**出现图号（如 "Figure 2"、"Fig. 2"、"FIG. 2"），则在 \\caption 下一行添加 \\label{fig:2}，数字与图号一致；
  - 若**无法**从图中确定图号，则**不要**编造 \\label，只保留 figure + includegraphics + caption。

【表格】
- 若版面为表格，优先输出 LaTeX tabular / table 环境；若过于复杂可退化为文字描述，但勿用 Markdown 表格语法。

【其它】
- 页眉、页脚、期刊名、页码等若占版面可简要保留为文本行。
- 不要输出解释性中文说明，只输出 LaTeX。"""


def _strip_latex_fences(text: str) -> str:
    """去除模型偶发包裹的 ```latex ... ```"""
    t = text.strip()
    m = re.match(r"^```(?:latex|tex)?\s*\n?", t, re.IGNORECASE)
    if m:
        t = t[m.end() :]
    if t.rstrip().endswith("```"):
        t = re.sub(r"\n?```\s*$", "", t.rstrip())
    return t.strip()


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
                    {"type": "text", "text": _OCR_USER_PROMPT},
                ],
            },
        ],
    )
    raw = completion.choices[0].message.content or ""
    return _strip_latex_fences(raw)


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
                    {"type": "text", "text": _OCR_USER_PROMPT},
                ],
            },
        ],
    )
    raw = completion.choices[0].message.content or ""
    return _strip_latex_fences(raw)
