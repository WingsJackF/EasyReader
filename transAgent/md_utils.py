"""
Markdown 工具：修复渲染问题、转换为 PDF
"""
import re
import subprocess
import sys
from pathlib import Path


def strip_markdown_code_blocks(text: str) -> str:
    """
    去除内容中的 ```markdown 或 ``` 代码块包裹，
    使内部 Markdown 能正常渲染。
    """
    text = text.strip()

    # 匹配 ```markdown 或 ```md 或 ``` 开头的代码块
    for prefix in ("```markdown\n", "```md\n", "```\n"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break

    for suffix in ("\n```", "```"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            break

    return text.strip()


# OCR/翻译里的假图链：括号内路径以 image.png 结尾（大小写不敏感），如 (image.png)、(./image.png)
_PLACEHOLDER_IMAGE = re.compile(
    r"!\[([^\]]*)\]\(([^)]*?)image\.png\)",
    re.IGNORECASE,
)


def _figure_path_from_md_line(line: str) -> str | None:
    m = re.search(r"\((figures/[^)\s]+)\)", line.strip())
    return m.group(1) if m else None


def inline_ocr_placeholder_figures(
    body: str,
    fig_lines: list[str] | None,
) -> tuple[str, list[str]]:
    """
    按**出现顺序**把本页 ``figures/`` 裁剪图填入正文中的 ``image.png`` 占位图：

    - 正文中第 1 个 ``![...](...image.png)`` → 本页第 1 张裁剪图，第 2 个占位 → 第 2 张，以此类推。
    - 占位多于图：余下的 ``image.png`` 保持不变。
    - 图多于占位：多出的图仍留在「本页插图」列表（页首展示）。

    会保留原占位图的 alt 文本，只替换链接路径。
    """
    if not fig_lines:
        return body, []

    paths: list[str] = []
    for line in fig_lines:
        p = _figure_path_from_md_line(line)
        if p:
            paths.append(p)
    if not paths:
        return body, list(fig_lines)

    counter: list[int] = [0]

    def sequential_repl(m: re.Match[str]) -> str:
        idx = counter[0]
        if idx >= len(paths):
            return m.group(0)
        path = paths[idx]
        counter[0] = idx + 1
        alt = m.group(1)
        if alt.strip():
            return f"![{alt}]({path})"
        return f"![]({path})"

    new_body = _PLACEHOLDER_IMAGE.sub(sequential_repl, body)
    consumed = counter[0]
    remaining = fig_lines[consumed:]
    return new_body, remaining


def fix_markdown_file(md_path: str | Path) -> None:
    """
    修复 Markdown 文件中每页内容被 ```markdown 包裹的问题。
    使用正则匹配 ## 第 N 页 后的代码块并去除包裹。
    """
    md_path = Path(md_path)
    content = md_path.read_text(encoding="utf-8")

    # 匹配: ## 第 N 页\n\n```markdown 或 ```\n(内容)\n``` 或 \s*```
    # 替换为: ## 第 N 页\n\n(内容)
    pattern = r"(## 第 \d+ 页)\n\n```(?:markdown|md)?\n(.*?)\n```"
    fixed = re.sub(pattern, r"\1\n\n\2", content, flags=re.DOTALL)

    # 将 LaTeX 转为 Markdown 可渲染格式
    fixed = re.sub(
        r"\\begin\{equation\}\s*(.*?)\s*\\end\{equation\}",
        r"$$\1$$",
        fixed,
        flags=re.DOTALL,
    )
    fixed = re.sub(r"\\subsection\{([^}]*)\}", r"### \1", fixed)
    fixed = re.sub(r"\\subsubsection\{([^}]*)\}", r"#### \1", fixed)
    fixed = re.sub(
        r"\\begin\{center\}\s*(.*?)\s*\\end\{center\}",
        r"<p align=\"center\">\1</p>",
        fixed,
        flags=re.DOTALL,
    )

    md_path.write_text(fixed, encoding="utf-8")
    print(f"已修复: {md_path}")


def md_to_pdf(md_path: str | Path, pdf_path: str | Path | None = None) -> Path:
    """
    使用 pandoc 将 Markdown 转为 PDF（支持 LaTeX 公式）。

    需要安装：pandoc、LaTeX（如 MacTeX / TeX Live）
    """
    md_path = Path(md_path)
    pdf_path = Path(pdf_path or md_path.with_suffix(".pdf"))

    # 使用 xelatex 以更好支持中文
    cmd = [
        "pandoc",
        str(md_path),
        "-o",
        str(pdf_path),
        "--pdf-engine=xelatex",
        "-V",
        "CJKmainfont=Songti SC",  # macOS 宋体，Linux 可改为 "Noto Serif CJK SC"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"pandoc 转换失败: {result.stderr}")

    print(f"已生成 PDF: {pdf_path}")
    return pdf_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法:")
        print("  修复 Markdown: python md_utils.py fix <input.md>")
        print("  转为 PDF:     python md_utils.py pdf <input.md> [output.pdf]")
        sys.exit(1)

    cmd = sys.argv[1].lower()
    if cmd == "fix":
        fix_markdown_file(sys.argv[2])
    elif cmd == "pdf":
        md_to_pdf(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    else:
        print("未知命令，请使用 fix 或 pdf")
        sys.exit(1)
