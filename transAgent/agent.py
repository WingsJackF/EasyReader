"""
文献阅读 Agent
流程：PDF → 图片 → 上传 OSS → OCR → 翻译；启用 --figures 时 **Paddle 插图** 在后台线程与上传/OCR/翻译 **并行**。
支持异步上传、异步 OCR、异步翻译，最后按页码顺序写入。

单次任务的所有产物集中在**一个任务目录**内：
  默认: ./output/<pdf文件名>/
  - <pdf文件名>.md   主 Markdown
  - figures/         插图（启用 --figures 时）
  - artifacts/       每页中间文件：render.png、oss_url.txt、ocr.txt、translation_raw.md、translation_merged.md
"""
import argparse
import asyncio
import re
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from config import (
    CONCURRENCY_LIMIT,
    FIGURE_CAPTION_GAP_PADDLE,
    FIGURE_CROP_PADDING,
    FIGURE_EXTRACT_DPI,
    OSS_BUCKET,
    OSS_ENDPOINT,
    OSS_REGION,
    PADDLE_LAYOUT_MODEL,
    PADDLE_LAYOUT_USE_CHART,
    PADDLE_OCR_TOKEN,
    PDF_DPI,
)
from md_utils import inline_ocr_placeholder_figures
from pdf_processor import get_page_count, pdf_to_images
from oss_uploader import create_oss_client, upload_image
from ocr_service import create_async_ocr_client, image_to_latex_async
from translator import create_async_translator_client, latex_to_chinese_markdown_async


def _parse_page_range(page_range: str, total_pages: int) -> list[int]:
    """解析形如 4-9 的页段（闭区间）。"""
    m = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", page_range)
    if not m:
        raise ValueError("page_range 格式错误，应为 start-end，例如 4-9")
    start = int(m.group(1))
    end = int(m.group(2))
    if start < 1 or end < 1:
        raise ValueError("page_range 页码必须 >= 1")
    if start > end:
        raise ValueError("page_range 起始页不能大于结束页")
    if end > total_pages:
        raise ValueError(f"page_range 超出 PDF 总页数（总页数 {total_pages}）")
    return list(range(start, end + 1))


def _strip_markdown_code_block(text: str) -> str:
    """去除翻译结果中可能出现的 ```markdown 代码块包裹"""
    text = text.strip()
    for prefix in ("```markdown\n", "```md\n", "```\n"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    if text.endswith("\n```"):
        text = text[:-4]
    elif text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _clean_latex_inline_text(text: str) -> str:
    return (
        text.replace(r"\textdagger", "†")
        .replace(r"\dagger", "†")
        .replace(r"\*", "*")
    )


def _fix_latex_for_markdown(text: str) -> str:
    """将 LaTeX 转为 Markdown 可渲染格式"""
    text = re.sub(
        r"\\textsuperscript\{([^{}]*)\}",
        lambda m: f"<sup>{_clean_latex_inline_text(m.group(1))}</sup>",
        text,
    )
    # \begin{equation}...\end{equation} → $$...$$
    text = re.sub(
        r"\\begin\{equation\}\s*(.*?)\s*\\end\{equation\}",
        r"$$\1$$",
        text,
        flags=re.DOTALL,
    )
    for env in ("align", "align*", "gather", "gather*"):
        text = re.sub(
            rf"\\begin\{{{re.escape(env)}\}}\s*(.*?)\s*\\end\{{{re.escape(env)}\}}",
            r"$$\1$$",
            text,
            flags=re.DOTALL,
        )
    # \subsection{标题} → ### 标题
    text = re.sub(r"\\subsection\{([^}]*)\}", r"### \1", text)
    text = re.sub(r"\\subsubsection\{([^}]*)\}", r"#### \1", text)
    # \begin{center}...\end{center} → 居中段落（部分渲染器支持）
    text = re.sub(
        r"\\begin\{center\}\s*(.*?)\s*\\end\{center\}",
        r"<p align=\"center\">\1</p>",
        text,
        flags=re.DOTALL,
    )
    return text


def _artifacts_page_dir(artifacts_root: Path, page_num: int) -> Path:
    return artifacts_root / f"page_{page_num:04d}"


def _write_page_artifacts(
    task_dir: Path,
    *,
    pages: list[tuple[int, bytes]],
    upload_results: list[tuple[int, str]],
    ocr_results: list[tuple[int, str]],
    translate_results: list[tuple[int, str]],
) -> Path:
    """
    在 task_dir/artifacts/page_XXXX/ 下写入每页：渲染图、OSS URL、OCR 文本、翻译原文。
    返回 artifacts 根路径。
    """
    artifacts_root = task_dir / "artifacts"
    artifacts_root.mkdir(parents=True, exist_ok=True)

    for page_num, img_bytes in pages:
        pdir = _artifacts_page_dir(artifacts_root, page_num)
        pdir.mkdir(parents=True, exist_ok=True)
        (pdir / "render.png").write_bytes(img_bytes)

    for page_num, url in upload_results:
        pdir = _artifacts_page_dir(artifacts_root, page_num)
        pdir.mkdir(parents=True, exist_ok=True)
        (pdir / "oss_url.txt").write_text(f"{url}\n", encoding="utf-8")

    for page_num, latex in ocr_results:
        pdir = _artifacts_page_dir(artifacts_root, page_num)
        pdir.mkdir(parents=True, exist_ok=True)
        (pdir / "ocr.txt").write_text(latex, encoding="utf-8")

    for page_num, chinese_md in translate_results:
        pdir = _artifacts_page_dir(artifacts_root, page_num)
        pdir.mkdir(parents=True, exist_ok=True)
        (pdir / "translation_raw.md").write_text(chinese_md, encoding="utf-8")

    (artifacts_root / "README.txt").write_text(
        "每页子目录 page_XXXX/ 说明：\n"
        "- render.png            本页 OCR 用渲染图（与上传 OSS 一致）\n"
        "- oss_url.txt           该页图片的 OSS 公网 URL\n"
        "- ocr.txt               千问 OCR 返回的 LaTeX/文本\n"
        "- translation_raw.md    DeepSeek 翻译原始输出\n"
        "- translation_merged.md 写入总稿前的本页正文（插图占位已替换、含「本页插图」块；无「## 第 N 页」标题）\n",
        encoding="utf-8",
    )
    return artifacts_root


def resolve_task_output_dir(
    pdf_path: Path,
    output_md_path: str | Path | None,
    output_dir: str | Path | None,
) -> tuple[Path, Path]:
    """
    解析单次任务目录与 Markdown 路径（插图、md 均在同一 task_dir）。

    规则:
    - 指定 output_md_path 为**绝对路径**的 .md → task_dir = 其父目录，md = 该文件
    - 指定 output_dir → task_dir = 该路径；若 output_md_path 为相对 .md 名则 md = task_dir / 名
    - 指定 output_md_path 为相对路径 .md → task_dir = 其父目录（相对 cwd），md = 解析后路径
    - 指定 output_md_path 为目录 → task_dir = 该目录，md = task_dir / <stem>.md
    - 都不指定 → task_dir = ./output/<stem>/，md = <stem>.md
    """
    stem = pdf_path.stem

    if output_md_path is not None:
        p = Path(output_md_path).expanduser()
        if p.suffix.lower() == ".md":
            if p.is_absolute():
                md_path = p.resolve()
                task_dir = md_path.parent
                task_dir.mkdir(parents=True, exist_ok=True)
                return task_dir, md_path
            # 相对路径 .md
            if output_dir is not None:
                task_dir = Path(output_dir).expanduser().resolve()
                md_path = (task_dir / p.name).resolve()
            else:
                md_path = (Path.cwd() / p).resolve()
                task_dir = md_path.parent
            task_dir.mkdir(parents=True, exist_ok=True)
            return task_dir, md_path
        # 当作目录
        task_dir = p.resolve()
        md_path = (task_dir / f"{stem}.md").resolve()
        task_dir.mkdir(parents=True, exist_ok=True)
        return task_dir, md_path

    if output_dir is not None:
        task_dir = Path(output_dir).expanduser().resolve()
        md_path = (task_dir / f"{stem}.md").resolve()
    else:
        task_dir = (Path.cwd() / "output" / stem).resolve()
        md_path = (task_dir / f"{stem}.md").resolve()

    task_dir.mkdir(parents=True, exist_ok=True)
    return task_dir, md_path


def process_paper(
    pdf_path: str | Path,
    output_md_path: str | Path | None = None,
    dpi: int = PDF_DPI,
    *,
    output_dir: str | Path | None = None,
    extract_figures: bool = False,
    figure_dpi: int | None = None,
    save_artifacts: bool = True,
    page_range: str | None = None,
) -> str:
    """
    处理一篇 PDF 文献，输出中文 Markdown。

    Args:
        pdf_path: PDF 文件路径
        output_md_path: 可选；Markdown 文件路径，或任务目录（非 .md）
        output_dir: 可选；任务输出根目录。与 output_md_path 组合见 resolve_task_output_dir
        dpi: PDF 转图片分辨率（OCR 用）
        extract_figures: 是否调用 Paddle 版面 API 按 Figure N 裁剪插图并嵌入 Markdown
            （与上传/OCR/翻译并行，写 md 前会等待插图线程结束）
        figure_dpi: 插图裁剪用渲染 DPI，默认读取配置 FIGURE_EXTRACT_DPI（建议 300）
        save_artifacts: 是否在 task_dir/artifacts/ 按页保存渲染图、OSS URL、OCR、翻译中间结果
        page_range: 可选，连续页段（闭区间），格式如 "4-9"

    Returns:
        汇总后的中文 Markdown 内容
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 不存在: {pdf_path}")

    task_dir, out_path = resolve_task_output_dir(
        pdf_path, output_md_path, output_dir
    )
    fdpi = figure_dpi if figure_dpi is not None else FIGURE_EXTRACT_DPI

    if extract_figures and not PADDLE_OCR_TOKEN:
        raise ValueError(
            "启用插图提取需在环境变量或 .env 中设置 PADDLE_OCR_TOKEN（Paddle AI Studio bearer）"
        )

    # 初始化客户端
    oss_client = create_oss_client(region=OSS_REGION, endpoint=OSS_ENDPOINT)
    ocr_client = create_async_ocr_client()
    translator_client = create_async_translator_client()

    paper_name = pdf_path.stem
    total_pages = get_page_count(pdf_path)
    if page_range:
        selected_pages = _parse_page_range(page_range, total_pages)
    else:
        selected_pages = list(range(1, total_pages + 1))
    selected_page_set = set(selected_pages)
    selected_total = len(selected_pages)

    if page_range:
        print(
            f"[1/4] PDF 转图（总 {total_pages} 页，处理页段 {page_range}，共 {selected_total} 页，OCR DPI={dpi}）..."
        )
    else:
        print(f"[1/4] PDF 转图（共 {selected_total} 页，OCR DPI={dpi}）...")

    # Step 1: PDF 转图片
    pages: list[tuple[int, bytes]] = []
    for page_num, img_bytes in pdf_to_images(
        pdf_path,
        dpi=dpi,
        page_numbers=selected_page_set,
    ):
        pages.append((page_num, img_bytes))
        print(f"      [转换] {len(pages)}/{selected_total}")

    # 插图仅依赖 PDF + Paddle，与后续上传/OCR/翻译并行（单独线程，避免阻塞 asyncio 主流程）
    figures_future: Future | None = None
    fig_executor: ThreadPoolExecutor | None = None
    figure_md_by_page: dict[int, list[str]] = {}

    if extract_figures:
        from paddle_figure_extract import extract_figures_from_pdf_page

        def _run_figures_pipeline() -> dict[int, list[str]]:
            fig_root = task_dir / "figures"
            fig_root.mkdir(parents=True, exist_ok=True)
            out_map: dict[int, list[str]] = {}
            print(
                f"[插图·并行] Paddle 版面与裁剪（后台线程，与上传/OCR/翻译同时进行），"
                f"共 {selected_total} 页，DPI={fdpi}…"
            )
            for idx, page_num in enumerate(selected_pages, start=1):
                try:
                    crops = extract_figures_from_pdf_page(
                        pdf_path,
                        page_num,
                        token=PADDLE_OCR_TOKEN,
                        figure_dpi=fdpi,
                        crop_padding=FIGURE_CROP_PADDING,
                        caption_gap_paddle=FIGURE_CAPTION_GAP_PADDLE,
                        use_chart=PADDLE_LAYOUT_USE_CHART,
                        model=PADDLE_LAYOUT_MODEL,
                        verbose_paddle=False,
                    )
                except Exception as e:
                    print(f"      [插图] 第 {page_num} 页失败（已跳过）: {e}")
                    continue
                lines: list[str] = []
                for stem, png_bytes in crops:
                    fname = f"p{page_num:04d}_{stem}.png"
                    (fig_root / fname).write_bytes(png_bytes)
                    alt = stem.replace("_", " ")
                    lines.append(f"![{alt}](figures/{fname})")
                if lines:
                    out_map[page_num] = lines
                print(f"      [插图] 第 {idx}/{selected_total} 页（PDF第{page_num}页） → {len(crops)} 张")
            return out_map

        fig_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="figures")
        figures_future = fig_executor.submit(_run_figures_pipeline)

    # Step 2: 异步上传 OSS
    print(f"[2/4] 上传中...")
    upload_count = [0]

    async def upload_all():
        loop = asyncio.get_running_loop()
        lock = asyncio.Lock()

        async def upload_one(p, b, k):
            result = await loop.run_in_executor(
                pool,
                lambda pp=p, bb=b, kk=k: (
                    pp,
                    upload_image(oss_client, OSS_BUCKET, bb, kk, OSS_ENDPOINT),
                ),
            )
            async with lock:
                upload_count[0] += 1
                print(f"      [上传] {upload_count[0]}/{selected_total}")
            return result

        with ThreadPoolExecutor(max_workers=8) as pool:
            tasks = [
                upload_one(page_num, img_bytes, f"{uuid.uuid4().hex}.png")
                for page_num, img_bytes in pages
            ]
            return await asyncio.gather(*tasks)

    # Step 3: 异步 OCR（限制并发）
    sem_ocr = asyncio.Semaphore(CONCURRENCY_LIMIT)
    ocr_count = [0]

    async def ocr_one(page_num: int, url: str):
        async with sem_ocr:
            result = page_num, await image_to_latex_async(ocr_client, url)
            ocr_count[0] += 1
            print(f"      [OCR] {ocr_count[0]}/{selected_total}")
            return result

    async def ocr_all():
        tasks = [ocr_one(p, url) for p, url in upload_results]
        return await asyncio.gather(*tasks)

    # Step 4: 异步翻译（限制并发）
    sem_translate = asyncio.Semaphore(CONCURRENCY_LIMIT)
    translate_count = [0]

    async def translate_one(page_num: int, latex: str):
        async with sem_translate:
            result = page_num, await latex_to_chinese_markdown_async(
                translator_client, latex
            )
            translate_count[0] += 1
            print(f"      [翻译] {translate_count[0]}/{selected_total}")
            return result

    async def translate_all():
        tasks = [translate_one(p, latex) for p, latex in ocr_results]
        return await asyncio.gather(*tasks)

    translate_results: list[tuple[int, str]]
    try:
        upload_results = asyncio.run(upload_all())
        print(f"[3/4] OCR 中...")
        ocr_results = asyncio.run(ocr_all())
        print(f"[4/4] 翻译中...")
        translate_results = asyncio.run(translate_all())

        if save_artifacts:
            _write_page_artifacts(
                task_dir,
                pages=pages,
                upload_results=list(upload_results),
                ocr_results=list(ocr_results),
                translate_results=list(translate_results),
            )
            print(
                f"      → 已写入中间文件: {task_dir / 'artifacts'}/page_XXXX/"
                f"（oss_url.txt, ocr.txt, translation_raw.md 等）"
            )
    finally:
        if figures_future is not None and fig_executor is not None:
            print("[插图·并行] 等待后台插图任务完成…")
            try:
                figure_md_by_page = figures_future.result()
            except Exception as e:
                print(f"[插图] 后台线程异常: {e}")
                figure_md_by_page = {}
            fig_executor.shutdown(wait=True)
        elif fig_executor is not None:
            fig_executor.shutdown(wait=True)

    # Step 5: 按页码顺序写入（try 若抛错则不会执行到这里）
    sorted_results = sorted(translate_results, key=lambda x: x[0])
    translated_parts: list[str] = []
    artifacts_root = task_dir / "artifacts"
    for page_num, chinese_md in sorted_results:
        body = _fix_latex_for_markdown(_strip_markdown_code_block(chinese_md))
        fig_lines = figure_md_by_page.get(page_num)
        if fig_lines:
            # 将译文中的 ![图 N](image.png) 等占位图换成本页真实 figures/ 路径，并避免与页首重复
            body, fig_lines = inline_ocr_placeholder_figures(body, list(fig_lines))
        if fig_lines:
            fig_block = "### 本页插图\n\n" + "\n\n".join(fig_lines) + "\n\n"
        else:
            fig_block = ""
        if save_artifacts:
            pdir = _artifacts_page_dir(artifacts_root, page_num)
            pdir.mkdir(parents=True, exist_ok=True)
            merged = f"{fig_block}{body}".rstrip() + "\n"
            (pdir / "translation_merged.md").write_text(merged, encoding="utf-8")
        translated_parts.append(f"## 第 {page_num} 页\n\n{fig_block}{body}\n")
    full_markdown = f"# {paper_name}\n\n" + "\n---\n\n".join(translated_parts)

    out_path.write_text(full_markdown, encoding="utf-8")
    print(f"      → 任务目录: {task_dir}")
    print(f"      → 已写入 {out_path}")
    if save_artifacts:
        print(f"      → 每页中间文件: {task_dir / 'artifacts'}/page_XXXX/")

    return full_markdown


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF → 中文 Markdown（可选插图提取）")
    parser.add_argument("pdf_path", type=Path, help="输入 PDF")
    parser.add_argument(
        "output_md",
        type=Path,
        nargs="?",
        default=None,
        help="可选：Markdown 路径，或任务目录；默认写入 ./output/<pdf名>/<pdf名>.md",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="任务输出目录（其下生成 .md 与 figures/）；指定后默认可配合第二参数作为 md 文件名",
    )
    parser.add_argument(
        "--figures",
        action="store_true",
        help="启用 Paddle 版面插图提取（需 PADDLE_OCR_TOKEN）",
    )
    parser.add_argument(
        "--figure-dpi",
        type=int,
        default=None,
        help=f"插图渲染 DPI（默认环境变量 FIGURE_EXTRACT_DPI 或 {FIGURE_EXTRACT_DPI}）",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=None,
        help=f"OCR 用 PDF 转图 DPI（默认 {PDF_DPI}）",
    )
    parser.add_argument(
        "--no-artifacts",
        action="store_true",
        help="不写入 task_dir/artifacts/ 中间文件（省磁盘）",
    )
    parser.add_argument(
        "--page-range",
        type=str,
        default=None,
        help='仅处理连续页段（闭区间），格式如 "4-9"',
    )
    args = parser.parse_args()
    process_paper(
        args.pdf_path,
        args.output_md,
        dpi=args.dpi if args.dpi is not None else PDF_DPI,
        output_dir=args.output_dir,
        extract_figures=args.figures,
        figure_dpi=args.figure_dpi,
        save_artifacts=not args.no_artifacts,
        page_range=args.page_range,
    )
    print("完成")
