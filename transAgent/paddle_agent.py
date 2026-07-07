"""
Paddle-only 文献转换 Agent

流程：PDF 单页 → PaddleOCR AI Studio OCR/Layout → 下载 Paddle 返回图片 →
本地化 Markdown 图片路径 → DeepSeek 翻译 → 按页合并中文 Markdown。

这个脚本不调用 Qwen OCR，也不把 Qwen 当作失败时的备用路线。

默认输出目录带时间戳，避免相同 PDF 的多次运行互相覆盖：
  ./output/<pdf文件名>_<YYYYmmdd_HHMMSS>/
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
from openai import OpenAI

from config import (
    PADDLE_LAYOUT_MODEL,
    PADDLE_LAYOUT_USE_CHART,
    PADDLE_OCR_TOKEN,
)
from md_utils import fix_latex_for_markdown, strip_markdown_code_blocks
from paddle_figure_extract import (
    DEFAULT_MODEL,
    download_jsonl,
    extract_single_page_pdf,
    poll_until_done,
    submit_job,
)
from pdf_processor import get_page_count

DEEPSEEK_MAX_TOKENS = 32768
PADDLE_OCR_CONCURRENCY = 2
DEEPSEEK_TRANSLATION_CONCURRENCY = 4


_PADDLE_TRANSLATOR_SYSTEM = """你是一位专业的学术文献翻译助手。请将用户提供的英文学术 Markdown/HTML 内容翻译成中文 Markdown。

要求：
1. 直接输出 Markdown 正文，不要用 ```markdown 或任何代码块包裹。
2. 保留所有图片引用行，图片路径不得改写、删除或移动。形如 ![](figures/xxx.jpg) 的行必须原样保留在对应位置。
3. 图片下方紧邻的英文 Figure/Fig./Table 标题需要翻译成中文，并继续放在图片下方。
4. 表格若已是 HTML table，可以翻译单元格文字并尽量转为 Markdown 表格；结构复杂时保留 HTML 表格也可以，但不要丢失表题。
5. 数学公式保留为 Markdown 可渲染形式，行内公式用 $...$。
6. 保持原始阅读顺序，不要重排段落、图片、图注和表格。
7. 页眉、页脚、页码如果不是正文核心信息，可以省略；但不要省略正文、图、图注、表格。
"""


def _load_env_files() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env")
    load_dotenv(Path(__file__).resolve().parent / ".env")


def _parse_page_range(page_range: str, total_pages: int) -> list[int]:
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


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def resolve_timestamped_task_dir(
    pdf_path: Path,
    output_root: str | Path | None,
    run_name: str | None,
) -> tuple[Path, Path]:
    stamp = _timestamp()
    name = run_name.strip() if run_name else f"{pdf_path.stem}_{stamp}"
    if stamp not in name:
        name = f"{name}_{stamp}"
    root = Path(output_root).expanduser().resolve() if output_root else (Path.cwd() / "output").resolve()
    task_dir = root / name
    task_dir.mkdir(parents=True, exist_ok=False)
    return task_dir, task_dir / f"{pdf_path.stem}.md"


def _safe_image_name(src: str, page_num: int, idx: int) -> str:
    parsed = urlparse(src)
    base = Path(parsed.path).name or f"image_{idx}.jpg"
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base)
    return f"p{page_num:04d}_{idx:02d}_{base}"


def _download_binary(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    path.write_bytes(r.content)


def _html_to_markdown_for_translation(text: str) -> str:
    img_pattern = re.compile(
        r"""<div[^>]*>\s*<img\s+[^>]*src=["']([^"']+)["'][^>]*>\s*</div>""",
        re.IGNORECASE,
    )
    text = img_pattern.sub(lambda m: f"![]({m.group(1)})", text)

    caption_pattern = re.compile(
        r"""<div[^>]*text-align:\s*center[^>]*>(.*?)</div>""",
        re.IGNORECASE | re.DOTALL,
    )
    text = caption_pattern.sub(
        lambda m: re.sub(r"<[^>]+>", "", m.group(1)).strip(),
        text,
    )
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def _extract_payloads(jsonl_text: str) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for line in jsonl_text.splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        payload = obj.get("result", obj)
        if isinstance(payload, dict):
            payloads.append(payload)
    return payloads


def _summarize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    layouts = payload.get("layoutParsingResults") or []
    summary: dict[str, Any] = {
        "top_level_keys": sorted(payload.keys()),
        "layout_result_count": len(layouts) if isinstance(layouts, list) else 0,
    }
    if not layouts:
        return summary

    first = layouts[0]
    pruned = first.get("prunedResult") or {}
    markdown = first.get("markdown") or {}
    output_images = first.get("outputImages") or {}
    items = pruned.get("parsing_res_list") or []
    labels = Counter(
        str(item.get("block_label") or "unknown")
        for item in items
        if isinstance(item, dict)
    )
    summary.update(
        {
            "page_count": pruned.get("page_count"),
            "width": pruned.get("width"),
            "height": pruned.get("height"),
            "model_settings": pruned.get("model_settings"),
            "markdown_text_chars": len(markdown.get("text") or ""),
            "markdown_image_count": len(markdown.get("images") or {}),
            "output_image_keys": sorted(output_images.keys()),
            "block_label_counts": dict(labels),
        }
    )
    return summary


def _localize_paddle_markdown_images(
    markdown: str,
    images: dict[str, str],
    *,
    page_num: int,
    task_dir: Path,
) -> tuple[str, list[dict[str, str]]]:
    local_map: dict[str, str] = {}
    image_records: list[dict[str, str]] = []
    figures_dir = task_dir / "figures"

    referenced_images = [
        (src, url) for src, url in images.items() if src in markdown
    ]
    for idx, (src, url) in enumerate(referenced_images, start=1):
        rel_path = f"figures/{_safe_image_name(src, page_num, idx)}"
        _download_binary(url, task_dir / rel_path)
        local_map[src] = rel_path
        image_records.append(
            {"paddle_src": src, "local_path": rel_path, "url": url}
        )

    figures_dir.mkdir(parents=True, exist_ok=True)
    localized = markdown
    for src, rel_path in local_map.items():
        localized = localized.replace(src, rel_path)
    return _html_to_markdown_for_translation(localized), image_records


def _write_page_report(
    pdir: Path,
    *,
    page_num: int,
    summary: dict[str, Any],
    images: list[dict[str, str]],
) -> None:
    lines = [
        f"# Paddle Page Report - {page_num}",
        "",
        f"- Layout results: `{summary.get('layout_result_count')}`",
        f"- Page size: `{summary.get('width')} x {summary.get('height')}`",
        f"- Markdown chars: `{summary.get('markdown_text_chars')}`",
        f"- Markdown images: `{summary.get('markdown_image_count')}`",
        f"- Output image keys: `{', '.join(summary.get('output_image_keys') or [])}`",
        "",
        "## Block Labels",
        "",
    ]
    for label, count in sorted((summary.get("block_label_counts") or {}).items()):
        lines.append(f"- `{label}`: {count}")
    lines.extend(["", "## Images", ""])
    if images:
        for item in images:
            lines.append(f"- `{item['paddle_src']}` -> `{item['local_path']}`")
    else:
        lines.append("- None")
    pdir.joinpath("report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _translate_markdown_with_deepseek(
    client: OpenAI,
    markdown: str,
    *,
    model: str,
) -> str:
    if not markdown.strip():
        return ""
    completion = client.chat.completions.create(
        model=model,
        max_tokens=DEEPSEEK_MAX_TOKENS,
        messages=[
            {"role": "system", "content": _PADDLE_TRANSLATOR_SYSTEM},
            {"role": "user", "content": markdown},
        ],
    )
    return completion.choices[0].message.content or ""


_IMAGE_LINE = re.compile(r"^\s*!\[[^\]]*\]\((figures/[^)]+)\)\s*$")


def _translate_markdown_preserving_images(
    client: OpenAI,
    markdown: str,
    *,
    model: str,
) -> str:
    """
    Translate text chunks while reinserting image lines exactly where Paddle put
    them. This makes image placement a deterministic part of the pipeline,
    rather than a behavior we hope the translation model preserves.
    """
    chunks: list[tuple[str, str]] = []
    text_lines: list[str] = []

    def flush_text() -> None:
        if text_lines:
            chunks.append(("text", "\n".join(text_lines).strip()))
            text_lines.clear()

    for line in markdown.splitlines():
        if _IMAGE_LINE.match(line):
            flush_text()
            chunks.append(("image", line.strip()))
        else:
            text_lines.append(line)
    flush_text()

    out: list[str] = []
    for kind, value in chunks:
        if kind == "image":
            out.append(value)
            continue
        if not value.strip():
            continue
        translated = _translate_markdown_with_deepseek(
            client,
            value,
            model=model,
        )
        out.append(translated.strip())
    return "\n\n".join(part for part in out if part.strip())


def _process_page_with_paddle(
    pdf_path: Path,
    page_num: int,
    *,
    task_dir: Path,
    token: str,
    paddle_model: str,
    use_chart: bool,
) -> tuple[str, dict[str, Any], list[dict[str, str]]]:
    artifacts_dir = task_dir / "artifacts" / f"page_{page_num:04d}"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    tmp_pdf = extract_single_page_pdf(pdf_path, page_num)
    try:
        job_id = submit_job(tmp_pdf, token, paddle_model, use_chart)
        json_url = poll_until_done(job_id, token, interval=3.0, verbose=True)
        jsonl_text = download_jsonl(json_url)
    finally:
        tmp_pdf.unlink(missing_ok=True)

    artifacts_dir.joinpath("paddle_raw.jsonl").write_text(
        jsonl_text, encoding="utf-8"
    )
    payloads = _extract_payloads(jsonl_text)
    if not payloads:
        raise RuntimeError(f"第 {page_num} 页 Paddle 返回为空")

    payload = payloads[0]
    artifacts_dir.joinpath("paddle_first_result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = _summarize_payload(payload)
    artifacts_dir.joinpath("paddle_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    first_layout = (payload.get("layoutParsingResults") or [{}])[0]
    markdown_obj = first_layout.get("markdown") or {}
    raw_md = markdown_obj.get("text") or ""
    images = markdown_obj.get("images") or {}
    artifacts_dir.joinpath("paddle_markdown_raw.md").write_text(
        raw_md, encoding="utf-8"
    )

    local_md, image_records = _localize_paddle_markdown_images(
        raw_md,
        images,
        page_num=page_num,
        task_dir=task_dir,
    )
    artifacts_dir.joinpath("paddle_markdown_local.md").write_text(
        local_md, encoding="utf-8"
    )
    _write_page_report(
        artifacts_dir,
        page_num=page_num,
        summary=summary,
        images=image_records,
    )
    return local_md, summary, image_records


def _run_paddle_ocr_pages_parallel(
    pdf_path: Path,
    pages: list[int],
    *,
    task_dir: Path,
    token: str,
    paddle_model: str,
    use_chart: bool,
    max_workers: int = PADDLE_OCR_CONCURRENCY,
) -> list[tuple[int, str, dict[str, Any], list[dict[str, str]]]]:
    results: dict[int, tuple[str, dict[str, Any], list[dict[str, str]]]] = {}
    total = len(pages)
    workers = min(max_workers, total) if total else 1
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="paddle-ocr") as pool:
        future_to_page = {
            pool.submit(
                _process_page_with_paddle,
                pdf_path,
                page_num,
                task_dir=task_dir,
                token=token,
                paddle_model=paddle_model,
                use_chart=use_chart,
            ): (idx, page_num)
            for idx, page_num in enumerate(pages, start=1)
        }
        for future in as_completed(future_to_page):
            idx, page_num = future_to_page[future]
            local_md, summary, image_records = future.result()
            results[page_num] = (local_md, summary, image_records)
            print(
                f"      [Paddle OCR完成] 第 {idx}/{total} 页（PDF第 {page_num} 页）"
            )
    return [(page_num, *results[page_num]) for page_num in pages]


def _translate_page_result(
    page_num: int,
    local_md: str,
    summary: dict[str, Any],
    image_records: list[dict[str, str]],
    *,
    task_dir: Path,
    deepseek_key: str,
    deepseek_model: str,
) -> tuple[int, str, dict[str, Any]]:
    translator = OpenAI(
        api_key=deepseek_key,
        base_url="https://api.deepseek.com/v1",
        timeout=180,
    )
    translated = _translate_markdown_preserving_images(
        translator,
        local_md,
        model=deepseek_model,
    )
    translated = fix_latex_for_markdown(strip_markdown_code_blocks(translated))
    pdir = task_dir / "artifacts" / f"page_{page_num:04d}"
    pdir.mkdir(parents=True, exist_ok=True)
    pdir.joinpath("translation_raw.md").write_text(
        translated, encoding="utf-8"
    )
    merged = translated.rstrip() + "\n"
    pdir.joinpath("translation_merged.md").write_text(merged, encoding="utf-8")
    translated_part = f"## 第 {page_num} 页\n\n{merged}"
    run_summary = {
        "page": page_num,
        "markdown_chars": summary.get("markdown_text_chars"),
        "image_count": len(image_records),
        "block_label_counts": summary.get("block_label_counts"),
    }
    return page_num, translated_part, run_summary


def _run_deepseek_translations_parallel(
    ocr_results: list[tuple[int, str, dict[str, Any], list[dict[str, str]]]],
    *,
    task_dir: Path,
    deepseek_key: str,
    deepseek_model: str,
    max_workers: int = DEEPSEEK_TRANSLATION_CONCURRENCY,
) -> tuple[list[str], list[dict[str, Any]]]:
    translated_parts_by_page: dict[int, str] = {}
    run_summary_by_page: dict[int, dict[str, Any]] = {}
    total = len(ocr_results)
    workers = min(max_workers, total) if total else 1
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="deepseek") as pool:
        future_to_page = {
            pool.submit(
                _translate_page_result,
                page_num,
                local_md,
                summary,
                image_records,
                task_dir=task_dir,
                deepseek_key=deepseek_key,
                deepseek_model=deepseek_model,
            ): (idx, page_num)
            for idx, (page_num, local_md, summary, image_records) in enumerate(
                ocr_results,
                start=1,
            )
        }
        for future in as_completed(future_to_page):
            idx, page_num = future_to_page[future]
            done_page, translated_part, run_summary = future.result()
            translated_parts_by_page[done_page] = translated_part
            run_summary_by_page[done_page] = run_summary
            print(
                f"      [DeepSeek翻译完成] 第 {idx}/{total} 页（PDF第 {page_num} 页）"
            )
    pages = [page_num for page_num, *_ in ocr_results]
    return (
        [translated_parts_by_page[page_num] for page_num in pages],
        [run_summary_by_page[page_num] for page_num in pages],
    )


def process_paper_with_paddle(
    pdf_path: str | Path,
    *,
    output_root: str | Path | None = None,
    run_name: str | None = None,
    page_range: str | None = None,
    paddle_model: str | None = None,
    use_chart: bool = PADDLE_LAYOUT_USE_CHART,
    deepseek_model: str = "deepseek-chat",
) -> str:
    _load_env_files()
    pdf_path = Path(pdf_path).expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 不存在: {pdf_path}")

    token = (PADDLE_OCR_TOKEN or os.getenv("PADDLE_OCR_TOKEN", "")).strip()
    if not token:
        raise RuntimeError("缺少 PADDLE_OCR_TOKEN，无法运行 Paddle-only agent")
    deepseek_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not deepseek_key:
        raise RuntimeError("缺少 DEEPSEEK_API_KEY，无法翻译 Paddle OCR 结果")

    task_dir, out_path = resolve_timestamped_task_dir(pdf_path, output_root, run_name)
    total_pages = get_page_count(pdf_path)
    pages = _parse_page_range(page_range, total_pages) if page_range else list(range(1, total_pages + 1))
    selected_total = len(pages)
    model = paddle_model or PADDLE_LAYOUT_MODEL or DEFAULT_MODEL
    print(f"[Paddle-only] PDF: {pdf_path}")
    print(f"[Paddle-only] 输出目录: {task_dir}")
    print(
        f"[Paddle-only] 处理页数: {selected_total}/{total_pages}，模型: {model}，"
        f"OCR并行数: {PADDLE_OCR_CONCURRENCY}，翻译并行数: {DEEPSEEK_TRANSLATION_CONCURRENCY}"
    )

    print(f"[1/2] Paddle OCR/Layout 并行处理中（并行数 {PADDLE_OCR_CONCURRENCY}）")
    ocr_results = _run_paddle_ocr_pages_parallel(
        pdf_path,
        pages,
        task_dir=task_dir,
        token=token,
        paddle_model=model,
        use_chart=use_chart,
    )

    print(f"[2/2] DeepSeek 翻译并行处理中（并行数 {DEEPSEEK_TRANSLATION_CONCURRENCY}）")
    translated_parts, run_summary = _run_deepseek_translations_parallel(
        ocr_results,
        task_dir=task_dir,
        deepseek_key=deepseek_key,
        deepseek_model=deepseek_model,
    )

    full_markdown = f"# {pdf_path.stem}\n\n" + "\n---\n\n".join(translated_parts)
    out_path.write_text(full_markdown, encoding="utf-8")
    task_dir.joinpath("run_summary.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    task_dir.joinpath("README.txt").write_text(
        "Paddle-only Agent 输出说明：\n"
        f"- {out_path.name}: 合并后的中文 Markdown\n"
        "- figures/: Paddle 返回图片下载后的本地相对路径图片\n"
        "- artifacts/page_XXXX/paddle_raw.jsonl: Paddle 原始 JSONL\n"
        "- artifacts/page_XXXX/paddle_markdown_local.md: 图片路径本地化后的 Paddle Markdown\n"
        "- artifacts/page_XXXX/translation_merged.md: 写入总稿前的本页中文正文\n"
        "- run_summary.json: 每页块类型与图片数量摘要\n",
        encoding="utf-8",
    )
    print(f"      → 任务目录: {task_dir}")
    print(f"      → 已写入: {out_path}")
    return full_markdown


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PDF → PaddleOCR AI Studio OCR/Layout → DeepSeek 中文 Markdown"
    )
    parser.add_argument("pdf_path", type=Path, help="输入 PDF")
    parser.add_argument(
        "-o",
        "--output-root",
        type=Path,
        default=None,
        help="输出根目录；实际任务目录会在其下追加时间戳",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="自定义任务名；若不含时间戳，脚本会自动追加时间戳",
    )
    parser.add_argument(
        "--page-range",
        type=str,
        default=None,
        help='仅处理连续页段（闭区间），格式如 "4-9"',
    )
    parser.add_argument(
        "--paddle-model",
        type=str,
        default=None,
        help=f"Paddle 模型名，默认环境变量 PADDLE_LAYOUT_MODEL 或 {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--no-chart",
        action="store_true",
        help="关闭 Paddle chart recognition",
    )
    parser.add_argument(
        "--deepseek-model",
        type=str,
        default="deepseek-chat",
        help="DeepSeek 翻译模型",
    )
    args = parser.parse_args()
    process_paper_with_paddle(
        args.pdf_path,
        output_root=args.output_root,
        run_name=args.run_name,
        page_range=args.page_range,
        paddle_model=args.paddle_model,
        use_chart=not args.no_chart,
        deepseek_model=args.deepseek_model,
    )
    print("完成")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"失败: {exc}", file=sys.stderr)
        raise
