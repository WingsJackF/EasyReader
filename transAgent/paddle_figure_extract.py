"""
PaddleOCR AI Studio 版面分析 → 按 Figure N 合并子块并裁剪插图（不含表格）。
供 transAgent/agent 在翻译流程中可选调用。

需环境变量 PADDLE_OCR_TOKEN（bearer），与 experiments/paddle_aistudio_layout_experiment.py 一致。
依赖: requests, pymupdf, pillow
"""
from __future__ import annotations

import io
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator

import fitz
import requests
from PIL import Image

JOB_URL = "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"
DEFAULT_MODEL = "PaddleOCR-VL-1.5"

TABLE_BLOCK_LABELS = frozenset(
    {"table", "table_caption", "table_title", "table_body"}
)
FIGURE_VISUAL_LABELS = frozenset({"chart", "figure", "image"})


def extract_single_page_pdf(src_pdf: Path, page_1based: int) -> Path:
    doc = fitz.open(src_pdf)
    try:
        if page_1based < 1 or page_1based > len(doc):
            raise ValueError(f"页码须在 1~{len(doc)}")
        out = fitz.open()
        out.insert_pdf(doc, from_page=page_1based - 1, to_page=page_1based - 1)
        fd, path = tempfile.mkstemp(suffix=".pdf", prefix="paddle_page_")
        os.close(fd)
        out.save(path)
        out.close()
        return Path(path)
    finally:
        doc.close()


def submit_job(
    file_path: Path,
    token: str,
    model: str,
    use_chart: bool,
) -> str:
    headers = {"Authorization": f"bearer {token}"}
    optional_payload = {
        "useDocOrientationClassify": False,
        "useDocUnwarping": False,
        "useChartRecognition": use_chart,
    }
    data = {
        "model": model,
        "optionalPayload": json.dumps(optional_payload),
    }
    with open(file_path, "rb") as f:
        r = requests.post(
            JOB_URL, headers=headers, data=data, files={"file": f}, timeout=120
        )
    r.raise_for_status()
    body = r.json()
    if body.get("code") not in (0, None) and "data" not in body:
        raise RuntimeError(f"Paddle 提交失败: {body}")
    return body["data"]["jobId"]


def poll_until_done(
    job_id: str,
    token: str,
    interval: float = 5.0,
    *,
    verbose: bool = False,
) -> str:
    headers = {"Authorization": f"bearer {token}"}
    while True:
        r = requests.get(f"{JOB_URL}/{job_id}", headers=headers, timeout=120)
        r.raise_for_status()
        data = r.json()["data"]
        state = data["state"]
        if verbose:
            if state == "pending":
                print("      [Paddle] pending …")
            elif state == "running":
                prog = data.get("extractProgress") or {}
                print(
                    f"      [Paddle] running … "
                    f"{prog.get('extractedPages', '?')}/{prog.get('totalPages', '?')} 页"
                )
        if state == "done":
            return data["resultUrl"]["jsonUrl"]
        if state == "failed":
            raise RuntimeError(data.get("errorMsg", "Paddle job failed"))
        time.sleep(interval)


def download_jsonl(url: str) -> str:
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.text


def _norm_bbox(b: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(b, (list, tuple)) or len(b) < 4:
        return None
    xs: list[float] = []
    ys: list[float] = []
    if isinstance(b[0], (list, tuple)) and len(b[0]) == 2:
        for p in b:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                xs.append(float(p[0]))
                ys.append(float(p[1]))
    else:
        return float(b[0]), float(b[1]), float(b[2]), float(b[3])
    if not xs:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def _skip_block_for_figures_only(item: dict) -> bool:
    bl = (item.get("block_label") or "").lower()
    if bl in TABLE_BLOCK_LABELS:
        return True
    ct = item.get("block_content") or ""
    if bl in ("figure_caption", "figure_title") and re.match(r"^\s*Table\b", ct, re.I):
        return True
    return False


def extract_paddle_pruned_boxes(
    payload: dict,
    *,
    figures_only: bool = False,
) -> tuple[list[tuple[str, tuple[float, float, float, float]]], int, int] | None:
    layouts = payload.get("layoutParsingResults")
    if not isinstance(layouts, list):
        return None
    out: list[tuple[str, tuple[float, float, float, float]]] = []
    seen: set[tuple[str, int, int, int, int]] = set()
    pw, ph = 1224, 1584

    for block in layouts:
        pr = block.get("prunedResult")
        if not isinstance(pr, dict):
            continue
        pw = int(pr.get("width") or pw)
        ph = int(pr.get("height") or ph)
        for item in pr.get("parsing_res_list") or []:
            if not isinstance(item, dict):
                continue
            if figures_only and _skip_block_for_figures_only(item):
                continue
            lab = str(item.get("block_label") or "block")
            bb = item.get("block_bbox")
            if not bb and item.get("block_polygon_points"):
                bb = item["block_polygon_points"]
            nb = _norm_bbox(bb) if bb else None
            if not nb:
                continue
            key = (lab, int(nb[0]), int(nb[1]), int(nb[2]), int(nb[3]))
            if key in seen:
                continue
            seen.add(key)
            out.append((lab, nb))

    if not out:
        return None
    return out, pw, ph


def iter_parsing_items_for_figure_crop(
    payload: dict,
    *,
    figures_only: bool,
) -> Iterator[dict]:
    layouts = payload.get("layoutParsingResults")
    if not isinstance(layouts, list):
        return
    relevant = FIGURE_VISUAL_LABELS | {"figure_caption", "figure_title"}
    for block in layouts:
        pr = block.get("prunedResult")
        if not isinstance(pr, dict):
            continue
        for item in pr.get("parsing_res_list") or []:
            if not isinstance(item, dict):
                continue
            if figures_only and _skip_block_for_figures_only(item):
                continue
            lab = (item.get("block_label") or "").lower()
            if lab not in relevant:
                continue
            bb = item.get("block_bbox")
            if not bb and item.get("block_polygon_points"):
                bb = item["block_polygon_points"]
            nb = _norm_bbox(bb) if bb else None
            if not nb:
                continue
            yield {
                "label": lab,
                "bbox": nb,
                "content": item.get("block_content") or "",
            }


def union_boxes_paddle(
    boxes: list[tuple[float, float, float, float]],
) -> tuple[float, float, float, float]:
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def group_figure_regions_by_caption(
    items: list[dict],
    *,
    gap_paddle: float = 18.0,
) -> list[tuple[str, tuple[float, float, float, float]]]:
    visual_boxes: list[tuple[float, float, float, float]] = []
    for it in items:
        if it["label"] in FIGURE_VISUAL_LABELS:
            visual_boxes.append(it["bbox"])

    captions: list[tuple[int, tuple[float, float, float, float]]] = []
    for it in items:
        if it["label"] not in ("figure_caption", "figure_title"):
            continue
        m = re.search(r"(?:Figure|Fig\.)\s*(\d+)", it["content"], re.I)
        if not m:
            continue
        captions.append((int(m.group(1)), it["bbox"]))
    captions.sort(key=lambda x: x[1][1])

    if not captions:
        return []

    used = [False] * len(visual_boxes)
    out: list[tuple[str, tuple[float, float, float, float]]] = []
    prev_bottom = 0.0

    for fig_num, cbb in captions:
        cy0, cy1 = cbb[1], cbb[3]
        picked: list[tuple[float, float, float, float]] = []
        for i, vb in enumerate(visual_boxes):
            if used[i]:
                continue
            vy0, vy1 = vb[1], vb[3]
            if vy1 <= cy0 + gap_paddle and vy0 >= prev_bottom - gap_paddle:
                picked.append(vb)
                used[i] = True
        merged = union_boxes_paddle([cbb, *picked])
        out.append((f"Figure_{fig_num}", merged))
        prev_bottom = cy1

    return out


def paddle_bbox_to_local_crop_rect(
    bb: tuple[float, float, float, float],
    sx: float,
    sy: float,
    sc: float,
    pad: int,
    w: int,
    h: int,
) -> tuple[int, int, int, int]:
    x0 = max(0, int(bb[0] * sx * sc - pad))
    y0 = max(0, int(bb[1] * sy * sc - pad))
    x1 = min(w, int(bb[2] * sx * sc + pad))
    y1 = min(h, int(bb[3] * sy * sc + pad))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("无效裁剪区")
    return x0, y0, x1, y1


def extract_figures_from_pdf_page(
    pdf_path: Path,
    page_1based: int,
    *,
    token: str,
    figure_dpi: int = 300,
    crop_padding: int = 12,
    caption_gap_paddle: float = 18.0,
    bbox_scale: float = 1.0,
    use_chart: bool = True,
    model: str = DEFAULT_MODEL,
    verbose_paddle: bool = False,
) -> list[tuple[str, bytes]]:
    """
    单页：上传单页 PDF → Paddle 版面 → 本地高 DPI 渲染 → 按 Figure N 裁剪 PNG。

    Returns:
        [(文件名 stem, PNG bytes), ...]，stem 如 \"Figure_4\"
    """
    tmp_pdf = extract_single_page_pdf(pdf_path, page_1based)
    try:
        job_id = submit_job(tmp_pdf, token, model, use_chart)
        json_url = poll_until_done(job_id, token, verbose=verbose_paddle)
        text = download_jsonl(json_url)
    finally:
        try:
            tmp_pdf.unlink(missing_ok=True)
        except OSError:
            pass

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []

    first = json.loads(lines[0])
    payload = first.get("result", first)
    if not isinstance(payload, dict):
        return []

    paddle_w, paddle_h = 1224, 1584
    extracted = extract_paddle_pruned_boxes(payload, figures_only=True)
    if extracted:
        _, paddle_w, paddle_h = extracted

    doc = fitz.open(pdf_path)
    try:
        page = doc[page_1based - 1]
        z = figure_dpi / 72.0
        mat = fitz.Matrix(z, z)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        w, h = pix.width, pix.height
        png_bytes = pix.tobytes("png")
    finally:
        doc.close()

    sx = w / float(paddle_w) if paddle_w else 1.0
    sy = h / float(paddle_h) if paddle_h else 1.0

    crop_items = list(
        iter_parsing_items_for_figure_crop(payload, figures_only=True)
    )
    groups = group_figure_regions_by_caption(
        crop_items, gap_paddle=caption_gap_paddle
    )
    if not groups:
        return []

    base = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    out: list[tuple[str, bytes]] = []
    sc = bbox_scale
    for name, pbb in groups:
        try:
            lx0, ly0, lx1, ly1 = paddle_bbox_to_local_crop_rect(
                pbb, sx, sy, sc, crop_padding, w, h
            )
        except ValueError:
            continue
        buf = io.BytesIO()
        base.crop((lx0, ly0, lx1, ly1)).save(buf, format="PNG")
        out.append((name, buf.getvalue()))
    return out
