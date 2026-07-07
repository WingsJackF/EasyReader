"""
PDF 转图片模块
将 PDF 每一页转换为高清 PNG 图片，便于后续 OCR 识别
"""
import fitz  # PyMuPDF
from pathlib import Path
from typing import Generator


def pdf_to_images(
    pdf_path: str | Path,
    output_dir: str | Path | None = None,
    dpi: int = 200,
    page_numbers: set[int] | None = None,
) -> Generator[tuple[int, bytes], None, None]:
    """
    将 PDF 每一页转换为 PNG 图片字节流。

    Args:
        pdf_path: PDF 文件路径
        output_dir: 可选，若指定则同时保存到本地目录
        dpi: 渲染分辨率，越高越清晰，默认 200（平衡清晰度与文件大小）
        page_numbers: 可选，仅渲染这些页码（1-based）

    Yields:
        (页码, 图片字节) 元组，页码从 1 开始
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

    # 高 DPI 提升清晰度，zoom 因子 = dpi/72 (PDF 默认 72dpi)
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)

    doc = fitz.open(pdf_path)
    try:
        for page_num in range(len(doc)):
            one_based_page = page_num + 1
            if page_numbers is not None and one_based_page not in page_numbers:
                continue
            page = doc[page_num]
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            img_bytes = pix.tobytes("png")

            if output_dir:
                out_path = Path(output_dir) / f"page_{one_based_page:04d}.png"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(img_bytes)

            yield one_based_page, img_bytes
    finally:
        doc.close()


def get_page_count(pdf_path: str | Path) -> int:
    """获取 PDF 总页数"""
    doc = fitz.open(pdf_path)
    try:
        return len(doc)
    finally:
        doc.close()
