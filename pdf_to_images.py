#!/usr/bin/env python3
"""Convert each page of a PDF into a low-resolution PNG image.

Usage:
    python pdf_to_images.py input.pdf [--out OUTPUT_DIR] [--dpi DPI] [--format png|jpg]
"""
import argparse
import sys
from pathlib import Path

import fitz  # PyMuPDF


def pdf_to_images(pdf_path: Path, out_dir: Path, dpi: int, fmt: str) -> None:
    doc = fitz.open(pdf_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    zoom = dpi / 72  # 72 dpi is PDF's default
    matrix = fitz.Matrix(zoom, zoom)

    pad = len(str(len(doc)))
    for i, page in enumerate(doc, start=1):
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        out_path = out_dir / f"page_{str(i).zfill(pad)}.{fmt}"
        pix.save(out_path)
        print(f"wrote {out_path}")

    doc.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="Path to input PDF")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: <pdf-stem>_pages next to the PDF)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=50,
        help="Render resolution; low values give small icon-sized images (default: 50)",
    )
    parser.add_argument(
        "--format",
        choices=["png", "jpg"],
        default="png",
        help="Output image format (default: png)",
    )
    args = parser.parse_args()

    if not args.pdf.is_file():
        sys.exit(f"error: {args.pdf} is not a file")

    out_dir = args.out or args.pdf.with_name(f"{args.pdf.stem}_pages")
    pdf_to_images(args.pdf, out_dir, args.dpi, args.format)


if __name__ == "__main__":
    main()
