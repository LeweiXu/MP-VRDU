#!/usr/bin/env python3
"""Compile latex/acl_latex.tex into a standalone PDF.

Produces two PDFs side-by-side with this script:
  - Lit_Review.pdf       (the default ACL two-column layout)
  - Lit_Review_1col.pdf  (a single-column variant, patched on the fly)

For each variant, runs pdflatex -> bibtex -> pdflatex -> pdflatex with all
auxiliary files (.aux/.log/.bbl/.blg/.out/.toc) under its own build/ subdir.
The single-column variant is produced by writing a patched copy of acl.sty
into its build directory (with \\twocolumn swapped for \\onecolumn) and
prepending that directory to TEXINPUTS so pdflatex picks it up first.

Usage:
  python compile.py            # compile both variants and clean up build dirs
  python compile.py --keep     # compile and keep build dirs for inspection
  python compile.py --only 2col   # only build the two-column version
  python compile.py --only 1col   # only build the single-column version
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LATEX_DIR = HERE / "latex"
SRC = LATEX_DIR / "acl_latex.tex"
JOB = "acl_latex"            # jobname (also the base of the .aux/.pdf names)

BUILD_2COL = HERE / "build"
BUILD_1COL = HERE / "build_1col"
OUTPUT_PDF_2COL = HERE / "Lit_Review.pdf"
OUTPUT_PDF_1COL = HERE / "Lit_Review_1col.pdf"

REQUIRED = [
    SRC,
    LATEX_DIR / "acl.sty",
    LATEX_DIR / "custom.bib",
    LATEX_DIR / "acl_natbib.bst",  # contains IEEEtran.bst content; see acl_latex.tex note
]


def run(cmd: list[str], env: dict[str, str], cwd: Path = HERE) -> None:
    print(f"\n$ {' '.join(cmd)}  (cwd={cwd})")
    result = subprocess.run(cmd, cwd=cwd, env=env)
    if result.returncode != 0:
        sys.exit(f"command failed with exit code {result.returncode}: {' '.join(cmd)}")


def check_inputs() -> None:
    missing = [str(p.relative_to(HERE)) for p in REQUIRED if not p.exists()]
    if missing:
        sys.exit("missing required files:\n  - " + "\n  - ".join(missing))


def build_env(search_dirs: list[Path]) -> dict[str, str]:
    """Expose given dirs (in order) to pdflatex / bibtex via TEXINPUTS et al."""
    env = os.environ.copy()
    extras = os.pathsep.join(str(p) for p in search_dirs) + os.pathsep
    for var in ("TEXINPUTS", "BIBINPUTS", "BSTINPUTS"):
        env[var] = extras + env.get(var, "")
    return env


def patch_acl_sty_for_onecolumn(text: str) -> str:
    """Force single-column layout in acl.sty.

    acl.sty hardcodes two-column mode in two places:
      - `\\flushbottom \\twocolumn \\sloppy` at the top
      - `\\twocolumn[\\@maketitle]` inside the patched \\maketitle
    Swap both for single-column equivalents.
    """
    replacements = {
        r"\flushbottom \twocolumn \sloppy": r"\flushbottom \onecolumn \sloppy",
        r"\twocolumn[\@maketitle]": r"\@maketitle",
    }
    for old, new in replacements.items():
        if old not in text:
            sys.exit(f"patch failed: expected to find {old!r} in acl.sty")
        text = text.replace(old, new)
    return text


def compile_variant(build_dir: Path, output_pdf: Path, *, one_column: bool, keep: bool) -> None:
    label = "single-column" if one_column else "two-column"
    print(f"\n=== building {label} variant -> {output_pdf.name} ===")

    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir()
    # \include{latex/X} writes build/latex/X.aux, so the subdir must exist.
    (build_dir / "latex").mkdir()
    # bibtex (run inside build/) searches the cwd; copy the .bib and .bst here.
    shutil.copyfile(LATEX_DIR / "custom.bib", build_dir / "custom.bib")
    shutil.copyfile(LATEX_DIR / "acl_natbib.bst", build_dir / "acl_natbib.bst")

    search_dirs = [LATEX_DIR]
    if one_column:
        patched = build_dir / "acl.sty"
        patched.write_text(patch_acl_sty_for_onecolumn((LATEX_DIR / "acl.sty").read_text()))
        # Prepend the build dir so the patched acl.sty is found before the original.
        search_dirs.insert(0, build_dir)
    env = build_env(search_dirs)

    pdflatex_cmd = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        f"-output-directory={build_dir}",
        f"-jobname={JOB}",
        str(SRC),
    ]
    # bibtex refuses absolute output paths under default openout_any=p;
    # run it from inside build/ with a relative jobname instead.
    bibtex_cmd = ["bibtex", JOB]

    run(pdflatex_cmd, env)
    run(bibtex_cmd, env, cwd=build_dir)
    run(pdflatex_cmd, env)
    run(pdflatex_cmd, env)

    pdf_src = build_dir / f"{JOB}.pdf"
    if not pdf_src.exists():
        sys.exit(f"expected output not found: {pdf_src}")
    shutil.copyfile(pdf_src, output_pdf)
    print(f"\nwrote {output_pdf}")

    if not keep:
        shutil.rmtree(build_dir, ignore_errors=True)
        print(f"cleaned {build_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--keep", action="store_true", help="keep build/ dirs instead of deleting them after success")
    parser.add_argument(
        "--only",
        choices=("2col", "1col"),
        help="build only the specified variant (default: build both)",
    )
    args = parser.parse_args()

    check_inputs()

    if args.only != "1col":
        compile_variant(BUILD_2COL, OUTPUT_PDF_2COL, one_column=False, keep=args.keep)
    if args.only != "2col":
        compile_variant(BUILD_1COL, OUTPUT_PDF_1COL, one_column=True, keep=args.keep)


if __name__ == "__main__":
    main()
