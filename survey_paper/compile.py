#!/usr/bin/env python3
"""Compile main.tex into a standalone PDF.

Runs pdflatex -> bibtex -> pdflatex -> pdflatex from the survey_paper/
directory, leaving auxiliary files in a build/ subdirectory and copying
the final PDF next to main.tex.

Required files in survey_paper/ (copy from ../survey_source/latex/ as noted):
  latex/acl.sty                       (from survey_source/latex/acl.sty)
  latex/acl_natbib.bst                (from survey_source/latex/acl_natbib.bst)
  latex/models_summary_arch.tex       (from survey_source/latex/)
  latex/models_summary_ocr.tex        (from survey_source/latex/)
  latex/model_performance_arch.tex    (from survey_source/latex/)
  latex/model_training.tex            (from survey_source/latex/)
  custom.bib                          (already present)
  main.tex                            (already present)
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAIN = "main"
BUILD = HERE / "build"


def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=HERE)
    if result.returncode != 0:
        sys.exit(f"command failed with exit code {result.returncode}: {' '.join(cmd)}")


def check_inputs() -> None:
    required = [
        HERE / "main.tex",
        HERE / "custom.bib",
        HERE / "latex" / "acl.sty",
        HERE / "latex" / "acl_natbib.bst",
        HERE / "latex" / "models_summary_arch.tex",
        HERE / "latex" / "models_summary_ocr.tex",
        HERE / "latex" / "model_performance_arch.tex",
        HERE / "latex" / "model_training.tex",
    ]
    missing = [str(p.relative_to(HERE)) for p in required if not p.exists()]
    if missing:
        sys.exit(
            "missing required files:\n  - "
            + "\n  - ".join(missing)
            + "\n\nsee the docstring at the top of compile.py for where to copy each file from."
        )


def main() -> None:
    check_inputs()
    BUILD.mkdir(exist_ok=True)

    # acl.sty and acl_natbib.bst live under latex/ but pdflatex / bibtex
    # search the current directory and TEXINPUTS / BSTINPUTS, so we expose
    # latex/ via -include-directory and copy the .bst into build/.
    bst_dst = BUILD / "acl_natbib.bst"
    shutil.copyfile(HERE / "latex" / "acl_natbib.bst", bst_dst)

    pdflatex_cmd = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        f"-output-directory={BUILD}",
        "-include-directory=latex",
        f"{MAIN}.tex",
    ]
    bibtex_cmd = ["bibtex", str(BUILD / MAIN)]

    # Make custom.bib discoverable from build/ by symlinking (or copying).
    bib_link = BUILD / "custom.bib"
    if not bib_link.exists():
        shutil.copyfile(HERE / "custom.bib", bib_link)

    run(pdflatex_cmd)
    run(bibtex_cmd)
    run(pdflatex_cmd)
    run(pdflatex_cmd)

    pdf_src = BUILD / f"{MAIN}.pdf"
    pdf_dst = HERE / f"{MAIN}.pdf"
    if not pdf_src.exists():
        sys.exit(f"expected output not found: {pdf_src}")
    shutil.copyfile(pdf_src, pdf_dst)
    print(f"\nwrote {pdf_dst}")


if __name__ == "__main__":
    main()
