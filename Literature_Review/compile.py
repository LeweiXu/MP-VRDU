#!/usr/bin/env python3
"""Compile latex/acl_latex.tex into a standalone PDF.

Runs pdflatex -> bibtex -> pdflatex -> pdflatex from the Literature_Review/
directory, places all auxiliary files (.aux/.log/.bbl/.blg/.out/.toc) under a
build/ subdirectory, and copies the final PDF next to this script as
Lit_Review.pdf.

Usage:
  python compile.py            # compile and clean up build/
  python compile.py --keep     # compile and keep build/ for inspection
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
BUILD = HERE / "build"
OUTPUT_PDF = HERE / "Lit_Review.pdf"

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


def build_env() -> dict[str, str]:
    """Expose latex/ to pdflatex (TEXINPUTS) and bibtex (BIBINPUTS, BSTINPUTS)."""
    env = os.environ.copy()
    extras = f"{LATEX_DIR}{os.pathsep}"
    for var in ("TEXINPUTS", "BIBINPUTS", "BSTINPUTS"):
        env[var] = extras + env.get(var, "")
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--keep", action="store_true", help="keep build/ instead of deleting it after success")
    args = parser.parse_args()

    check_inputs()
    BUILD.mkdir(exist_ok=True)
    # \include{latex/X} writes build/latex/X.aux, so the subdir must exist.
    (BUILD / "latex").mkdir(exist_ok=True)
    # bibtex (run inside build/) searches the cwd; copy the .bib and .bst here.
    shutil.copyfile(LATEX_DIR / "custom.bib", BUILD / "custom.bib")
    shutil.copyfile(LATEX_DIR / "acl_natbib.bst", BUILD / "acl_natbib.bst")
    env = build_env()

    pdflatex_cmd = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        f"-output-directory={BUILD}",
        f"-jobname={JOB}",
        str(SRC),
    ]
    # bibtex refuses absolute output paths under default openout_any=p;
    # run it from inside build/ with a relative jobname instead.
    bibtex_cmd = ["bibtex", JOB]

    run(pdflatex_cmd, env)
    run(bibtex_cmd, env, cwd=BUILD)
    run(pdflatex_cmd, env)
    run(pdflatex_cmd, env)

    pdf_src = BUILD / f"{JOB}.pdf"
    if not pdf_src.exists():
        sys.exit(f"expected output not found: {pdf_src}")
    shutil.copyfile(pdf_src, OUTPUT_PDF)
    print(f"\nwrote {OUTPUT_PDF}")

    if not args.keep:
        shutil.rmtree(BUILD, ignore_errors=True)
        print(f"cleaned {BUILD}")


if __name__ == "__main__":
    main()
