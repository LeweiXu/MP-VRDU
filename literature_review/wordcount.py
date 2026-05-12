#!/usr/bin/env python3
"""Word count for the literature review, ignoring comments and references."""

import re
import sys
from pathlib import Path

TEX_FILE = Path(__file__).parent / "latex" / "acl_latex.tex"


def strip_comments(text: str) -> str:
    # Remove % comments but preserve escaped \%
    out_lines = []
    for line in text.splitlines():
        result = []
        i = 0
        while i < len(line):
            if line[i] == "\\" and i + 1 < len(line) and line[i + 1] == "%":
                result.append("\\%")
                i += 2
            elif line[i] == "%":
                break
            else:
                result.append(line[i])
                i += 1
        out_lines.append("".join(result))
    return "\n".join(out_lines)


def strip_environments(text: str, envs: list[str]) -> str:
    for env in envs:
        pattern = re.compile(
            rf"\\begin\{{{env}\*?\}}.*?\\end\{{{env}\*?\}}", re.DOTALL
        )
        text = pattern.sub(" ", text)
    return text


def strip_citations_and_refs(text: str) -> str:
    # Remove the bibliography command itself
    text = re.sub(r"\\bibliography\{[^}]*\}", " ", text)
    text = re.sub(r"\\bibliographystyle\{[^}]*\}", " ", text)

    # Remove \cite{...}, \citep{...}, \citet{...}, \citeyear{...}, \ref{...}, \label{...}, etc.
    cite_like = [
        "cite", "citep", "citet", "citeauthor", "citeyear", "citealp",
        "ref", "label", "eqref", "pageref", "autoref", "nameref",
    ]
    for cmd in cite_like:
        text = re.sub(rf"\\{cmd}\*?\{{[^}}]*\}}", " ", text)
    return text


def strip_latex_commands(text: str) -> str:
    # Remove \input{...}, \include{...}
    text = re.sub(r"\\(input|include|usepackage|documentclass|setcitestyle)\b\*?(\[[^\]]*\])?\{[^}]*\}", " ", text)

    # Remove standalone commands with one or two bracketed args we don't want to count
    text = re.sub(r"\\(title|author|date|maketitle|appendix|newpage|bibliographystyle)\b\*?(\[[^\]]*\])?(\{[^}]*\})?", " ", text)

    # Strip remaining backslash commands but keep their textual arguments.
    # Repeat until stable so nested commands collapse.
    prev = None
    while prev != text:
        prev = text
        # \foo[opt]{arg} -> arg
        text = re.sub(r"\\[a-zA-Z]+\*?\s*(?:\[[^\]]*\])?\s*\{([^{}]*)\}", r" \1 ", text)
        # \foo[opt] (no braces) -> drop
        text = re.sub(r"\\[a-zA-Z]+\*?\s*\[[^\]]*\]", " ", text)
        # bare \foo -> drop
        text = re.sub(r"\\[a-zA-Z]+\*?", " ", text)

    # Remove leftover braces, math delimiters, and tilde-non-breaking-spaces
    text = text.replace("~", " ")
    text = re.sub(r"[{}]", " ", text)
    text = re.sub(r"\$[^$]*\$", " ", text)
    return text


def body_only(text: str) -> str:
    # Keep only content inside \begin{document}...\end{document} if present
    m = re.search(r"\\begin\{document\}(.*?)\\end\{document\}", text, re.DOTALL)
    if m:
        text = m.group(1)
    # Drop anything from \appendix onward (appendix not counted toward main length)
    text = re.split(r"\\appendix\b", text, maxsplit=1)[0]
    return text


def count_words(text: str) -> int:
    tokens = re.findall(r"[A-Za-z][A-Za-z\-']*", text)
    return len(tokens)


def main() -> int:
    path = TEX_FILE if len(sys.argv) < 2 else Path(sys.argv[1])
    raw = path.read_text(encoding="utf-8")

    text = strip_comments(raw)
    text = body_only(text)
    text = strip_environments(text, ["thebibliography", "verbatim", "lstlisting", "equation", "align", "tabular", "longtable"])
    text = strip_citations_and_refs(text)
    text = strip_latex_commands(text)

    words = count_words(text)
    print(f"{path.name}: {words} words")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
