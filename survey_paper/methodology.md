# MP-VRDU Survey Methodology

## Pipeline

1. **Source search.** Issue keyword queries across general indexes and venue-specific archives. Cap window at the past five years (2021-01 to 2026-05) since MP-VRDU as a distinct subarea predates this only marginally.
2. **Manual MP relevance filter.** Skim each candidate's abstract, model section, and benchmark table. Retain only papers that (a) propose a model, framework, or pipeline whose target setting is multi-page documents, and (b) report results on at least one multi-page benchmark (MP-DocVQA, DUDE, SlideVQA, MMLongBench-Doc, LongDocURL, M3DocVQA, M-LongDoc, DocBench, MMVQA, PDF-VQA, or a paper-introduced multi-page suite). Single-page-only models (DocVQA, InfographicVQA, ChartQA only) are excluded even when reporting strong scores.
3. **Knowledge extraction.** Convert each retained paper to markdown. Run a strong LLM (Claude/GPT-class) over the markdown to perform key-information extraction with evidence-span location, populating per-model fields (architecture family, OCR dependency, vision encoder, LLM backbone, adaptors and projectors, resolution, training stages, datasets used, reported scores). Human verification follows every extraction.

## Information sources

| Source | URL | Notes |
|---|---|---|
| Google Scholar | https://scholar.google.com | Catch-all; sort by relevance and by date (last 2 years) for each query |
| arXiv | https://arxiv.org | cs.CL, cs.CV, cs.IR — most preprints land here first |
| ACL Anthology | https://aclanthology.org | ACL, EMNLP, NAACL, EACL, Findings, COLING, TACL |
| OpenReview | https://openreview.net | ICLR, NeurIPS, COLM submissions and reviews |
| CVF Open Access | https://openaccess.thecvf.com | CVPR, ICCV, ECCV, WACV proceedings |
| AAAI Proceedings | https://ojs.aaai.org | AAAI, IAAI |
| Semantic Scholar | https://www.semanticscholar.org | Citation graph for forward/backward search from anchor papers |
| DBLP | https://dblp.org | Author-page sweeps for prolific document-AI groups |

## Anchor venues (direct browsing)

ACL, EMNLP, NAACL, EACL, COLING, TACL, CVPR, ICCV, ECCV, WACV, NeurIPS, ICLR, AAAI, ICDAR, ICDAR-IJDAR, AICCSA, MM (ACM Multimedia), KDD, SIGIR, COLM.

ICDAR is the most under-indexed venue by ACL Anthology and is worth a dedicated browse for OCR-adjacent multi-page work.

## Search keywords

Run each cluster as a Boolean query of the form `(MP-cluster) AND (task-cluster)` where applicable, then sweep additional standalone terms.

### Multi-page / long-document cluster
- `"multi-page document"` / `"multi page document"`
- `"long document"` understanding
- `"long-context document"`
- `"document-level"` reasoning / VQA / understanding
- `"multi-page PDF"` / `"long PDF"`
- `"multi-document"` (catches some MP-VRDU systems framed as multi-doc)
- `"page-level retrieval"`
- `"cross-page"` reasoning / evidence
- `"document-scale"`
- `"hundred-page"` / `"hundreds of pages"`

### Task / setting cluster
- `"document visual question answering"` / `"DocVQA"`
- `"document VQA"`
- `"visually rich document understanding"` / `"VRDU"`
- `"visually-rich document"`
- `"document understanding"` (broad, needs MP filter)
- `"document question answering"`
- `"document information extraction"`
- `"slide question answering"` / `"SlideVQA"`

### Method-family cluster
- `"document retrieval-augmented generation"` / `"document RAG"`
- `"page retrieval"` / `"visual document retrieval"`
- `"ColPali"` / `"ColQwen"` / late-interaction document
- `"document agent"` / `"agentic document"`
- `"multi-agent document"`
- `"long-context vision-language"` model
- `"document multimodal"` LLM
- `"PDF understanding"` LLM / agent
- `"hierarchical document"` transformer
- `"document compression"` token / visual

### Benchmark / dataset name probes (good for forward-citation sweeps)
Search each of these as a phrase; harvest papers that cite them as a benchmark.
- `"MP-DocVQA"`
- `"MMLongBench-Doc"` (and the colloquial `"MMLongBench"` — distinguish from the unrelated Wang 2025 benchmark)
- `"LongDocURL"`
- `"DUDE"` document understanding
- `"SlideVQA"`
- `"M3DocVQA"` / `"M3DocRAG"`
- `"DocBench"`
- `"M-LongDoc"`
- `"MMVQA"` document
- `"PDF-VQA"`
- `"ViDoSeek"` / `"ViDoRAG"`
- `"OpenDocVQA"`
- `"BRIDGE"` benchmark document
- `"PaperPDF"`

### Anchor-model probes (forward citations)
Run a Semantic Scholar / Google Scholar "cited by" sweep on each, sorted by date, for new entrants we may have missed:
- Hi-VT5, GRAM, Arctic-TILT
- mPLUG-DocOwl2, Docopilot, Texthawk2, Leopard
- M3DocRAG, VDocRAG, ColPali (the retriever paper itself)
- DocAgent, MDocAgent, ViDoRAG, DocLens
- DocR1, Doc-V*, DocDancer, MACT, CoR
- MoLoRAG, MHier-RAG, MLDocRAG, KGP

## Inclusion criteria

A paper is **included** if all of the following hold:
1. Proposes a model, framework, system, or pipeline (not solely a dataset, benchmark, or position paper).
2. The target setting is multi-page documents (≥ 2 pages per input, with at least one evaluation suite averaging ≥ 5 pages, or where multi-page handling is the explicit design contribution).
3. Reports quantitative results on at least one multi-page benchmark, or introduces such a benchmark and runs its own model on it.
4. Published 2021-01 onward, peer-reviewed venue or arXiv preprint with ≥ 1 citation or a clearly identifiable author affiliation.

## Exclusion criteria

A paper is **excluded** if any of the following hold:
1. Single-page only (DocVQA / InfographicVQA / ChartQA / FUNSD / CORD only, no multi-page reporting).
2. Pure benchmark or dataset paper without an accompanying model contribution (these are tracked separately under the dataset corpus).
3. Survey, position, or opinion paper (tracked under related surveys).
4. General long-context LLM work without document-specific contribution (long-context LLM benchmarks where documents are one of many input types).
5. Generic OCR, layout-parsing, or table-extraction tools without VQA / understanding / reasoning evaluation.
6. Non-English primary text without an English-language methodology section that would let us extract metadata reliably.

## What to do with this list

- Run each keyword cluster on Google Scholar, arXiv, and ACL Anthology, sorted by date, for the past 2 years. Anything before that is likely already in the corpus.
- For each anchor model, run a forward-citation sweep on Semantic Scholar.
- Cross-check the resulting candidate list against `survey_paper/models/models_summary.csv`. New entrants get a markdown extract under `models_main_md/` and a row added to the summary CSV before any prose update.
