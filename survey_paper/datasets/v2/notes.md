# v2 build notes

Process log for the 46-paper rebuild of `datasets_used.md` / `datasets_used.csv` / `datasets_analysis.md`. The v1 outputs in `/home/lingwei/CITS4010/survey_paper/datasets/` were not modified.

## Methodology

1. **Identified the v1 → v2 delta.** The 10 newly-added papers (IDs 37–46 in `models_summary.csv`) are: DREAM, MM-Doc-R1, SV-RAG, SLEUTH, MM-R5, VisDoMRAG, VRAG-RL, URaG, DocSeeker, DMAP.
2. **Dispatched two parallel Explore agents** to extract PT/FT/BM dataset usage from the 10 new papers (5 each). Each agent was instructed to grep-verify before claiming and to cite line numbers.
3. **Spot-checked the four MMLongBench mentions** in DMAP, MM-Doc-R1, VRAG-RL, SLEUTH against their bibliography references — all four cite Ma et al. 2024 (i.e., MMLongBench-Doc), not Wang et al. 2025 (MMLongBench).
4. **Verified DREAM, SV-RAG, MM-R5 specifics** via grep before merging.
5. **Merged with the v1 36-paper data** (kept verbatim where unchanged, with one correction).
6. **Recomputed all category counts** from the resulting CSV using awk pipelines.
7. **Wrote `datasets_used.md`, `datasets_used.csv`, `datasets_analysis.md`** afresh; preserved the structure of v1 but updated numbers, examples, and trend claims.

## Decisions and normalisation

- **MMLongBench-Doc vs MMLongBench.** Disambiguated explicitly in the conventions block of `datasets_used.md`. All four "MMLongBench" references in DMAP, MM-Doc-R1, VRAG-RL, SLEUTH cite Ma 2024 — i.e., MMLongBench-Doc. Grep-verified the bibliography lines in each paper. The 2025 Wang MMLongBench does not appear in any of the 46 papers.
- **MP-DocVQA spellings.** Normalised "MPDocVQA", "Multi-Page DocVQA", "MP-DocVQA" → MP-DocVQA across all rows.
- **InfographicVQA / InfoVQA.** Normalised to InfographicVQA.
- **WikiTableQuestions / WTQ.** Normalised to WikiTableQuestions.
- **VisR-Bench (SV-RAG)** and **VisDoMBench (VisDoMRAG)** flagged with † as newly-introduced benchmarks.
- **MM-R5 Reasoning Set, VRAG-RL trajectories, DocSeeker ALR-CoT** are listed as introduced training corpora (not benchmarks) in Table 2.
- **MM-Doc-R1 SPO** is an algorithm, not a dataset; not listed in any table. The LongDocURL slice it uses is listed under LongDocURL FT.

## One correction to v1

- **DREAM also fine-tunes on DUDE's training split** — not just MP-DocVQA. Confirmed by line 243 of the DREAM paper: "we utilize the MP-DocVQA and DUDE training datasets, where the ranking results of DUDE's Multi-modal Retrieval Combination Reranking are obtained from the MP-DOCVQA-trained ranking model." v1's table 2 listed DREAM only under MP-DocVQA FT; v2 lists it under both MP-DocVQA FT and DUDE FT.

## Disagreements with existing CSVs

- **`datasets_main.csv` lists MMLongBench (Wang 2025)** as a separate row (ID 29). No paper in the 46-paper corpus uses this dataset. It is included in the survey's dataset metadata for completeness — survey authors may want to flag this when introducing the dataset, since none of the surveyed models have engaged with it yet.
- **`datasets_main.csv` lists BRIDGE (Xiang 2026)** as ID 30. Same situation — no surveyed paper uses it, but it represents a future direction (step-level multi-hop evaluation).
- **`datasets_summary.csv` lists 33 rows.** The 46-paper extraction surfaces ≥60 distinct benchmarking datasets. Datasets in the corpus but not in `datasets_summary.csv` include: MMDocIR (used by MM-R5), VisR-Bench (introduced by SV-RAG), VisDoMBench (introduced by VisDoMRAG), SciGraphQA, SPIQA, PaperText, MOAMOB, MPVQA, DocHieNet, HRDH, several Texthawk2-only OCR benchmarks. If the survey's LaTeX dataset table is meant to be exhaustive for the corpus, these should be added; if it is meant to cover only the "core" datasets, no change needed.

## Spot-check log

Five randomly-chosen (dataset, paper) claims grep-verified during the build:

1. **MM-Doc-R1 uses LongDocURL for RL training** — verified at line 194: "For RL training, we use a subset of 300 samples from LongDocURL as the validation set, and the remaining data as the training set."
2. **VRAG-RL evaluates on SlideVQA, ViDoSeek, MMLongBench** — verified at line 234 and line 478: "We evaluate our method on three visually rich document datasets: SlideVQA, ViDoSeek, and MMLongbench."
3. **SV-RAG introduces VisR-Bench** — verified at line 39: "We collect a visually-rich document QA dataset, VisR-Bench, comprising nine domains ... 226 documents and 471 question answer pairs."
4. **MM-R5 trains and evaluates on MMDocIR** — verified at line 179 and 195: "We conduct our experiments on MMDocIR (Dong et al. 2025) ... We adopt MMDocIR for both training and evaluation."
5. **DREAM trains on MP-DocVQA + DUDE training sets** — verified at line 243 (see correction above).

## Files written

- `/home/lingwei/CITS4010/survey_paper/datasets/v2/datasets_used.md` — three role tables, 46-paper coverage, with normalisation conventions and notes.
- `/home/lingwei/CITS4010/survey_paper/datasets/v2/datasets_used.csv` — flattened (Dataset, Role, Paper, Usage) row per claim. ~330 rows.
- `/home/lingwei/CITS4010/survey_paper/datasets/v2/datasets_analysis.md` — narrative analysis with updated counts.
- `/home/lingwei/CITS4010/survey_paper/datasets/v2/notes.md` — this file.

## What changed substantively from v1

- MMLongBench-Doc surpassed MP-DocVQA as the most-used benchmark (25 vs 20 papers).
- RL footprint tripled (2 papers in v1 → 6 in v2).
- Two new benchmarks introduced (VisR-Bench, VisDoMBench).
- Five new synthesised training corpora (MM-R5 reasoning set, VRAG-RL trajectories, DocSeeker ALR-CoT, MM-Doc-R1 SPO pool, SV-RAG ColPali-filtered mix).
- Zero new pretraining datasets — the "from-scratch pretraining" recipe is dead among 2025–26 papers.
- One v1 mapping correction (DREAM + DUDE FT).
- §3.4 expanded from two niches of self-introduced datasets to three (adding the "small RL/SFT pools with shaped supervision signals" niche).
- §5.1 contamination table updated with 5 new concerning cases and 3 new responsibly-handled cases.

## Quality caveats

- I did not perform a fresh extraction of all 36 v1 papers — I relied on v1's `datasets_used.md` as the source of truth for them, with one spot-fix (DREAM). If the user wants a *full* re-extraction (e.g., to catch other v1 errors), that would require ~4 more Explore-agent dispatches over the 36 originals.
- I did not audit the venue/year/page-count metadata in `datasets_main.csv`. The survey's quantitative claims about dataset statistics (e.g., "MMLongBench-Doc averages 47.5 pages") rely on that CSV being correct.
- All 17 synthetic-data papers and all 6 RL papers in v2 lack human-rated quality samples — a structural weakness in the corpus that the survey should foreground rather than smooth over.
