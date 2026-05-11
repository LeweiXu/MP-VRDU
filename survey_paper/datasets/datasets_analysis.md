# Datasets in MP-VRDU: Patterns, Trends, and Survey-Writing Guide

This document distils observations from `datasets_used.md` / `datasets_used.csv` (36 MP-VRDU papers) into a narrative that can directly seed the **Datasets** section of the survey. It is organised into:

1. The headline dataset inventory by category.
2. Patterns *within* each category (what is used, why, how).
3. Cross-cutting trends across the field.
4. Gaps, biases, and open problems worth flagging in the survey.
5. A deep dive on the two structural methodological flags (contamination and synthetic-data circularity).
6. A suggested section structure with notes on which datasets to emphasise where.

A glossary of dataset abbreviations is provided at the end.

---

## 1. Headline inventory

### 1.1 Most-used datasets per category

| Rank | Pretraining | # papers | Fine-Tuning | # papers | Benchmarking | # papers |
|---|---|---|---|---|---|---|
| 1 | DocStruct4M | 2 | DocVQA | 9 | MP-DocVQA | 18 |
| 2 | Cauldron | 2 | DUDE | 13 | MMLongBench-Doc | 17 |
| 3 | Arxiv PDFs | 2 | MP-DocVQA | 15 | DUDE | 15 |
| 4 | (most others appear in only one paper, dominated by Texthawk2 and Leopard) | — | InfographicVQA | 8 | DocVQA | 14 |
| 5 | | | ChartQA | 5 | InfographicVQA | 14 |
| 6 | | | VisualMRC | 4 | SlideVQA | 13 |
| 7 | | | SlideVQA | 3 | LongDocURL | 10 |
| 8 | | | MultiHiertt | 2 | ChartQA | 10 |
| 9 | | | ArxivQA / Kleister-Charity / DeepForm / WTQ / TabFact / TextVQA / TextCaps / OCR-VQA / FigureQA | 2 each | TAT-DQA / FUNSD / CORD / MultiHiertt / MultiChartQA / WTQ / TabFact / TextVQA | 2–3 each |

**Net composition (across all 36 papers):** 31 distinct pretraining-stage datasets, 64 fine-tuning datasets, and 70+ benchmarking datasets. Of these, 8 are *introduced* by papers in the corpus (MP-DocVQA, M3DocVQA, ViDoSeek, OpenDocVQA, Doc-750K, PaperPDF, CoR-Dataset, Leopard-Instruct/MHDocVQA/MP-DocStruct1M/MP-DocReason51K).

### 1.2 The "core five" benchmarks

Five datasets dominate evaluation across the corpus and appear in roughly half or more of the papers: **MP-DocVQA, MMLongBench-Doc, DUDE, DocVQA, InfographicVQA**. Any survey datasets section should treat these as the de-facto comparison axes for MP-VRDU work; everything else is supplementary or domain-specific.

---

## 2. Patterns within each category

### 2.1 Pretraining

**Who pretrains?** Only the *backbone-centric* models pretrain at scale: **Texthawk2, Leopard, mPLUG-DocOwl2, PDF-WuKong, Arctic-TILT, Hi-VT5, Docopilot, CoR, VDocRAG**. Retriever-generator and agentic-pipeline papers (DocReact, MDocAgent, DocAgent, MLDocRAG, MHier-RAG, M3DocRAG, ViDoRAG, SimpleDoc, M2RAG, MoLoRAG, Doc-V*) inherit pretrained backbones and skip this stage entirely.

**Three pretraining recipes are observable:**

1. **OCR / layout self-supervision.** Hi-VT5 (OCR-IDL, 200K pages with hierarchical denoising), Arctic-TILT (CCpdf, 900K steps), Texthawk2 (IIT-CDIP, Wukong, Common Crawl PDFs, plus a 10-dataset multilingual OCR pile). Goal: teach the encoder to *read* before it tries to *reason*.
2. **Document-structure pretraining on synthesised QA.** mPLUG-DocOwl2's DocStruct4M and MP-DocStruct1M (and VDocRAG's reuse of DocStruct4M) provide weakly-supervised structure / text-to-position alignment data. PDF-WuKong's PaperPDF and Docopilot's Doc-750K serve a similar role for *paper-style* multi-page PDFs.
3. **Generic VLM pretraining + document continue-pretraining.** Leopard combines LLaVA-558K, CC-3M, Cauldron, and Donut for the connector, then continues on Arxiv Pages (4M). Texthawk2 mixes web scale (LAION-400M, GrIT-20M, UMG-41M) with document-specific corpora.

**Trend:** the field is converging on a "general VLM → document continue-pretraining" pipeline rather than training document VLMs from scratch. Token-level structure prediction (DocStruct4M-style) is becoming the dominant pretext task for documents, replacing simple masked-language-modelling on OCR text.

**Implication for the survey:** classify pretraining datasets along two axes — *signal type* (OCR text, layout, structure-QA, captions) × *granularity* (single image, multi-image / multi-page).

### 2.2 Instruct-Tuning / Fine-Tuning

**Two clear strategies:**

| Strategy | Representative papers | Datasets used |
|---|---|---|
| **Aggregate many public sources** | Docopilot, Leopard, mPLUG-DocOwl2, DocR1 | DocVQA, ChartQA, InfographicVQA, DUDE, MP-DocVQA, TAT-DQA, WTQ, TabFact, TextVQA, OCR-VQA, FUNSD, KleisterCharity, DeepForm, ArxivQA, MultiHiertt, FigureQA, etc. |
| **Curate one new bespoke corpus** | Doc-750K (Docopilot), Leopard-Instruct (Leopard), CoR-Dataset (CoR), MHDocVQA (VDocRAG), PaperPDF (PDF-WuKong), MoLoRAG synthetic, DocDancer trajectories | Often built by prompting GPT-4o over scraped PDFs / slides |

**Reuse of evaluation datasets for training is rampant.** MP-DocVQA appears in 15 papers' SFT mixes *and* in 18 evaluation suites — i.e., almost every paper that fine-tunes uses MP-DocVQA's training split, and every paper evaluates on its test split. The same is true of DUDE (13 train / 15 eval) and DocVQA (9 train / 14 eval). The survey should explicitly call out that "supervised SOTA" and "zero-shot SOTA" numbers in the literature are computed against the *same* dataset families — there is essentially no held-out benchmark suite untouched by training.

**Synthetic-data generation has become standard.** MoLoRAG (5,500 GPT-4o QAs over MMLongBench), DocDancer (5,000 trajectories from LongDocURL/MMDocRAG/CUAD/DUDE), CoR (26K QA + 5K DPO), Leopard (Pew-Research, SlideShare via GPT-4o), PDF-WuKong (1.1M QA over scientific PDFs), Docopilot (Doc-750K from Arxiv/OpenReview/Sci-Hub). The pattern is: scrape PDFs or slides → ask a strong LLM/VLM to write QA pairs / reasoning traces → fine-tune. This shift is one of the most important developments to discuss in the survey.

**Reinforcement / preference learning is emerging but small.** DocR1 (GRPO with EviBench), CoR (DPO with 5K preference pairs). Worth flagging as a forward-looking trend.

**Implication for the survey:** distinguish between *task-mixture FT* (e.g., Leopard, mPLUG-DocOwl2, Docopilot) and *capability-targeted FT* (e.g., MoLoRAG's logical-relevance scorer, DFVC's adapter, KGP's retriever). The training-data composition reveals the *intent* of the model — generalist vs specialist.

### 2.3 Benchmarking

**The MP-VRDU benchmark stack has matured along three "page-count tiers":**

1. **Single-page (DocVQA, InfographicVQA, ChartQA, VisualMRC, FUNSD, CORD, TextVQA).** Originally the entire field. Still used as sanity-check or component evaluation.
2. **Tens-of-pages multi-page (MP-DocVQA: avg ~8 pages; DUDE: heterogeneous; SlideVQA: 20-page decks; TAT-DQA: tables-in-document).** The current "default" MP-VRDU benchmarks.
3. **Long / very-long-document (MMLongBench-Doc: 47 pages avg; LongDocURL: 85 pages avg; PaperTab/FetaTab; MMNIAH; PaperPDF; Arctic-TILT extends MMLongBench-Doc to 400 pages; Doc-V* tests on 468 pages; Self-Attention Scoring extends MP-DocVQA to 793 pages).** The new frontier.

**MMLongBench-Doc and LongDocURL are the de-facto "long-document" benchmarks** — together they cover 17 + 10 of the 36 papers and are the principal arena where retriever-generator and agentic-pipeline approaches differentiate themselves.

**Open-domain (cross-document) evaluation is rare and recent.** Only **M3DocVQA, ViDoSeek, and OpenDocVQA** test retrieval over a *corpus* of documents rather than within a single (long) document. KGP evaluates on multi-document QA (HotpotQA, IIRC, 2WikiMQA, MuSiQue) but those are text-only.

**Domain-specific evaluations are fragmented.** Financial (FinRAGBench-V, KleisterCharity, KleisterNDA, DeepForm, VQA-CD, VQAonBD, CUAD), scientific (ArxivFullQA, PaperPDF, PaperTab, FetaTab, ArXivLay, PubMed-Lay, MMNIAH), tables (TAT-DQA, MultiHiertt, WTQ, TabFact, TableBench, TableVQA-Bench), charts (ChartQA, MultiChartQA, CharXiv), slides (SlideVQA, ViDoSeek). Each appears in only 1–3 papers, signalling that domain coverage is uneven and there is room for a unified domain-segmented benchmark.

**Implication for the survey:** organise benchmarks by *page-count tier × domain × open/closed-domain*. This 3-axis taxonomy is more informative than a flat list.

---

## 3. Cross-cutting trends

### 3.1 The page-count arms race

Single-page DocVQA (2020) → multi-page MP-DocVQA (2023, Hi-VT5) → multi-document M3DocVQA (2024) and ViDoSeek (2025) → very-long MMLongBench-Doc / LongDocURL (2024–25). Each generation strains a different bottleneck: text recognition → cross-page reasoning → retrieval over corpora → context-window scaling. Newer papers (Arctic-TILT, Doc-V*, Self-Attention Scoring) deliberately *extend* existing benchmarks to even longer documents to stress modern architectures.

### 3.2 From "annotated by humans" to "synthesised by GPT-4o"

Every paper that introduces a new training corpus after 2024 uses an LLM/VLM to generate QA, trajectories, or reasoning traces. This raises three concerns the survey should discuss:

- **Distillation circularity.** Models being evaluated against GPT-4o on benchmarks are also being trained on GPT-4o-generated data. Performance gains are partly recovery of the teacher's behaviour.
- **Evaluation contamination risk.** Documents in MMLongBench-Doc / LongDocURL test sets have been used as *source documents* for synthesising training data in several papers (MoLoRAG, DocDancer). Careful test/train separation is sometimes underspecified.
- **Quality variability.** No paper systematically audits the synthesised data — a worthwhile research direction.

### 3.3 Modality coverage

Across the corpus, 24 papers explicitly handle **text + visual + layout** information; 8 are vision-only (Doc-V*, M3DocRAG, MoLoRAG, AVIR, Self-Attention Scoring, RM-T5, DocReact, DocR1). Pure-text approaches are extinct in the corpus — KGP is the only text-dominant pipeline. This is an important narrative point: MP-VRDU is now firmly multimodal.

### 3.4 Self-introduced datasets cluster around two niches

- **New benchmarks for new capabilities** (M3DocVQA, ViDoSeek, OpenDocVQA, MP-DocVQA): expose under-tested capabilities (cross-document, retrieval, slide-decks).
- **New training corpora for scale** (Doc-750K, PaperPDF, Leopard-Instruct, MP-DocStruct1M, CoR-Dataset, MHDocVQA): provide enough multi-page supervised data to fine-tune general VLMs.

This separation tells the story of the field: benchmarks measure where we want to go, corpora pay for the journey.

### 3.5 Architecture × dataset coupling

| Architecture family | Typical training data | Typical evaluation data |
|---|---|---|
| Hierarchical document transformers (Hi-VT5, GRAM, RM-T5, Arctic-TILT) | OCR-IDL / CCpdf + MP-DocVQA / DUDE / TAT-DQA | MP-DocVQA, DUDE, MMLongBench-Doc |
| Backbone-centric MLLMs (mPLUG-DocOwl2, Leopard, Texthawk2, Docopilot, Doc-V*, DocSLM, DocR1, CoR, InstructDoc, LayTokenLLM) | Bespoke instruction sets (Leopard-Instruct, Doc-750K, PaperPDF, CoR-Dataset, InstructDoc) + many public sources | MP-DocVQA, MMLongBench-Doc, DocVQA, ChartQA, InfographicVQA |
| Retriever-generator (M3DocRAG, MoLoRAG, MHier-RAG, MLDocRAG, AVIR, MultiDocFusion, DFVC, CREAM, Self-Attention Scoring, PDF-WuKong, RAG-DocVQA, VDocRAG) | Often training-light; some adapters on MP-DocVQA / MMLongBench-Doc | MMLongBench-Doc, LongDocURL, MP-DocVQA, M3DocVQA |
| Agentic pipelines (Doc-V*, DocAgent, DocDancer, DocLens, DocReact, M2RAG, MDocAgent, SimpleDoc, ViDoRAG, MACT, KGP) | None or synthesised trajectories | MMLongBench-Doc, LongDocURL, DocBench, SlideVQA |

The survey can use this matrix to argue that *evaluation choice tracks architecture choice*: agentic and retrieval-based papers cluster on long-document and open-domain evaluations, while backbone-centric models still report on the classic mid-length suite.

---

## 4. Gaps, biases, and open problems

1. **No held-out evaluation.** Almost every "evaluation" benchmark has been used as training data by *someone*. The community lacks a contamination-free benchmark for MP-VRDU.
2. **Domain narrowness.** Academic papers, slides, financial filings, and infographics dominate. Healthcare, legal-contract reasoning, government forms, multilingual documents (beyond Texthawk2's bilingual focus), and handwritten / historical documents are under-represented.
3. **Pages, not corpora.** Most benchmarks evaluate within a single document; multi-document retrieval is tested by only three datasets (M3DocVQA, ViDoSeek, OpenDocVQA).
4. **Question-style monoculture.** Most QA pairs are extractive or short-answer. Reasoning-heavy chains (multi-hop, comparative, counterfactual) are rare outside MMLongBench-Doc and LongDocURL.
5. **Modality bias toward English text + Western layouts.** Texthawk2 is the main bilingual exception; non-Latin scripts and right-to-left layouts are essentially absent.
6. **No standardised metric set.** ANLS for extractive, EM/F1 for QA, GPT-judge for open-ended — papers mix and match. The survey should flag the lack of metric standardisation alongside dataset standardisation.
7. **Synthetic-data quality is unaudited.** As discussed, this is the single largest methodological risk in the corpus.

---

## 5. Deep dive on the two structural flags

The two most important issues raised in §4 — train/test contamination and synthetic-data circularity — deserve a section of their own because they interact and because they shape almost every claim in the literature.

### 5.1 Flag 1 — Train/test contamination undermines "zero-shot" claims

#### The numbers behind the claim

| Benchmark | Papers using train split for SFT | Papers evaluating on test split | Overlap |
|---|---|---|---|
| MP-DocVQA | 15 | 18 | 13 papers do both |
| DUDE | 13 | 15 | 12 papers do both |
| DocVQA | 9 | 14 | 9 papers do both |
| InfographicVQA | 8 | 14 | 7 papers do both |
| ChartQA | 5 | 10 | 4 papers do both |
| SlideVQA | 3 | 13 | 3 papers do both |
| MMLongBench-Doc | 3 | 17 | 3 papers do both (and several more synthesise training data from its source documents) |
| LongDocURL | 0 (direct FT) | 10 | 2 papers (MoLoRAG, DocDancer) use its documents to synthesise training data |

Read: when paper X reports a number on MP-DocVQA, the prior probability that X (or its backbone) trained on MP-DocVQA's train split is ~70 %.

#### Three layers of contamination, ordered worst → mildest

1. **Direct in-domain evaluation labelled as zero-shot.** Several agentic and retrieval pipelines describe their evaluation as "no task-specific training" while their underlying VLM (Qwen2.5-VL, InternVL, Idefics, GPT-4o) was instruction-tuned on DocVQA / InfographicVQA / ChartQA. Strictly "zero-shot" with respect to the *pipeline* is not zero-shot with respect to the *evaluation*.
2. **Document-level overlap from synthetic data.** MoLoRAG synthesises 5,500 GPT-4o QAs over MMLongBench-Doc's source PDFs and then evaluates on MMLongBench-Doc's official test questions. Even if the *questions* differ, the *documents* are seen during training — which defeats the premise of long-document evaluation, where the test signal is precisely "can the model handle a previously unseen long document?"
3. **Backbone leakage.** Open-weight VLMs (Qwen2.5-VL, mPLUG-Owl2, Phi-3V, InternVL) include DocVQA / InfographicVQA / ChartQA in their pretraining or instruction-tuning recipes. Every paper using one of these as a frozen backbone inherits the leakage. None of the 36 papers in the corpus audits which datasets their backbone has seen.

#### Specific concerning cases

| Paper | Concern |
|---|---|
| MoLoRAG | Synthesises training QAs from MMLongBench-Doc source documents → evaluates on MMLongBench-Doc test split. Document-level overlap unaudited. |
| DocDancer | Source documents drawn from LongDocURL, MMDocRAG, CUAD, DUDE — four benchmarks they could be tested on. They evaluate on MMLongBench-Doc / DocBench, but document overlap with sister benchmarks is plausible. |
| DFVC | Trains adapter on 80 % of MMLongBench-Doc and MP-DocVQA, evaluates on the remaining 20 %. This is technically a clean split, but reported as "long-document evaluation" without flagging the in-domain status. |
| Most retriever-generator papers | Use frozen Qwen-VL or GPT-4o, then "evaluate zero-shot." Backbone has likely seen the test datasets. |

#### Responsibly handled cases (worth citing as positive examples)

- **Doc-V\*** explicitly labels MP-DocVQA and DUDE evaluations as "in-domain" and SlideVQA / LongDocURL / MMLongBench-Doc as "out-of-domain."
- **VDocRAG** keeps the zero-shot (ChartQA, SlideVQA) and supervised (InfographicVQA, DUDE) tracks separated and reports both.
- **Hi-VT5**, by virtue of *introducing* MP-DocVQA, naturally trains and tests on it — the contamination notion does not really apply when the dataset and split protocol come from the same paper.

#### What the survey should recommend

1. A **declaration table** in every paper listing every dataset the model (and its backbone) has seen, regardless of stage.
2. **In-domain vs. out-of-domain segmentation** of every results table.
3. **Document-overlap audits** when training data is synthesised from publicly-available documents.
4. The community should commission a **closed-test benchmark** — questions and documents that have never been released — analogous to MMLU's hidden suite or HELM-style holdouts.
5. A **leakage taxonomy**: split-level (train↔test), document-level (synthesis↔eval), backbone-level (pretraining↔eval). Papers should report all three.

### 5.2 Flag 2 — Synthetic-data circularity is now structural, not incidental

#### The numbers

Of the 36 papers, **12 introduce or rely on an LLM-synthesised training corpus**:

| Corpus | Paper | Generator |
|---|---|---|
| Doc-750K | Docopilot | GPT-4o over Arxiv / OpenReview / Sci-Hub |
| PaperPDF (1.1M) | PDF-WuKong | LLM over scraped scientific PDFs |
| Leopard-Instruct (925K) | Leopard | GPT-4o over Pew Research charts + SlideShare |
| CoR-Dataset (26K) + DPO 5K | CoR | GPT-4o reasoning traces |
| InstructDoc (30 sources) | InstructDoc | Templated + LLM-rewritten instructions |
| MoLoRAG (5,500) | MoLoRAG | GPT-4o over MMLongBench-Doc PDFs |
| MHDocVQA | VDocRAG | LLM-generated multi-hop QAs |
| Synthetic trajectories (5K) | DocDancer | GPT-4o exploration-then-synthesis |
| MP-DocReason51K | mPLUG-DocOwl2 | LLM-generated reasoning |
| DocStruct4M / MP-DocStruct1M | mPLUG-DocOwl2 | Templated + LLM rewrites |
| Pew Research / SlideShare QAs | Leopard | GPT-4o |
| Various GRPO trajectories | DocR1 | LLM-judged candidate filtering |

Of these 12, **9 source the underlying documents from existing evaluation benchmarks or the same public PDF pools as those benchmarks** (Arxiv, OpenReview, MMLongBench-Doc PDFs, LongDocURL PDFs, DUDE PDFs, CUAD).

#### The circularity loop

```
GPT-4o → writes QAs over PDFs
   ↓
Model X is fine-tuned to imitate GPT-4o's QA style
   ↓
Model X is evaluated on benchmarks where the gold answer was also produced (or LLM-judged) by GPT-4o
   ↓
Model X looks excellent because it has learned to mimic GPT-4o
```

Three knock-on effects:

1. **Performance gains may be distillation, not capability.** When DocR1 reports +X points on MMLongBench-Doc after training on synthesised reasoning traces, part of that delta is "the model now sounds more like the GPT-4o judge that scores MMLongBench-Doc." This isn't worthless — distillation has practical value — but it is *not* progress against the upper bound, only progress toward the teacher.
2. **Style transfer rather than reasoning gain.** GPT-4o has stylistic patterns: it tends to begin reasoning chains by enumerating sources, hedge with "Based on the document…", and prefer extractive over abstractive answers. Models trained on its output replicate these patterns regardless of whether they are good for the task.
3. **Common biases amplified.** Documented GPT-4o biases — hallucinating page numbers, preferring early-page content, over-extracting — propagate into every model trained on its synthesised traces. The field has not measured how much.

#### Three quality-control failures across the corpus

1. **No human-rated quality samples.** None of the 12 papers reports a human evaluation of a sampled subset of its synthesised data.
2. **No document-overlap audits.** Of papers that synthesise training data from existing benchmark documents, only Leopard explicitly partitions its source pool, and even there the partition is by document URL rather than by content hash.
3. **No final-answer fact-checking.** Synthesised QAs are accepted if the LLM produces them; correctness against the source PDF is rarely verified beyond a self-consistency check (i.e., the LLM is asked again).

#### Why the field tolerates this

The economic argument is simple: a single annotator might write 20 high-quality multi-page QA pairs in a day, costing ~$300 per 1,000 QAs. GPT-4o produces 1,000 QAs for under $20. To assemble Doc-750K (758K QAs) from human annotation would cost ~$11M and take ~3 years. Synthetic data is not a luxury; it is the only viable scaling path. The survey should acknowledge this trade-off rather than dismiss synthetic data — but should also press for the cheaper compensating measures (sampling audits, deduplication, cross-LLM QA agreement).

#### Specific best-practice recommendations to surface in the survey

1. **Mandatory quality reporting.** Future papers introducing a synthesised corpus should report (a) inter-annotator-style agreement using a *different* LLM as the second judge, and (b) a 200-QA human spot-check.
2. **Document-hash dedup** against MP-DocVQA / DUDE / MMLongBench-Doc / LongDocURL / SlideVQA test pools, with the dedup result reported.
3. **Generator diversity.** Papers should generate with at least two different LLMs (e.g., GPT-4o + Gemini 1.5 Pro + Claude) to dilute single-model bias.
4. **Generator–evaluator separation.** If GPT-4o was used to synthesise training data, the evaluation judge should not be GPT-4o.
5. **Public release of generation prompts.** Several papers omit the prompts used to synthesise QAs, making reproducibility and bias-analysis impossible.

### 5.3 Why the two flags are one structural problem, not two

The two flags interact: the synthetic-data wave (Flag 2) is the proximate *cause* of the contamination problem (Flag 1) — because every synthesised corpus is built over public PDFs, and most public PDFs are already in someone's evaluation benchmark. The survey should treat these not as two separate methodological footnotes but as a single, structural feature of MP-VRDU as a 2024–25 field: *the data-scaling strategy that made the field possible is the same strategy that has compromised its evaluation integrity*. Flagging this prominently is both honest and useful — and it is a clean motivation for proposing a held-out evaluation suite as a community deliverable.

---

## 6. Suggested section structure for the survey

A 6–8 page **Datasets** section can be organised as:

1. **Overview and scope** (½ page). Define MP-VRDU dataset, list inclusion criteria, explain the three-category split (PT / FT / Bench) and reference Tables 1-3 of `datasets_used.md`.
2. **Pretraining datasets** (1 page). Two-axis taxonomy (signal × granularity). Lead with Texthawk2's mega-mix and Leopard / mPLUG-DocOwl2's DocStruct4M / MP-DocStruct1M family. Discuss the OCR-pretext → structure-pretext shift.
3. **Fine-tuning datasets** (1–1.5 pages). Distinguish *task-mixture* vs *capability-targeted* FT. Highlight the rise of LLM-synthesised corpora (Doc-750K, PaperPDF, Leopard-Instruct, CoR-Dataset). Include a sub-section on RL/preference learning (DocR1, CoR).
4. **Benchmarking datasets** (2–2.5 pages). Use the *page-count tier × domain × open/closed-domain* taxonomy. Tier 1: single-page (sanity). Tier 2: mid-length (MP-DocVQA, DUDE, SlideVQA). Tier 3: long-document (MMLongBench-Doc, LongDocURL, PaperPDF). Tier 4: open-domain / cross-document (M3DocVQA, ViDoSeek, OpenDocVQA). Devote a paragraph each to the "core five" with dataset statistics.
5. **Trends across categories** (½–1 page). Page-count arms race; synthetic data; modality coverage; architecture × dataset coupling.
6. **Limitations and open problems** (½ page). Items 1–7 from §4 above.
7. **Recommended evaluation protocol** (optional, ½ page). Suggest a held-out subset strategy, contamination audits, multi-domain coverage, and standardised metrics — set up the rest of the paper.

**Tables to include:**
- Master dataset table (`datasets_used.md`'s three tables, possibly condensed).
- Statistics summary: dataset → #docs / #pages / #questions / domain / public release year.
- Architecture-family × benchmark-tier matrix (§3.5 above).

**Figures to consider:**
- A Sankey-style diagram from architecture family → preferred benchmark tier.
- A timeline of benchmark releases with page-count growth on the y-axis.
- A bar chart of "training reuse" — number of papers using each evaluation dataset for training too.

---

## 7. Glossary of dataset abbreviations

- **ANLS** — Average Normalised Levenshtein Similarity (metric, not a dataset, listed for completeness).
- **CCpdf** — Common-Crawl PDF subset used by Arctic-TILT.
- **CHART-Infographics** — chart understanding dataset used in Arctic-TILT FT.
- **CoR-Dataset** — Chain-of-Reading dataset (26,088 QA traces) introduced in CoR.
- **DocBench** — multi-page document evaluation (229 docs, ~1,082 Qs).
- **DocCVQA** — single-page Document Collection VQA, baseline in Hi-VT5.
- **DocStruct4M / MP-DocStruct1M** — structure-aware pretraining corpora used in mPLUG-DocOwl2 and (DocStruct4M only) VDocRAG.
- **DocVQA** — single-page document VQA (12K images, 50K QAs).
- **DUDE** — Document Understanding & Reasoning (multi-domain, multi-page; ~3K docs, 23.7K QAs).
- **DUDE / DocBench / MMLongBench-Doc / LongDocURL** — the four pillars of long-document evaluation.
- **DVQA / FigureQA / PlotQA / ChartQA / MultiChartQA** — chart-QA family.
- **FetaTab / PaperTab** — table-QA on academic papers (used as long-document tabular evaluation).
- **FUNSD / CORD / SROIE / POIE / XFUND / SIBR** — form / receipt / invoice IE datasets.
- **HotpotQA / IIRC / 2WikiMQA / MuSiQue** — text-only multi-document QA used by KGP.
- **InfographicVQA / InfoVQA** — same dataset, both spellings used.
- **IIT-CDIP / OCR-IDL / Wukong / Common-Crawl PDFs** — large OCR pretraining corpora.
- **KleisterCharity / KleisterNDA / DeepForm / VQA-CD / VQAonBD / CUAD** — financial / legal IE.
- **Leopard-Instruct** — 925K-instance multi-image instruction set introduced by Leopard.
- **LongDocURL** — long multimodal document benchmark (396 docs, 85 pages avg, 2,325 Qs).
- **M3DocVQA** — open-domain multi-document VQA (3,368 PDFs, 41K pages, 2,441 Qs) introduced by M3DocRAG.
- **MMLongBench-Doc / MMLongBench** — long-context multimodal document benchmark (135 docs, 47.5 pages avg).
- **MMNIAH** — multimodal Needle-in-a-Haystack benchmark for long PDFs.
- **MP-DocVQA** — Multi-Page DocVQA (5,928 docs, 60K pages, 46K QAs) introduced by Hi-VT5.
- **OCR-VQA** — OCR-based VQA, single-page.
- **OpenDocVQA** — unified open-domain DocVQA introduced by VDocRAG.
- **PaperPDF** — 1.1M bilingual scientific-PDF QA introduced by PDF-WuKong.
- **PDFTriage** — structural document QA dataset used by KGP.
- **RVL-CDIP / DocBank / DocLayNet / PubLayNet** — layout / document-classification corpora (mostly cited rather than actively used in this corpus).
- **SIBR** — real-world VIE dataset with degraded text.
- **SlideVQA** — 20-page slide-deck VQA.
- **TAT-DQA** — tables-in-document QA.
- **ViDoSeek** — visually-rich document retrieval/QA benchmark introduced by ViDoRAG (refined from SlideVQA).
- **VisualMRC** — visual reading comprehension over web pages.
- **WikiTableQuestions / WTQ** — table QA, used in mPLUG-DocOwl2 and DocR1.
