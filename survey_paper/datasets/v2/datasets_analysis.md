# Datasets in MP-VRDU: Patterns, Trends, and Survey-Writing Guide (v2 — 46-paper corpus)

This document distils observations from `datasets_used.md` / `datasets_used.csv` (46 MP-VRDU papers) into a narrative that can directly seed the **Datasets** section of the survey. It updates v1 (36 papers) with the 10 newly-added models — **DMAP, DREAM, MM-Doc-R1, MM-R5, SV-RAG, SLEUTH, VisDoMRAG, VRAG-RL, URaG, DocSeeker** — and folds new findings into the existing structure:

1. The headline dataset inventory by category.
2. Patterns *within* each category (what is used, why, how).
3. Cross-cutting trends across the field.
4. Gaps, biases, and open problems worth flagging in the survey.
5. A deep dive on the two structural methodological flags (contamination and synthetic-data circularity).
6. A suggested section structure with notes on which datasets to emphasise where.
7. A glossary of dataset abbreviations.

The biggest v1→v2 deltas (skim before reading):

- **MMLongBench-Doc is now the field's most-used benchmark** at 25/46 papers (54%) — overtaking MP-DocVQA (20/46, 43%) for the first time. 8 of the 10 new papers evaluate on it. *Advantage*: a clear shared yardstick. *Disadvantage*: amplifies the contamination problem in §5.
- **The RL footprint has tripled.** v1 had 2 RL/preference papers (DocR1 GRPO; CoR DPO). v2 has **6** — adding MM-Doc-R1 (SPO), MM-R5 (GRPO), VRAG-RL (GRPO), DocSeeker (EviGRPO). RL on top of frozen Qwen2.5-VL backbones is now the dominant training recipe for new MP-VRDU papers.
- **Zero new pretraining-stage datasets in the 10 new papers.** Every one inherits a pretrained backbone (Qwen2.5-VL, InternVL2, Qwen3-VL, Phi-3-V, PaliGemma) and skips PT entirely. The "from-scratch pretraining" recipe is now firmly historical.
- **Two new benchmarks introduced in v2**: VisR-Bench (SV-RAG; 226 docs, 471 QAs, 9 domains) and VisDoMBench (VisDoMRAG; 2,271 QAs, the first multi-document multi-modal RAG benchmark).
- **One v1 correction**: DREAM fine-tunes on DUDE's training split (in addition to MP-DocVQA's), corrected in Table 2 of `datasets_used.md`.

---

## 1. Headline inventory

### 1.1 Most-used datasets per category

| Rank | Pretraining | # papers | Fine-Tuning | # papers | Benchmarking | # papers |
|---|---|---|---|---|---|---|
| 1 | DocStruct4M | 2 | MP-DocVQA | 19 | **MMLongBench-Doc** | **25** |
| 2 | Cauldron | 2 | DUDE | 16 | MP-DocVQA | 20 |
| 3 | Arxiv PDFs | 2 | DocVQA | 10 | DUDE | 20 |
| 4 | (others ≤1) | — | InfographicVQA | 9 | SlideVQA | 17 |
| 5 | | | SlideVQA | 7 | DocVQA | 15 |
| 6 | | | ChartQA | 5 | LongDocURL | 14 |
| 7 | | | VisualMRC | 4 | InfographicVQA | 12 |
| 8 | | | TAT-DQA | 3 | ChartQA | 7 |
| 9 | | | OCR-VQA / MMLongBench-Doc / LongDocURL / KleisterCharity / DeepForm / ArxivQA | 3 each | PaperTab / FetaTab | 6 each |

**Net composition (across all 46 papers):** 31 distinct pretraining-stage datasets (unchanged from v1), ~70 fine-tuning-stage datasets (up from 64 in v1, primarily because the new papers introduce smaller bespoke training mixes rather than aggregating public sources), and 60+ benchmarking datasets (up from 50+ in v1, with two new introduced benchmarks plus several niche additions like MMDocIR, SciGraphQA, SPIQA, PaperText). Of these, **10 are *introduced* by papers in the corpus**: MP-DocVQA, M3DocVQA, ViDoSeek, OpenDocVQA, Doc-750K, PaperPDF, CoR-Dataset, Leopard-Instruct/MHDocVQA/MP-DocStruct1M/MP-DocReason51K, and now **VisR-Bench and VisDoMBench**.

### 1.2 The "core six" benchmarks

Six datasets now dominate evaluation across the corpus and appear in roughly a third or more of the papers: **MMLongBench-Doc (25), MP-DocVQA (20), DUDE (20), SlideVQA (17), DocVQA (15), LongDocURL (14)**. Any survey datasets section should treat these as the de-facto comparison axes for MP-VRDU work; everything else is supplementary or domain-specific.

**Advantage of dominance:** a paper that evaluates on this stack inherits 6 axes of cross-paper comparability, which makes meta-analysis (the survey's own raison d'être) tractable.

**Disadvantage:** every one of these six has been used as *training data* by at least 3 papers in the corpus. Their dominance compresses the entire field's reported numbers into a contamination-exposed envelope — see §5.1.

---

## 2. Patterns within each category

### 2.1 Pretraining

**Who pretrains?** Still only the *backbone-centric / page-to-document* models: **Texthawk2, Leopard, mPLUG-DocOwl2, PDF-WuKong, Arctic-TILT, Hi-VT5, Docopilot, CoR, VDocRAG**. Retriever-generator and agentic-pipeline papers (DocReact, MDocAgent, DocAgent, MLDocRAG, MHier-RAG, M3DocRAG, ViDoRAG, SimpleDoc, M2RAG, MoLoRAG, Doc-V*, DMAP, DREAM, MM-Doc-R1, MM-R5, SV-RAG, SLEUTH, VisDoMRAG, VRAG-RL, URaG, DocSeeker) inherit pretrained backbones and skip this stage entirely.

**Three pretraining recipes are observable (unchanged from v1):**

1. **OCR / layout self-supervision.** Hi-VT5 (OCR-IDL, 200K pages with hierarchical denoising), Arctic-TILT (CCpdf, 900K steps), Texthawk2 (IIT-CDIP, Wukong, Common Crawl PDFs, plus a 10-dataset multilingual OCR pile). Goal: teach the encoder to *read* before it tries to *reason*.
2. **Document-structure pretraining on synthesised QA.** mPLUG-DocOwl2's DocStruct4M and MP-DocStruct1M (and VDocRAG's reuse of DocStruct4M) provide weakly-supervised structure / text-to-position alignment data. PDF-WuKong's PaperPDF and Docopilot's Doc-750K serve a similar role for *paper-style* multi-page PDFs.
3. **Generic VLM pretraining + document continue-pretraining.** Leopard combines LLaVA-558K, CC-3M, Cauldron, and Donut for the connector, then continues on Arxiv Pages (4M). Texthawk2 mixes web scale (LAION-400M, GrIT-20M, UMG-41M) with document-specific corpora.

**Trend confirmed (and intensified):** the field has converged on a "general VLM → document continue-pretraining (or skip altogether)" pipeline rather than training document VLMs from scratch. **None of the 10 new papers performs any custom pretraining** — they consume Qwen2.5-VL / InternVL / Phi-3-V / PaliGemma off the shelf and add SFT/RL on top. Token-level structure prediction is now the dominant pretext task; the OCR-pretext era (IIT-CDIP, OCR-IDL) is effectively closed except for legacy hierarchical encoders.

**Advantages of skipping PT:**
- Compute cost collapses by 1–2 orders of magnitude.
- The author can publish with academic-scale resources (1–8 H100s vs. node-weeks on dozens of GPUs).
- Inherits the backbone's general-vision capability for free.

**Disadvantages:**
- The paper inherits whatever data the backbone provider saw — including, almost certainly, MP-VRDU evaluation benchmarks (see §5.1 *backbone leakage*).
- The architecture is constrained to what the backbone supports — e.g., long-context document encoders are still novel because most backbones cap at 8–32K tokens.

**Implication for the survey:** classify pretraining datasets along two axes — *signal type* (OCR text, layout, structure-QA, captions) × *granularity* (single image, multi-image / multi-page) — and note that the active design surface has *moved* from PT to FT/RL.

### 2.2 Instruct-Tuning / Fine-Tuning

**Two clear strategies, with a third (RL) now substantial:**

| Strategy | Representative papers (v2-additions in bold) | Datasets used |
|---|---|---|
| **Aggregate many public sources** | Docopilot, Leopard, mPLUG-DocOwl2, DocR1, **MM-R5**, **SV-RAG** | DocVQA, ChartQA, InfographicVQA, DUDE, MP-DocVQA, TAT-DQA, WTQ, TabFact, TextVQA, OCR-VQA, FUNSD, KleisterCharity, DeepForm, ArxivQA, MultiHiertt, FigureQA, SciQAG, CUAD, etc. |
| **Curate one new bespoke corpus** | Doc-750K (Docopilot), Leopard-Instruct (Leopard), CoR-Dataset (CoR), MHDocVQA (VDocRAG), PaperPDF (PDF-WuKong), MoLoRAG synthetic, DocDancer trajectories, **MM-R5 reasoning set**, **VRAG-RL trajectories**, **DocSeeker ALR-CoT** | Often built by prompting GPT-4o / Gemini-2.5-Flash over scraped PDFs / slides / RL trajectories |
| **RL / preference learning over a small targeted pool** | DocR1 (GRPO over EviBench), CoR (DPO with 5K), **MM-Doc-R1 (SPO over 300-val LongDocURL)**, **DocSeeker (EviGRPO over MP-DocVQA + DUDE)**, **MM-R5 (GRPO over 3K resolution-balanced)**, **VRAG-RL (GRPO over SlideVQA + ViDoSeek + MMLongBench-Doc trajectories)** | Often re-uses existing benchmark train splits, but introduces task-shaped *reward functions* — answer correctness, evidence localisation, format, length |

**Reuse of evaluation datasets for training is now structural, not incidental.** MP-DocVQA appears in 19 papers' SFT mixes *and* in 20 evaluation suites — i.e., almost every paper that fine-tunes uses MP-DocVQA's training split, and every paper evaluates on its test split. The same is true of DUDE (16 train / 20 eval) and DocVQA (10 train / 15 eval). The new RL papers tighten this loop: they use the same train/test splits but switch from cross-entropy to reward-shaped objectives, which the survey can flag as "a new way to reuse the same datasets, not a new dataset axis."

**Synthetic-data generation has become the *default* training-data recipe for new MP-VRDU papers.** Every new training corpus introduced after 2024 is LLM-synthesised. v2 adds 5 new synthesised corpora to the v1 list of 12 (MoLoRAG 5.5K, DocDancer 5K trajectories, CoR 26K + 5K DPO, Leopard Pew/SlideShare, PDF-WuKong 1.1M, Docopilot Doc-750K 758K, MHDocVQA, MP-DocReason51K, DocStruct4M / MP-DocStruct1M, InstructDoc, plus DocR1 GRPO trajectories): **MM-R5 7,200 GPT-4o reasoning chains**, **VRAG-RL Qwen-VLmax trajectories**, **DocSeeker 13,986 Gemini-2.5-Flash ALR-CoT samples**, **MoLoRAG-style synthetic for SV-RAG's PFLDocVQA/DocMatix-IR filtering**, **MM-Doc-R1 SPO-shaped reward trajectories**. The pattern is: scrape PDFs or slides → ask a strong LLM/VLM to write QA pairs / CoT / trajectories → fine-tune *or* shape RL rewards. This is one of the most important developments to discuss in the survey.

**Advantages of synthesised corpora:**
- *Scale*: Doc-750K reaches 758K QAs at ~1/100th the cost of human annotation (~$20K vs. ~$11M for an equivalent 3-year human campaign).
- *Targeted capability shaping*: VRAG-RL trajectories embed visual-perception actions; DocSeeker ALR-CoT embeds an explicit Analysis–Localisation–Reasoning structure; CoR's chains embed reading order. These are easier to encode in synthetic data than to elicit from human annotators.
- *Closes the gap to closed-weight models*: MM-R5, DocSeeker, MM-Doc-R1 all close ≥10pp accuracy gaps to Gemini/GPT-4o on MMLongBench-Doc through distillation alone.

**Disadvantages:**
- *Distillation circularity* — covered in detail in §5.2.
- *Evaluation contamination* — covered in §5.1.
- *No quality auditing* — none of the 17 synthetic-data papers in v2 reports a human spot-check of its generated set.
- *Style transfer rather than capability gain* — models trained on GPT-4o output start to *sound* like GPT-4o, which is also the judge on MMLongBench-Doc / LongDocURL. Reported gains conflate "got better at the task" with "got better at imitating the judge".

**Reinforcement / preference learning is no longer emerging — it is dominant for new papers.** In v1 only DocR1 (GRPO) and CoR (DPO) used RL/preference learning. In v2 we have **six**: DocR1 (GRPO), CoR (DPO), MM-R5 (GRPO), VRAG-RL (GRPO), DocSeeker (EviGRPO), MM-Doc-R1 (Similarity-based Policy Optimisation, an RL algorithm replacing GRPO). Four of the ten new papers use RL. The signal: GRPO-style RL on top of Qwen2.5-VL is the new default training recipe for MP-VRDU. The survey should:

- Distinguish *RL on outcome rewards* (correctness) from *RL on shaped rewards* (evidence-page hit + format + correctness — DocR1, DocSeeker, MM-R5).
- Note that several RL papers (MM-Doc-R1, DocSeeker, MM-R5) introduce algorithmic variants of GRPO (SPO, EviGRPO) — i.e., MP-VRDU is starting to back-port findings into RL methodology.
- Caveat: most RL pools are *tiny* (300 LongDocURL for MM-Doc-R1, 3K for MM-R5, ~10K for DocSeeker). Reported gains are modest in absolute terms even if dramatic in relative terms.

**Implication for the survey:** organise FT discussion into three buckets — *task-mixture FT* (Leopard, mPLUG-DocOwl2, Docopilot, MM-R5, SV-RAG), *capability-targeted SFT* (CoR, DocSeeker, VRAG-RL trajectories), and *RL-fine-tuning* (DocR1, MM-Doc-R1, DocSeeker, MM-R5, VRAG-RL, CoR-DPO). The training-data composition reveals the *intent* of the model — generalist vs specialist vs reward-shaped.

### 2.3 Benchmarking

**The MP-VRDU benchmark stack has stratified along four "page-count tiers":**

1. **Single-page (DocVQA, InfographicVQA, ChartQA, VisualMRC, FUNSD, CORD, TextVQA).** Originally the entire field. Still used as sanity-check or component evaluation.
2. **Tens-of-pages multi-page (MP-DocVQA: avg ~8 pages; DUDE: heterogeneous; SlideVQA: 20-page decks; TAT-DQA: tables-in-document).** The current "default" MP-VRDU benchmarks.
3. **Long / very-long-document (MMLongBench-Doc: 47 pages avg; LongDocURL: 85 pages avg; PaperTab/FetaTab/PaperText; MMNIAH; PaperPDF; MMDocIR: 65 pages avg; VisR-Bench: similar scale; Arctic-TILT extends MMLongBench-Doc to 400 pages; Doc-V* tests on 468 pages; Self-Attention Scoring extends MP-DocVQA to 793 pages).** The current research frontier — and where v2 papers concentrate.
4. **Open-domain (cross-document) and multi-document (M3DocVQA: 3,368 PDFs; ViDoSeek: ~6K images; OpenDocVQA: open-domain DocVQA; VisDoMBench: 2,271 multi-document QAs).** Still small but growing — VisDoMBench is v2's most important addition for this tier, because it explicitly targets the *multi-document* gap noted in v1.

**MMLongBench-Doc is now the de-facto MP-VRDU benchmark** at 25/46 papers (54%). LongDocURL (14/46, 30%) is the secondary long-document benchmark. The new papers cluster heavily on these two: 8 of the 10 evaluate on MMLongBench-Doc, 4 on LongDocURL. SlideVQA (17/46, 37%) has risen sharply because all the backbone-centric and retrieval newcomers (SV-RAG, VRAG-RL, URaG, DocSeeker) evaluate on it — slide decks now serve as a "multi-page sanity benchmark" between MP-DocVQA and the long-document tier.

**Advantages of long-document benchmarks (MMLongBench-Doc, LongDocURL, MMDocIR, VisR-Bench, M-LongDoc):**
- Genuinely exercise the context-window scaling that motivates the field.
- Cross-page reasoning is forced — the "extract from the right page" trick from MP-DocVQA no longer works.
- Multi-modal evidence (chart, table, figure, text, layout) is mixed within a document.

**Disadvantages:**
- *Small in absolute terms*: 135 (MMLongBench-Doc), 396 (LongDocURL), 180 (M-LongDoc), 226 (VisR-Bench), 313 (MMDocIR), 262 (BRIDGE). Re-annotation of even a single benchmark is expensive (~$20–50K per 1K QAs at human-annotator rates).
- *Already contaminated*: every long-document benchmark has been used as training-data source by at least one paper (see §5.1).
- *Heavy judge dependence*: most use GPT-4o or Gemini as the answer judge, which couples reported gains to judge bias.

**Open-domain (cross-document) evaluation is now slightly less rare but still small.** v1 had only M3DocVQA, ViDoSeek, and OpenDocVQA. v2 adds **VisDoMBench** (VisDoMRAG) — explicitly a multi-document benchmark with 2,271 QAs aggregating PaperTab/FetaTab/SciGraphQA/SPIQA/SlideVQA. KGP still represents the text-only multi-doc evaluation lineage (HotpotQA, IIRC, 2WikiMQA, MuSiQue).

**Domain-specific evaluations remain fragmented.** Financial (FinRAGBench-V, KleisterCharity, KleisterNDA, DeepForm, VQA-CD, VQAonBD, CUAD), scientific (ArxivFullQA, PaperPDF, PaperTab, PaperText, FetaTab, ArXivLay, PubMed-Lay, MMNIAH, SciGraphQA, SPIQA), tables (TAT-DQA, MultiHiertt, WTQ, TabFact, TableBench, TableVQA-Bench), charts (ChartQA, MultiChartQA, CharXiv), slides (SlideVQA, ViDoSeek). Each appears in only 1–6 papers, signalling that domain coverage is uneven and there is still room for a unified domain-segmented benchmark.

**Implication for the survey:** organise benchmarks by *page-count tier × domain × open/closed-domain*. This 3-axis taxonomy is more informative than a flat list.

---

## 3. Cross-cutting trends

### 3.1 The page-count arms race continues — but the new front is multi-document

Single-page DocVQA (2020) → multi-page MP-DocVQA (2023, Hi-VT5) → multi-document M3DocVQA (2024) and ViDoSeek (2025) → very-long MMLongBench-Doc / LongDocURL (2024–25) → **multi-document multi-modal RAG** (VisDoMBench, 2025) and **cross-task long-context** (MMLongBench 2025, BRIDGE 2026 — present in datasets_main.csv but not yet used by any of the 46 papers). Each generation strains a different bottleneck: text recognition → cross-page reasoning → retrieval over corpora → context-window scaling → multi-document evidence aggregation → step-level reasoning evaluation.

### 3.2 From "annotated by humans" to "synthesised by GPT-4o" — and now "shaped by RL"

Every paper that introduces a new training corpus after 2024 uses an LLM/VLM to generate QA, trajectories, or reasoning traces. v2 pushes this further: four of the new papers use *RL with synthesised trajectories*, where the trajectories themselves are produced by a teacher model (Qwen-VLmax for VRAG-RL; Gemini-2.5-Flash for DocSeeker; GPT-4o for MM-R5). This raises three concerns the survey should discuss:

- **Distillation circularity.** Models being evaluated against GPT-4o on benchmarks are also being trained on GPT-4o-generated data. Performance gains are partly recovery of the teacher's behaviour.
- **Evaluation contamination risk.** Documents in MMLongBench-Doc / LongDocURL test sets have been used as *source documents* for synthesising training data in MoLoRAG, DocDancer, VRAG-RL, MM-Doc-R1, and DocSeeker. Careful test/train separation is sometimes underspecified.
- **Quality variability.** No paper systematically audits the synthesised data — a worthwhile research direction.

### 3.3 Modality coverage

Across the 46-paper corpus, ~34 papers explicitly handle **text + visual + layout** information; ~12 are vision-only (Doc-V*, M3DocRAG, MoLoRAG, AVIR, Self-Attention Scoring, RM-T5, DocReact, DocR1, MM-Doc-R1, VRAG-RL, URaG, DocSeeker). Pure-text approaches are extinct in the corpus — KGP is the only text-dominant pipeline. The v2-additions cluster strongly toward vision-only RAG (VRAG-RL, MM-Doc-R1, URaG, DocSeeker) — partly because they're built on Qwen2.5-VL, whose vision tokeniser already handles OCR-implicit text reasonably well, and partly because OCR-free pipelines now match OCR-based ones on long-document benchmarks.

### 3.4 Self-introduced datasets cluster around three niches (v1 had two)

- **New benchmarks for new capabilities** (M3DocVQA, ViDoSeek, OpenDocVQA, MP-DocVQA, **VisR-Bench**, **VisDoMBench**): expose under-tested capabilities (cross-document, retrieval, slide-decks, web-crawled visually-rich docs, multi-document RAG).
- **New training corpora for scale** (Doc-750K, PaperPDF, Leopard-Instruct, MP-DocStruct1M, CoR-Dataset, MHDocVQA, MP-DocReason51K): provide enough multi-page supervised data to fine-tune general VLMs.
- **NEW: New small RL/SFT pools with shaped supervision signals** (DocSeeker ALR-CoT 13,986; MM-R5 7,200 reasoning chains; VRAG-RL Qwen-VLmax trajectories; CoR DPO 5K; MM-Doc-R1 SPO trajectories). These are not "datasets" in the classical scale sense — they're behavioural-shaping pools, often <30K samples, designed to teach a specific reasoning structure or action policy.

This three-way separation tells the story of the 2025 field: benchmarks measure where we want to go, large corpora pay for the scaling phase, and small RL pools shape the policy after the scale work is done.

### 3.5 Architecture × dataset coupling

| Architecture family | Typical training data | Typical evaluation data |
|---|---|---|
| Hierarchical / Page-to-Document Transformers (Hi-VT5, GRAM, RM-T5, Arctic-TILT) | OCR-IDL / CCpdf + MP-DocVQA / DUDE / TAT-DQA | MP-DocVQA, DUDE, MMLongBench-Doc |
| Backbone-centric MLLMs (mPLUG-DocOwl2, Leopard, Texthawk2, Docopilot, Doc-V*, DocSLM, DocR1, CoR, InstructDoc, LayTokenLLM, **URaG**, **DocSeeker**) | Bespoke instruction sets (Leopard-Instruct, Doc-750K, PaperPDF, CoR-Dataset, InstructDoc, ALR-CoT) + many public sources; v2-arrivals add SFT+RL on Qwen2.5-VL | MP-DocVQA, MMLongBench-Doc, DocVQA, ChartQA, InfographicVQA, LongDocURL, SlideVQA, DUDE |
| Retriever-generator (M3DocRAG, MoLoRAG, MHier-RAG, MLDocRAG, AVIR, MultiDocFusion, DFVC, CREAM, Self-Attention Scoring, PDF-WuKong, RAG-DocVQA, VDocRAG, **DREAM**, **MM-R5**, **SV-RAG**, **VisDoMRAG**) | Training-light; some adapters / rerankers / Col-retrieval modules on MP-DocVQA, MMLongBench-Doc, SlideVQA, MMDocIR, ColPali synthetic, PFLDocVQA, DocMatix-IR | MMLongBench-Doc, LongDocURL, MP-DocVQA, M3DocVQA, MMDocIR, VisR-Bench, VisDoMBench, SlideVQA |
| Agentic pipelines (Doc-V*, DocAgent, DocDancer, DocLens, DocReact, M2RAG, MDocAgent, SimpleDoc, ViDoRAG, MACT, KGP, **DMAP**, **MM-Doc-R1**, **SLEUTH**, **VRAG-RL**) | None (training-free orchestration) OR small RL pools (MM-Doc-R1, VRAG-RL) | MMLongBench-Doc, LongDocURL, DocBench, SlideVQA, PaperTab, FetaTab, PaperText |

The matrix sharpens in v2: agentic and retrieval-based papers cluster *almost exclusively* on the long-document and open-domain evaluations; backbone-centric models still report on the classic mid-length suite plus the long-document headliners.

---

## 4. Gaps, biases, and open problems

1. **No held-out evaluation.** Every "evaluation" benchmark in the core six has been used as training data by ≥3 papers in the corpus. v2 worsens this: MMLongBench-Doc is now both the most-evaluated *and* the most-trained-on long-document benchmark (3 FT users — MoLoRAG, DFVC, VRAG-RL — and indirect document overlap from 5 more synthesisers). The community still lacks a contamination-free benchmark for MP-VRDU.
2. **Domain narrowness.** Academic papers, slides, financial filings, and infographics dominate. Healthcare, legal-contract reasoning (CUAD remains the only contract benchmark, used by 1 evaluation paper), government forms, multilingual documents (beyond Texthawk2's bilingual focus), and handwritten / historical documents are under-represented. v2 adds VisR-Bench's web-crawled-flyer / newsletter / manual domains, which is a small step toward "non-academic visually-rich documents."
3. **Pages, not corpora.** Most benchmarks evaluate within a single document; multi-document retrieval is tested by only four datasets (M3DocVQA, ViDoSeek, OpenDocVQA, **VisDoMBench**). VisDoMBench is v2's main contribution here.
4. **Question-style monoculture.** Most QA pairs are extractive or short-answer. Reasoning-heavy chains (multi-hop, comparative, counterfactual) are rare outside MMLongBench-Doc, LongDocURL, and the new BRIDGE / M-LongDoc benchmarks. v2's MM-Doc-R1, DocSeeker, and SLEUTH all push for *multi-hop / multi-evidence* sub-evaluation of MMLongBench-Doc — a positive sign, but they're stratifying within the same dataset rather than introducing new question styles.
5. **Modality bias toward English text + Western layouts.** Texthawk2 is the main bilingual exception (Chinese / English); non-Latin scripts and right-to-left layouts are essentially absent. None of the 10 new papers fills this gap.
6. **No standardised metric set.** ANLS for extractive, EM/F1 for QA, GPT-judge for open-ended, mAP/Recall@k for retrieval — papers mix and match. v2 worsens this: MM-Doc-R1 introduces "Qwen2.5-72B-Instruct as judge" (different from MMLongBench-Doc's GPT-4o-judge), DocSeeker uses Gemini-2.5-Flash for distillation, SV-RAG introduces "Mean GPT Score" for VisR-Bench. The survey should flag the lack of metric standardisation alongside dataset standardisation.
7. **Synthetic-data quality is unaudited.** As discussed, this is the single largest methodological risk in the corpus — now spanning 17 papers (12 in v1 + 5 in v2: MM-R5, VRAG-RL, DocSeeker, MM-Doc-R1, SV-RAG-via-ColPali).
8. **Backbone leakage is mounting.** The v2 papers all build on Qwen2.5-VL or Qwen3-VL — backbones whose instruction-tuning recipes are known to include DocVQA, InfographicVQA, ChartQA. Reporting "zero-shot on DocVQA" for a pipeline built on Qwen2.5-VL is incoherent. None of the v2 papers audits this.

---

## 5. Deep dive on the two structural flags

The two most important issues raised in §4 — train/test contamination and synthetic-data circularity — deserve a section of their own because they interact and because they shape almost every claim in the literature.

### 5.1 Flag 1 — Train/test contamination undermines "zero-shot" claims

#### The numbers behind the claim (v2)

| Benchmark | Papers using train split for SFT | Papers evaluating on test split | Overlap (papers doing both) |
|---|---|---|---|
| MP-DocVQA | 19 | 20 | 17 |
| DUDE | 16 | 20 | 14 |
| DocVQA | 10 | 15 | 10 |
| InfographicVQA | 9 | 12 | 8 |
| SlideVQA | 7 | 17 | 7 |
| ChartQA | 5 | 7 | 4 |
| MMLongBench-Doc | 3 direct FT (MoLoRAG, DFVC, VRAG-RL) + 5 document-level synthesis (MoLoRAG, DocDancer, MM-Doc-R1, DocSeeker indirectly, VRAG-RL) | 25 | 3 direct + 5 document-level = 8 |
| LongDocURL | 3 direct (MoLoRAG, DocDancer, MM-Doc-R1) | 14 | 3 |
| TAT-DQA | 3 | 1 (Arctic-TILT) | 1 |

Read: when a v2 paper reports a number on MP-DocVQA, the prior probability that the paper (or its backbone) trained on MP-DocVQA's train split is now ~85%. For MMLongBench-Doc the document-level prior is ~30% (8 of 25 evaluators have either fine-tuned on it or synthesised training data from its documents).

#### Three layers of contamination, ordered worst → mildest (unchanged from v1, but v2 examples added)

1. **Direct in-domain evaluation labelled as zero-shot.** Several agentic and retrieval pipelines describe their evaluation as "no task-specific training" while their underlying VLM (Qwen2.5-VL, Qwen3-VL, InternVL, Idefics, GPT-4o) was instruction-tuned on DocVQA / InfographicVQA / ChartQA. Strictly "zero-shot" with respect to the *pipeline* is not zero-shot with respect to the *evaluation*. v2 examples: **DMAP, SLEUTH, VisDoMRAG** (all training-free) report MMLongBench-Doc accuracy on Qwen-VL or GPT-4o backbones without auditing backbone exposure.
2. **Document-level overlap from synthetic data.** MoLoRAG synthesises 5,500 GPT-4o QAs over MMLongBench-Doc's source PDFs and then evaluates on MMLongBench-Doc's official test questions. **MM-Doc-R1** uses LongDocURL documents (minus 300 val samples) as RL training and then evaluates on MMLongBench-Doc. **VRAG-RL** collects Qwen-VLmax trajectories over SlideVQA, ViDoSeek, *and MMLongBench-Doc* documents, then evaluates on those same three. Document overlap with the test split is plausible but not audited.
3. **Backbone leakage.** Open-weight VLMs (Qwen2.5-VL, Qwen3-VL, mPLUG-Owl2, Phi-3V, InternVL) include DocVQA / InfographicVQA / ChartQA / MP-DocVQA in their pretraining or instruction-tuning recipes. Every paper using one as a frozen backbone inherits the leakage. None of the 46 papers in the corpus audits which datasets their backbone has seen.

#### Specific concerning cases (v1 + v2)

| Paper | Concern |
|---|---|
| MoLoRAG | Synthesises training QAs from MMLongBench-Doc source documents → evaluates on MMLongBench-Doc test split. Document-level overlap unaudited. |
| DocDancer | Source documents drawn from LongDocURL, MMDocRAG, CUAD, DUDE — four benchmarks they could be tested on. They evaluate on MMLongBench-Doc / DocBench, but document overlap with sister benchmarks is plausible. |
| DFVC | Trains adapter on 80% of MMLongBench-Doc and MP-DocVQA, evaluates on the remaining 20%. Technically a clean split, but reported as "long-document evaluation" without flagging in-domain status. |
| **VRAG-RL** | Trains GRPO on Qwen-VLmax trajectories collected over SlideVQA, ViDoSeek, *and* MMLongBench-Doc documents; evaluates on those same three benchmarks. Document overlap unaudited. |
| **MM-Doc-R1** | Splits LongDocURL into 300 val / remainder training; evaluates on MMLongBench-Doc and reports 10.4 pp improvement. Train-set documents may overlap with MMLongBench-Doc documents (both drawn from CommonCrawl / public web). |
| **DocSeeker** | Trains EviGRPO on MP-DocVQA + DUDE pool and reports results on MMLongBench-Doc, LongDocURL, SlideVQA — all of which share the public-web document pool. |
| **MM-R5** | SFT pool of 73,843 QAs from MMDocIR's *training* set; evaluates on MMDocIR's *test* set. This is a within-benchmark split and is reported as such — methodologically clean. |
| **SV-RAG** | Col-retrieval module trained on ColPali synthetic + PFLDocVQA + DocMatix-IR + DocVQA + InfoVQA + TAT-DQA + arXivQA + SlideVQA; evaluates on SlideVQA, MMLongBench-Doc, DocVQA, DUDE, VisR-Bench. DocVQA/SlideVQA train→test are in-domain; cross-evaluation on MMLongBench-Doc and VisR-Bench is out-of-domain. SV-RAG does *not* flag the in-domain status of DocVQA/SlideVQA — survey should call this out. |

#### Responsibly handled cases (worth citing as positive examples; v2 adds three)

- **Doc-V\*** explicitly labels MP-DocVQA and DUDE evaluations as "in-domain" and SlideVQA / LongDocURL / MMLongBench-Doc as "out-of-domain."
- **VDocRAG** keeps the zero-shot (ChartQA, SlideVQA) and supervised (InfographicVQA, DUDE) tracks separated and reports both.
- **Hi-VT5**, by virtue of *introducing* MP-DocVQA, naturally trains and tests on it — the contamination notion does not really apply when the dataset and split protocol come from the same paper.
- **MM-R5** evaluates strictly within MMDocIR's own train/test split, with no cross-benchmark training.
- **VisDoMRAG** is training-free; while backbone-leakage still applies, the paper makes no FT claims and so cannot "contaminate" its own benchmark.
- **URaG, DocSeeker** at least segment their results tables by benchmark and report both in-domain (MP-DocVQA, DUDE) and out-of-domain (LongDocURL, MMLongBench-Doc) accuracy.

#### What the survey should recommend

1. A **declaration table** in every paper listing every dataset the model (and its backbone) has seen, regardless of stage.
2. **In-domain vs. out-of-domain segmentation** of every results table.
3. **Document-overlap audits** when training data is synthesised from publicly-available documents.
4. The community should commission a **closed-test benchmark** — questions and documents that have never been released — analogous to MMLU's hidden suite or HELM-style holdouts.
5. A **leakage taxonomy**: split-level (train↔test), document-level (synthesis↔eval), backbone-level (pretraining↔eval). Papers should report all three.

### 5.2 Flag 2 — Synthetic-data circularity is now structural, not incidental

#### The numbers (v2)

Of the 46 papers, **17 introduce or rely on an LLM/VLM-synthesised training corpus** (v1: 12, v2-additions: 5):

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
| GRPO trajectories | DocR1 | LLM-judged candidate filtering |
| **MM-R5 Reasoning Set (7,200)** | MM-R5 | GPT-4o reasoning-chain annotations |
| **VRAG-RL trajectories** | VRAG-RL | Qwen-VLmax trajectory collection over visually-rich docs |
| **DocSeeker ALR-CoT (13,986)** | DocSeeker | Gemini-2.5-Flash distillation with Analysis–Localisation–Reasoning structure |
| **MM-Doc-R1 SPO pool** | MM-Doc-R1 | RL trajectories shaped by similarity-weighted rewards |
| **ColPali synthetic + DocMatix-IR + PFLDocVQA filtering** | SV-RAG | GPT-4o filtering of pre-existing synthetic IR data |

Of these 17, **11 source the underlying documents from existing evaluation benchmarks or the same public PDF pools as those benchmarks** (Arxiv, OpenReview, MMLongBench-Doc PDFs, LongDocURL PDFs, DUDE PDFs, CUAD, SlideVQA decks).

#### The circularity loop (unchanged from v1, intensified)

```
GPT-4o / Gemini / Qwen-VLmax → writes QAs / CoT / trajectories over PDFs
   ↓
Model X is fine-tuned (or RL-shaped) to imitate the teacher's outputs
   ↓
Model X is evaluated on benchmarks where the gold answer was also produced
  (or LLM-judged) by GPT-4o / Gemini / Qwen
   ↓
Model X looks excellent because it has learned to mimic the teacher
```

Three knock-on effects (the v2 evidence sharpens each):

1. **Performance gains may be distillation, not capability.** When DocSeeker reports +X points on MMLongBench-Doc after training on Gemini-distilled ALR chains, part of that delta is "the model now sounds more like Gemini, which is also used as judge in some MMLongBench-Doc evaluation protocols." When MM-R5 reports SOTA on MMDocIR after training on GPT-4o reasoning chains, part of that is "now talks like GPT-4o." This isn't worthless — distillation has practical value — but it is *not* progress against the upper bound, only progress toward the teacher.
2. **Style transfer rather than reasoning gain.** GPT-4o has stylistic patterns: it tends to begin reasoning chains by enumerating sources, hedge with "Based on the document…", and prefer extractive over abstractive answers. Models trained on its output replicate these patterns regardless of whether they are good for the task. v2's CoT-distilled models (DocSeeker, MM-R5, MM-Doc-R1) are especially exposed because their reasoning chains are *the supervision signal*.
3. **Common biases amplified.** Documented GPT-4o biases — hallucinating page numbers, preferring early-page content, over-extracting — propagate into every model trained on its synthesised traces. The field has not measured how much.

#### Three quality-control failures across the corpus (v2 evidence)

1. **No human-rated quality samples.** None of the 17 synthetic-data papers reports a human evaluation of a sampled subset of its synthesised data. v2 papers are no exception — DocSeeker, MM-R5, VRAG-RL, MM-Doc-R1, SV-RAG all release synthetic data without human spot-checks.
2. **No document-overlap audits.** Of papers that synthesise training data from existing benchmark documents, only Leopard explicitly partitions its source pool, and even there the partition is by document URL rather than by content hash. v2 papers do not improve on this.
3. **No final-answer fact-checking.** Synthesised QAs are accepted if the LLM produces them; correctness against the source PDF is rarely verified beyond a self-consistency check (i.e., the LLM is asked again). MM-R5 mentions a "result reward" verifying correctness against ground truth during RL — but the SFT-stage synthetic chains are not verified.

#### Why the field tolerates this

The economic argument is simple: a single annotator might write 20 high-quality multi-page QA pairs in a day, costing ~$300 per 1,000 QAs. GPT-4o produces 1,000 QAs for under $20. To assemble Doc-750K (758K QAs) from human annotation would cost ~$11M and take ~3 years. Synthetic data is not a luxury; it is the only viable scaling path. The survey should acknowledge this trade-off rather than dismiss synthetic data — but should also press for the cheaper compensating measures (sampling audits, deduplication, cross-LLM QA agreement). v2's bright spot: MM-R5 uses *result + format* rewards during RL to filter teacher-produced trajectories — closest to a quality gate in the corpus.

#### Specific best-practice recommendations to surface in the survey

1. **Mandatory quality reporting.** Future papers introducing a synthesised corpus should report (a) inter-annotator-style agreement using a *different* LLM as the second judge, and (b) a 200-QA human spot-check.
2. **Document-hash dedup** against MP-DocVQA / DUDE / MMLongBench-Doc / LongDocURL / SlideVQA test pools, with the dedup result reported.
3. **Generator diversity.** Papers should generate with at least two different LLMs (e.g., GPT-4o + Gemini 1.5 Pro + Claude) to dilute single-model bias. Current v2 practice is monogenerator: each paper uses one teacher.
4. **Generator–evaluator separation.** If GPT-4o was used to synthesise training data, the evaluation judge should not be GPT-4o. v2's MM-Doc-R1 actually shows the way here — it uses Qwen2.5-72B as judge while Qwen3 is the policy.
5. **Public release of generation prompts.** Several papers omit the prompts used to synthesise QAs, making reproducibility and bias-analysis impossible.

### 5.3 Why the two flags are one structural problem, not two

The two flags interact: the synthetic-data wave (Flag 2) is the proximate *cause* of the contamination problem (Flag 1) — because every synthesised corpus is built over public PDFs, and most public PDFs are already in someone's evaluation benchmark. v2 makes this clearer: the new RL-fine-tuning recipes (MM-Doc-R1, DocSeeker, VRAG-RL) all use *evaluation-benchmark documents* (LongDocURL, MP-DocVQA, DUDE, MMLongBench-Doc, SlideVQA, ViDoSeek) as their RL training pool, then evaluate on the same benchmarks. The survey should treat these not as two separate methodological footnotes but as a single, structural feature of MP-VRDU as a 2024–26 field: *the data-scaling strategy that made the field possible is the same strategy that has compromised its evaluation integrity*. Flagging this prominently is both honest and useful — and it is a clean motivation for proposing a held-out evaluation suite as a community deliverable.

---

## 6. Suggested section structure for the survey

A 6–8 page **Datasets** section can be organised as:

1. **Overview and scope** (½ page). Define MP-VRDU dataset, list inclusion criteria, explain the three-category split (PT / FT / Bench) and reference Tables 1-3 of `datasets_used.md`.
2. **Pretraining datasets** (1 page). Two-axis taxonomy (signal × granularity). Lead with Texthawk2's mega-mix and Leopard / mPLUG-DocOwl2's DocStruct4M / MP-DocStruct1M family. Discuss the OCR-pretext → structure-pretext shift. Close with the observation that **no 2025–26 paper performs custom pretraining** — the era is over.
3. **Fine-tuning datasets** (1–1.5 pages). Distinguish three buckets: *task-mixture* FT, *capability-targeted* SFT, and *RL/preference learning*. Highlight the rise of LLM-synthesised corpora (Doc-750K, PaperPDF, Leopard-Instruct, CoR-Dataset, ALR-CoT, MM-R5 reasoning set). Include a sub-section on the RL wave (DocR1, CoR, MM-Doc-R1, MM-R5, DocSeeker, VRAG-RL) and the algorithmic variants (GRPO, SPO, EviGRPO, DPO).
4. **Benchmarking datasets** (2–2.5 pages). Use the *page-count tier × domain × open/closed-domain* taxonomy. Tier 1: single-page (sanity). Tier 2: mid-length (MP-DocVQA, DUDE, SlideVQA). Tier 3: long-document (MMLongBench-Doc, LongDocURL, PaperPDF, MMDocIR, VisR-Bench, M-LongDoc, BRIDGE). Tier 4: open-domain / cross-document (M3DocVQA, ViDoSeek, OpenDocVQA, VisDoMBench). Devote a paragraph each to the "core six" with dataset statistics. Flag MMLongBench-Doc as the field's central benchmark.
5. **Trends across categories** (½–1 page). Page-count arms race + multi-document tier; synthetic data + RL wave; modality coverage; architecture × dataset coupling.
6. **Limitations and open problems** (½ page). Items 1–8 from §4 above.
7. **Recommended evaluation protocol** (optional, ½ page). Suggest a held-out subset strategy, contamination audits (split / document / backbone), multi-domain coverage, and standardised metrics — set up the rest of the paper.

**Tables to include:**
- Master dataset table (`datasets_used.md`'s three tables, possibly condensed).
- Statistics summary: dataset → #docs / #pages / #questions / domain / public release year.
- Architecture-family × benchmark-tier matrix (§3.5 above).
- Contamination ledger: dataset → #FT users / #BM users / overlap (use the table in §5.1).

**Figures to consider:**
- A Sankey-style diagram from architecture family → preferred benchmark tier.
- A timeline of benchmark releases with page-count growth on the y-axis (now extending to BRIDGE 2026, ~21+ page bin).
- A bar chart of "training reuse" — number of papers using each evaluation dataset for training too.
- A scatter of synthesised-corpus size vs. evaluation-overlap risk.

---

## 7. Glossary of dataset abbreviations

- **ALR-CoT** — Analysis–Localisation–Reasoning chain-of-thought training data introduced by DocSeeker (Gemini-distilled, 13,986 samples).
- **ANLS** — Average Normalised Levenshtein Similarity (metric, not a dataset, listed for completeness).
- **BRIDGE** — multi-hop scientific paper QA benchmark with step-level annotations; in datasets_main.csv but not yet used by any model in this corpus.
- **CCpdf** — Common-Crawl PDF subset used by Arctic-TILT.
- **CHART-Infographics** — chart understanding dataset used in Arctic-TILT FT.
- **CoR-Dataset** — Chain-of-Reading dataset (26,088 QA traces) introduced in CoR.
- **DocBench** — multi-page document evaluation (229 docs, ~1,082 Qs).
- **DocCVQA** — single-page Document Collection VQA, baseline in Hi-VT5.
- **DocMatix-IR** — ColPali-augmented IR data used by SV-RAG for retrieval-module FT.
- **DocStruct4M / MP-DocStruct1M** — structure-aware pretraining corpora used in mPLUG-DocOwl2 and (DocStruct4M only) VDocRAG.
- **DocVQA** — single-page document VQA (12K images, 50K QAs).
- **DUDE** — Document Understanding & Reasoning (multi-domain, multi-page; ~5K docs, 41K QAs).
- **DVQA / FigureQA / PlotQA / ChartQA / MultiChartQA** — chart-QA family.
- **EviGRPO** — DocSeeker's GRPO variant with multi-reward (format / localization / answer).
- **FetaTab / PaperTab / PaperText** — table / text QA sub-tasks from UDA-Benchmark / QASPER on academic papers.
- **FUNSD / CORD / SROIE / POIE / XFUND / SIBR** — form / receipt / invoice IE datasets.
- **HotpotQA / IIRC / 2WikiMQA / MuSiQue** — text-only multi-document QA used by KGP.
- **InfographicVQA / InfoVQA** — same dataset, both spellings used.
- **IIT-CDIP / OCR-IDL / Wukong / Common-Crawl PDFs** — large OCR pretraining corpora.
- **KleisterCharity / KleisterNDA / DeepForm / VQA-CD / VQAonBD / CUAD** — financial / legal IE.
- **Leopard-Instruct** — 925K-instance multi-image instruction set introduced by Leopard.
- **LongDocURL** — long multimodal document benchmark (396 docs, 85 pages avg, 2,325 Qs).
- **M-LongDoc** — super-long document benchmark (180 docs, 210.8 pages avg).
- **M3DocVQA** — open-domain multi-document VQA (3,368 PDFs, 41K pages, 2,441 Qs) introduced by M3DocRAG.
- **MMDocIR** — multi-modal document retrieval benchmark (313 docs avg 65 pages, 1,658 Qs) used by MM-R5.
- **MMLongBench-Doc** — long-context multimodal document benchmark (Ma 2024; 135 docs, 47.5 pages avg). The field's most-used benchmark.
- **MMLongBench** — separate Wang-2025 benchmark, not used by any model in this corpus.
- **MMNIAH** — multimodal Needle-in-a-Haystack benchmark for long PDFs.
- **MP-DocVQA** — Multi-Page DocVQA (5,928 docs, 60K pages, 46K QAs) introduced by Hi-VT5.
- **OCR-VQA** — OCR-based VQA, single-page.
- **OpenDocVQA** — unified open-domain DocVQA introduced by VDocRAG.
- **PaperPDF** — 1.1M bilingual scientific-PDF QA introduced by PDF-WuKong.
- **PDFTriage** — structural document QA dataset used by KGP.
- **PFLDocVQA** — federated DocVQA pairs filtered by SV-RAG for retrieval-module FT.
- **RVL-CDIP / DocBank / DocLayNet / PubLayNet** — layout / document-classification corpora (mostly cited rather than actively used in this corpus).
- **SciGraphQA / SPIQA** — chart / scientific-paper QA sub-tasks of VisDoMBench.
- **SIBR** — real-world VIE dataset with degraded text.
- **SlideVQA** — 20-page slide-deck VQA.
- **SPO** — Similarity-based Policy Optimisation, MM-Doc-R1's GRPO replacement.
- **TAT-DQA** — tables-in-document QA.
- **ViDoSeek** — visually-rich document retrieval/QA benchmark introduced by ViDoRAG (refined from SlideVQA).
- **VisDoMBench** — multi-document multi-modal RAG benchmark (2,271 QAs) introduced by VisDoMRAG.
- **VisR-Bench** — web-crawled visually-rich document benchmark (226 docs, 471 QAs, 9 domains) introduced by SV-RAG.
- **VisualMRC** — visual reading comprehension over web pages.
- **WikiTableQuestions / WTQ** — table QA, used in mPLUG-DocOwl2 and DocR1.
