# Decision Log: 10 Models Added to the Corpus

This file records the inclusion-justifications and non-KIE judgements for the 10 models added to the survey corpus across two audit rounds. Each entry covers: (i) why the paper was included against the inclusion criteria, (ii) its architecture categorisation, and (iii) other field choices that required interpretation.

Inclusion criteria (from §8 / Appendix D):
1. Proposes a model, framework, system, or pipeline rather than a benchmark or dataset alone.
2. Reports quantitative results on at least one multi-page benchmark with multi-page handling as an explicit design contribution.
Peer-reviewed venue publications are preferred; preprints retained only where the contribution introduces clear architectural, training, or orchestration novelty.

---

## Batch 1 (6 papers)

### DREAM — `zhang2025dream` — ACM MM 2025

- **Inclusion**: Multi-page DVQA is the explicit design goal; benchmarks on MP-DocVQA, DUDE, MMLongBench-Doc; full retriever-generator system with custom multi-page MLLM. Peer-reviewed at ACM MM 2025.
- **Architecture: Retriever-Generator Pipeline**. Single-pass framework: hierarchical retrieval (multimodal embedding + keyword scorer + LightGBM LTR) feeds into a custom multi-page MLLM with cross-page MoE attention. No iterative loop. Same shape as CREAM.
- **OCR: No** — base DREAM is OCR-free; the DREAM† variant uses MLLM-extracted (not OCR-extracted) text context.
- **Reasoning: HD** — hierarchical retrieval (low-level embedding similarity + high-level keyword scoring) is coarse-to-fine.
- **Search: J** — joins multimodal embedding similarity + keyword MLLM scoring via LightGBM fusion.
- **Training category: partial** — VE frozen, LLM frozen with LoRA, connector trained.

### MM-Doc-R1 — `lin2026mmdocr1` — preprint (arXiv 2604.13579)

- **Inclusion**: Long-doc VQA is the explicit design goal; +10.4 ACC on MMLongBench-Doc; novel SPO algorithm and three-agent workflow. Preprint, but novelty justifies inclusion.
- **Architecture: Agentic Pipeline**. Three agents (planner, seeker, answer) with iterative tool use and adaptive trajectory length per question. Trained end-to-end via multi-turn RL. Same shape as Doc-V*.
- **OCR: Yes\*** — uses Doc2X for OCR parsing of TOC and BM25 search, alongside visual VLM read tool. Same `Yes*` (Aux.) convention as M2RAG and DocAgent.
- **Reasoning: ReAct, HD** — planner decomposes the query (HD); seeker iterates ReAct loop.
- **Search: S, IR, QR, Nav** — BM25 (sparse), iterative refinement, sub-query refinement, page navigation via read tool.
- **Training category: partial** — VE (read-tool Qwen2.5-VL) frozen, LLM agents trained via SPO/GRPO.

### SV-RAG — `chen2025svrag` — ICLR 2025

- **Inclusion**: Long document understanding is the explicit goal; benchmarks on SlideVQA, MMLongBench-Doc, DUDE, DocVQA (SP+MP), VisR-Bench; full retriever-generator system with dual LoRA adapters on a shared MLLM backbone. Peer-reviewed at ICLR 2025.
- **Architecture: Retriever-Generator Pipeline**. Single-pass: Col-style late-interaction retrieval over MLLM hidden states → frozen MLLM generator. Dual LoRA adapters on shared backbone, but the control flow is a fixed retrieve-then-read pipeline.
- **OCR: No** — paper explicitly markets itself as OCR-free.
- **Reasoning: -** — no inference-time CoT or ReAct.
- **Search: D** — pure dense retrieval via Col-style late interaction.
- **Training category: partial** — VE frozen, LLM frozen with dual LoRA adapters.

### SLEUTH — `liu2025sleuth` — preprint (arXiv 2511.22850)

- **Inclusion**: Long-document understanding is the explicit goal; SOTA on MMLongBench, LongDocURL, PaperTab, FetaTab; novel context-engineering paradigm with four-agent framework. Preprint, but the context-engineering paradigm and SOTA evidence justify inclusion.
- **Architecture: Agentic Pipeline**. Four cooperating agents (Clue Discovery, Page Screening, Difficulty Assessment, Core Decision) plus ColPali retriever. Difficulty-Assessment Agent routes between Instruct and Thinking modes per question — adaptive trajectory.
- **OCR: No** — uses ColPali + VLMs throughout, no OCR step.
- **Reasoning: CoT, HD** — Clue Discovery does per-page CoT; the overall pipeline is coarse-to-fine (retriever → screening → core decision).
- **Search: D** — single ColPali pass; downstream agents filter retrieved pages but do not re-retrieve. Page Screening is filtering, not joining of retrieval signals, so `J` does not apply.
- **Training category: training-free** — no parameter updates.

### MM-R5 — `xu2026mmr5` — AAAI 2026

- **Inclusion**: Multi-page reasoning is explicit (MMDocIR, 65.1 pages avg); novel two-stage SFT+GRPO training with reasoning-chain output. Fits §5.3 Trained Retrieval Components alongside DFVC and MoLoRAG. Peer-reviewed at AAAI 2026.
- **Architecture: Retriever-Generator Pipeline**. A trained reranker that fits as a Trained Retrieval Component between an upstream ColQwen retriever and a downstream LVLM. Not standalone agentic.
- **OCR: No** — purely visual reranker over page images.
- **Reasoning: CoT** — explicit reasoning chains during reranking.
- **Search: J** — joins dense retrieval scores with reasoning-based reranking, similar to MHier-RAG's LLM re-ranking tagging.
- **Agentic: -** — a reranker, not a multi-agent system.
- **Training category: partial** — VE frozen, LLM trained via SFT+GRPO.

### VisDoMRAG — `suri2025visdom` — NAACL 2025

- **Inclusion**: Multi-document multi-page QA is explicit; introduces VisDoMBench (~128 pages avg per query); novel modality-fusion via reasoning-chain consistency. Peer-reviewed at NAACL 2025.
- **Architecture: Retriever-Generator Pipeline**. Two parallel single-pass RAG pipelines (text + visual) plus a fixed modality-fusion step. Trajectory determined at design time → fixed-trajectory pipeline rather than adaptive-trajectory agentic.
- **OCR: Yes\*** — textual pipeline uses PyTesseract OCR; visual pipeline uses ColPali. Same `Yes*` (Aux.) convention as M2RAG.
- **Reasoning: CoT** — three-step Evidence Curation + CoT + Answer Generation per pipeline. `HD` does not apply since there is no coarse-to-fine retrieval cascade.
- **Search: J** — joins text dense (BGE-1.5) + visual dense (ColPali) retrieval signals.
- **Agentic: 3A, SC** — text pipeline + visual pipeline + modality fusion, with self-consistency-style consistency check during fusion.
- **Training category: training-free** — no parameter updates.

---

## Batch 2 (4 papers)

### VRAG-RL — `wang2025vragrl` — preprint (arXiv 2505.22019)

- **Inclusion**: Substantive RL framework with novel visual-perception action space (crop/zoom regions) + retrieval-aware reward + GRPO training. Trains VLMs to interact with search engines via ReAct over visually rich documents. Same scope as Doc-V* and MM-Doc-R1. Preprint, but novelty justifies inclusion.
- **Architecture: Agentic Pipeline**. RL-trained ReAct agent that interleaves search and visual-perception (crop/zoom) actions over multiple turns. Variable trajectory length per question.
- **OCR: No** — purely visual RAG; OCR-based methods are explicit baselines they outperform.
- **Reasoning: ReAct, HD** — ReAct loop with coarse-to-fine perception via crop/zoom.
- **Search: D, IR, QR** — dense retrieval via search engine, iterative, query rewrite.
- **Training category: end-to-end** — paper says "We use full parameter fine-tuning … during SFT" then RL with GRPO; no frozen-backbone strategy mentioned. Same pattern as MACT.

### URaG — `shi2026urag` — AAAI 2026 Oral

- **Inclusion**: Peer-reviewed top-tier oral. Novel architecture — unifies retrieval and generation within a single MLLM by converting early Transformer layers into an evidence selector via a lightweight cross-modal retrieval module. Benchmarks on **all five** standard MP suites (MP-DocVQA, DUDE, SlideVQA, MMLongBench-Doc, LongDocURL). Multi-page handling is the explicit design goal.
- **Architecture: Backbone-Centric MLLM Adaptation**. The retrieval module is integrated **inside** the MLLM (lightweight projection between Transformer layers 6 and 7); no external retriever, no iterative loop, single forward pass with intermediate page filtering. Closest analogue is DocR1.
- **OCR: No** — OCR-free purely visual MLLM.
- **Reasoning: HD** — coarse-to-fine reasoning pattern (early layers retrieve, deep layers generate) is the paper's own framing. No CoT prompting or ReAct loop, so just `HD`.
- **Search: D** — dense retrieval via cross-modal late interaction.
- **Agentic: -** — single forward pass with internal retrieval.
- **Training category: partial** — VE frozen, LLM frozen with LoRA, retrieval module trained. Same pattern as CoR.

### DocSeeker — `yan2025docseeker` — preprint (no arXiv ID surfaced)

- **Inclusion**: Substantive contribution — novel Analysis-Localization-Reasoning paradigm + EviGRPO training + Evidence-Guided Resolution Allocation. Benchmarks on **all five** standard MP suites with stronger evaluation footprint than DocR1. Preprint, but the ALR + EGRA novelty justifies inclusion.
- **Architecture: Backbone-Centric MLLM Adaptation**. Single MLLM (Qwen-2.5-VL-7B) trained end-to-end via SFT + EviGRPO with structured `<think><answer>` outputs. No external retriever, no iterative loop. Direct successor to DocR1's EviGRPO design.
- **OCR: No** — paper explicitly markets the ALR paradigm as a "pure-visual solution".
- **Reasoning: CoT, HD** — Analysis-Localization-Reasoning workflow with explicit page identifiers is structured CoT and coarse-to-fine.
- **Search: -** — no retrieval, single forward pass.
- **Agentic: -** — single MLLM.
- **Training category: end-to-end** — two-stage SFT + EviGRPO on the full backbone; no mention of frozen visual encoder. Mirrors DocR1's classification.

### DMAP — `fu2026dmap` — preprint (arXiv 2601.18203)

- **Inclusion**: Substantive contribution — novel hierarchical Document MAP representation explicitly modelling cross-page section hierarchy, figure-text correspondence, and cross-reference relations. Directly addresses MP-distinctive structure (§3.3 Structure Modality in our paper). Preprint, but the cross-page structural contribution justifies inclusion.
- **Architecture: Agentic Pipeline**. Two-agent framework (SSUA + RRA) with a reflective loop that iteratively retrieves additional DMAP elements when the answer is judged incomplete. Adaptive trajectory length per question.
- **OCR: Yes\*** — uses pymupdf for parsing and pdffigure2 for figure extraction; auxiliary OCR-style pipeline alongside the visual ColPali path. Same `Yes*` (Aux.) convention as DocAgent and MDocAgent.
- **Reasoning: CoT, HD** — hierarchical structure traversal + reflection-based reasoning.
- **Search: J** — tri-path retrieval (structured semantic + textual via ColBERTv2 + visual via ColPali) joined into a fused result set.
- **Agentic: 2A, SR** — SSUA (structural understanding) + RRA (reflective reasoning); reflection loop is self-refine.
- **Modality: T, V, L** — text via ColBERTv2, visual via ColPali, layout via the DMAP hierarchy itself.
- **Training category: training-free** — uses pretrained models throughout.

---

## Excluded candidates (for reference)

These were evaluated alongside the 10 included papers and excluded against the inclusion criteria.

| Paper | Venue | Reason for exclusion |
|---|---|---|
| PREMIR | EMNLP 2025 | Focus is OOD/multilingual retriever; multi-page handling is incidental rather than the design contribution. |
| HM-RAG | ACM MM 2025 | Benchmarked only on ScienceQA + CrisisMMD; no multi-page document benchmark. |
| GME | CVPR 2025 | Encoder/embedder only — like ColPali variants we treat as building blocks. |
| CoRe-MMRAG | ACL 2025 | Benchmarked on InfoSeek + Encyclopedic-VQA (knowledge-based entity VQA); wrong domain. |
| MARL-RAGDoc | Nature Scientific Reports | Domain-specific (industrial/power-grid documents); no standard MP-VRDU benchmark. |
| RegionRAG | AAAI 2026 | Region-level retrieval is a granularity improvement; evaluation dominated by single-page benchmarks. |

## Adjacent benchmarks not yet added to the dataset catalogue

These surfaced during the audit and may be worth adding to `datasets_summary.csv` if any future corpus models adopt them:

- ArXivQA (~100K queries, 16.6K scientific papers)
- VisR-Bench (471 queries, 226 long docs) — introduced by SV-RAG
- UniDoc-Bench (1.6K queries, 70K images)
- BBox-DocVQA (32K queries, grounded spatial reasoning)
- ViDoRe (3.8K queries, 8.3K docs) — ColPali's eval suite
- MMDocIR (EMNLP 2025) — used by MM-R5
- VisDoMBench (NAACL 2025) — introduced by VisDoMRAG
- MMDocQA — used by DMAP

## Search gaps the audit did not close

- ICDAR 2025 main proceedings (only Workshop entries surfaced via Google indexing)
- WACV 2026, ECCV 2024 direct browse
- OpenReview ICLR 2026 / NeurIPS 2025 by-title search for "document"
- Author-page sweeps for X-PLUG, Adobe Research, Snowflake, the rubentito group at CVC-UAB
