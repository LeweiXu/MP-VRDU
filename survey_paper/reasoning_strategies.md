# Reasoning Strategies — Classification of 11 DocVQA Models

This document classifies the reasoning strategies of 11 multi-modal document VQA / DocQA models for the survey paper summary table. The classification is based strictly on what each paper describes about how the model reasons, plans, decomposes, retrieves iteratively, uses multiple agents, and uses tools.

## Acronym Vocabulary

**Reasoning trace shape**
- **CoT** — Chain-of-Thought
- **ReAct** — Reasoning + Acting (tool-augmented CoT with Thought / Action / Observation interleaving)
- **ToT** — Tree-of-Thought
- **PoT** — Program-of-Thought

**Planning and decomposition**
- **PE** — Plan-Execute (plan first, then execute)
- **QD** — Question Decomposition (split into sub-queries)
- **HD** — Hierarchical Decomposition (coarse-to-fine)

**Self-improvement**
- **SR** — Self-Refine (iterate on own output)
- **SC** — Self-Consistency (sample-and-vote)
- **SV** — Self-Verify (check own answer before returning)
- **SA** — Sampling-Adjudication (sample N, judge picks)

**Iteration over evidence**
- **IR** — Iterative Retrieval (multi-pass retrieval)
- **QR** — Query Reformulation (rewrite query between passes)
- **Nav** — Navigation (controller-driven page traversal)

**Agent topology**
- **1A** — Single-Agent
- **MA** — Multi-Agent
- **MA-R** — Multi-Agent, Role-specialised
- **MA-D** — Multi-Agent, Debate / Adjudication
- **MA-H** — Multi-Agent, Hierarchical (controller + workers)

**Important distinctions**
- ReAct = the agent's own CoT decides the next tool call dynamically. A fixed pipeline of tool calls is **not** ReAct even if tools are used.
- SC vs SA: SC = sample-and-vote (consistency check); SA = sample-and-judge (separate adjudicator agent picks).
- IR vs QR: IR = re-retrieve more pages; QR = rewrite the query text. Many models do both.
- 1A vs MA: count distinct prompted roles, not distinct backbone calls. A single backbone called multiple times in different roles = MA-R.

---

## Model-by-Model Classification

### Doc-V*

- **Tags:** `ReAct, HD, Nav, IR, QR, 1A`
- **Distinct agents:** 1 (a single OCR-free agent built on Qwen-2.5-VL-7B; no other prompted roles).
- **Tools:** `retrieval_page` (semantic search via ColQwen) and `fetch_page` (direct page index fetch). Tool calls are **dynamically selected** by the agent at each step (think → action).
- **Justification:** The paper explicitly states it follows the ReAct "think–acting" protocol with `<think>`/`<action>` tokens, and the agent decides between retrieval and fetch actions based on its evolving working memory. It begins coarse-to-fine from a global thumbnail overview (HD) before iteratively navigating to high-resolution pages, refining queries between rounds.

### DocAgent

- **Tags:** `ReAct, HD, Nav, IR, MA-R, MA-D`
- **Distinct agents:** 3 — Actor, Reviewer, Reflection module (the Reflection module updates a memory bank when actor and reviewer disagree).
- **Tools:** 5 tools — `search`, `get_section_content`, `get_image`, `get_page_images`, `get_table_image`. Tool calls are **dynamically selected** by both actor and reviewer based on their reasoning over a tree-structured XML outline.
- **Justification:** The paper explicitly cites ReAct as its agent paradigm; the actor reasons over the XML outline (HD) and dynamically issues tool calls until terminating with an answer. The reviewer cross-verifies using complementary modalities (MA-D-like adjudication that can override the actor), and a memory module accumulates reflections across tasks.

### DocDancer

- **Tags:** `ReAct, IR, QR, 1A`
- **Distinct agents:** 1 (a single end-to-end trained agent based on Qwen3-4B / Qwen3-30B-A3B-Thinking).
- **Tools:** 2 tools — `Search` (keyword full-text search returning section IDs and snippets) and `Read` (fine-grained, goal-conditioned section reading using an auxiliary multimodal summarizer). Tool calls are **dynamically selected** by the agent.
- **Justification:** The paper states "We adopt the vanilla ReAct as the agent's framework," with interleaved thought/action/observation triplets. It deliberately uses a single-agent design (citing The Bitter Lesson) over multi-agent pipelines, iteratively refining queries via repeated `Search`/`Read` calls.

### Doc-React

- **Tags:** `ReAct, QD, IR, QR, 1A`
- **Distinct agents:** 1 (a single MLLM, GPT-4o, acting as both judge and generator).
- **Tools:** Multimodal retriever (ColPali / VisRAG) used as a search action. Tool calls are **dynamically selected** by the LLM at each iteration.
- **Justification:** The paper explicitly inherits the ReAct framework (Thought / Action / Observation traces shown in case studies). The LLM acts as judge-and-generator, decomposing the residual information gap into sub-queries (QD) and reformulating search queries each iteration to maximise InfoNCE-guided mutual information gain.

### KGP (Knowledge Graph Prompting)

- **Tags:** `IR, Nav, 1A`
- **Distinct agents:** 1 (an LLM-based KG traversal agent — a fine-tuned T5 / LLaMA / MDR encoder).
- **Tools:** Knowledge-graph traversal over a pre-built KG of passage / page / table nodes; TF-IDF for seed nodes; the LLM ranks neighbours to visit. The traversal sequence is **dynamically selected** by the LLM agent.
- **Justification:** The paper builds a passage-level KG and uses a single LLM-based traversal agent that, conditioned on already-visited nodes, predicts the next supporting fact and visits the most-matching neighbour. There is no Thought/Action/Observation ReAct protocol nor multiple prompted roles. Iteration is driven by graph navigation rather than retrieval-query rewriting.

### M2RAG

- **Tags:** `MA-R`
- **Distinct agents:** 3 — Text Filter (Qwen2.5-7B-Instruct), Visual Extractor (fine-tuned Qwen2.5-VL-7B), Modal Fuser (fine-tuned Qwen2.5-VL-7B).
- **Tools:** BM25 text retrieval and VisRAG-Ret visual retrieval, run **once in a fixed dual-tower pipeline** (not dynamically chosen by an agent).
- **Justification:** Pipeline is strictly fixed: dual-tower retrieval → text filter produces (a_T, E_T) → visual extractor produces (a_V, E_V) → modal fuser combines them into the final answer. Each agent runs once; there is no iterative retrieval, no query reformulation, no self-reflection loop, and no ReAct. The only branching is whether the text or visual answer is "no information" — a deterministic switch, not an LLM-controlled action selection.

### MACT

- **Tags:** `CoT, PE, SC, SV, MA-R, MA-D`
- **Distinct agents:** 4 — Planning Agent, Execution Agent, Judgment Agent, Answer Agent.
- **Tools:** A tool library used by the Execution Agent (the paper details this in its appendix). Tools are invoked step-by-step within the execution agent's plan; the planning agent selects high-level steps that the execution agent then realises with tools — partially dynamic within a plan-execute scaffold (not full ReAct because planning and execution are separated agents).
- **Justification:** MACT explicitly decomposes into Plan → Execute → Judge → Answer (PE). The planning agent generates _N_p relevant plans (parallel sampling). The execution agent generates _N_e candidate executions per step, scored by a reward model and best-of-N selected (SC-style sampling-and-vote at the step level). The judgment agent verifies plan and execution and routes back for correction (SV). The judgment agent is independent of the executor (MA-D-style separation of judge and corrector), with up to N_c=3 correction loops.

### MDocAgent

- **Tags:** `MA-R`
- **Distinct agents:** 5 — General Agent, Critical Agent, Text Agent, Image Agent, Summarizing Agent.
- **Tools:** Two parallel RAG retrievers (ColBERTv2 for text, ColPali for images), invoked once per query as a **fixed pipeline**. No agent dynamically selects tools.
- **Justification:** The paper specifies a strict five-stage pipeline (preprocess → retrieve → general+critical → specialised text/image → summarise). Each agent runs once, no iteration, no re-retrieval, no query reformulation, no self-verification. The agents are role-specialised (MA-R) but there is no debate/adjudication and no controller dynamically choosing actions, hence not ReAct and not MA-D.

### SimpleDoc

- **Tags:** `IR, QR, SR, SV, 1A`
- **Distinct agents:** 1 reasoning agent (a VLM, e.g., Qwen2.5-VL-32B-Instruct). The paper also uses an LLM for summary-based re-ranking during retrieval but this is part of the retriever, not an additional reasoning role.
- **Tools:** Dual-cue retriever = embedding similarity (ColQwen / ColPali) + LLM-based summary re-ranking. The reasoner agent calls retrieval iteratively when it emits a "Query Update" response. Tool invocation is **dynamically driven** by the reasoner's three-way self-judgment (Answer / Not Answerable / Query Update).
- **Justification:** The paper deliberately argues against multi-agent designs in favour of a single VLM reasoner that iteratively (i) reads retrieved pages, (ii) decides if evidence is sufficient (SV-style), and (iii) emits a refined follow-up query if not (QR + IR + SR via working-memory updates). Although it iterates with retrieval, it does not adopt the ReAct Thought/Action/Observation token protocol; the loop is structured around the three response types.

### ViDoRAG

- **Tags:** `ReAct, HD, IR, SR, SV, MA-R, MA-D`
- **Distinct agents:** 3 — Seeker Agent, Inspector Agent, Answer Agent.
- **Tools:** Multi-modal hybrid retrieval (visual + textual pipelines fused by a GMM-based dynamic top-K). The seeker dynamically selects images each round (action space = image selection); the inspector reviews and either drafts an answer or returns reflection feedback. **Dynamically selected** within an iterative loop.
- **Justification:** The paper explicitly states the seeker uses an "improved ReAct" paradigm with memory updates per step. The seeker (coarse) selects relevant images from thumbnails; the inspector (fine) reviews at high resolution and provides reflection feedback (MA-D adjudication / SR / SV) until evidence is sufficient; the answer agent then performs a consistency check on the inspector's draft answer to produce the final answer (SV). The seeker–inspector loop with reflection is iterative retrieval (IR) and the coarse-to-fine seeker / inspector split is HD.

### DocLens (gold-standard reference, polished)

- **Tags:** `CoT, SA, MA-R`
- **Distinct agents:** 4 — Page Navigator, Element Localizer, Answer Sampler, Adjudicator.
- **Tools:** OCR, layout detection, cropping. Tools are invoked in a **fixed pipeline order** (not action-selected by a controller).
- **Justification:** The Reasoning Module performs CoT with self-consistency-style sampling at non-zero temperature, then the Adjudicator (a separate role) judges among the sampled candidates (SA, not SC, because a distinct adjudicator picks rather than vote). Not ReAct since the pipeline of tool calls is fixed and tool calls aren't dynamically action-selected by the controller.

---

## Summary Table

| Model | Reasoning Strategy | # Agents | Tools |
|---|---|---|---|
| Doc-V* | ReAct, HD, Nav, IR, QR, 1A | 1 (Doc-V* agent) | retrieval_page (ColQwen), fetch_page — dynamic |
| DocAgent | ReAct, HD, Nav, IR, MA-R, MA-D | 3 (Actor, Reviewer, Reflection) | search, get_section_content, get_image, get_page_images, get_table_image — dynamic |
| DocDancer | ReAct, IR, QR, 1A | 1 (DocDancer agent) | Search, Read — dynamic |
| Doc-React | ReAct, QD, IR, QR, 1A | 1 (LLM as judge+generator) | Multimodal retriever (ColPali / VisRAG) — dynamic |
| KGP | IR, Nav, 1A | 1 (LLM-based KG traversal agent) | TF-IDF seeds + LLM-guided KG traversal — dynamic |
| M2RAG | MA-R | 3 (Text Filter, Visual Extractor, Modal Fuser) | BM25, VisRAG-Ret — fixed pipeline |
| MACT | CoT, PE, SC, SV, MA-R, MA-D | 4 (Planning, Execution, Judgment, Answer) | Tool library used by Execution agent — partially dynamic within plan |
| MDocAgent | MA-R | 5 (General, Critical, Text, Image, Summarizing) | ColBERTv2 + ColPali — fixed pipeline |
| SimpleDoc | IR, QR, SR, SV, 1A | 1 (reasoner agent) | Dual-cue retriever (ColQwen + summary re-rank) — dynamic |
| ViDoRAG | ReAct, HD, IR, SR, SV, MA-R, MA-D | 3 (Seeker, Inspector, Answer) | GMM-based hybrid retrieval (visual + textual) — dynamic |
| DocLens | CoT, SA, MA-R | 4 (Page Navigator, Element Localizer, Answer Sampler, Adjudicator) | OCR, layout detection, cropping — fixed pipeline |

---

## Round 2: Retriever-Generator (RAG) Family

These models perform retrieval + generation, typically in a single-pass pipeline. Most have minimal or no inference-time reasoning patterns. Tags are deliberately conservative.

### AVIR

- **Tags:** `HD, 1A`
- **Distinct agents:** 1 (frozen Qwen2.5-VL-3B-AWQ generator; the retriever and adaptive page selector are non-agentic modules).
- **Tools:** Pix2Struct-based lightweight page retriever + adaptive page selector (clustering / threshold). Fixed pipeline; no LLM-driven action selection.
- **Justification:** Score → cluster/threshold → Top-K → frozen LVLM answer. The clustering vs. threshold branch is a deterministic data-driven switch, not LLM-controlled. No CoT prompting described. The coarse-then-fine score-clustering selector is mildly hierarchical (HD), but there is no iterative refinement.

### CREAM

- **Tags:** `HD, 1A`
- **Distinct agents:** 1 (LLaMA-Adapter-V2-based MLLM as the answer generator; RankVicuna LLM is used as a re-ranker tool inside the retriever).
- **Tools:** Coarse-to-fine retrieval pipeline — bge-large embedding retrieval, then LLM-based grouped re-ranking (RankVicuna), then attention-pooled multi-page vision encoder. Fixed pipeline.
- **Justification:** Coarse-then-fine retrieval with LLM re-ranking is hierarchical decomposition (HD), but generation is a single forward pass through a fine-tuned MLLM with no CoT prompting, no iteration, no self-verification.

### DFVC

- **Tags:** `—`
- **Distinct agents:** n/a (no LLM reasoning agent; this paper proposes a retrieval-only adapter, with downstream answering left to off-the-shelf retrievers like VisRAG/ColQwen).
- **Tools:** Lightweight MLP adapter that fuses neighbouring page embeddings via gating + residual connection. Pure feed-forward.
- **Justification:** This work modifies only the retrieval embedding by fusing context from neighbouring pages. There is no agentic component, no CoT, no iterative reasoning. It is a parameter-efficient retrieval enhancement.

### M3DocRAG

- **Tags:** `1A`
- **Distinct agents:** 1 (Qwen2-VL-7B as MLLM answerer).
- **Tools:** ColPali multi-modal retriever with MaxSim scoring + IVF approximate index. Single retrieval call, fixed pipeline.
- **Justification:** Three-stage embed → retrieve top-K → answer with MLLM. Single-pass generation, no CoT prompt, no iteration, no query reformulation. Pure RAG baseline.

### MHier-RAG

- **Tags:** `CoT, HD, 1A`
- **Distinct agents:** 1 (an LLM/LVLM such as Qwen-turbo/GPT-4o for answer generation; an LLM also re-ranks parent pages, but acts as a tool in the retriever, not a separate role).
- **Tools:** Hierarchical index = flattened in-page chunks + topological cross-page (GMM-clustered) summary tree. Multi-granularity retriever fuses page-level parent-page retrieval and document-level summary retrieval. LLM-based re-ranking. Fixed pipeline.
- **Justification:** The paper explicitly uses "a Chain-of-Thought (CoT) prompting strategy with structured output" for answer reasoning (CoT). The two-level (page-level + document-level / cluster summary) index is hierarchical decomposition (HD). Retrieval and generation each run once; no iteration, no self-verification, no multi-agent debate.

### MLDocRAG

- **Tags:** `HD, 1A`
- **Distinct agents:** 1 (Qwen2.5-VL-32B as the LVLM generator; an LVLM is also used offline to generate Doc2Query queries during graph construction, but this is a one-time index-building step rather than an inference-time agent).
- **Tools:** Multimodal Chunk-Query Graph (MCQG) built via MDoc2Query expansion; KNN over query nodes + multi-hop graph traversal + chunk ranking. Fixed retrieval pipeline followed by single-pass generation.
- **Justification:** Multi-hop graph traversal over the chunk-query graph imposes a hierarchical / structured retrieval (HD) but is purely an index-side mechanism, not an LLM-driven action selection. The generator runs once with no CoT prompt explicitly described and no iteration.

### MoLoRAG

- **Tags:** `IR, Nav, 1A`
- **Distinct agents:** 1 (a small VLM, e.g., Qwen2.5-VL-3B, acts as the retrieval engine performing graph traversal; downstream LVLM answers from top-K).
- **Tools:** Page graph built from ColPali embeddings. The retrieval-engine VLM scores logical relevance per visited page and decides which neighbours to traverse next; multi-hop BFS-like traversal with bounded width and hop count.
- **Justification:** The retrieval engine performs **iterative, controller-driven graph traversal** (Nav, IR) with up to `n_hop` multi-hop expansion, using VLM-assigned logical relevance scores at each step. This is the only RAG paper in this round whose retrieval is genuinely agent-controlled rather than a fixed pipeline. Generation is then single-pass, no CoT/SV described.

### MultiDocFusion

- **Tags:** `HD, 1A`
- **Distinct agents:** 1 (an LLM, e.g., Mistral-8B-based DSHP-LLM, is used at indexing time to construct the section hierarchy; downstream answer is generated by a separate LLM after BGE-based retrieval).
- **Tools:** DP (vision layout) + OCR + DSHP-LLM (LoRA-tuned) building a hierarchical document tree + DFS-based hierarchical chunking + BGE/E5/BM25 retrieval. Fixed pipeline.
- **Justification:** Hierarchical document-tree construction and hierarchical chunking is HD, but it is a pre-processing/indexing pipeline. Retrieval and generation each run once with no inference-time reasoning loop, no CoT, no iteration.

### PDF-WuKong

- **Tags:** `1A`
- **Distinct agents:** 1 (an MLLM based on IXC2-VL-4KHD).
- **Tools:** End-to-end sparse sampler — text encoder (BGE-M3) + shared image encoder pre-encode all paragraphs/diagrams; at query time the model selects top-5 relevant chunks via similarity and feeds them with the query into the LLM. Single sampling step, integrated end-to-end.
- **Justification:** Sparse sampler + LLM trained jointly. One-shot retrieval-then-generate. No CoT prompt, no iteration, no multi-agent collaboration. Pure feed-forward design.

### RAG-DocVQA (López et al.)

- **Tags:** `1A`
- **Distinct agents:** 1 (one of VT5, Qwen2.5-VL-7B, or Pix2Struct as the answer generator).
- **Tools:** Bi-encoder (bge-en-small) chunk retrieval + cross-encoder reranker (textual variant) or ColBERT-style late-interaction visual patch retrieval (visual variant). Fixed three-stage pipeline (index → retrieve+rerank → generate).
- **Justification:** Standard textual/visual RAG pipeline; the cross-encoder reranker improves precision but does not constitute reasoning. No CoT, no iteration, no agent control flow.

### Self-Attention Scoring (Kang et al.)

- **Tags:** `1A`
- **Distinct agents:** 1 (Pix2Struct encoder-decoder; the self-attention scoring head is a small 1-layer 16-head module on top of the frozen encoder).
- **Tools:** Self-attention scoring head produces per-page relevance probabilities; top-1 page is selected; the same Pix2Struct model then generates the answer. Fixed pipeline.
- **Justification:** A single OCR-free model with an auxiliary scoring head. No reasoning trace, no CoT, no iteration. Closer in spirit to Hi-VT5/RM-T5 but with retrieval-style top-1 page selection.

### VDocRAG

- **Tags:** `1A`
- **Distinct agents:** 1 (VDocGenerator, a Phi3V-based LVLM; VDocRetriever shares architecture but is a retrieval module, not a reasoning agent).
- **Tools:** Dual-encoder visual retrieval (LVLM-based with EOS-token compression via RCR/RCG pre-training) + LVLM generator. Fixed retrieve-then-generate pipeline.
- **Justification:** The contribution is in retriever pre-training tasks (Representation Compression via Retrieval / Generation). At inference time it is a single-pass RAG. No CoT prompt, no iteration, no multi-agent.

---

## Round 2: Backbone-Centric MLLM Adaptation Family

These are mostly trained models with single-pass inference. Tags are minimal unless the paper explicitly describes an inference-time reasoning pattern.

### CoR (Chain-of-Reading, Qwen2.5-VL-CoR)

- **Tags:** `CoT, PE, HD, 1A`
- **Distinct agents:** 1 (an MLLM, Qwen2.5-VL fine-tuned on CoR traces).
- **Tools:** None at inference (end-to-end MLLM with `<think>`/`<answer>` traces). Mask-AR is a self-supervised pre-training objective, not an inference tool.
- **Justification:** CoR explicitly trains an MLLM to follow a four-stage Chain-of-Reading: Task Planning → Phased & Focused Search (coarse-to-fine) → Cross-modal Evidence Integration → Synthesized Reasoning & Verification, all emitted in a single `<think>` block. This is CoT with an explicit plan-then-execute structure (PE) and coarse-to-fine localization (HD), but it remains a single MLLM without external tools or multi-agent coordination.

### Docopilot

- **Tags:** `1A`
- **Distinct agents:** 1 (a native long-context MLLM trained from InternVL2 on Doc-750K).
- **Tools:** None — argued explicitly against RAG/tools in favour of native long-context processing.
- **Justification:** Trained model with single-pass inference. No reasoning trace, no CoT prompting, no iteration. The contribution is the dataset and training recipe, not an inference-time strategy.

### DocR1

- **Tags:** `CoT, HD, 1A`
- **Distinct agents:** 1 (Qwen2.5-VL-7B-Instruct trained with EviGRPO).
- **Tools:** None at inference. EviGRPO is an RL training framework with format/accuracy/evidence-page rewards.
- **Justification:** DocR1 emits structured `<think>`/`<evidence_page>`/`<answer>` outputs. The paper explicitly describes a "coarse-to-fine" reading strategy where the model first identifies relevant pages, then reasons over them — a CoT-with-evidence-localization pattern (HD), trained via GRPO. It is still a single MLLM inferring in one forward pass.

### DocSLM

- **Tags:** `SC, 1A`
- **Distinct agents:** 1 (a 2B SVLM with hierarchical multimodal compression).
- **Tools:** Hierarchical Multimodal Compressor (page-level token compression) + Streaming Abstention with an entropy-based uncertainty calibrator that aggregates per-segment predictions.
- **Justification:** The Streaming Abstention mechanism processes document segments sequentially, produces per-segment answers with uncertainty scores, and selects the lowest-uncertainty answer at the document level. This is an uncertainty-weighted vote across independent samples — a self-consistency-style aggregation (SC), albeit over document segments rather than reasoning paths. No CoT prompt, no iteration on a single segment.

### InstructDoc / InstructDr

- **Tags:** `—`
- **Distinct agents:** n/a (a trained instruction-tuned model with a Document-former bridging module).
- **Tools:** None at inference.
- **Justification:** Pure instruction-tuned MLLM trained on 30 VDU datasets with handcrafted instructions. No inference-time reasoning pattern is described.

### LayTokenLLM

- **Tags:** `—`
- **Distinct agents:** n/a (a trained LLM with single-token layout representation).
- **Tools:** None.
- **Justification:** Pure architectural / tokenisation contribution (layout-as-single-token, NTLP pre-training). No inference-time reasoning pattern.

### Leopard

- **Tags:** `—`
- **Distinct agents:** n/a (instruction-tuned MLLM for text-rich multi-image tasks).
- **Tools:** None.
- **Justification:** Trained model with adaptive high-resolution multi-image encoding. Single-pass inference with no described reasoning strategy.

### mPLUG-DocOwl2

- **Tags:** `—`
- **Distinct agents:** n/a (an MLLM with a High-resolution DocCompressor).
- **Tools:** None.
- **Justification:** Token-compression architecture trained in three stages (single-image pretraining → multi-image continue-pretraining → multi-task finetuning). Single-pass inference; no inference-time reasoning pattern.

### TextHawk2

- **Tags:** `—`
- **Distinct agents:** n/a (a bilingual LVLM with 16x token compression).
- **Tools:** None.
- **Justification:** Trained model focused on token compression and visual encoder reinforcement. No inference-time reasoning pattern described.

---

## Round 2: Page-to-Document (Hierarchical Document Transformer) Family

These end-to-end encoder-decoder models have no inference-time reasoning pattern. They process documents in a single forward pass with architectural mechanisms (memory, global tokens, hierarchical encoders) for cross-page integration.

### Arctic-TILT

- **Tags:** `—`
- **Distinct agents:** n/a (a sub-billion-parameter encoder-decoder built on TILT/T5).
- **Tools:** None.
- **Justification:** Single end-to-end encoder-decoder with tensor-product modality fusion and SLED-style sparse attention (max 400k tokens). Single-pass inference; no inference-time reasoning pattern.

### GRAM

- **Tags:** `—`
- **Distinct agents:** n/a (encoder-decoder with global-local interleaved layers, optionally followed by a C-Former compression transformer).
- **Tools:** None.
- **Justification:** Single forward pass with page-level and document-level (global) tokens interacting through interleaved encoder blocks; the C-Former is an architectural compressor, not a reasoning module.

### Hi-VT5

- **Tags:** `—`
- **Distinct agents:** n/a (T5-based hierarchical multimodal encoder-decoder).
- **Tools:** None.
- **Justification:** Each page is encoded independently into [PAGE] summary tokens; the decoder consumes the concatenation of all [PAGE] tokens. Single forward pass; the page-prediction head is a classification output, not a reasoning step.

### RM-T5 (Recurrent Memory Transformer)

- **Tags:** `—`
- **Distinct agents:** n/a (T5 encoder-decoder with recurrent memory cells across pages).
- **Tools:** None.
- **Justification:** Pages are processed sequentially with memory tokens carrying context across pages, then all memory cells are concatenated and fed to the decoder. Although the recurrence is iterative across pages, it is a fixed architectural unrolling, not an inference-time reasoning loop driven by the model's outputs.

---

## Combined Summary Table (All ~36 Models)

| Model | Family | Reasoning Strategy | # Agents | Tools |
|---|---|---|---|---|
| Doc-V* | Agentic | ReAct, HD, Nav, IR, QR, 1A | 1 | retrieval_page (ColQwen), fetch_page — dynamic |
| DocAgent | Agentic | ReAct, HD, Nav, IR, MA-R, MA-D | 3 | search, get_section_content, get_image, get_page_images, get_table_image — dynamic |
| DocDancer | Agentic | ReAct, IR, QR, 1A | 1 | Search, Read — dynamic |
| Doc-React | Agentic | ReAct, QD, IR, QR, 1A | 1 | Multimodal retriever (ColPali / VisRAG) — dynamic |
| KGP | Agentic | IR, Nav, 1A | 1 | TF-IDF seeds + LLM-guided KG traversal — dynamic |
| M2RAG | Agentic | MA-R | 3 | BM25, VisRAG-Ret — fixed pipeline |
| MACT | Agentic | CoT, PE, SC, SV, MA-R, MA-D | 4 | Tool library used by Execution agent — partially dynamic within plan |
| MDocAgent | Agentic | MA-R | 5 | ColBERTv2 + ColPali — fixed pipeline |
| SimpleDoc | Agentic | IR, QR, SR, SV, 1A | 1 | Dual-cue retriever (ColQwen + summary re-rank) — dynamic |
| ViDoRAG | Agentic | ReAct, HD, IR, SR, SV, MA-R, MA-D | 3 | GMM-based hybrid retrieval — dynamic |
| DocLens | Agentic | CoT, SA, MA-R | 4 | OCR, layout detection, cropping — fixed pipeline |
| AVIR | RAG | HD, 1A | 1 | Pix2Struct retriever + adaptive selector — fixed pipeline |
| CREAM | RAG | HD, 1A | 1 | bge-large + RankVicuna re-ranker + multi-page vision encoder — fixed pipeline |
| DFVC | RAG | — | n/a | MLP gating adapter over frozen VLM (retrieval only) |
| M3DocRAG | RAG | 1A | 1 | ColPali (MaxSim, IVF index) — single-pass retrieve-then-generate |
| MHier-RAG | RAG | CoT, HD, 1A | 1 | In-page + cross-page hierarchical index, LLM re-ranker — fixed pipeline |
| MLDocRAG | RAG | HD, 1A | 1 | MCQG graph (MDoc2Query) + KNN + multi-hop traversal — fixed retrieval |
| MoLoRAG | RAG | IR, Nav, 1A | 1 | ColPali page graph + VLM-driven multi-hop traversal — dynamic |
| MultiDocFusion | RAG | HD, 1A | 1 | DSHP-LLM hierarchical tree + DFS chunking + BGE retrieval — fixed pipeline |
| PDF-WuKong | RAG | 1A | 1 | End-to-end sparse sampler (BGE-M3 + image encoder) — single-pass |
| RAG-DocVQA | RAG | 1A | 1 | bi-encoder + cross-encoder reranker (text) or ColBERT-style late interaction (visual) — fixed pipeline |
| Self-Attn Scoring | RAG | 1A | 1 | Self-attention scoring head over Pix2Struct — fixed pipeline |
| VDocRAG | RAG | 1A | 1 | LVLM dual-encoder (RCR/RCG pre-training) — fixed pipeline |
| CoR | Backbone | CoT, PE, HD, 1A | 1 | None (end-to-end Qwen2.5-VL-CoR) |
| Docopilot | Backbone | 1A | 1 | None (native long-context MLLM) |
| DocR1 | Backbone | CoT, HD, 1A | 1 | None (Qwen2.5-VL-7B trained with EviGRPO) |
| DocSLM | Backbone | SC, 1A | 1 | Hierarchical compressor + streaming abstention with uncertainty calibrator |
| InstructDoc | Backbone | — | n/a | None |
| LayTokenLLM | Backbone | — | n/a | None |
| Leopard | Backbone | — | n/a | None |
| mPLUG-DocOwl2 | Backbone | — | n/a | None |
| TextHawk2 | Backbone | — | n/a | None |
| Arctic-TILT | Page-to-Doc | — | n/a | None (sparse attention encoder-decoder) |
| GRAM | Page-to-Doc | — | n/a | None (global-local encoder + optional C-Former) |
| Hi-VT5 | Page-to-Doc | — | n/a | None (hierarchical T5) |
| RM-T5 | Page-to-Doc | — | n/a | None (recurrent memory transformer) |
