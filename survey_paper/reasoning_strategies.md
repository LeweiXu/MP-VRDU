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
