# MP-VRDU Survey Paper Handoff Prompt

You are working in `/home/lingwei/CITS4010` on an honours survey paper titled *A Survey on the Evolving Frontier of Multi-Page Visually Rich Document Understanding: Architectures, Trends, Methods, and Challenges*. The paper is in `survey_paper/main.tex` and uses ACL article style under an 8-page main-text budget plus appendix.

## Before editing, read

- `AGENTS.md` if present, or any repository instructions supplied by the session.
- Current state of the paper at `survey_paper/main.tex`.
- Bibliography at `survey_paper/custom.bib`.
- CSV summaries under `survey_paper/` (architecture, OCR dependency, training, performance).
- Original model markdown notes under `models_main_md/`.
- General MLLM VRDU survey notes at `survey_paper/General_MLLM_VRDU_Survey.md` for style and to identify overlap to avoid.

The first response in a new session should not edit files. Instead, report:

1. The current structure and progress of the paper.
2. Which sections look polished, drafted, incomplete, or still bullet-point form.
3. Citation key inconsistencies, duplicated content, and overlap with the general VRDU survey.
4. A concise recommendation for the next section to work on.

Then ask what to do next. Do not assume. When given a task, read the relevant model papers and CSV summaries before drafting prose.

## Survey scope and structure

The paper is structured around a clear separation between **overview sections** (high-level patterns, taxonomies, trade-offs) and **methods sections** (model-specific techniques and existing works). Overview sections must not name individual techniques beyond the level of citation support for taxonomy-level claims. Methods sections are where model-specific mechanisms, training procedures, and inference-time strategies are described in detail.

```latex
\section{Introduction}                                    % bullet form, prose conversion pending

\section{Framework Architecture Overview and Trends}      % overview — REWRITTEN
  \subsection{Page-to-Document Transformer}               %   definition, advantages, limitations
  \subsection{Backbone-Centric MLLM Adaptation}           %   definition, advantages, limitations
  \subsection{Retriever--Generator}                       %   definition, advantages, limitations
  \subsection{Agentic Pipeline}                           %   definition, advantages, limitations

\section{Modalities Overview and Trends}                  % overview — TO BE REWRITTEN
  \subsection{Text Modality}                              %   how text is handled in MP setting
  \subsection{Visual Modality}                            %   how vision is handled in MP setting
  \subsection{Layout Modality}                            %   how layout/structure is handled
  \subsection{Multimodal Fusion Overview}                 %   when/how modalities are fused

\section{Training Strategies}                             % methods — partially drafted
  % parameter-efficient tuning, curriculum, retrieval training, RL for reasoning
  % concrete techniques and existing works, organised by training pattern

\section{Training-Free Strategies}                        % methods — drafted
  \subsection{Chain-of-Thought and Tool Augmented Reasoning}
  \subsection{Retrieval and Navigation Techniques}
  \subsection{Multi-Agent Decomposition}

\section{Datasets}                                        % drafted, placeholders remain

\section{Challenges}                                      % bullet form

\section{Limitations}
\section{Ethics}
```

### What goes where

- **§2 Framework Architecture Overview.** Pattern-level only. For each family give a four-beat paragraph: definition and characteristics, general advantage, specific suitability (domains/cases), limitations. Citations support taxonomy-level claims, never instantiate techniques. **This section is now rewritten.**
- **§3 Modalities Overview.** Pattern-level only. Describe how each modality (text, vision, layout) is *handled in the multi-page setting* — what role it plays, what scaling pressures it creates, and what coarse strategies architecture families use to keep it tractable. Do not enumerate individual model techniques. The Multimodal Fusion subsection covers *when* fusion happens (early, late, staged) at the family level, not specific fusion mechanisms. **This is the immediate task.**
- **§4 Training Strategies.** Methods section. Concrete techniques and existing works, organised by training pattern (parameter-efficient tuning, multi-stage curriculum, self-supervised retrieval training, reinforcement learning for reasoning). Models named here are described mechanism-first, not chronologically.
- **§5 Training-Free Strategies.** Methods section. Inference-time techniques on frozen backbones, organised into CoT/tool-augmented reasoning, retrieval and navigation, and multi-agent decomposition. **Drafted.**
- **§6 Challenges.** Synthesises the limitations surfaced across §2–§5 into a forward-looking discussion. Currently bullet form.

### Scoping conventions

- **By-technique scoping.** Partially-trained systems (e.g., DocDancer, M2RAG, AVIR, CREAM, VDU-CoR) are cited in whichever methods section their inference-time pattern fits, and re-cited in §4 with their training details. The opening framing of §5 — "adapting inference rather than parameters" — applies at the technique level, not per-model exclusion.
- **Multi-faceted models distributed by component.** Models combining several patterns (DocAgent, ViDoRAG, DocLens) are cited across multiple methods subsections, with each subsection describing only the relevant component.
- **Multi-pass RAG counts as agentic.** The boundary between iterated retrieval and controller-driven action selection is treated as orchestration rather than architecture.

## Writing style

Mimic the tone of the general MLLM VRDU survey at `survey_paper/General_MLLM_VRDU_Survey.md`. Prefer concise article-ready LaTeX prose over bullet points. Preserve the paper's existing style when revising.

- Compact academic survey prose. Each sentence adds a definition, mechanism, comparison, trend, benefit, limitation, or transition.
- **Do not use "because".** Prefer "as", "since", "thereby", "due to", "which", or a separate consequence sentence.
- **No colons in prose.** Use full sentences. Reserve colons for LaTeX syntax, table captions, or technical notation.
- Full declarative sentences. No conversational phrasing, rhetorical questions, or informal connectors.
- Cut filler such as "in this context", "it is important to note", "this design is effective when", "the main point is".
- Taxonomy-level phrasing before model-level detail. Prefer "Some frameworks…", "Other approaches…", "Recent work…", "These methods…" over repeatedly naming models as subjects.
- Keep citations close to the claim. Citations support patterns, not standalone lists.
- Use contrastive transitions sparingly: "However", "Moreover", "Additionally", "In contrast", "In sum", "Overall" — none of these more than once per subsection.
- State limitations specifically. Identify the source — context length, visual token growth, OCR error propagation, retrieval recall, compression loss, tool latency, attention dilution, lost-in-the-middle, parser quality. Do not write "scalability issues" without naming the cause.
- Cautious formulations when summarising trends — "often", "typically", "may", "can", "tend to".
- No model-by-model chronology in overview sections. Synthesise into patterns and trade-offs.
- Keep MP-VRDU focus explicit. Mention multi-page evidence aggregation, cross-page reasoning, page-level structure, long-context constraints, or document-scale evidence whenever relevant.
- Do not repeat the same limitation in adjacent sentences. Merge overlapping statements.
- Avoid broad general VRDU background unless it positions an MP-VRDU-specific argument.

## Working pattern the user prefers

- Plan before drafting. For each beat (definition, advantage, suitability, limitations), provide a primary recommendation **and alternatives** so the user can pick.
- After approval, draft the prose, write to `main.tex`, and also output the prose to the CLI for review.
- Preserve user-added bullet-point notes and `\paragraph{...}` headers in `main.tex`; update bullets to match revised prose.
- Use citation keys from `custom.bib`. If a needed paper is missing, use a clear placeholder and flag it.

## Pending tasks (rough priority)

1. **§3 Modalities Overview rewrite (next).** Convert each subsection into a proper overview of how the modality is handled in the MP setting. Remove any specific-technique drift. Use the same four-beat structure as §2 where it fits, or a tighter overview shape if the modality does not need full advantages/limitations beats.
2. §4 Training Strategies — clean up citation key casing (`Zhu2025-jb-laytokenllm`, `Zhang2024-zs-cream`, `Wu2025-mi-molorag`, `Hannan2025-zu`, `Hu2024-wl`, `Tanaka2025-er`) to match the lowercase project convention, and expand the section with concrete techniques.
3. §1 Introduction prose conversion from bullets.
4. §6 Challenges prose conversion from bullets.
5. §7 Datasets — fill `[PLACEHOLDER]` numerical values.
