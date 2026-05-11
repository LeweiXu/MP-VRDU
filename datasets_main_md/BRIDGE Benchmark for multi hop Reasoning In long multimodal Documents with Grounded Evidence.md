## **BRIDGE: Benchmark for multi-hop Reasoning In long multimodal Documents with Grounded Evidence** 

Yihao Ding The University of Western Australia Perth, Australia yihao.ding@uwa.edu.au 

Soyeon Caren Han The University of Melbourne Melbourne, Australia Caren.Han@unimelb.edu.au 

Biao Xiang The University of Melbourne Melbourne, Australia biaox@student.unimelb.edu.au 

**Table 1: Comparison with existing multi-hop QA benchmarks. MH: Multi-hop; Type: Question types (Ab: Abstractive; Cp: Comparative; Re: Causal Reasoning); Ann.: Explicit multi-hop annotation; Evl.: Step-by-step multi-hop evaluation; MM: Multimodal evidence.** 

## **Abstract** 

Multi-hop question answering (QA) is widely used to evaluate the reasoning capabilities of large language models, yet most benchmarks focus on final answer correctness and overlook intermediate reasoning, especially in long multimodal documents. We introduce BRIDGE, a benchmark for multi-hop reasoning over long scientific papers that require integrating evidence across text, tables, and figures. The dataset supports both chain-like and fan-out structures and provides explicit multi-hop reasoning annotations for step-level evaluation beyond answer accuracy. Experiments with state-of-the-art LLMs and multimodal retrieval-augmented generation (RAG) systems reveal systematic deficiencies in evidence aggregation and grounding that remain hidden under conventional answer-only evaluation. BRIDGE provides a targeted testbed for diagnosing reasoning failures in long multimodal documents. 

||**Dataset**<br>MultiTabQA [16]<br>MEQA [10]<br>SPIQA [18]|**MH **<br>✓<br>✓<br>×|**Type**<br>Cp<br>Cp+Re<br>—|**Ann. **<br>×<br>×<br>×|**Evl. **<br>×<br>×<br>×|**MM **<br>×<br>×<br>✓|**Doc. Scope**<br>Relational DB<br>Multi-doc text<br>Sci. papers|**Scale**<br>9k<br>2,243<br>270k|
|---|---|---|---|---|---|---|---|---|
||DBQR-QA [14]<br>JEMHopQA [7]<br>Complex-TR [21]<br>FanOutQA [26]|✓<br>✓<br>✓<br>✓|Cp+Re<br>Cp+Re<br>Cp+Re<br>Cp|×<br>✓<br>×<br>×|×<br>×<br>×<br>×|✓<br>×<br>×<br>×|Finance reports<br>Wikipedia (JP)<br>Temporal facts<br>Wikipedia|400<br>1179<br>60k<br>1894|
||MultiHoax [19]<br>BioHopR [9]<br>Cofca [24]|✓<br>✓<br>✓|Cp+Re<br>Re<br>Re|×<br>×<br>✓|×<br>×<br>✓|×<br>×<br>×|Wikipedia<br>Biomed. KG<br>Short passages|6,944<br>10k<br>3.6k|
||MMQA [23]|✓|Cp|×|×|✓|Relational tables|3.3k|
||**BRIDGE (Ours)**|✓|Ab+Cp+Re|✓|✓|✓|Sci. papers|11k|



## **CCS Concepts** 

## • **Information systems** → **Information retrieval** . 

## **Keywords** 

Document Understanding, Information Retrieving, Multi-hop 

## **ACM Reference Format:** 

Biao Xiang, Soyeon Caren Han, and Yihao Ding. 2026. BRIDGE: Benchmark for multi-hop Reasoning In long multimodal Documents with Grounded Evidence. In _Proceedings of Make sure to enter the correct conference title from your rights confirmation email (Conference acronym ’XX)._ ACM, New York, NY, USA, 5 pages. https://doi.org/XXXXXXX.XXXXXXX 

## **1 Introduction** 

Recent advances in Large Language Models (LLMs) have significantly improved document-based question answering (QA). However, in high-stakes domains such as finance [3, 22, 13], healthcare [20, 8], and academic research [18], answers are rarely explicitly stated and must be derived through multi-hop reasoning over heterogeneous evidence distributed across long documents. Existing multi-hop QA benchmarks are typically framed as either fan-out or chain-like [26], differing in how reasoning structure is enforced and 

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org. _Conference acronym ’XX, Woodstock, NY_ 

© 2026 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 978-1-4503-XXXX-X/2018/06 https://doi.org/XXXXXXX.XXXXXXX 

evaluated. Fan-out formulations emphasize parallel evidence collection with answer-level aggregation, often without constraining intermediate reasoning. In contrast, chain-like formulations impose sequential dependency, where errors propagate along the reasoning path and shortcut behaviors can be exposed [2]. Nevertheless, as summarized in Table 1, most benchmarks lack explicit supervision and step-level evaluation of intermediate reasoning [24]. This limitation becomes more pronounced in multimodal settings. Although several datasets incorporate multiple modalities [23, 14, 7, 21, 9], modalities are frequently treated as independent or redundant information sources. As a result, models can rely primarily on textual cues while underutilizing tables or figures, reducing multimodal reasoning to shallow pattern matching rather than structured evidence aggregation. We argue that long scientific papers provide a natural and challenging testbed for structured multi-hop multimodal reasoning. Unlike loosely connected Wikipedia passages [25, 6], scientific documents exhibit coherent discourse structures in which claims introduced in text are quantified in tables and validated in figures, forming intrinsic cross-modal dependency chains. Evaluating reasoning in such settings requires not only answer correctness, but also verification of intermediate evidence usage and cross-modal consistency. 

Our main contributions can be summarized as: (1). We introduce a **B** enchmark for multi-hop **R** easoning **I** n long multimodal **D** ocuments with **G** rounded **E** vidence (BRIDGE), that supports both chain-like and fan-out question formulations. The benchmark covers diverse question types, including comparative, causal reasoning, and abstractive questions. (2). We construct a multimodal multi-hop QA benchmark grounded in long scientific documents, requiring explicit reasoning across text, tables, and figures within long documents. (3). We provide explicit multi-hop reasoning annotations and 

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY 

Biao Xiang, Soyeon Caren Han, and Yihao Ding 

|**Type **|**Type **|**Questions (and Mod.)**|**Questions (and Mod.)**|**Questions (and Mod.)**|**Questions (and Mod.)**||**Answers (shorter version)**|**Answers (shorter version)**|**Answers (shorter version)**|**Answers (shorter version)**|**Answers (shorter version)**||
|---|---|---|---|---|---|---|---|---|---|---|---|---|
||Cp|In Fig 3, which task sets a larger|||In Fig 3, which task sets a larger<br>They are the same. Both of them|||They are the same. Both of them|||||
|||“Ann” size, task 1 or 3? (F+T)|||“Ann” size, task 1 or 3? (F+T)||are 110.||||||
||Ab|Based on the whole paper, how<br>Justification dominates, while ar-||||||||||Justification dominates, while ar-|
|||do explanation types shift with<br>gument increases as tasks be-|||||||||||
|||subjectivity? (T)|||||come more subjective.||||||
||Re|Why use only a subset of fallacy|Why use only a subset of fallacy<br>To reduce label ambiguity,||||||||and||
|||types? (T+Tb)|||||make annotation and evaluation|||make annotation and evaluation|||
||||||||reliable in Table 2.||||||
||||||||||||||
||||||||||||||
||||||||||||||
||||**...**||||**...**||||||



**Figure 1: Representative examples of comparative (Cp), abstractive (Ab), and causal reasoning (Re) questions (top), and the corresponding pages where evidences locate (bottom). Mod.: involved modalities (T: text; Tb: table; F: figure).** 

evaluation protocols that assess intermediate reasoning states and evidence usage, going beyond final answer correctness. In addition, we introduce a structured error taxonomy to facilitate fine-grained analysis of reasoning failures. 

## **2 Dataset** 

We present the BRIDGE dataset, supporting both fan-out and chainlike multi-hop QA over long multimodal scientific documents. **Task Definition.** The BRIDGE task is formally defined as follows: given a question _𝑞_ and a multi-page scientific document _𝐷_ —a longform scientific paper comprising heterogeneous and interdependent modalities such as textual passages, tables, and figures—the goal is to generate a final answer _𝑎_ along with a set of supporting evidences _𝐸_ = { _𝑒_ 1 _,𝑒_ 2 _, . . . ,𝑒𝑘_ }. The document is represented as a collection of semantic entities, each associated with a specific location (e.g., page index and region) and consisting of text spans, table entries, or visual figures. The objective is to require the model F to perform grounded reasoning over _𝐷_ , such that F ( _𝑞, 𝐷_ ) →( _𝑎, 𝐸_ ). The task typically demands _multi-hop reasoning_ , where answering _𝑞_ necessitates synthesizing information from multiple evidences, potentially across modalities. Two types of reasoning structures are considered: _chain-like_ , where evidences must be followed in a sequential, dependent manner; and _fan-out_ , where multiple evidences contribute in parallel to the final answer. This formulation allows for comprehensive evaluation of a model’s capacity to perform complex, modality-spanning reasoning within scientific documents. 

**Data Collection and Preprocessing** We construct the dataset from 262 research papers on ArXiv[1] , including PDF files and available LaTeX sources. To ensure structural quality, we focus on papers published at top-tier venues. The corpus spans recent computer science research (2023–2025), primarily in NLP and computer vision (e.g., ACL, EMNLP, CVPR, ICCV). Each paper is parsed using the 

**==> picture [230 x 163] intentionally omitted <==**

**----- Start of picture text -----**<br>
7500 7286 5213<br>1271 4122 704<br>5000 4316 4000 1118<br>650 4500 3305 2051<br>2500 2466 2000 1833<br>0 239 1200 1515 0 1171 1204 1557382 418<br>1 2 3+ 1 2 3 4+<br>Hop depth Distinct pages involved<br>1052 1038<br>1000 148 872<br>750 680 630 684 152 565 505<br>500 672 251 480 300 389 280 245 236<br>2500 224 362 151 281 80 200 313 102 9088 98 134<br>F-T Tb-T F-T-T Tb-Tb Tb- T-T-T T-T F-F Tb- Tb-<br>T-T Tb-Tb Tb-T<br>Hop Pattern<br>Abstractive Causal Comparative<br>**----- End of picture text -----**<br>


**Figure 2: Distribution of QA instances by hop depth, number of distinct pages involved, and hop pattern, broken down by question type (Abstractive, Causal, Comparative)** 

Adobe PDF Extract API[2] to extract text, tables, and figures with page indices and bounding-box metadata, enabling layout-aware document representations. 

## **2.1 QA Pair Generation** 

We generate multi-hop question–answer pairs using a chain-ofthought (CoT) prompting strategy with LLMs. 

**Defining Question Types** We define three types of multi-hop questions based on the underlying reasoning patterns: (1) _Causalreasoning questions_ , where hops are connected through causal relations between entities or events; (2) _Comparative questions_ , where multiple hops involve comparisons across entities, primarily numerical values, and occasionally concepts; (3) _Abstractive questions_ , which require a holistic understanding of the entire paper to produce a summary-style answer. 

**Question Type Guided Prompt Designing** We adopt a twostage prompting framework: Stage 1 (Structure Mining) extracts question-type-specific entity structures, and Stage 2 (ConstraintGuided Generation), conditions on these structures, formal question definitions, and multi-hop constraints to generate structured multihop QA pairs with explicit evidence hops. 

**Dual-Stage QA Quality Filtering** To ensure high data reliability, we employ a two-stage filtering pipeline: (1) a rule-based pre-filter that eliminates malformed instances, unresolved references, and instances lacking numeric grounding or metadata; and (2) an LLMas-a-judge framework that evaluates grounding, faithfulness, and reasoning depth. By utilizing structured PDF anchors to flag hallucinations and "single-hop shortcuts," this process ensures a scalable, reproducible collection of high-confidence multi-hop QA pairs. 

## **2.2 Dataset Analysis** 

BRIDGE contains 11,857 QA pairs with annotated evidence chains, spanning three task types and diverse hop patterns (e.g., table–text, figure–table–text). We summarize its structural properties along 

2https://developer.adobe.com/document-services/apis/pdf-extract/ 

1https://info.arxiv.org/help/api/index.html 

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY 

BRIDGE: Benchmark for multi-hop Reasoning In long multimodal Documents with Grounded Evidence 

**Table 2: Unified evaluation results. LLM-based judge metrics (Audit, Accuracy, Fidelity) and N-gram-based lexical metrics (ROUGE-L, BLEU). Higher is better for all metrics.** 

|**Model**<br>**Strategy**|**LLM as Judge**<br>**N-gram Metrics**<br>**Audit**<br>**Acc**<br>**Fidelity**<br>**ROUGE-L**<br>**BLEU**|
|---|---|
|ChatGPT<br>Direct<br>CoT<br>Refection|4.379<br>4.528<br>4.567<br>0.247<br>0.048<br>4.404<br>4.524<br>4.610<br>0.192<br>0.038<br>4.387<br>4.539<br>4.587<br>0.243<br>0.047|
|Gemma<br>Direct<br>CoT<br>Refection|4.131<br>4.373<br>4.291<br>0.292<br>0.069<br>4.175<br>4.413<br>4.403<br>0.292<br>0.069<br>4.170<br>4.415<br>4.390<br>0.291<br>0.068|
|Gemini<br>Direct<br>CoT<br>Refection|3.989<br>4.154<br>4.281<br>0.288<br>0.078<br>3.791<br>3.929<br>4.166<br>0.268<br>0.073<br>3.728<br>3.861<br>4.131<br>0.260<br>0.071|
|Qwen<br>Direct<br>CoT<br>Refection|3.489<br>3.578<br>3.508<br>0.266<br>0.058<br>3.605<br>3.685<br>3.549<br>0.224<br>0.043<br>3.594<br>3.666<br>3.517<br>0.219<br>0.045|



three dimensions. _i) Hop Depth._ Most questions require two reasoning steps, with a substantial portion involving three or more hops. It indicates a broader range of reasoning depths. _ii) Cross-Page Scope._ A large proportion of questions aggregate evidence from multiple pages, commonly two to three and extending beyond, reflecting the long-document nature. _iii) Modality Transitions._ We observe diverse modality transitions, including text–table, text–figure, and multi-step cross-modal chains. While purely unimodal reasoning forms only a subset. 

## **3 Experimental Setup** 

We evaluate representative MLLMs, including closed-source systems (Gemini-3 [5], ChatGPT-5[15]) and open-weight models (Qwen3 [1], Gemma-3 [4]), under a unified evaluation pipeline. All models are tested with standardized input formatting and identical decoding settings (temperature = 0.2, max tokens = 2048) to ensure fair comparison. For retrieval-augmented experiments, we employ ColPali as a multimodal retriever at the page level, retrieving top- _𝑘_ = 3 relevant pages per query. Retrieved content is concatenated with the prompt before generation. Then for result evaluation, we adopt Qwen-Plus (temperature = 0.2) as an LLM-as-a-judge to score final answer correctness and evidence alignment on a 0–5 scale. We further analyze reasoning behaviors using a structured error taxonomy. In addition, we report ROUGE [11] and BLEU [17] as complementary automatic metrics. 

## **4 Results and Discussion** 

## **4.1 Overall Performance** 

Table 2 summarizes judge-based metrics. Using audit_score as the primary metric, ChatGPT performs best across strategies (4.379– 4.404), followed by Gemma (4.131–4.175), Gemini (3.728–3.989), and Qwen (3.489–3.605). Strategy effects are model-dependent: Gemini performs best with direct prompting and degrades under CoT/reflection (audit drops by 0.198 and 0.261, respectively), while Qwen improves with CoT/reflection (audit +0.116/+0.105). Gemma yields fewer valid generations in certain subsets compared to other 

**Table 3: End-to-end impact of Colpali-RAG relative to Gemini-only baselines. Negative** Δ **indicates degradation (Colpali minus Gemini).** 

|**Comparison**|Δ**Audit**|Δ**Acc**|Δ**Fid**|Δ**R-L**|Δ**BLEU**|
|---|---|---|---|---|---|
|Colpali – Gemini Direct|-1.700|-1.775|-0.741|-0.037|-0.026|
|Colpali – Gemini CoT|-1.502|-1.551|-0.626|-0.017|-0.022|
|Colpali – Gemini Refection|-1.439|-1.483|-0.591|-0.008|-0.019|



models. To ensure transparency in coverage, we explicitly report the number of evaluated instances N in subsequent breakdown tables. 

## **4.2 Performance on Various Evaluation Metrics** 

Table 2 reports lexical overlap. Strategy-induced stylistic drift is visible even when judge metrics remain stable. For ChatGPT, CoT does not materially change audit/accuracy but substantially reduces ROUGE-L (0.247→0.192), indicating more paraphrastic or verbose generations. For Qwen, CoT/reflection improve judge-based correctness but decrease ROUGE-L/BLEU (e.g., ROUGE-L drops by 0.042 under CoT), suggesting that reasoning prompts yield answers that are judged more correct yet less surface-aligned with the GT wording. In contrast, Gemma’s ROUGE/BLEU are nearly invariant across strategies, matching its minimal stylistic drift. 

Colpali was originally proposed for visually-rich, page-level multimodal document retrieval (e.g., ViDoRe [12]) and demonstrates strong retrieval performance in that setting. However, Table 3 shows that, in our long multimodal document QA setting requiring multihop evidence, Colpali-RAG substantially degrades end-to-end answer quality even when Gemini is used as the downstream generator. Compared to Gemini direct prompting, Colpali reduces audit by 1.700 and accuracy by 1.775, with a concurrent fidelity drop of 0.741. The smaller reductions in ROUGE-L/BLEU (e.g., ROUGE-L -0.037) indicate that the dominant failure mode is not merely paraphrasing, but incorrect or weakly grounded answers, which is consistent with retrieval mismatch and difficulty of locating multi-hop evidence across long documents. 

## **4.3 Breakdown Analysis** 

We conduct a comprehensive breakdown analysis across multiple dimensions, including question type, number of pages, and evidence modality, to systematically evaluate the capabilities of existing MLLMs and baseline configurations. 

_Question-Type Analysis._ Table 4 reveals systematic differences across task types. Causal-Reasoning is generally the most stable category for strong models (ChatGPT audit 4.766; Gemini audit 4.366), with high accuracy and fidelity, suggesting that causal questions are often directly supported by localized evidence. Comparative questions are the hardest across settings: even ChatGPT drops to 3.691 audit, and Colpali-RAG collapses to 1.002 audit and 1.017 accuracy, indicating difficulty in aligning multiple entities/conditions across distant evidence. Abstractive questions show the largest spread across models, where Gemma remains strong (audit 4.484) while Qwen and RAG underperform substantially. 

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY 

Biao Xiang, Soyeon Caren Han, and Yihao Ding 

**Table 4: Question-type breakdown. We normalize variant labels (e.g., TABLE/FIGURE-first) into three core types. We report judge-based metrics: audit_score (Audit), answer accuracy (Acc), and evidence fidelity (Fid).** 

|**Question Type**|**Model**|**Best Strat.**|**Audit**|**Acc**|**Fid**|**N**|
|---|---|---|---|---|---|---|
||ChatGPT|Refection|4.233|4.353|4.456|2766|
||Gemma|Refection|4.484|4.784|4.730|538|
|Abstractive|Gemini|Direct|3.848|3.982|4.098|2258|
||Qwen|CoT|2.682|2.556|2.435|162|
||RAG|Colpali|2.008|2.060|3.442|2807|
||ChatGPT|CoT|4.766|4.944|4.947|5065|
||Gemma|CoT|4.351|4.633|4.615|741|
|Causal-Reasoning|Gemini|Direct|4.366|4.586|4.665|4183|
||Qwen|CoT|3.658|3.749|3.611|5483|
||RAG|Colpali|2.936|3.074|3.825|5145|
||ChatGPT|CoT|3.691|3.706|3.925|1908|
||Gemma|Direct|3.594|3.679|3.639|418|
|Comparative|Gemini|Direct|3.217|3.279|3.554|1579|
||Qwen|Refection|2.874|2.843|2.657|102|
||RAG|Colpali|1.002|1.017|2.954|1951|



**Table 5: Performance across page-depth bins, grouped by the furthest evidence page involved (best strategy per model by audit_score).** 

|**Page Bin**|**Model**|**Best Strat.**|**Audit**|**Acc**|**Fid**|**N**|
|---|---|---|---|---|---|---|
||ChatGPT|CoT|4.692|4.867|4.881|3338|
|1–2|Qwen|CoT|3.731|3.844|3.706|2417|
||ChatGPT|CoT|4.417|4.538|4.632|2739|
|3–5|Qwen|CoT|3.552|3.617|3.472|1659|
||ChatGPT|CoT|4.240|4.319|4.455|2636|
|6–10|Qwen|CoT|3.518|3.559|3.430|1333|
||ChatGPT|CoT|3.879|3.942|4.123|1399|
|11–20|Qwen|Refection|3.453|3.565|3.142|822|
||ChatGPT|CoT|3.321|3.303|3.521|33|
|21+|Qwen|Refection|3.570|3.484|3.124|23|



_Cross-Page Evidence Distribution Analysis._ Table 5 shows a clear degradation as the required evidence moves deeper into the document. For all models, performance is highest when evidence appears early (pages 1–2) and drops for later bins (pages 11–20 and 21+). For example, ChatGPT decreases from 4.692 (pages 1–2) to 3.879 (11–20) and 3.321 (21+), while Gemini drops from 4.370 to 3.349 and 3.109. This trend is consistent with long-context search becoming harder and multi-hop evidence coverage decreasing with page depth. Notably, Qwen is comparatively less monotonic at 21+ (3.570), but the sample size is very small (N=23), suggesting high variance in the tail bins. 

_Hop Depth Analysis._ Table 6 shows that performance remains comparable between 2-hop and 3+-hop questions for strong models, suggesting that hop depth alone does not fully determine difficulty. However, smaller models exhibit higher variance across hop bins. 

**Table 6: Performance by hop depth (1-hop, 2-hop, 3+-hop).** 

|**Hop Bin**|**Model**|**Best Strat.**|**Audit**|**Acc**|**Fid**|**N**|
|---|---|---|---|---|---|---|
||ChatGPT|CoT|3.692|3.656|3.966|212|
|1-hop|Gemma<br>Gemini|Direct<br>Direct|2.952<br>2.888|3.000<br>2.978|3.317<br>3.245|25<br>180|
||Qwen|Refection|2.649|2.729|2.602|59|
||ChatGPT|CoT|4.425|4.548|4.641|3344|
|2-hop|Gemma<br>Gemini|CoT<br>Direct|4.200<br>4.004|4.454<br>4.179|4.408<br>4.283|817<br>2820|
||Qwen|CoT|3.415|3.447|3.310|1742|
||ChatGPT|CoT|4.418|4.541|4.615|6075|
|3+-hop|Gemma<br>Gemini|CoT<br>Direct|4.194<br>4.020|4.421<br>4.181|4.429<br>4.317|850<br>5030|
||Qwen|Refection|3.719|3.808|3.668|3954|



**Table 7: Performance by primary evidence modality (Text, Table, Figure), aggregated over prompting strategies. N denotes the number of instances whose dominant supporting evidence belongs to each modality.** 

|**Evidence**|**Model**|**Audit**|**Acc**|**Fid**|**N**|
|---|---|---|---|---|---|
||ChatGPT|4.629|4.812|4.816|6933|
|Text|Gemma<br>Gemini|4.384<br>4.250|4.699<br>4.447|4.643<br>4.564|1859<br>5643|
||Qwen|3.496|3.557|3.437|5145|
||ChatGPT|4.096|4.180|4.323|11217|
|Table|Gemma<br>Gemini|3.816<br>3.368|3.955<br>3.458|3.950<br>3.789|1794<br>9314|
||Qwen|3.492|3.571|3.448|5091|
||ChatGPT|4.537|4.706|4.712|11163|
|Figure|Gemma<br>Gemini|4.360<br>4.066|4.644<br>4.238|4.577<br>4.385|1376<br>8724|
||Qwen|3.682|3.777|3.662|6711|



_Diverse Evidence Modality Analysis._ Across all models, questions whose supporting evidence primarily comes from **tables** are the most challenging. Judge-based performance drops when evidence is presented in tables compared to text or figures. For example, Gemini’s audit score decreases from **4.250** on text evidence to **3.368** on tables (a drop of **0.882** ), and ChatGPT drops from **4.629** to **4.096** (a drop of **0.533** ). In contrast, **figure-based** evidence is relatively easier for stronger models: ChatGPT achieves an audit score of **4.537** on figures, close to its text performance ( **4.629** ), suggesting improved robustness when evidence is presented visually rather than in dense tabular form. Notably, Qwen exhibits nearly identical audit scores on text and tables ( **3.496** vs. **3.492** ) but remains substantially below the stronger models overall, indicating that its dominant bottleneck may be general evidence alignment and reasoning accuracy rather than table-specific difficulty. 

## **5 Conclusion** 

We introduce BRIDGE, a benchmark for long multimodal document QA that explicitly stresses multi-hop evidence aggregation across visually-rich PDFs, and address a gap not covered by prior 

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY 

BRIDGE: Benchmark for multi-hop Reasoning In long multimodal Documents with Grounded Evidence 

evaluations that focus on short-context QA or page-level retrieval. Our results show that strong LLMs can achieve high judge-based correctness under direct access to evidence, but performance varies substantially with prompting, and lexical overlap (ROUGE/BLEU) can diverge from factual grounding. More importantly, a Colpalibased RAG pipeline degrades markedly in end-to-end multi-hop QA, which highlights retrieval mismatch and evidence-missing as dominant failure modes. Overall, the benchmark provides a new, targeted testbed for diagnosing grounding, comparison reversal, and evidence coverage errors in current LLM systems, and motivates future work on retrieval calibration, evidence verification, and citation-faithful generation for long multimodal documents. 

## **References** 

- [1] Alibaba Group. 2023. Qwen technical report. https : / / qwen . ai / qwenchat. Accessed: 2025-02-09. (2023). 

- [2] Jifan Chen and Greg Durrett. 2019. Understanding dataset design choices for multi-hop reasoning. In _Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)_ . Jill Burstein, Christy Doran, and Thamar Solorio, (Eds.) Association for Computational Linguistics, Minneapolis, Minnesota, (June 2019), 4026–4032. doi:10.18653/v1/N19-1405. 

- [3] Yihao Ding, Siqu Long, Jiabin Huang, Kaixuan Ren, Xingxiang Luo, Hyunsuk Chung, and Soyeon Caren Han. 2023. Form-nlu: dataset for the form natural language understanding. (2023). https://arxiv.org/abs/2304.01577 arXiv: 2304.01577 [cs.IR]. 

- [4] Google. 2024. Gemma: open models based on gemini research and technology. https://ai.google.dev/gemma. Accessed: 2025-02-09. (2024). 

- [5] Google DeepMind. 2023. Gemini: a family of highly capable multimodal models. https://deepmind.google/technologies/gemini/. Accessed: 2025-02-09. (2023). 

- [6] Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. 2020. Constructing a multi-hop qa dataset for comprehensive evaluation of reasoning steps. (2020). https://arxiv.org/abs/2011.01060 arXiv: 2011.01060 [cs.CL]. 

- [7] Ai Ishii, Naoya Inoue, Hisami Suzuki, and Satoshi Sekine. 2024. JEMHopQA: dataset for Japanese explainable multi-hop question answering. In _Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)_ . Nicoletta Calzolari, Min-Yen Kan, Veronique Hoste, Alessandro Lenci, Sakriani Sakti, and Nianwen Xue, (Eds.) ELRA and ICCL, Torino, Italia, (May 2024), 9515–9525. https://aclan thology.org/2024.lrec-main.831/. 

- [8] Qiao Jin, Bhuwan Dhingra, Zhengping Liu, William Cohen, and Xinghua Lu. 2019. PubMedQA: a dataset for biomedical research question answering. In _Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)_ . Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan, (Eds.) Association for Computational Linguistics, Hong Kong, China, (Nov. 2019), 2567–2577. doi:10.18653/v1/D19-1259. 

- [16] Vaishali Pal, Andrew Yates, Evangelos Kanoulas, and Maarten de Rijke. 2023. MultiTabQA: generating tabular answers for multi-table question answering. In _Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ . Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki, (Eds.) Association for Computational Linguistics, Toronto, Canada, (July 2023), 6322–6334. doi:10.18653/v1/2023.acl-long.348. 

- [17] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. In _Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics_ . Pierre Isabelle, Eugene Charniak, and Dekang Lin, (Eds.) Association for Computational Linguistics, Philadelphia, Pennsylvania, USA, (July 2002), 311–318. doi:10.3115/1073083.1073135. 

- [18] Shraman Pramanick, Rama Chellappa, and Subhashini Venugopalan. 2025. Spiqa: a dataset for multimodal question answering on scientific papers. (2025). https://arxiv.org/abs/2407.09413 arXiv: 2407.09413 [cs.CL]. 

- [19] Mohammadamin Shafiei, Hamidreza Saffari, and Nafise Sadat Moosavi. 2025. Multihoax: a dataset of multi-hop false-premise questions. (2025). https://arxiv .org/abs/2506.00264 arXiv: 2506.00264 [cs.CL]. 

- [20] Sarvesh Soni, Meghana Gudala, Atieh Pajouhi, and Kirk Roberts. 2022. RadQA: a question answering dataset to improve comprehension of radiology reports. In _Proceedings of the Thirteenth Language Resources and Evaluation Conference_ . Nicoletta Calzolari et al., (Eds.) European Language Resources Association, Marseille, France, (June 2022), 6250–6259. https://aclanthology.org/2022.lrec-1 .672/. 

- [21] Qingyu Tan, Hwee Tou Ng, and Lidong Bing. 2024. Towards robust temporal reasoning of large language models via a multi-hop qa dataset and pseudoinstruction tuning. (2024). https://arxiv.org/abs/2311.09821 arXiv: 2311.09821 [cs.CL]. 

- [22] Rubèn Tito, Dimosthenis Karatzas, and Ernest Valveny. 2023. Hierarchical multimodal transformers for multipage docvqa. _Pattern Recognition_ , 144, 109834. doi:https://doi.org/10.1016/j.patcog.2023.109834. 

- [23] Jian Wu, Linyi Yang, Dongyuan Li, Yuliang Ji, Manabu Okumura, and Yue Zhang. 2025. MMQA: evaluating LLMs with multi-table multi-hop complex questions. In _The Thirteenth International Conference on Learning Representations_ . https://openreview.net/forum?id=GGlpykXDCa. 

- [24] Jian Wu, Linyi Yang, Zhen Wang, Manabu Okumura, and Yue Zhang. 2024. Cofca: a step-wise counterfactual multi-hop qa benchmark. (2024). https://arxi v.org/abs/2402.11924 arXiv: 2402.11924 [cs.CL]. 

- [25] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. 2018. Hotpotqa: a dataset for diverse, explainable multi-hop question answering. (2018). https://arxiv.or g/abs/1809.09600 arXiv: 1809.09600 [cs.CL]. 

- [26] Andrew Zhu, Alyssa Hwang, Liam Dugan, and Chris Callison-Burch. 2024. Fanoutqa: a multi-hop, multi-document question answering benchmark for large language models. (2024). https://arxiv.org/abs/2402.14116 arXiv: 2402.14116 [cs.CL]. 

Received 20 February 2007; revised 12 March 2009; accepted 5 June 2009 

- [9] Yunsoo Kim, Yusuf Abdulle, and Honghan Wu. 2025. Biohopr: a benchmark for multi-hop, multi-answer reasoning in biomedical domain. (2025). https://ar xiv.org/abs/2505.22240 arXiv: 2505.22240 [cs.CL]. 

- [10] Ruosen Li, Zimu Wang, Son Quoc Tran, Lei Xia, and Xinya Du. 2024. Meqa: a benchmark for multi-hop event-centric question answering with explanations. In _Advances in Neural Information Processing Systems_ . A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, (Eds.) Vol. 37. Curran Associates, Inc., 126835–126862. doi:10.52202/079017-4028. 

- [11] Chin-Yew Lin. 2004. ROUGE: a package for automatic evaluation of summaries. In _Text Summarization Branches Out_ . Association for Computational Linguistics, Barcelona, Spain, (July 2004), 74–81. https://aclanthology.org/W04-1013/. 

- [12] António Loison et al. 2026. Vidore v3: a comprehensive evaluation of retrieval augmented generation in complex real-world scenarios. (2026). https://arxiv.or g/abs/2601.08620 arXiv: 2601.08620 [cs.AI]. 

- [13] Minesh Mathew, Dimosthenis Karatzas, and C. V. Jawahar. 2021. Docvqa: a dataset for vqa on document images. (2021). https://arxiv.org/abs/2007.00398 arXiv: 2007.00398 [cs.CV]. 

- [14] Rungsiman Nararatwong, Chung-Chi Chen, Natthawut Kertkeidkachorn, Hiroya Takamura, and Ryutaro Ichise. 2024. DBQR-QA: a question answering dataset on a hybrid of database querying and reasoning. In _Findings of the Association for Computational Linguistics: ACL 2024_ . Lun-Wei Ku, Andre Martins, and Vivek Srikumar, (Eds.) Association for Computational Linguistics, Bangkok, Thailand, (Aug. 2024), 15169–15182. doi:10.18653/v1/2024.findings-acl.900. 

- [15] OpenAI. 2024. Chatgpt. https://openai.com/chatgpt. Accessed: 2025-02-09. (2024). 

