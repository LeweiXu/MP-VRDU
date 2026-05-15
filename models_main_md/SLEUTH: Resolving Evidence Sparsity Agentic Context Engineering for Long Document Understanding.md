# **Resolving Evidence Sparsity: Agentic Context Engineering for Long-Document Understanding** 

Keliang Liu[1] Zizhi Chen[1] Mingcheng Li[1] Jingqun Tang[3] Dingkang Yang[1] _[,]_[2] _[ †]_ Lihua Zhang[1] _[,]_[2] _[ †]_ 

> _†_ corresponding authors project lead 

1College of Intelligent Robotics and Advanced Manufacturing, Fudan University 

> 2Fysics Intelligence Technologies Co., Ltd. (Fysics AI) 

3ByteDance 

klliu25@m.fudan.edu.cn, _{_ dkyang20, lihuazhang _}_ @fudan.edu.cn 

## **Abstract** 

_Document understanding is a long-standing practical task. Vision-Language Models (VLMs) have gradually become a primary approach in this domain, demonstrating effective performance on single-page tasks. However, their effectiveness diminishes when handling long documents. In such scenarios, clues are often scattered across multiple pages and modalities, and redundancy from lengthy inputs can impair the model’s judgment. While retrieval-augmented generation mitigates this issue by filtering for questionrelevant content, the retrieved results still contain substantial redundancy. To address these limitations, we propose SLEUTH, a multi-agent framework. Concretely, SLEUTH orchestrates a retriever and four collaborative agents in a coarse-to-fine process. The framework identifies key textual and visual clues within the retrieved pages, filters for salient visual evidence such as tables and charts, and analyzes the query to devise a reasoning strategy. It ultimately synthesizes a distilled, evidence-dense multimodal context to generate the final prediction. SLEUTH is model-agnostic and scalable. When paired with advanced VLM backbones, it consistently improves performance on multiple long-document benchmarks, achieving SOTA results. Ablation studies verify each module’s effectiveness and confirm the benefits of our hierarchical refinement paradigm._ 

## **1. Introduction** 

Documents are one of the fundamental forms of human information preservation and transmission [5, 19]. Understanding documents with complex layouts and multimodal components has long been a pragmatic challenge and also serves as a benchmark for evaluating the multimodal long-context reasoning capability of AI systems [9, 10, 30]. Documents convey 

rich visual information, not only through textual content but also via charts, page layouts, tables, and even fonts. Traditional document question answering methods usually employ OCR-based pipelines [29, 36] to extract textual content and then feed it into large language models (LLMs) for response generation. However, such approaches often lose critical multimodal cues, leading to imperfect and shallow document comprehension. Recently, Multimodal Large Language Models (MLLMs) [1, 2, 14, 16, 27, 35] have achieved remarkable progress in document understanding tasks owing to their intrinsic multimodal capabilities, particularly in single-page document understanding [20–22, 58]. Nevertheless, their ability to comprehend long-context documents remains uncertain [31, 57, 59]. Long documents introduce extensive contexts where most information is redundant, while the key evidence required to answer a query is often sparse and scattered across multiple pages and modalities. This characteristic poses substantial challenges to existing VLMs. The core objective of long-document understanding lies in precisely locating informative evidence from massive content and organizing it into a high-quality contextual representation suitable for reasoning and response generation. 

Recent document understanding studies mainly follow three paths, namely enhancing agent reasoning, improving retrieval recall, and combining the two. As shown in Figure 1a, multi-agent approaches such as MACT [54] strengthen reasoning through agent collaboration and training, and achieve good results in long-document understanding. However, when answering questions, these methods still take long contexts as input, with large amounts of redundant information that interferes with reasoning. Figure 1b demonstrates that Retrieval-Augmented Generation (RAG) methods [4, 6, 11, 32, 39] retrieve document content relevant to the query and achieve much better performance than directly reading the entire long document. However, the retrieved results 

1 

**==> picture [442 x 125] intentionally omitted <==**

**----- Start of picture text -----**<br>
a Reasoning Enhancement Method Try to enhance reasoning ability. c Retrieval-reasoning integration.<br>r r r r Multimodal<br>Retriever<br>en Long Document (Wed Agent1 Agent2 ...... Answer | Long Document Bs * SESW Agent1 Agent2 ...... 3d Answer<br>b Retrieval Based Method Text Query Try to improve retrieval recall. d  Ours ： Context engineering Method Construct High-Quality Context Context engineering<br>| J & ea !<br>Long Document Retriever Agent Answer<br>| @-a--ed ae od ae Extract<br>Long Document Retriever Relevant Pages Agent Answer Build concise, evidence-dense, and highly reliable multimodal contexts. Agents Filter Retriever Agent<br>**----- End of picture text -----**<br>


Figure 1. Comparison with mainstream methods. (a) Strengthening reasoning via agent optimization; (b) Improving recall through retrieval augmentation; (c) Combining (a) and (b); (d) Our method focuses on constructing evidence-dense contexts. 

still contain much irrelevant information, and VLMs have difficulty identifying the few key pieces of evidence scattered across multiple pages. Refer to Figure 1c, methods such as MDocAgent [12] combine RAG with multi-agent guided thinking, achieving performance gains. However, the reasoning process still involves long and noisy contexts. Our method is illustrated in Figure 1d. From the perspective of context engineering, we introduce Sequential Longdocument Evidence Uncovering through mulTi-agent Hierarchical refinement (SLEUTH), a plug-and-play multi-agent framework that, in a training-free manner, builds concise and evidence-dense contexts from noisy long-context retrieval results, thereby efficiently improving document understanding performance. A standard retriever first narrows the search space; four collaborative agents then mine page-level textual and visual clues, filter irrelevant page images, and adapt the reasoning strategy to the query’s difficulty before producing the final answer. By operating page-wise, SLEUTH keeps the effective context length fixed, enabling accuracy to scale with larger retrieval top-K without amplifying hallucinations. Our method can also be combined with other approaches to further improve document understanding capabilities. 

The core contributions of this work are summarized as follows: (i) We propose SLEUTH, a trainingfree multi-agent framework that efficiently mines key evidence from noisy long documents and constructs evidence-dense multimodal contexts. (ii) We design two complementary agents: the Clue Discovery Agent and the Page Screening Agent, responsible for clue extraction and visual filtering, respectively. Their collaboration enables a structured and refined contextual representation. (iii) We introduce an evidence and difficulty-aware mechanism that allows the system to adaptively select reasoning strategies based on query complexity and evidence context, leading to consistent performance improvements across multiple longdocument understanding benchmarks. 

To the best of our knowledge, this is the first work that investigates long-document understanding from the perspective of constructing concise, evidence-dense 

contexts. Extensive experiments on four benchmarks, including MMLongBench [19], LongDocURL [9], PaperTab [13], and FetaTab [13], demonstrate that our approach is model-agnostic and achieves SOTA performance across different VLM backbones. Furthermore, ablation studies verify the effectiveness of each agent and emphasize the importance of constructing compact, trustworthy, and evidence-dense contexts for robust long-document understanding. We believe this method, together with reinforcement reasoning and retrieval-based methods, will jointly advance the progress in the field of long-document understanding. 

## **2. Related Work** 

**Document Analysis and Understanding.** Document Analysis and Understanding, a long-standing task with strong practical needs [5, 19], has gained renewed attention with the rise of multimodal large language models [1, 2, 7, 14, 27, 35] These models show strong performance on benchmarks like DocVQA [21], ChartQA [20], InfoVGA [22], and TAT-DQA [58]. However, real-world documents are often long and complex, spanning text, tables, charts, and images. Consequently, research has shifted toward long-document settings that demand cross-page and multimodal reasoning [5, 6, 9, 10, 13, 19, 28]. Traditional OCR-based LLM methods can only process text, whereas VLM-based approaches directly interpret document images via multimodal capabilities and have become the mainstream paradigm. DeepSeek-OCR [37] shows that unified visual mappings can compress contexts while retaining layout cues. We adopt a related intuition by representing pages visually, which preserves structure and supports fine-grained perception and reasoning. 

**Context Engineering.** Benefiting from the development of learning-based approaches [17, 18, 41–53], context engineering involves the construction of dynamic systems that furnish accurate information and tools in appropriate formats, thereby enabling LLMs to effectively execute tasks. Suboptimal agent performance is frequently attributable to the inadequate pro- 

2 

vision of context, instructions, and tools to the model. Numerous studies [4, 19, 30, 31, 57, 59] reveal that, despite the ultra-long context windows of MLLMs, performance degrades sharply with increasing context length. The cornerstone of effective document analysis and understanding lies in high-quality context engineering. Retrieval-based methods like Colpali [11], Colbert [15], and BGE M3 [3] support document embeddings, while RAG frameworks such as M3DocRAG [6], SV-RAG [4], ViDoRAG [32] and MoLoRAG [39] retrieve relevant pages to reduce context length. Dynamic methods like Doc-React [38] refine sub-queries, and ACE [55] or Sun _et al._ [26] advocate proactive context management. Multi-agent systems such as CoA [56] and MDocAgent [12] distribute reasoning across agents, while reinforcement learning (RL) approaches (VRAG-RL [33], MACT [54], ReMemR1 [25], CARE [34]) enhance evidence gathering and retrieval. However, the above methods do not optimize for the fact that MLLM performance degrades as context length increases. By dynamically recording clues and filtering visual information, we construct concise yet information-dense contexts from overly long and redundant retrieval results, thereby enhancing understanding capability. 

## **3. Methodology** 

## **3.1. Overall Framework** 

As shown in Figure 2, given a question _Q_ and a document _D_ = _{p_ 1 _, p_ 2 _, . . . , pN }_ , where each _pi_ denotes an individual page represented as an RGB image and _N_ is the total number of pages, the goal of longdocument question answering is to generate an accurate answer _A_ to _Q_ based on the evidence contained in _D_ . Our SLEUTH is a training-free and plug-andplay framework with the following workflow: First, a standard retrieval model narrows down the evidence search space, reducing computational cost. The retrieved coarse-grained relevant pages are then processed by our multi-agent system. Leveraging the strong single-image reasoning capability of VLMs, Clue Discovery Agent scans each retrieved page and records structured textual and visual cues. In parallel, Page Screening Agent selects page images containing task-relevant tables, charts, or diagrams and filters out irrelevant ones. Difficulty Assessment Agent analyzes the query content, generates strategic instructions, and guides Core Decision Agent to adopt the optimal reasoning mode. This process produces a denoised multimodal context that is both highly reliable and rich in evidential clues. Finally, Core Decision Agent infers the final answer based on the query and the refined high-confidence context. 

## **3.2. Coarse-grained Visual Retrieval** 

To save computation and time, we first efficiently localize a small set of pages most relevant to the query _Q_ 

across the entire long document. We use Colpali [11] to encode the question _Q_ into a sequence of textual embeddings **q** _tt_ = 1 _[n][Q]_ and each page image _pi_ into a set of visual embeddings **v** _i, jj_ = 1 _[n][i]_ . The page-level relevance is then defined as: 

**==> picture [167 x 29] intentionally omitted <==**

Based on the relevance scores _{si}[N] i_ =1[,][we][select][the] top- _K_ indices _IK_ to obtain the set of candidate pages: 

**==> picture [181 x 12] intentionally omitted <==**

The coarse-grained retrieval stage performs page-level selection in a purely visual manner, ensuring that _K ≪ N_ and thus significantly reduces computation time. 

## **3.3. Evidence Refinement and Visual Screening** 

After page-level retrieval, the candidate set still contains a large amount of redundant information. The system must further identify the specific parts that truly contain critical evidence. In SLEUTH, we introduce two complementary agents: Clue Discovery Agent and Page Screening Agent. These two agents operate in parallel—the former focuses on discovering and recording interpretable structured clues, while the latter analyzes the overall visual composition of each page and filters out irrelevant images. Through this collaborative mechanism, the system constructs an evidence-dense multimodal context, providing a solid foundation for the final reasoning stage. 

Clue Discovery Agent treats each page as the minimal processing unit. The MLLMs sequentially process each candidate page image and extract queryrelevant evidence. For any _pi ∈ PK_ , the agent identifies potential clues at the region level such as text lines, table cells, or chart areas and produces a structured clue set as follows: 

**==> picture [112 x 11] intentionally omitted <==**

Each evidence unit is organized as a structured record: 

**==> picture [214 x 29] intentionally omitted <==**

where **w** _i,m_ denotes the spatial position or cell coordinates of the clue, _Content ci,m_ represents the clue extracted from _pi_ , _ki,m_ denotes the key insight inferred by the agent based on the clue, and _ri,m_ records the complete chain-of-thought (CoT) reasoning trace for relevance analysis. This design ensures that each clue not only carries semantic content but also preserves its source, spatial location, and adoption rationale, facilitating provenance tracking and interpretability verification during subsequent reasoning stages. 

Although Clue Discovery Agent captures finegrained clues within each page, for pages containing 

3 

**==> picture [438 x 223] intentionally omitted <==**

**----- Start of picture text -----**<br>
fast retrieval Toe (pas TITTIES<br>Embedding  Retrieval Model Top K<br>Question<br>= Gea '<br>1ca Tee | | (J a —<br>Question OF) 3 H} = fl<br>SS ... 2<br>Retriever<br>Clue Discovery Agent Page Screening Agent<br>CO — + | = [ aq<br>2.A. — Maxim ( ’ ) | Difficulty  Write notes to<br>Documents Images Assessment  record clues _ Discard<br>Agent e__.___..--- Irrelevant |<br>Irrelevantimage<br>Based on the clue  ee le @ Geil Irrelevantimage Ih<br>evidence and  image<br>relevant  Clue1: Page 1 introduces that the<br>views...... This  Core Decision  >0 model is composed of...<br>architecture  | - fea Clue2: Page 2 indicates that the<br>consists of a visual  Agent datasets consist of...<br>encoder, a  = ... _— a3 |<br>projector, and an  Needs  Clue5: No evidence was found. Keep<br>LLM.... So the  “4 || strong  | SSS ... 7) Ee) related |<br>C: answers are ... | Enable Thinking reasoning? VB Evidence Context = el pages<br>**----- End of picture text -----**<br>


Figure 2. **Overall Framework.** SLEUTH adopts a coarse-to-fine pipeline: (1) a visual retriever selects Top- _K_ pages; (2) Clue Discovery Agent records and refines evidence, Page Screening Agent filters irrelevant page images, (3) Difficulty Assessment Agent analyzes query complexity, and (4) Core Decision Agent reasons over the distilled, evidence-dense context. 

complex visual elements such as tables, charts, or embedded illustrations the textual records alone may still be insufficient to reflect all informative cues. To compensate for this limitation, Page Screening Agent also analyzes each candidate page in _PK_ , focusing on the visual information and its semantic alignment with the query intent. Let the set of salient visual elements on the _i_ -th page be defined as: 

**==> picture [112 x 11] intentionally omitted <==**

Each visual element _vi,k_ represents a table, chart, or key illustration within the page. Page Screening Agent performs joint semantic–visual reasoning over the entire page and outputs both the relevance level between the page and the query, as well as the corresponding reasoning explanation. Specifically, for a candidate page _pi_ , the agent generates a discrete relevance label based on the holistic page content: 

**==> picture [76 x 11] intentionally omitted <==**

and simultaneously provides an explanatory description _ri_[page] that clarifies the reasoning process behind its decision. CR, R, and IR refer to Completely Relevant, Relevant, and Irrelevant, respectively. This classification result is entirely derived from the model’s own multimodal reasoning: the agent integrates the page’s layout structure, table organization, chart content, and textual semantics to infer the logical relationships between these visual elements and the query, thereby determining whether the page should be retained. The number of retained pages adapts dynamically according to the query. For example, on LongDocURL [9], when retrieving the top-5 page images, 

only an average of 2.1 relevant visual pages are preserved. The final retained set is defined as: 

**==> picture [136 x 13] intentionally omitted <==**

The two agents are complementary at different levels: Clue Discovery Agent provides fine-grained and interpretable textual evidence, while Page Screening Agent ensures that critical visual elements are sufficiently preserved. Together, they collaboratively construct a multimodal context: 

**==> picture [156 x 23] intentionally omitted <==**

where _P_[�] denotes the set of retained page images, and _E_ represents the collection of evidence extracted from the original top- _K_ pages. Through this parallel–complementary collaboration mechanism, SLEUTH achieves compact input while maximizing evidence density, thereby substantially enhancing the contextual quality and robustness of long-document question answering. In certain cases, modality absence may occur. For example, some questions may result in all page images being filtered out, leading to a lack of visual inputs, or the query itself may be irrelevant to the document, yielding an unanswerable case. Such situations are reasonable and naturally handled: depending on the presence or absence of visual modality in the constructed context, the system automatically adjusts the prompt template fed into Core Decision Agent, avoiding inconsistencies such as providing visual instructions without corresponding visual inputs. Algorithm 1 illustrates the overall process of multimodal context construction. 

4 

**Algorithm 1** Retrieval and Context Construction 

**Require:** Natural-language question _Q_ ; document _D_ = _{p_ 1 _, . . . , pN }_ ; parameter _K_ ; **Ensure:** Filtered pages _P_[�] ; global evidence set _E_ 1: Encode the query: _{_ **q** _t}[n] t_ =1 _[Q][←]_[EncodeQuery(] _[Q]_[)] 2: **for** _i_ = 1 to _N_ **do** _▷_ Relevance scoring 3: _{_ **v** _i,j}[n] j_ =1 _[i][←]_[VisualEncode(] _[p][i]_[)] 4: _si ←_[�] _[n] t_ =1 _[Q]_[max][1] _[≤][j][≤][n] i[⟨]_ **[q]** _[t][,]_ **[ v]** _[i,j][⟩] ▷_ Eq. (1) 5: **end for** 6: _IK ←_ TopK _K_ ( _{si}[N] i_ =1[)][;] _PK ←{pi | i ∈ IK}_ 7: Initialize _P_[�] _←∅_ , _E ←∅_ 8: **for** each page _pi ∈ PK_ **do** _▷_ Two agents run in parallel _(a) Clue Discovery Agent: clue extraction_ 9: _Ri ←_ ProposeRegions( _pi_ ); _Ei ←∅_ 10: **for** each region **w** _∈Ri_ **do** 11: ( _c, k, r_ ) _←_ ClueDiscovery( _Q, pi,_ **w** ) 12: _e ←_ (page = _i,_ region = **w** _,_ content = _c,_ insight= _k,_ rationale= _r_ ) 13: _Ei ←Ei ∪{e}_ 14: **end for** _(b) Page Screening Agent: visual relevance filtering_ 15: Initialize page-level candidate set _P_[�] _←∅_ 16: ( _yi, ri_[page] ) _←_ PageScreen( _Q, pi_ ) 17:18: **if** _yiP_ � _∈{←_ CR _P_ � _∪{,_ R _}p_ **then** _i}_ 19: **end if** 20: **end for** 21: **return** _P,_[�] _E_ 

## **3.4. Evidence and Difficulty-Aware Decision** 

Different questions require different levels of reasoning. Some queries can be directly answered based on the aggregated evidence, while others demand integration across multiple pages or numerical reasoning. Difficulty Assessment Agent analyzes the query and selects the appropriate model type, providing guiding instructions for subsequent reasoning. Given a question _Q_ and a structured context _C_ , the agent first determines the task difficulty level _d ∈{_ 0 _,_ 1 _}_ and outputs an instruction set Γ _d_ : 

**==> picture [203 x 17] intentionally omitted <==**

where _d_ = 0 indicates a ordinary mode and _d_ = 1 denotes a reasoning mode . The ordinary mode corresponds to general Instruct-type models and is suitable for most basic queries, while the reasoning mode corresponds to Thinking-type models (Some MLLMs have a thinking version or can be enabled with thinking capabilities) capable of multi-step inference across pages and performing numerical computation. The instruction set Γ _d_ summarizes key reasoning cues, such as “ _requires cross-page aggregation_ ”, “ _involves table calculation_ ” or “ _needs trend comparison_ ” which 

**Algorithm 2** Reasoning and Decision-Making Based on Evidence Context and Difficulty Perception 

- **Require:** Question _Q_ ; structured context _C_ = ( _P,_[�] _E_ ); MLLMs Φ 

- **Ensure:** Final answer _A[⋆]_ ; evidence reference table S 

- 1: **Step 1.** _difficulty assessment and instruction extraction._ 

- 2: ( _d,_ Γ _d_ ) _←_ Φassess( _Q, C_ ) _▷_ Single inference: output difficulty _d ∈{_ 0 _,_ 1 _}_ and task instructions Γ _d_ 

- 3: **Step 2.** _Select reasoning mode._ 

- 4: **if** _d_ = 0 **then** 

- 5: Select the Instruct-type model. 

- 6: **else** 

- 7: Select the Thinking-type model. 

- 8: **end if** 

- 9: **Step 3.** _Final reasoning and answer generation._ 

- 10: ( _A[⋆] ,_ S) _←_ Φreason( _Q, C,_ Γ _d_ ) _▷_ Model produces answer and evidence references 

- 11: **return** _A[⋆] ,_ S 

guide the downstream decision-making process. The core reasoning is executed by Core Decision Agent. 

This agent receives the instruction set Γ _d_ and the structured context _C_ from Difficulty Assessment Agent, and then invokes the corresponding type of MLLM to generate the final answer: 

**==> picture [147 x 11] intentionally omitted <==**

where Φ denotes multimodal large language models’ unified reasoning function capable of switching between different backbone models. During answer generation, Core Decision Agent simultaneously produces an evidence reference table S = _{_ page index _,_ evidence content _,_ evidence source _}_ , ensuring verifiability of the reasoning process. 

Through this difficulty-aware model selection mechanism, SLEUTH dynamically adapts to queries of varying complexity while maintaining a trainingfree paradigm, effectively balancing efficiency and reasoning depth. Algorithm 2 illustrates the overall reasoning procedure based on multimodal evidence context and difficulty perception. 

## **4. Experiments** 

## **4.1. Experimental Setup** 

**Datasets and Evaluation Metrics.** Experiments are conducted on four datasets: **MMLongBenchDoc** [19], **LongDocURL** [9], **PaperTab** [13], and **FetaTab** [13]. These benchmarks span multiple domains—including administrative documents, tutorials, and research reports and feature diverse multimodal evidence. They also differ in average page length and information density, making them wellsuited for testing cross-page and cross-modality rea- 

5 

Table 1. Results on MMLongBench-Doc [19]. Accuracy (%, higher is better) across different content types. SLEUTH achieves the highest average value and the highest scores in multiple subtasks. **Bold** denotes the best in each column. 

|**Method**|**Chart**|**Table**|**Pure-text**|**Layout**|**Figure**|**None**|**Avg.**|
|---|---|---|---|---|---|---|---|
|Direct|34.37|37.57|46.13|54.93|37.03|55.61|42.71|
|M3DocRAG [6]|53.94|44.15|54.63|53.52|41.51|48.64|45.90|
|MoLoRAG [39]|**54.85**|46.89|54.86|54.93|46.41|51.67|48.75|
|MDocAgent [12]|52.97|45.98|53.81|**56.39**|45.52|49.71|47.82|
|Base|54.72|44.76|53.33|53.52|44.92|52.68|46.76|
|**SLEUTH (ours)**|53.27|**47.55**|**59.26**|53.52|**50.27**|**67.38**|**52.77**|



Table 2. Results on LongDocURL [9], PaperTab [13] and FetaTab [13]. Accuracy (%, higher is better). SLEUTH achieves the best overall results across all tasks and datasets. **Bold** denotes the best in each column. 

|**Method**<br>M3DocRAG [6]<br>MoLoRAG [39]<br>MDocAgent [12]<br>Base<br>**SLEUTH (ours)**|**LongDocURL**[9]<br>**PaperTab**[13]<br>**FetaTab**[13]<br>**Locating**<br>**Understanding**<br>**Reasoning**<br>**Avg.**|
|---|---|
||45.46<br>60.07<br>52.37<br>54.59<br>39.11<br>64.08<br>49.32<br>63.35<br>52.73<br>57.57<br>42.59<br>69.41<br>44.74<br>57.46<br>51.46<br>53.11<br>39.86<br>66.55<br>46.04<br>61.56<br>51.09<br>55.18<br>38.88<br>64.16|
||**53.63**<br>**65.67**<br>**52.99**<br>**59.96**<br>**43.09**<br>**70.46**|



soning capabilities. To avoid confusion with shortdocument settings, we exclude short-context benchmarks such as DocVQA [21] and ChartQA [20]. Statistics on page counts and dataset scales are provided in the **Appendix A** . Following the evaluation protocols of Ma _et al._ [19] and Deng _et al._ [9], we extract short answers with DeepSeek-V3.2-Exp [8] from the model outputs and report answer accuracy with generalized accuracy (based on a rule-based evaluation script covering different answer types). 

**SLEUTH Configuration.** SLEUTH consists of four agents: Clue Discovery Agent, Page Screening Agent, Difficulty Assessment Agent, and Core Decision Agent. By default, we employ Qwen3VL-8B as the vision–language backbone and ColPali-v1.3 [11] as the page-level visual retriever. Unless otherwise stated, the retriever returns the Top-5 pages as input to SLEUTH. The prompts that drive each agent are provided in the **Appendix F** . 

**Baselines.** For fairness, unless otherwise noted, all methods adopt Qwen3VL-8B as the VLM backbone and use the same retriever, ColPali-v1.3 [11]. We retrieve the Top-5 pages according to retrieval scores and set the temperature to 0.1. The compared baselines include: (1) VLM Direct Inference: directly inputs the entire document snapshot into the VLM for QA; (2) Base Inference: employs ColPali retrieval and feeds the retrieved pages into the VLM; (3) strong RAG-based methods, including M3DocRAG [6] and MoLoRAG [39]; (4) MDocAgent [12]: a multi-agent baseline using separate text and image pipelines with Qwen3-8B [40] and Qwen3VL-8B; text and visual retrievers are BGE M3 [3] and ColPali-v1.3 [11]. We also evaluated the performance of commercial closedsource models such as GPT-5 [24] when directly inputting long documents. We also evaluated the per- 

formance of several commercial closed-source models when directly processing long documents. 

## **4.2. Overall Performance** 

**Quantitative Results on MMLongBench.** Table 1 presents SLEUTH’s results on MMLongBench-Doc grouped by evidence type. Among them, “None” indicates the type of questions that cannot be answered. SLEUTH achieves an overall accuracy of 52.77%, surpassing the best retrieval-based baseline MoLoRAG (48.75%) by +4.02% absolute points. Compared with M3DocRAG, MDocAgent, MoLoRAG, Base, and Direct, SLEUTH improves by +6.87%, +4.02%, +4.95%, +6.01%, and +10.06%, respectively. From a category-wise perspective, SLEUTH attains the best performance in Table, Pure-text, Figure, and None categories, outperforming the second-best method. Compared with directly feeding the entire document, SLEUTH synthesizes evidence through an iterative “ _page-by-page, evidence-recording, and pagescreening_ ” short-context process. This effectively suppresses misleading signals and hallucinations caused by redundant long contexts, with significant gains observed in the Pure-text and Figure categories. This indicates that the method for acquiring evidence-dense context is more robust and generalizable. 

**Quantitative Results on LongDocURL, PaperTab, and FetaTab.** As shown in Table 2, SLEUTH consistently outperforms all baselines on LongDocURL, PaperTab, and FetaTab. This implies that our multimodal evidence context engineering design can achieve cross-task generalization. 

**Quantitative Results of Comparison with Commercial Models.** As depicted in Figure. 3, SLEUTH still maintains a clear advantage over closed source commercial models, validating the insight that high- 

6 

Table 3. Ablation of SLEUTH variants across different base models on MMLongBench-Doc [19] and LongDocURL [9]. Accuracy (%, higher is better). We progressively enable each agent, Clue Discovery Agent (C), Page Screening Agent (P), and Difficulty Assessment Agent (D)—and vary the retriever Top- _K_ (1/3/5). Performance improves consistently as the agent system becomes more complete, with Top-5 achieving the best overall average. **Bold** indicates the best average (Avg.). 

|**Model**<br>**Variants**|**Agent Confguration**<br>**Clue.**<br>**Page.**<br>**Diff.**|**MMLongBench-Doc**<br>**Chart**<br>**Table**<br>**Pure-text**<br>**Layout**<br>**Figure**<br>**None**<br>**Avg.**|**LongDocURL**<br>**Loc.**<br>**Und.**<br>**Reas.**<br>**Avg.**|
|---|---|---|---|
|**Qwen3-VL-8B**<br>Base<br>SLEUTH (C)<br>SLEUTH (P)<br>SLEUTH (Top1)<br>SLEUTH (Top3)|–<br>–<br>–<br>✓<br>✗<br>✗<br>✓<br>✓<br>✗<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓|**54.72**<br>44_._76<br>53_._33<br>53_._52<br>44_._92<br>52_._68<br>46_._76<br>45_._28<br>46_._15<br>54_._81<br>**56.39**<br>48_._02<br>**69.23**<br>48_._61<br>51_._40<br>46_._15<br>55_._56<br>50_._70<br>50_._27<br>67_._38<br>51_._29<br>41_._46<br>40_._41<br>49_._33<br>45_._07<br>38_._24<br>74_._25<br>44_._92<br>49_._21<br>43_._49<br>55_._08<br>49_._25<br>49_._94<br>67_._38<br>49_._65|46_._04<br>61_._56<br>51_._09<br>55_._18<br>49_._03<br>62_._05<br>54_._27<br>57_._15<br>53_._21<br>65_._35<br>52_._21<br>59_._49<br>52_._69<br>58_._03<br>36_._69<br>52_._88<br>**57.48**<br>62_._64<br>49_._14<br>58_._38|
|SLEUTH (Top5)|✓<br>✓<br>✓|53_._27<br>**47.55**<br>**59.26**<br>53_._52<br>**50.27**<br>67_._38<br>**52.77**|53_._63<br>**65.67**<br>**52.99**<br>**59.96**|
|**GLM-4.1V-**<br>**Thinking-8B**<br>Base<br>SLEUTH (C)<br>SLEUTH (Top1)<br>SLEUTH (Top3)|–<br>–<br>–<br>✓<br>✗<br>✗<br>✓<br>✓<br>–<br>✓<br>✓<br>–|51_._01<br>39_._07<br>46_._67<br>49_._25<br>41_._37<br>94_._82<br>53_._05<br>45_._85<br>41_._88<br>49_._52<br>53_._52<br>42_._27<br>**95.12**<br>53_._22<br>42_._62<br>32_._45<br>48_._59<br>50_._70<br>46_._79<br>84_._08<br>49_._38<br>50_._65<br>39_._93<br>57_._26<br>54_._93<br>**47.38**<br>79_._10<br>54_._34|48_._03<br>64_._40<br>54_._04<br>57_._75<br>51_._24<br>63_._36<br>53_._37<br>58_._08<br>37_._61<br>57_._98<br>48_._66<br>50_._34<br>45_._19<br>64_._64<br>**55.40**<br>57_._29|
|SLEUTH (Top5)|✓<br>✓<br>–|**52.38**<br>**44.77**<br>**57.34**<br>**59.15**<br>47_._29<br>94_._62<br>**57.93**|**54.48**<br>**68.64**<br>54_._30<br>**62.02**|
|**Gemini-**<br>**2.5-Flash**<br>Base<br>SLEUTH (C)<br>SLEUTH (Top1)<br>SLEUTH (Top3)|–<br>–<br>–<br>✓<br>✗<br>✗<br>✓<br>✓<br>–<br>✓<br>✓<br>–|49_._09<br>41_._32<br>45_._34<br>54_._93<br>40_._67<br>61_._43<br>46_._37<br>45_._11<br>39_._78<br>44_._39<br>56_._39<br>36_._08<br>86_._10<br>49_._35<br>45_._55<br>40_._87<br>46_._23<br>49_._25<br>36_._78<br>76_._23<br>47_._01<br>50_._23<br>44_._62<br>**47.06**<br>59_._15<br>41_._45<br>**80.27**<br>51_._49|41_._36<br>60_._95<br>54_._25<br>54_._01<br>45_._58<br>64_._41<br>55_._34<br>57_._27<br>42_._02<br>56_._19<br>46_._99<br>49_._77<br>44_._38<br>60_._67<br>53_._93<br>54_._04|
|SLEUTH (Top5)|✓<br>✓<br>–|**51.08**<br>**44.78**<br>46_._51<br>**61.97**<br>**42.55**<br>77_._13<br>**51.86**|**49.48**<br>**66.52**<br>**61.51**<br>**60.62**|



Table 4. Comparison between Multimodal and Visual Retrieval Input on MMLongBench-Doc [19] and LongDocURL [9]. All numbers are accuracy (%). **Bold** indicates the better score for each benchmark. 

|**MMLongBench-Doc**[19]<br>**Method**<br>**Chart**<br>**Table**<br>**Pure-text**<br>**Layout**<br>**Figure**<br>**None**<br>**Avg**|**LongDocURL**[9]<br>**Locating**<br>**Understanding**<br>**Reasoning**<br>**Avg.**|
|---|---|
|Multimodal Retrieval Input<br>45.28<br>42.96<br>52.59<br>**59.15**<br>48.02<br>**71.67**<br>50.19<br>Visual Retrieval Input<br>**53.27**<br>**47.55**<br>**59.26**<br>53.52<br>**50.27**<br>67.38<br>**52.77**|50.94<br>62.15<br>**56.56**<br>57.62<br>**53.63**<br>**65.67**<br>52.99<br>**59.96**|



**==> picture [215 x 84] intentionally omitted <==**

**----- Start of picture text -----**<br>
60 Performance Comparison on MMLongBench Benchmark<br>50 51.2 48.1 50.1 52.8<br>42.4<br>40<br>30<br>20<br>10<br>0<br>Gemini 2.5 Pro GPT-5 Claude Opus 4.1 Seed1.5VL SLEUTH<br>(T:128) (w/o thinking) (w/o thinking) (w/o thinking) (Qwen3VL-8B)<br>Models<br>Score<br>**----- End of picture text -----**<br>


Figure 3. Performance of SLEUTH compared with closedsource commercial models such as Gemini 2.5 pro [7], GPT5 [24] on MMLongBench-Doc [19]. 

quality evidence context is more critical than merely increasing model size. Detailed experimental analysis is provided in the **Appendix B** . 

## **4.3. Ablation Studies** 

In Table 3, we conduct comprehensive ablation experiments on two datasets by removing agents one by one, replacing the base models of agents, and setting different retrieval K values, in order to evaluate the effectiveness and generality of all designs in SLEUTH. **Necessity of Framework Structure.** When any agent is removed, the performance decreases, and the drop is notable when removing the Clue Discovery Agent and the Page Screening Agent. This highlights the importance of constructing concise, highly effective, and information-dense contexts. The integration of context-construction agents plays a crucial role in 

combating hallucinations and enhancing fine-grained perception capabilities. Taking Qwen3-VL-8B as an example: enabling only the Clue Discovery Agent for evidence recording without any visual input yields an average gain of +1.85%. When the Page Screening Agent is added for visual filtering, the improvement increases to +4.53%. By integrating both agents, we achieve a complementarity between evidence mining and visual information, which yields performance boosts. After incorporating difficulty assessment, the Top-5 aggregated accuracy reaches 52.77%, achieving an overall gain of +6.01% over the Base method. 

**The generality of the framework.** Consistent improvements are also observed on GLM-4.1VThinking-9B [27] and Gemini-2.5-Flash [7], indicating that SLEUTH is backbone-agnostic and transferable, and each agent contributes to the performance. **K Value Ablation.** The retrieval Top- _K_ setting exhibits a monotonic performance increase. The context length input by our context construction agent is fixed, so the severity of hallucinations will not increase with the growth of the total amount of retrieved content. Moreover, as the K increases, the recall rate rises, and the performance improves steadily. 

SLEUTH adopts a context-engineering paradigm of “retrieval narrowing _→_ clue recording _→_ visual screening _→_ difficulty-adaptive reasoning”, achieving consistent gains across multiple benchmarks. Abla- 

7 

**==> picture [445 x 171] intentionally omitted <==**

**----- Start of picture text -----**<br>
Redundant context brings excessive noise, confusing VLMs.1 2  3 4 5 |SSSceaaaeaaeeiaaiaaaeemaeamminiaa Ours ：Redundancy-Reduced, Evidence-Dense Context for Response Enhancement<br>Clue Discovery Agent ‘ea Page Screening Agent iieieeiniiiaieianiaiie“ek |<br>Input： The table is structured as:Task, Sub Task, Dataset, Method. Let's list all...... Find key visual clues：<br>Top 5  1. Natural Q：...Let's count: 20 items.❌ 5 Discard<br>pages 2. ......23 numbers.❌ irrelevant<br>5. PopQA Method column has three green boxes......  pages<br>has Method numbers [7, 25, 67]......<br>=| Beet fs eee issiakee oy RL<br>So three methods.<br>Qexactly three methods?：According to table II, which are the datasets that has  f = 6. MS MARCO Method column... 11.NQA: 45, 60, 63, 123 → 4 numbers→ count 3.❌....... 1 ‘. zeae Eee - , [1, 2, 3, 4]<br>GT：['PopQA', 'MS MARCO', 'SST-2'] 43. SST-2 Method: 20, 33, 38. So 3 numbers...<br>Key Insights:"...,  only these [5，6，43] three<br>Base Method ：<br>|| Answermethods....NarrativeQA (NQA) [122]: Methods listed are ：PopQA: Methods listed are [7, 25, 67]. This is 3  datasets meet the exact count of three methods... " Evidence context Visual context<br>[45, 60, 123]. This is 3 methods... fF } Gy GS Ws!<br>['PopQA', 'MS MARCO', 'NarrativeQA (NQA)'] ° Wig, Core Decision Agent SES SES SEESSE |<br>oocccceeeecceceeeeee Too much noise from redundant information makes  1 Answer:Based on the above evidence...... ['PopQA', 'MS MARCO', 'SST-2'] |<br>A VLMs hallucinate, missing key information. (Sa ee)<br>**----- End of picture text -----**<br>


Figure 4. Case study. Compared with basic methods that rely solely on direct input of the top-5 retrieved pages, SLEUTH performs dynamic correction through multi-step evidence recording and page-wise filtering, effectively preventing hallucination accumulation caused by multimodal long-context inputs with complex layouts inputs. 

tions and multi-backbone evaluations demonstrate its universality and flexibility. 

**Visual** _**vs.**_ **Multimodal Retrieval Input.** To verify whether purely visual page evidence is more advantageous for our architecture, we modified the architecture to support two parallel Clue Discovery paths. We employed one agent to process the Top-5 visual pages retrieved by ColPali. Simultaneously, we deployed a second agent to process the Top-5 text pages. We obtained these text pages by extracting content via MinerU2.5 [23] and retrieving them with BGE M3. Both agents extracted evidence using the same prompt logic. Finally, we merged the evidence sets from both the visual and textual agents for the subsequent reasoning process. As shown in Table 4, using purely visual page inputs yields higher average accuracy on both benchmarks, outperforming the multimodal input setting by +2.58% and +2.34% points, respectively. This result supports our hypothesis that visual pages serve as a highly compressed and structure-preserving unified representation. It also aligns with recent works such as DeepSeek-OCR [37], which advocate for unified visual input as an effective means of compressing long contexts. More analysis of the ablation experiments can be found in the **Appendix C** . 

## **4.4. Qualitative Analysis** 

Figure 4 illustrates how SLEUTH alleviates hallucination caused by redundant long-document inputs. When asked questions, the base method directly feeds all retrieved pages to the VLM, where excessive numerical noise confuses the model and yields incorrect answers. In contrast, SLEUTH builds evidence page-wise with self-corrective reasoning: the Clue Discovery Agent reads each retrieved page holistically and writes structured clues; the Page Screening Agent prunes visually irrelevant pages; and the Core Decision Agent fuses the clues to produce the fi- 

nal right answer. The collaborative cooperation among the various agents in this process achieves redundancy elimination, preserves key clues, and overcomes hallucination interference. This example highlights how SLEUTH’s stepwise evidence recording and self-corrective reasoning enables precise answers even in dense, noisy long document environments. 

## **5. Conclusion** 

This paper presents SLEUTH, a framework that reexamines long-document question answering from the perspective of context engineering. Without modifying model architectures, SLEUTH organizes contextual inputs to improve the utilization of long contexts and maintain stable reasoning. Through a coarseto-fine process that narrows the search space and constructs evidence-focused multimodal contexts, it achieves better document understanding while preserving interpretability. Experiments show consistent gains across benchmarks, demonstrating the effectiveness of context engineering for long-document tasks. 

Despite promising results, we identify two primary limitations. First, the framework relies heavily on the initial retriever. If critical pages are missed early on, downstream agents cannot recover them. Second, the purely prompt-driven design limits the system’s ability to self-evolve through experience. To address these issues, our future work will focus on three directions. First, we will introduce feedback-based retrieval to reduce dependency on a single step. Second, we will integrate Reinforcement Learning and external toolkits for autonomous optimization. Furthermore, we will integrate existing methods for enhancing reasoning capabilities with techniques for improving retrieval precision to further advance the document understanding capabilities of AI. Third, we will extend the framework to support multilingual and handwritten documents to verify its broader generalization capabilities. 

8 

## **References** 

- [1] Anthropic. Claude sonnet 4, 2025. 1, 2 

- [2] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5vl technical report. _arXiv preprint arXiv:2502.13923_ , 2025. 1, 2 

- [3] Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. Bge m3-embedding: Multilingual, multi-functionality, multi-granularity text embeddings through self-knowledge distillation. _arXiv preprint arXiv:2402.03216_ , 2024. 3, 6, 4 

- [4] Jian Chen, Ruiyi Zhang, Yufan Zhou, Tong Yu, Franck Dernoncourt, Jiuxiang Gu, Ryan A. Rossi, Changyou Chen, and Tong Sun. SV-RAG: LoRA-contextualizing adaptation of MLLMs for long document understanding. In _The Thirteenth International Conference on Learning Representations_ , 2025. 1, 3 

- [5] Yew Ken Chia, Liying Cheng, Hou Pong Chan, Maojia Song, Chaoqun Liu, Mahani Aljunied, Soujanya Poria, and Lidong Bing. M-longdoc: A benchmark for multimodal super-long document understanding and a retrieval-aware tuning framework. In _Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing_ , pages 9244–9261, 2025. 1, 2 

- [6] Jaemin Cho, Debanjan Mahata, Ozan Irsoy, Yujie He, and Mohit Bansal. M3docrag: Multi-modal retrieval is what you need for multi-page multi-document understanding. _arXiv preprint arXiv:2411.04952_ , 2024. 1, 2, 3, 6 

- [7] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. _arXiv preprint arXiv:2507.06261_ , 2025. 2, 7, 3 

- [8] DeepSeek-AI. Deepseek-v3.2-exp: Boosting longcontext efficiency with deepseek sparse attention, 2025. 6 

- [9] Chao Deng, Jiale Yuan, Pi Bu, Peijie Wang, ZhongZhi Li, Jian Xu, Xiao-Hui Li, Yuan Gao, Jun Song, Bo Zheng, and Cheng-Lin Liu. LongDocURL: a comprehensive multimodal long document benchmark integrating understanding, reasoning, and locating. In _Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 1135–1159, Vienna, Austria, 2025. Association for Computational Linguistics. 1, 2, 4, 5, 6, 7, 3 

- [10] Yuchen Duan, Zhe Chen, Yusong Hu, Weiyun Wang, Shenglong Ye, Botian Shi, Lewei Lu, Qibin Hou, Tong Lu, Hongsheng Li, et al. Docopilot: Improving multimodal models for document-level understanding. In _Proceedings of the Computer Vision and Pattern Recognition Conference_ , pages 4026–4037, 2025. 1, 2 

- [11] Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani, Gautier Viaud, C´eline Hudelot, and Pierre 

Colombo. Colpali: Efficient document retrieval with vision language models. _arXiv preprint arXiv:2407.01449_ , 2024. 1, 3, 6, 4 

- [12] Siwei Han, Peng Xia, Ruiyi Zhang, Tong Sun, Yun Li, Hongtu Zhu, and Huaxiu Yao. Mdocagent: A multi-modal multi-agent framework for document understanding. _arXiv preprint arXiv:2503.13964_ , 2025. 2, 3, 6 

- [13] Yulong Hui, Yao Lu, and Huanchen Zhang. Uda: A benchmark suite for retrieval augmented generation in real-world document analysis. _Advances in Neural Information Processing Systems_ , 37:67200–67217, 2024. 2, 5, 6, 1 

- [14] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. _arXiv preprint arXiv:2410.21276_ , 2024. 1, 2 

- [15] Omar Khattab and Matei Zaharia. Colbert: Efficient and effective passage search via contextualized late interaction over bert. In _Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval_ , pages 39–48, 2020. 3 

- [16] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. _arXiv preprint arXiv:2408.03326_ , 2024. 1 

- [17] Lin Lin, Jiefeng Long, Zhihe Wan, Yuchi Wang, Dingkang Yang, Shuang Yang, Yueyang Yao, Xu Chen, Zirui Guo, Shengqiang Li, et al. Sail-embedding technical report: Omni-modal embedding foundation model. _arXiv preprint arXiv:2510.12709_ , 2025. 2 

- [18] Keliang Liu, Dingkang Yang, Ziyun Qian, Weijie Yin, Yuchi Wang, Hongsheng Li, Jun Liu, Peng Zhai, Yang Liu, and Lihua Zhang. Reinforcement learning meets large language models: A survey of advancements and applications across the llm lifecycle. _arXiv preprint arXiv:2509.16679_ , 2025. 2 

- [19] Yubo Ma, Yuhang Zang, Liangyu Chen, Meiqi Chen, Yizhu Jiao, Xinze Li, Xinyuan Lu, Ziyu Liu, Yan Ma, Xiaoyi Dong, et al. Mmlongbench-doc: Benchmarking long-context document understanding with visualizations. _Advances in Neural Information Processing Systems_ , 37:95963–96010, 2024. 1, 2, 3, 5, 6, 7 

- [20] Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. _arXiv preprint arXiv:2203.10244_ , 2022. 1, 2, 6 

- [21] Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. Docvqa: A dataset for vqa on document images. In _Proceedings of the IEEE/CVF winter conference on applications of computer vision_ , pages 2200–2209, 2021. 2, 6 

- [22] Minesh Mathew, Viraj Bagal, Rub`en Tito, Dimosthenis Karatzas, Ernest Valveny, and CV Jawahar. Infographicvqa. In _Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision_ , pages 1697–1706, 2022. 1, 2 

- [23] Junbo Niu, Zheng Liu, Zhuangcheng Gu, Bin Wang, Linke Ouyang, Zhiyuan Zhao, Tao Chu, Tianyao 

9 

He, Fan Wu, Qintong Zhang, et al. Mineru2. 5: A decoupled vision-language model for efficient high-resolution document parsing. _arXiv preprint arXiv:2509.22186_ , 2025. 8, 4 

- [24] OpenAI. Gpt-5 system card. https://openai. com/index/gpt-5-system-card/, 2025. Accessed: 2025-11-26. 6, 7 

- [25] Yaorui Shi, Yuxin Chen, Siyuan Wang, Sihang Li, Hengxing Cai, Qi Gu, Xiang Wang, and An Zhang. Look back to reason forward: Revisitable memory for long-context llm agents. _arXiv preprint arXiv:2509.23040_ , 2025. 3 

- [26] Weiwei Sun, Miao Lu, Zhan Ling, Kang Liu, Xuesong Yao, Yiming Yang, and Jiecao Chen. Scaling longhorizon llm agent via context-folding. _arXiv preprint arXiv:2510.11967_ , 2025. 3 

- [27] V Team, Wenyi Hong, Wenmeng Yu, Xiaotao Gu, Guo Wang, Guobing Gan, Haomiao Tang, Jiale Cheng, Ji Qi, Junhui Ji, Lihang Pan, Shuaiqi Duan, Weihan Wang, Yan Wang, Yean Cheng, Zehai He, Zhe Su, Zhen Yang, Ziyang Pan, Aohan Zeng, Baoxu Wang, Bin Chen, Boyan Shi, Changyu Pang, Chenhui Zhang, Da Yin, Fan Yang, Guoqing Chen, Jiazheng Xu, Jiale Zhu, Jiali Chen, Jing Chen, Jinhao Chen, Jinghao Lin, Jinjiang Wang, Junjie Chen, Leqi Lei, Letian Gong, Leyi Pan, Mingdao Liu, Mingde Xu, Mingzhi Zhang, Qinkai Zheng, Sheng Yang, Shi Zhong, Shiyu Huang, Shuyuan Zhao, Siyan Xue, Shangqin Tu, Shengbiao Meng, Tianshu Zhang, Tianwei Luo, Tianxiang Hao, Tianyu Tong, Wenkai Li, Wei Jia, Xiao Liu, Xiaohan Zhang, Xin Lyu, Xinyue Fan, Xuancheng Huang, Yanling Wang, Yadong Xue, Yanfeng Wang, Yanzi Wang, Yifan An, Yifan Du, Yiming Shi, Yiheng Huang, Yilin Niu, Yuan Wang, Yuanchang Yue, Yuchen Li, Yutao Zhang, Yuting Wang, Yu Wang, Yuxuan Zhang, Zhao Xue, Zhenyu Hou, Zhengxiao Du, Zihan Wang, Peng Zhang, Debing Liu, Bin Xu, Juanzi Li, Minlie Huang, Yuxiao Dong, and Jie Tang. Glm-4.5v and glm-4.1vthinking: Towards versatile multimodal reasoning with scalable reinforcement learning, 2025. 1, 2, 7, 3 

- [28] Jordy Van Landeghem, Rub`en Tito, Łukasz Borchmann, Michał Pietruszka, Pawel Joziak, Rafal Powalski, Dawid Jurkiewicz, Micka¨el Coustaty, Bertrand Anckaert, Ernest Valveny, et al. Document understanding dataset and evaluation (dude). In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 19528–19540, 2023. 2 

- [29] Bin Wang, Chao Xu, Xiaomeng Zhao, Linke Ouyang, Fan Wu, Zhiyuan Zhao, Rui Xu, Kaiwen Liu, Yuan Qu, Fukai Shang, et al. Mineru: An open-source solution for precise document content extraction. _arXiv preprint arXiv:2409.18839_ , 2024. 1 

- [30] Dongsheng Wang, Natraj Raman, Mathieu Sibue, Zhiqiang Ma, Petr Babkin, Simerjot Kaur, Yulong Pei, Armineh Nourbakhsh, and Xiaomo Liu. DocLLM: A layout-aware generative language model for multimodal document understanding. In _Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 8529–8548, Bangkok, Thailand, 2024. Association for Computational Linguistics. 1, 3 

- [31] Hengyi Wang, Haizhou Shi, Shiwei Tan, Weiyi Qin, Wenyuan Wang, Tunyu Zhang, Akshay Nambi, Tanuja 

   - Ganu, and Hao Wang. Multimodal needle in a haystack: Benchmarking long-context capability of multimodal large language models. In _Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)_ , pages 3221–3241, 2025. 1, 3 

- [32] Qiuchen Wang, Ruixue Ding, Zehui Chen, Weiqi Wu, Shihang Wang, Pengjun Xie, and Feng Zhao. Vidorag: Visual document retrieval-augmented generation via dynamic iterative reasoning agents. _arXiv preprint arXiv:2502.18017_ , 2025. 1, 3 

- [33] Qiuchen Wang, Ruixue Ding, Yu Zeng, Zehui Chen, Lin Chen, Shihang Wang, Pengjun Xie, Fei Huang, and Feng Zhao. Vrag-rl: Empower vision-perceptionbased rag for visually rich information understanding via iterative reasoning with reinforcement learning. _arXiv preprint arXiv:2505.22019_ , 2025. 3, 2 

- [34] Suyuchen Wang, Jinlin Wang, Xinyu Wang, Shiqi Li, Xiangru Tang, Sirui Hong, Xiao-Wen Chang, Chenglin Wu, and Bang Liu. Improving context fidelity via native retrieval-augmented reasoning. _arXiv preprint arXiv:2509.13683_ , 2025. 3 

- [35] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. Internvl3.5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. _arXiv preprint arXiv:2508.18265_ , 2025. 1, 2 

- [36] Haoran Wei, Chenglong Liu, Jinyue Chen, Jia Wang, Lingyu Kong, Yanming Xu, Zheng Ge, Liang Zhao, Jianjian Sun, Yuang Peng, et al. General ocr theory: Towards ocr-2.0 via a unified end-to-end model. _arXiv preprint arXiv:2409.01704_ , 2024. 1 

- [37] Haoran Wei, Yaofeng Sun, and Yukun Li. Deepseekocr: Contexts optical compression, 2025. 2, 8 

- [38] Junda Wu, Yu Xia, Tong Yu, Xiang Chen, Sai Sree Harsha, Akash V Maharaj, Ruiyi Zhang, Victor Bursztyn, Sungchul Kim, Ryan A Rossi, et al. Docreact: Multi-page heterogeneous document questionanswering. In _Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)_ , pages 67–78, 2025. 3 

- [39] Xixi Wu, Yanchao Tan, Nan Hou, Ruiyang Zhang, and Hong Cheng. Molorag: Bootstrapping document understanding via multi-modal logic-aware retrieval. In _The 2025 Conference on Empirical Methods in Natural Language Processing_ , 2025. 1, 3, 6, 2 

- [40] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. _arXiv preprint arXiv:2505.09388_ , 2025. 6 

- [41] Dingkang Yang, Shuai Huang, Haopeng Kuang, Yangtao Du, and Lihua Zhang. Disentangled representation learning for multimodal emotion recognition. In _Proceedings of the 30th ACM International Conference on Multimedia (ACM MM)_ , pages 1642–1651, 2022. 2 

- [42] Dingkang Yang, Shuai Huang, Shunli Wang, Yang Liu, Peng Zhai, Liuzhen Su, Mingcheng Li, and Lihua Zhang. Emotion recognition for multiple context awareness. In _Proceedings of the European Conference on Computer Vision (ECCV)_ , pages 144–162, 2022. 

10 

- [43] Dingkang Yang, Haopeng Kuang, Shuai Huang, and Lihua Zhang. Learning modality-specific and - agnostic representations for asynchronous multimodal language sequences. In _Proceedings of the 30th ACM International Conference on Multimedia (ACM MM)_ , pages 1708–1717, 2022. 

- [44] Dingkang Yang, Zhaoyu Chen, Yuzheng Wang, Shunli Wang, Mingcheng Li, Siao Liu, Xiao Zhao, Shuai Huang, Zhiyan Dong, Peng Zhai, and Lihua Zhang. Context de-confounded emotion recognition. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , pages 19005– 19015, 2023. 

- [45] Dingkang Yang, Yang Liu, Can Huang, Mingcheng Li, Xiao Zhao, Yuzheng Wang, Kun Yang, Yan Wang, Peng Zhai, and Lihua Zhang. Target and source modality co-reinforcement for emotion understanding from asynchronous multimodal sequences. _KnowledgeBased Systems_ , 265:110370, 2023. 

- [46] Dingkang Yang, Mingcheng Li, Linhao Qu, Kun Yang, Peng Zhai, Song Wang, and Lihua Zhang. Asynchronous multimodal video sequence fusion via learning modality-exclusive and-agnostic representations. _IEEE Transactions on Circuits and Systems for Video Technology_ , 2024. 

- [47] Dingkang Yang, Mingcheng Li, Dongling Xiao, Yang Liu, Kun Yang, Zhaoyu Chen, Yuzheng Wang, Peng Zhai, Ke Li, and Lihua Zhang. Towards multimodal sentiment analysis debiasing via bias purification. In _Proceedings of the European Conference on Computer Vision (ECCV)_ , 2024. 

- [48] Dingkang Yang, Jinjie Wei, Dongling Xiao, Shunli Wang, Tong Wu, Gang Li, Mingcheng Li, Shuaibing Wang, Jiawei Chen, Yue Jiang, et al. Pediatricsgpt: Large language models as chinese medical assistants for pediatric applications. _Advances in Neural Information Processing Systems_ , 37:138632–138662, 2024. 

      - Zhang. Improving factuality in large language models via decoding-time hallucinatory and truthful comparators. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , pages 25606–25614, 2025. 2 

   - [54] Xinlei Yu, Zhangquan Chen, Yudong Zhang, Shilin Lu, Ruolin Shen, Jiangning Zhang, Xiaobin Hu, Yanwei Fu, and Shuicheng Yan. Visual document understanding and question answering: A multi-agent collaboration framework with test-time scaling. _arXiv preprint arXiv:2508.03404_ , 2025. 1, 3, 2 

   - [55] Qizheng Zhang, Changran Hu, Shubhangi Upasani, Boyuan Ma, Fenglu Hong, Vamsidhar Kamanuru, Jay Rainton, Chen Wu, Mengmeng Ji, Hanchen Li, et al. Agentic context engineering: Evolving contexts for self-improving language models. _arXiv preprint arXiv:2510.04618_ , 2025. 3 

   - [56] Yusen Zhang, Ruoxi Sun, Yanfei Chen, Tomas Pfister, Rui Zhang, and Sercan Arik. Chain of agents: Large language models collaborating on long-context tasks. _Advances in Neural Information Processing Systems_ , 37:132208–132237, 2024. 3 

   - [57] Yucheng Zhou, Zhi Rao, Jun Wan, and Jianbing Shen. Rethinking visual dependency in long-context reasoning for large vision-language models. _arXiv preprint arXiv:2410.19732_ , 2024. 1, 3 

   - [58] Fengbin Zhu, Wenqiang Lei, Fuli Feng, Chao Wang, Haozhou Zhang, and Tat-Seng Chua. Towards complex document understanding by discrete reasoning. In _Proceedings of the 30th ACM International Conference on Multimedia_ , pages 4857–4866, 2022. 1, 2 

   - [59] Yufan Zhuang, Xiaodong Yu, Jialian Wu, Ximeng Sun, Ze Wang, Jiang Liu, Yusheng Su, Jingbo Shang, Zicheng Liu, and Emad Barsoum. Self-taught agentic long context understanding. _arXiv preprint arXiv:2502.15920_ , 2025. 1, 3 

- [49] Dingkang Yang, Kun Yang, Haopeng Kuang, Zhaoyu Chen, Yuzheng Wang, and Lihua Zhang. Towards context-aware emotion recognition debiasing from a causal demystification perspective via de-confounded training. _IEEE Transactions on Pattern Analysis and Machine Intelligence_ , 2024. 

- [50] Dingkang Yang, Kun Yang, Mingcheng Li, Shunli Wang, Shuaibing Wang, and Lihua Zhang. Robust emotion recognition in context debiasing. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , pages 12447–12457, 2024. 

- [51] Dingkang Yang, Mingcheng Li, Xuecheng Wu, Zhaoyu Chen, Kaixun Jiang, Keliang Liu, Peng Zhai, and Lihua Zhang. Improving multimodal sentiment analysis via modality optimization and dynamic primary modality selection. _arXiv preprint arXiv:2511.06328_ , 2025. 

- [52] Dingkang Yang, Jinjie Wei, Mingcheng Li, Jiyao Liu, Lihao Liu, Ming Hu, Junjun He, Yakun Ju, Wei Zhou, Yang Liu, et al. Medaide: Information fusion and anatomy of medical intents via llm-based agent collaboration. _Information Fusion_ , page 103743, 2025. 

- [53] Dingkang Yang, Dongling Xiao, Jinjie Wei, Mingcheng Li, Zhaoyu Chen, Ke Li, and Lihua 

11 

## **Resolving Evidence Sparsity: Agentic Context Engineering for Long-Document Understanding** 

## Supplementary Material 

## **Appendix Contents** 

|**1. Introduction**|**1**|
|---|---|
|**2. Related Work**|**2**|
|**3. Methodology**|**3**|
|3.1. Overall Framework . . . . . . . . . . .|3|
|3.2. Coarse-grained Visual Retrieval . . . .|3|
|3.3.<br>Evidence<br>Refnement<br>and<br>Visual||
|Screening . . . . . . . . . . . . . . .|3|
|3.4. Evidence and Diffculty-Aware Decision|5|
|**4. Experiments**|**5**|
|4.1. Experimental Setup. . . . . . . . . . .|5|
|4.2. Overall Performance . . . . . . . . . .|6|
|4.3. Ablation Studies . . . . . . . . . . . .|7|
|4.4. Qualitative Analysis . . . . . . . . . .|8|
|**5. Conclusion**|**8**|
|**A. Evaluation Benchmarks**|**1**|
|**B. Overall Performance**|**1**|
|B.1. Analysis of Experimental Results . . .|1|
|B.2. Additional Comparative Experiments .|2|
|**C. Ablation Studies**|**2**|
|C.1. Analysis of Ablation Experiment Results|2|
|C.2. Visual vs. Multimodal Retrieval Input .|4|
|**D. Innovations and Contributions**|**4**|
|**E. Limitations and Future Work**|**4**|
|**F. The Prompt design of SLEUTH**|**5**|



## **A. Evaluation Benchmarks** 

The statistics of the datasets are listed in Table 1. These datasets cover various topics, such as administrative documents, tutorials, and research reports. They also include diverse multimodal components like charts, texts, and tables. Moreover, their average document length and information density differ, providing a broad and balanced evaluation. 

MMLongBench-Doc [19] is a comprehensive benchmark for evaluating the long-context document understanding abilities of large vision-language models (LVLMs). Built upon 135 lengthy documents averaging 47.5 pages and over 21,000 tokens, it contains 1,082 expert-annotated questions that require reasoning across text, layout, charts, tables, and images. The benchmark includes 33.7% cross-page and 

Table 5. **Statistics of datasets used in our experiments.** 

|**Dataset**|**Question**|**Document**|**Avg. Pages**|**Avg. Tokens**|
|---|---|---|---|---|
|PaperTab [13]|393|307|11.0|12,685.4|
|FetaTab [13]|1,016|871|15.8|16,524.5|
|MMLongBench [19]|1,082|135|47.5|24,992.6|
|LongDocURL [9]|2,325|396|85.6|56,715.1|



20.6% unanswerable questions, assessing localization, cross-page comprehension, and hallucination resistance. Through rigorous annotation and quality control, MMLongBench-Doc provides a challenging, high-quality testbed for advancing multimodal longdocument understanding in LVLMs. 

LongDocURL [9] is a comprehensive benchmark designed for evaluating long document understanding in large vision-language models (LVLMs). It integrates three major task categories—Understanding, Reasoning, and Cross-Element Locating—across 20 sub-tasks. The dataset contains 2,325 high-quality question–answer pairs covering more than 33,000 pages from 396 diverse documents, such as reports, manuals, books, and theses. Constructed through a semi-automated pipeline combining machine generation and human verification, LongDocURL provides a large-scale, fine-grained testbed to assess models’ abilities to process complex layouts, long contexts, and multi-element reasoning. 

PaperTab [13] and FetaTab [13] are benchmarks designed to evaluate retrieval-augmented generation (RAG) systems on academic and knowledge-based documents. PaperTab focuses on table-centric question answering from academic papers, containing 307 documents and 393 Q&A pairs, mainly of extractive and yes/no types. In contrast, FetaTab consists of 871 documents and 1,016 Q&A pairs derived from Wikipedia tables, emphasizing free-form natural language answers. Together, these benchmarks test models’ abilities to interpret tabular data, reason across structured information, and generate coherent responses grounded in complex document contexts. 

## **B. Overall Performance** 

## **B.1. Analysis of Experimental Results** 

On MMLongBench-Doc [19], SLEUTH achieves an average accuracy of 52.77%, outperforming the strongest retrieval-based baseline, MoLoRAG [39] (48.75%), by +4.02 points, and the Base model (46.76%) by +6.01 points (see Table 1). At the category level, the largest improvements appear in Pure-text and Figure questions, which increase from 53.33% to 59.26% and from 44.92% to 50.27%, re- 

1 

spectively, while Table also shows a stable rise from 44.76% to 47.55%. The substantial increase in the None category (52.68% to 67.38%) results from the evidence-driven decision rule. When the two evidence construction stages fail to collect valid support, the system outputs “No answers found!” following the predefined protocol. This behavior shows that the model can correctly recognize cases without evidence and avoid generating unsupported answers. This design helps the model avoid hallucinated answers and reduces errors caused by redundant or mismatched context. These results are consistent with the design principle of “evidence first, decision later.” By recording clues on a page basis and performing whole-page filtering, the system maintains a controlled context length while increasing the evidence density, producing robust gains in tasks that require multi-page reasoning with focusable evidence. The performance of SLEUTH and the various comparison methods across different dimensions is shown in Figure 5. 

**==> picture [215 x 193] intentionally omitted <==**

**----- Start of picture text -----**<br>
MMLongBench-Doc: Baseline Comparison by Dimension (Per-dimension Max-Normalized)<br>Chart<br>None Table<br>0.6×max 0.8×max 1.0×max<br>Figure Pure-text<br>Layout<br>SLEUTH (ours) M3DocRAG Base<br>MoLoRAG MDocAgent Direct<br>**----- End of picture text -----**<br>


Figure 5. Baseline comparison on MMLongBench-Doc. Our method yields a larger polygon across dimensions, consistent with compact, page-grounded evidence contexts. 

On LongDocURL [9], which consists of three subtasks (Understanding, Reasoning, and Cross-element Locating), the model achieves an average accuracy of 59.96% (Base: 55.18%). The improvements are most evident in Locating (46.04% to 53.63%, +7.59%) and Understanding (61.56% to 65.67%, +4.11%), while Reasoning shows a smaller yet consistent gain (51.09% to 52.99%, +1.90%) as reported in Table 2. Given the dataset statistics, most improvements come from the first two stages of our multiagent framework. The Clue Discovery agent gathers cross-element evidence, while the Page Screening agent keeps only the visually and semantically relevant pages. Together they improve recall of the correct regions and reduce interference from irrelevant content, enabling the difficulty-aware reasoning stage to perform in a cleaner input space. 

**==> picture [212 x 156] intentionally omitted <==**

**----- Start of picture text -----**<br>
45 . 1<br>45 43 . 7<br>40 . 5<br>40<br>35 . 9<br>35<br>30<br>VRAG-RL MoLoRAG MACT Ours<br>Accuracy (%)<br>**----- End of picture text -----**<br>


Figure 6. Comparison on MMLongBench-Doc using Qwen2.5-VL-7B-Instruct as the VLM backbone. 

On PaperTab [13] and FetaTab [13], SLEUTH also delivers clear gains (43.09% and 70.46%, respectively). The Page Screening stage applies the same strategy as in other benchmarks, preserving pages that contain figures, tables, or diagrams and discarding irrelevant content. This design increases the usability of layout and numerical elements while maintaining concise inputs. The improvements are consistent across the four datasets, indicating that our method can generalize beyond a specific document form. 

All baselines share the same VLM backbone and retriever settings (Top-5, temperature 0.1). The shared configuration guarantees fairness for all comparisons. 

## **B.2. Additional Comparative Experiments** 

For a fair comparison, we replaced the VLM backbone of the Core Decision Agent with Qwen2.5-VL7B-Instruct [2] and compared our method against several recent training-based approaches, including VRAG-RL [33], MoLoRAG [39], and MACT [54], on the MMLongBench-Doc [19] benchmark. All competing methods employed the same Qwen2.5-VL-7BInstruct backbone to ensure consistency. The final results obtained on MMLongBench are shown in Figure 6. Our modified answering agent achieved a score of 45.1, outperforming the aforementioned methods, where MACT scored 43.7, VRAG-RL 35.9, and the fine-tuned version of MoLoRAG reached 40.5. These results further show that our training-free contextengineering framework remains robust and broadly applicable, even when compared with approaches that rely on fine-tuning or reinforcement learning. 

## **C. Ablation Studies** 

## **C.1. Analysis of Ablation Experiment Results** 

The ablation studies provide further insight into how each component contributes to the overall performance. Starting from the Base system (46.76% on MMLongBench-Doc [19] and 55.18% on Long- 

2 

**==> picture [205 x 137] intentionally omitted <==**

**----- Start of picture text -----**<br>
60<br>55<br>50<br>MMLongBench-Doc<br>45 LongDocURL<br>Base +Clue +Page +Difficulty<br>Discovery Screening<br>Qwen3-VL-8B variants<br>Accuracy (%)<br>**----- End of picture text -----**<br>


Figure 7. Component ablation on Qwen3-VL-8B. Activating the Clue Discovery, Page Screening, and Difficultyaware agents yields consistent improvements. By generating a compact, page-grounded evidence context from broader retrieval, the system enhances overall performance. 

DocURL [9]), enabling the Clue Discovery agent alone improves the averages to 48.61% and 57.15%. At this stage, the model starts to record explicit evidence, and the effect is most evident in the None category (52.68% to 69.23%). The system can now represent the absence of supporting information, which helps it correctly handle unanswerable cases and suppress hallucinated outputs. Adding the Page Screening agent raises the averages to 51.29% and 59.49%. This step enriches the context with complete, visually coherent pages and filters out those that contain no relevant elements, helping the backbone model attend to useful regions and reducing confusion from unstructured noise. When the Difficulty Assessment agent is further introduced under the Top-5 retrieval configuration, the averages reach 52.77% and 59.96%. This final module helps the system switch to an appropriate reasoning strategy for difficult queries while leaving the easier ones in the standard mode. The steady improvement from the Base model to the full system shows that the three agents work in a complementary way. Clue Discovery provides fine-grained and traceable evidence, Page Screening reduces noise in the input, and Difficulty Assessment adjusts the reasoning strategy according to task complexity. 

The Top-K ablation also shows a consistent upward trend. On MMLongBench-Doc [19], the accuracies for Top-1, Top-3, and Top-5 are 44.92%, 49.65%, and 52.77%. On LongDocURL [9], they are 52.88%, 58.38%, and 59.96%. The improvement with larger K is not due to longer input sequences but to higher recall. The context provided to the evidence extraction and screening agents remains fixed. Increasing K only expands the range of retrieved candidate pages without lengthening their input or introducing additional noise. Consequently, the system transforms a broader retrieval into a context of stable size but higher evidence density. This explains why the accuracy increases steadily with K while hallucination does not. 

**==> picture [200 x 130] intentionally omitted <==**

**----- Start of picture text -----**<br>
55<br>50<br>Qwen3-VL-8B<br>GLM-4.1V-Thinking-9B<br>45 Gemini-2.5-Flash<br>1 3 5<br>Top-K<br>Accuracy (%)<br>**----- End of picture text -----**<br>


Figure 8. Cross-backbone Top-K curves on MMLongBenchDoc. All backbones exhibit steady gains as K increases. The agents operate on a fixed-length input while constructing evidence contexts, producing consistent accuracy improvements across architectures. 

**==> picture [200 x 129] intentionally omitted <==**

**----- Start of picture text -----**<br>
60<br>55<br>Qwen3-VL-8B<br>GLM-4.1V-Thinking-9B<br>50 Gemini-2.5-Flash<br>1 3 5<br>Top-K<br>Accuracy (%)<br>**----- End of picture text -----**<br>


Figure 9. Cross-backbone Top-K curves on LongDocURL. Enlarging K enables the agents to collect richer page-level cues and generate stronger evidence contexts, resulting in uniform performance gains across different backbones. 

In the future, we will introduce retrieval methods that are more powerful than Colpali, and we believe this will further improve performance. 

The same pattern holds across different backbones. When GLM-4.1V-Thinking-9B [27] or Gemini-2.5Flash [7] replaces Qwen3VL-8B, each step of component addition produces similar improvements, and Top-5 remains the optimal configuration. This indicates that the observed gains arise from the evidence organization process itself, rather than any property of a specific backbone. The reasoning stage mainly benefits from the structured evidence, confirming that the proposed pipeline provides a universal enhancement independent of model architecture. To visualize the performance changes caused by parameter variations in the ablation studies, we have plotted visualization curves. The variation curves of the ablation experiments for component ablation, retrieval parameters, and VLM backbone are shown in Figures 7, 8 and 9 . 

In summary, the results across all benchmarks support a consistent interpretation. The retrieval stage expands the search space, the clue discovery and page screening stages distill compact and trustworthy con- 

3 

texts, and the difficulty-aware reasoning stage delivers the final prediction. This process converts the quantity of retrieved pages into the quality of a short, evidence-dense input. As a result, accuracy scales with retrieval coverage while hallucination remains controlled. The improvements on MMLongBench-Doc, LongDocURL, and the two additional benchmarks together demonstrate that optimizing how evidence is constructed and filtered is more effective for longdocument understanding than simply extending the input length or increasing model size. 

## **C.2. Visual vs. Multimodal Retrieval Input** 

To verify whether purely visual page evidence is more advantageous for our architecture, we compare a visual-only setting with a multimodal setting. In the visual-only setting, the Clue Discovery Agent directly processes the top- _K_ pages returned by the visual retriever (ColPali-v1.3 [11]). In the multimodal setting, there are two independent retrieval streams: a visual stream (as above) and a textual stream. For the textual stream, we extract page-level text with MinerU 2.5 [23] and perform textual retrieval with BGE M3 [3]. The Clue Discovery Agent prompt is minimally adjusted so that, when handling text-based pages, it reads the extracted text blocks using the same evidence format. Each stream independently selects its own top-5 results, and both are then passed to subsequent agents, thereby constructing the data flow for the multimodal retrieval setting. 

Table 4 shows that the visual-only input achieves higher average accuracy than the multimodal input on both benchmarks: 52.77% vs. 50.19% on MMLongBench-Doc, and 59.96% vs. 57.62% on LongDocURL. Although the multimodal setting retrieves more content overall, this does not yield better performance. We speculate that two factual properties of visual pages are relevant here. First, visual pages naturally provide a more _compact_ representation: their spatial layout organizes information in place and reduces redundancy. Second, the visual modality offers a _unified_ representation: text, tables, and charts are encoded in the same image form. This uniform form preserves layout relations and visible cues, which helps maintain coherent reasoning under a fixed evidence budget. In contrast, OCR text can break such layout relations and introduce duplicated segments and noise, reducing contextual integrity. 

We also observe limited but tangible cases where the multimodal setting is helpful, mainly when exact textual normalization is required ( _e.g._ , strict digit matching or exact entity strings). These gains are narrow in scope and do not alter the overall trend above. Overall, the ablation supports that visual evidence aligns well with the proposed context-engineering design for long-document understanding. 

## **D. Innovations and Contributions** 

Unlike prior work that focuses solely on enhancing reasoning ability or improving retrieval recall, SLEUTH approaches long-document understanding from the perspective of context engineering, which complements both directions by constructing concise and evidence-rich contexts that support more reliable reasoning. The framework introduces a training-free, hierarchical multi-agent pipeline that builds concise and evidence-dense contexts from noisy retrieval results. Four cooperative agents—Clue Discovery, Page Screening, Difficulty Assessment, and Core Decision—work sequentially to extract structured clues, filter visual noise, perceive task difficulty, and synthesize final reasoning. This coarse-to-fine design works on short and fixed-length inputs during evidence extraction and screening, which makes it easier to identify useful information and suppress noise. The final reasoning then operates on a concise yet evidencedense structured context that adapts to each query, enabling more accurate and focused understanding. 

Empirical results across four benchmarks demonstrate clear advantages of SLEUTH over strong RAGbased and agent-based baselines. Its training-free and model-agnostic design allows consistent improvement under different backbones. Notably, visual-only page inputs outperform multimodal retrieval, suggesting that visual layouts inherently preserve document structure and reasoning cues. Through these findings, SLEUTH establishes a new paradigm and emphasizes that context quality is the key factor determining the effectiveness of long-document understanding. 

## **E. Limitations and Future Work** 

Although SLEUTH shows promising results, several challenges remain. The framework relies on retriever coverage; when critical pages are missed, downstream agents cannot compensate. Incorporating feedbackbased or hierarchical retrieval may mitigate this dependency. The binary difficulty estimation is efficient but coarse, which may not capture intermediate reasoning cases. Extending it to a continuous scale or adopting a light expert-routing strategy could improve adaptability. Moreover, current experiments focus on English administrative and academic documents, leaving open questions about cross-lingual and domainspecific generalization. Extending the framework to multilingual, handwritten, or specialized materials such as legal and medical documents would provide a more comprehensive evaluation. Lastly, SLEUTH is entirely prompt-driven and training-free, which benefits interpretability but limits self-evolution. Future work will train the agents with RL and teach them to use external tools for better evidence discovery and reasoning. In addition, SLEUTH will integrate improved retrieval and reasoning reinforcement to further enhance long-document understanding. 

4 

## **F. The Prompt design of SLEUTH** 

## Clue Discovery Agent 

You are a Detective, an expert evidence collector for document question answering. Your task is to carefully examine the given PDF page and extract ALL evidence that might be relevant to answering the question. 

**Question:** _{_ question _}_ **Page Information:** 

- Page Number: _{_ page ~~n~~ um _}_ 

## **Your Task:** 

1. Carefully examine the page image! 

2. Identify ALL facts, data points, and information that could help answer the question. 

3. Extract specific evidence with: 

   - Exact quotes or data values 

   - Context where information appears 

   - Explanation of why it’s relevant 

**Output Format:** Provide your analysis in the following JSON format: 

- _{_ "page ~~n~~ umber": _{_ page ~~n~~ um _}_ , "has ~~r~~ elevant ~~e~~ vidence": 

- true/false, 

"evidence ~~i~~ tems": [ _{_ "evidence ~~t~~ ype": "text/chart/table/figure", "content": "The actual evidence (quote, data, or description)", 

"location": "Description of where this appears on the page", 

"relevance": "Explanation 

of why this is relevant...", "confidence": 

"high/medium/low" _}_ ], "page ~~s~~ ummary": "Overall summary of findings from this page", 

"key ~~i~~ nsights": "Any important insights or patterns noticed" _}_ 

## Page Screening Agent 

You are an expert at analyzing document pages and identifying relevant charts/figures/tables for answering questions. 

Your task is to examine this PDF page image and determine: 

1. Whether there are any charts, figures, tables, or diagrams on this page. 

2. If charts/figures/tables exist, whether they are relevant to answering the given question. 

**Question:** _{_ question _}_ 

## **Page Number:** _{_ page ~~n~~ umber _}_ 

**Instructions:** 

1. First, carefully examine the page image to identify any visual elements like: 

   - Charts (bar charts, line charts, pie charts, etc.) 

   - Figures (diagrams, illustrations, photos, etc.) 

   - Tables (data tables, comparison tables, etc.) 

- Infographics or other data visualizations 

- 2. If you find charts/figures/tables, assess their relevance to the question: 

   - **Completely Relevant** : Directly contains information needed to answer the question. 

   - **Relevant** : Might contain related information, but relevance is uncertain. 

   - **Irrelevant** : The chart/figure/table exists but is clearly unrelated to the question. 

3. If there are NO charts/figures/tables on this page (only pure text), output ”none”. 

**Output Format (strictly follow this format):** Has ~~C~~ hart: [Yes/No] 

Relevance: [Completely Relevant/ Relevant/ Irrelevant] 

Reasoning: [Brief explanation of your judgment, 1-2 sentences] 

Now, analyze the provided page image and respond following the exact format above. 

## **Important Guidelines:** 

- Be thorough - collect ALL potentially relevant evidence. 

- Include exact numbers, percentages, and specific facts. 

- Note relationships between data points. 

- If the page is not relevant, explain why! 

- Please think carefully and avoid generating content that does not conform to reality. 

- Now examine the page and provide your evidence collection in valid JSON format. 

5 

## Difficulty Assessment Agent 

You are an expert whose task is to evaluate the user’s query and any structured multimodal context to determine the optimal reasoning strategy. 

**Input:** 

- **Question (** _Q_ **):** _{_ question _}_ 

- **Structured Context (** _C_ **)** 

**Instructions:** Analyze the query and context to determine the difficulty level _d ∈{_ 0 _,_ 1 _}_ and generate a corresponding instruction set Γ _d_ . 

**1. Determine Difficulty Level (** _d_ **):** 

- **Mode 0 (Ordinary Mode,** _d_ = 0 **):** Select this if the question can be answered by direct lookup or simple extraction from the provided context. 

- **Mode 1 (Reasoning Mode,** _d_ = 1 **):** Select this if the question requires: 

- **Cross-page aggregation** (combining clues from multiple pages). 

- **Numerical computation** (summation, percentages, ratio calculations). 

- **Trend comparison** (inferring information not explicitly stated). 

- **Multi-step inference** (deducing implicit information). 

**2. Generate Instruction Set (** Γ _d_ **):** Create spe- 

- cific, actionable instructions to guide the Core Decision Agent. 

- _Example for d_ = 1 _:_ ”Requires summing values from Page 2 (Table 1) and Page 5 (Text). Calculate the percentage growth.” 

- **Output Format (Strict JSON):** _{_ 

"difficulty ~~l~~ evel": 0 or 1, 

"instruction set": "Specific reasoning instructions Γ _d_ for the next agent." _}_ 

## Core Decision Agent (Text Evidence Only) 

You are an extractive QA model that gives answer to given query. You are given a query and a set of evidence. You have to provide specific answer from the given evidence, give your answer based only on the evidence. If you don’t find the answer within the evidence provided say ’No answers found!’. Use bullet points if you have to make a list, only if necessary. For counting questions, count carefully across all evidence. Mention which page the information came from. QUERY: _{_ question _}_ STRATEGIC INSTRUCTIONS Γ _d_ : _{_ instruction ~~s~~ et _}_ EVIDENCE (from _{_ num ~~p~~ ages _}_ pages): _{_ evidence ~~s~~ ummary _}_ YOUR ANSWER: 

If page images are included, the system will automatically switch to a prompt version with visual cues. 

## Core Decision Agent (With Visuals) 

You are an extractive QA model that gives answer to given query. You are given a query and evidence from relevant pages. You have to provide a specific, concise answer from the given evidence. 

## **Instructions:** 

- Give your answer based only on the evidence provided. 

- If you don’t find the answer within the evidence provided say ’No answers found!’. 

- Provide ONLY the shortest possible answer: a number, a name, a short phrase, or a brief list - just the key information. 

- Synthesize information across ALL pages of evidence when necessary (e.g., if one page has percentage A and another has percentage B, you may need to combine them). 

- For calculation questions, perform the required calculations using data from the evidence. 

- For counting questions, count carefully across all evidence. 

- Use bullet points only if the answer is a list. 

- QUERY: _{_ question _}_ STRATEGIC INSTRUCTIONS Γ _d_ : _{_ instruction ~~s~~ et _}_ EVIDENCE (from _{_ num ~~p~~ ages _}_ pages): _{_ evidence ~~s~~ ummary _}_ 

- _{_ visual ~~e~~ vidence ~~s~~ ection _}_ YOUR ANSWER: 

6 

