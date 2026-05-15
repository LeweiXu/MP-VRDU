# **DocSeeker: Structured Visual Reasoning with Evidence Grounding for Long Document Understanding** 

Hao Yan[1] , Yuliang Liu[1] _[∗]_ , Xingchen Liu[1] , Yuyi Zhang[1] ,Minghui Liao[2] , Jihao Wu[2] , Wei Chen[1][*] , Xiang Bai[1] 

1Huazhong University of Science and Technology 2Huawei Inc. {haoyan, ylliu, lemuria_chen, xbai}@hust.edu.cn https://github.com/yh-hust/DocSeeker 

## **Abstract** 

_Existing Multimodal Large Language Models (MLLMs) suffer from significant performance degradation on the long document understanding task as document length increases. This stems from two fundamental challenges: 1) a low Signal-to-Noise Ratio (SNR), with crucial evidence buried in irrelevant pages; and 2) supervision scarcity, as datasets offering only final short answers provide a weak learning signal. In this paper, we address these challenges by proposing a paradigm that requires the model to execute a structured “_ _**Analysis** ,_ _**Localization** and_ _**Reasoning** ” workflow. To instill this capability, we design a two-stage training framework: we first perform Supervised Fine-Tuning on high-quality data generated via an efficient knowledge distillation strategy. Subsequently, we employ an Evidenceaware Group Relative Policy Optimization which jointly optimizes for both evidence localization and answer accuracy. Additionally, we introduce a Evidence-Guided Resolution Allocation strategy to mitigate memory constraints of training on multi-pages documents. Extensive experiments demonstrate that DocSeeker achieves superior performance on both in-domain and out-of-domain tasks. We show it robustly generalizes from short-page training to ultra-long documents and is naturally synergistic with visual Retrieval-Augmented Generation systems, serving as a solid foundation for their implementation._ 

## **1. Introduction** 

Multi-Page Document Visual Question Answering presents a grand challenge in reasoning over lengthy, visually-rich documents [1–3]. Traditional parsing-based pipelines often rely on Optical Character Recognition (OCR) models [4– 6], which is prone to _cascading errors_ and risks _losing layout and stylistic information_ . In contrast, building upon the huge success of Large Language Models [7], a more elegant 

> *Corresponding author. 

and unified direction involves **pure-visual solutions** powered by Multimodal Large Language Models (MLLMs) [8– 16]. By treating each page as an image, these models preserve the document’s visual integrity holistically, offering a general framework that bypasses fragile intermediate steps and applies universally to almost any document format. 

However, MLLMs powered pure-visual methods confront two fundamental hurdles when scaling to long documents. The first is severe **low Signal-to-Noise Ratio (SNR)** , where crucial evidence are buried within vast irrelevant content. Although several visual-Retrieval-Augmented Generation (RAG) methods [17–19] for document page retrieval have been proposed, which can pre-filter question relevant pages, it introduces the classic top- _k_ dilemma: a large _k_ ensures high recall but introduce more noise, whereas a small _k_ risks missing the evidence. 

The second is the **scarcity of fine-grained supervision** . Most existing multi-page document VQA datasets provide only final short answers, lacking intermediate reasoning steps like evidence localization and information synthesis. Training models with such labels may cause them to learn fragile shortcuts through memorization, rather than developing genuine reasoning capabilities. This can lead to two weaknesses: First, it creates models that are effectively black boxes [20–22], severely undermining their interpretability, whose outputs lack clear reasoning path and source attribution. Second, this learning strategy can result in poor generalization performance when dealing with outof-distribution (OOD) documents. 

To address above limitations of existing methods, we introduce DocSeeker, a document multimodal large language model built upon Qwen-2.5-VL-7B [8]. Inspired by human cognitive processes, DocSeeker adopts a novel _Analysis–Localization–Reasoning_ (ALR) visual reasoning paradigm, as illustrated in Fig.1(b). When processing a multi-page document and a user’s question, each page’s visual tokens are prefixed with a page id, acting as a pointer, anchoring all information to its specific page. Instead of a direct answer, DocSeeker first generates a structured think- 

1 

**==> picture [435 x 205] intentionally omitted <==**

**----- Start of picture text -----**<br>
48 Baseline DocSeeker (Ours) Question : Comparing to the  Telecom Operators  in 2014 and 2013-2014, What<br>! are the Operators that are not in common?<br>40.1 41.1<br>+13.0 37.6 … …<br>35 \ +11.4 tM +17.9 re +20.2 ; 31.8 A 11 Page 1 Page 9 XL, Indosat, 3, Smartfren, Telkomsel, Esia XL, IndosatTelkomsel,  EE) Page 10 Page 38<br>28.7 y<br>H +20.1 !!<br>23.2 v'1 <think><br>Doc_length (MMLongbench-doc) 17.4 11.7 ¥ [I] I Question Analysis Structured<br>! … identify operators that are unique to either "2014" or "2013-2014"  thinking<br>0:20 20:40 40:60 60:80 > 80 II when comparing them to "Telecom Operators"… finding lists of operators<br>(a) Model performance across different document page ranges+37% s 86.0 a Baseline a DocSeeker (Ours) 77.8 f!!If for these two periods and identifying operators that appear in one but not both. Evidence Localization … locate sections discussing “Telecom Operators” for “2014” and “2013-2014”. The title “Telecom Operators - 2013-20-14 (3)”  on page 9  ... The title “…”  on page 10  ...<br>70.1 +30% I Reasoning Process<br>sI 1. … 2013-2014: Page 9: Tel…, XL, Ind…, 3, Smartfren, Esia.<br>+64% 57.9 +37% 51.7 59.8 1!I 2. … late 2014: Page 10: Tel…, XL, Ind….   3. Compare: Unique to 2013-2014: 3, Smartfren, Esia.    Explicit Evidence Grouding<br>35.2 CS +58% 40.1 37.8 w 1 </think> Page aware<br>sI <answer><br>25.4 I! {"evidence_pages": [ 9, 10 ], "answer": [ "3", "Smartfren", "Esia" ]}<br>DUDE MP-DocVQA MMLongbench-doc LongDocURL SlideVQA !I </answer><br>f<br>(c) Model performance on 5 multi-page document VQA datasets 1 (b) Analyze–locate–reason workflow of DocSeeker<br>c c<br>**----- End of picture text -----**<br>


Figure 1. Overview of Principal Experimental Results in Long Document Understanding (Left) and the ALR Reasoning Paradigm of DocSeeker (Right). 

ing process. It starts by analyzing the user’s intent, then identifies supporting evidence with explicitly mentioning potential pages, and finally, builds a line of reasoning from that evidence. Following this thinking process, the model will deliver the final answer, along with the specific page ids where the evidence can be found. 

To enable the model to effectively learn this reasoning paradigm, we propose a two-stage training framework. In the first stage, we employ an efficient data distillation strategy to generate high-quality ALR Chain-of-Thought (CoT) annotations from a powerful teacher model for supervised fine-tuning (SFT). In the second stage, we introduce _Evidence-aware Group Relative Policy Optimization_ (EviGRPO) that jointly optimizes evidence localization and reasoning abilities. To render this process computationally tractable and more effective for long visual sequences, we propose _Evidence-Guided Resolution Allocation_ (EGRA) strategy, which is applied during training to optimize resource allocation and strengthen the supervisory signal. 

Extensive experiments demonstrate the effectiveness of DocSeeker. First, as shown in Fig. 1(c), DocSeeker achieves a remarkable 30-60% performance gain across all five document VQA benchmarks compared to the Baseline, which shares the same architecture and parameters. Furthermore, DocSeeker’s performance not only surpasses other advanced open-source MLLMs but also proves highly competitive with closed-source commercial models. More importantly, despite being trained only on existing shortpage documents, DocSeeker achieves robust generalization to ultra-long document reasoning, effectively mitigating the performance decay associated with long-sequence inputs, 

as shown in Fig. 1(a). DocSeeker’s strong localization capability, derived from the ALR paradigm, is proven to be naturally synergistic with visual RAG systems, serving as an ideal foundation for building next-generation, efficient, and robust implementations. 

Ours core contributions can be summarized as follows: 1) We introduce the **ALR** paradigm, a structured workflow inspired by human cognition, along with the corresponding ALR CoT dataset. This paradigm compells the model to internalize an ALR process rather than superficially imitating outputs; 2) We propose a two-stage training framework: **Stage I injecting the ALR Paradigm via SFT** , which uses an efficient, high-fidelity data distillation strategy to generate ALR CoT data for SFT; **Stage II EviGRPO** , which jointly optimizes for both evidence localization and reasoning abilities; and **EGRA** strategy mitigates memory constraints while simultaneously strengthening the supervision signal; 3) Extensive experiments demonstrate our model’s strong localization capabilities and robust generalization in OOD ultra-long document scenarios, and confirm it is naturally synergistic with visual RAG systems. 

## **2. Related Work** 

## **2.1. Multi-Page DocVQA dataset** 

Multi-page Document Visual Question Answering challenges models to comprehend and reason over lengthy, visually-rich documents. Early datasets [1, 2, 23–25] significantly advanced progress in multi-page document understanding. However, these works primarily focused on documents within a 20-page limit. More recent benchmarks, 

2 

including MMLongBench-doc [26], LongDocURL [27], MMdocIR [28], M-LongDoc [29] and DocBench [30], have extended the challenge to hundred-page scale scenarios. Despite this progress, annotating such ultra-long documents remains a significant bottleneck due to high human costs and the low accuracy of MLLM-based auto-labeling, making relevant, high-quality training data exceedingly scarce. Exacerbating this issue, most existing datasets with training splits primarily offer “long-sequence input to shortanswer output” pairs, which provides only a sparse signal and forces models to implicitly learn evidence localization from the final answer. 

## **2.2. Long Document Understanding** 

Existing methods for long document understanding can be primarily categorized into OCR-based methods and purevision methods. OCR-based methods, such as LongFormer [31] and LayTokenLLM [32], rely on OCR tools to extract text and layout for downstream models. However, this pipeline suffers from domain brittleness, cascading errors, and heavyweight OCR-tool dependency. In contrast, purely visual approaches feed pages directly into the MLLM, this paradigm fundamentally avoids error propagation and holistically preserves critical visual and layout information. Nevertheless, the performance and efficiency of this pure-vision paradigm are significantly challenged when processing ultra-long documents, leading existing MLLMs to primarily employ two strategies: (1) **Visual RetrievalAugmented Generation** leverage an external visual retriever, such as ColPail [18] and DSE [33], to identify the Top-K relevant pages and then process them for finegrained reasoning. Classic works that adopt this paradigm include VisRAG [17], SV-RAG [34], and VDocRAG [35]. (2) **End-to-End methods.** These methods must either use high-performance visual token compressors, like mPLUGDocOwl2 [10], or significantly expand the context window, such as in Qwen2.5VL [8] and InternVL3 [9]. These two strategies collectively represent the dominant research paradigms in the pure-vision long document understanding. Details of those models are provided in Appendix E. 

## **3. DocSeeker** 

We build DocSeeker upon the Qwen-2.5-VL-7B-Instruct architecture [8] as our backbone model, an powerful open source MLLM comprising a Transformer-decoder [36] based LLM, a Vision Transformer (ViT) [37] encoder, and a VL-Adapter for mapping image features into visual tokens. Rather than introducing any extraneous modules, we aim to enhance the backbone model’s intrinsic ability to mitigate the signal degradation caused by the noise and redundancy inherent in long visual token sequences, a critical bottleneck for scaling to ultra-long document VQA. To this end, we propose the novel ALR visual reasoning paradigm. Inspired 

by the human cognitive process of _find-then-reason_ [38], the ALR paradigm enables the model to efficiently localize salient information before conducting detailed reasoning. We instill this paradigm through a two-stage training framework composed of _SFT_ and _Evi-GRPO_ , where _EGRA_ is utilized in both training stages to enable efficient longsequence training. 

## **3.1. ALR Visual Reasoning Paradigm** 

The ALR visual reasoning paradigm is the core for our DocSeeker, it’s implemented via two primary mechanisms: _Page-Aware Input Representation_ and _Structured Reasoning Paradigm with Explicit Evidence Grounding_ , as illustrated in the Fig. 1(b). 

**Page-Aware Input Representation.** To enable the model to ground its reasoning on specific document pages, we introduce a page-aware input representation. This is achieved by interleaving textual page identifiers with their corresponding visual tokens. Specifically, for a user question _Q_ and a document of _N_ pages with each page image denoted as **I** _i_ ( _i ∈_ 1 _, . . . , N_ ), the input sequence **X** fed to the LLM is constructed as: 

**==> picture [169 x 30] intentionally omitted <==**

Here, **e** _Q_ represents the text embeddings of the question _Q_ . For each page _i_ , **v** _i_ denotes its visual tokens, which are preceded by the text embeddings of a page identifier, **e** _i_ (e.g., (e.g., _“Page i”_ ). The visual tokens are generated by the vision backbone: **v** _i_ = _f_ adapter( _f_ enc( **I** _i_ ; _ri_ )), where _ri_ is the image resolution. The _⊕_ operator denotes sequence concatenation. 

**Structured Reasoning Paradigm with Explicit Evidence Grounding.** Inspired by recent large reasoning models (LRMs) [39, 40] that separate the thought process from the answer (e.g., using <think> and <answer> tags), we enforce a novel structured reasoning paradigm tailored for multi-page long document VQA. Instead of generating freetext thought chains, our model’s output, denoted as **Y** , must adhere to a specific, interpretable reasoning path. The output sequence is defined as: 

**==> picture [231 x 11] intentionally omitted <==**

Here, the thought process **Y** th is decomposed into three parts: 1) First, the model generates _Question Analysis_ **Y** A to deconstruct the user’s intent; 2) The critical evidencefinding part is denoted as _Evidence Localization_ **Y** L, where the model is required to scan the document and explicitly report which pages it deems relevant and why, forcing the model learns visual evidence grounding by referencing specific page identifiers; 3) Finally, it articulates a _Reasoning Process_ **Y** R that synthesizes information from the located 

3 

**==> picture [428 x 259] intentionally omitted <==**

**----- Start of picture text -----**<br>
ALR CoT Data Distillation EGRA<br>Short Answer<br>Question Short Answer Question discard Page 1 Page 2<br>Filter<br>··· Pre-process Page {ID}: VerificationSecondary<br>Evidence Generator<br>Evidence<br>Raw data Minimal Context<br>¥ x<br>Stage ⅠSFT Evidence page    <think> Page 3 Page 4<br>Evidence \\boxed{Question Analysis}   <answer>"evidence_pages "<br>··· \\boxed{Evidence Localization}  “answer"<br>\\boxed{Reasoning Process} </answer><br></think><br>ae ALR CoT data lca! Tes<br>Initialize<br>···<br>Page 7 Page 8<br>Reward 1: Format<br>KL<br>Policy  Reward 2: Localization<br>optimization Reward 3: Answer<br>Policy model Reference model Reward functions<br>3 Advantage To=--. -____§f___ = x 3<br>Online Filtering Data ··· o 1 o 2 o 3 Rollout output o 4 o 5 … ResolutionLow ResolutionHigh<br>OO ~ €@ @e B a o da ee<br>Stage ⅡEvidence-aware GRPO (Answer √  Format√ Localization√ ) Fully correct output (Answer Fully incorrect output × Format ×  Localization ×  )  Evidence Page Non-evidence Page<br>a) C V) i)<br>**----- End of picture text -----**<br>


Figure 2. Overview of our training framework. (Left) Stage I SFT, where the model is supervised fine-tuned on high-quality ALR CoT data; Stage II RL, where the Policy model is optimized via reinforcement learning, guided by a multi-dimensional reward function that includes format, localization, and accuracy. (Right) Our proposed Evidence-Guided Resolution Allocation (EGRA) strategy. 

evidence to derive a conclusion. Following the thought process, the model generates the answer **Y** ans, which consists of a list of the page numbers identified **Y** E as evidence, and the final, concise answer **Y** F. 

The proposed ALR Visual Reasoning paradigm, a **structured** and **page-aware** paradigm, offers significant advantages. It mandates explicit evidence grounding, which yields **high interpretability** and allows users to easily verify the answer by referring to the cited pages. More critically, subsequent experiments demonstrate that by compelling the model to localize and cite page-specific information, it learns to effectively distinguish visual tokens belonging to different pages, thus counteracting the interference and noise in long visual inputs. Furthermore, and to our surprise, **the model can seamlessly integrate with a visual retrieval system** . It can still perform reasoning effectively even with the discontinuous, non-sequential Top- _k_ pages returned by the visual page retriever. 

## **3.2. Training Procedure** 

To enable the model to master the ALR paradigm mentioned in Sec. 3.1, we designed the following training procedure, as shown in Fig. 2. 

**Stage I: Injecting the ALR Paradigm via SFT.** This 

stage aims to address supervision scarcity and instill the core capabilities for backbone model to follow the ALR visual reasoning paradigm by constructing high quality training data. Instead of building from scratch, we build upon existing multi-page VQA datasets that provide labels consisting of final short answer and evidence page IDs, but lacking the explicit, step-by-step reasoning paths required for the ALR format. To achieve this, we construct a highquality training dataset via knowledge distillation. We utilize Gemini-2.5-Flash [41] as our teacher model. Instead of a naive and costly full-document prompting approach, we employ a more efficient strategy: we provide the teacher with a **minimal context** composed of only ground-truth evidence pages, a very few distractor pages within the same document, corresponding page IDs and the question. The teacher model is required to generate an complete ALRformatted response in the instruction prompt. To guarantee data quality, we implement a two-stage verification pipeline: an initial automated Exact Match (EM) check on the ground-truth short answer and evidence page ids, followed by a semantic validation for failed samples with wrong matched answers using GPT-4o [42]. This secondary check effectively rescues correct examples where the generated output is semantically correct but fails the strict EM 

4 

criteria due to trivial formatting differences or acceptable paraphrasing. See more details in Appendix A. 

With this high-quality, verified dataset, we then **finetune** our backbone model, using a standard cross-entropy loss to predict the entire ALR-formatted response **Y** , given the **complete document** (i.e., all pages and page ids) and question as input. By supervising on the full reasoning process over complete documents, this stage enhances reasoning rather than memorizing by compelling it to learn the holistic ALR reasoning structure. 

**Stage II: Evidence-aware GRPO.** While Supervised Fine-Tuning (SFT) equips the model with the ability to follow the ALR paradigm, the resulting reasoning paths from this imitation learning are often sub-optimal. To transcend this limitation, we introduce a second fine-tuning stage using Group Relative Policy Optimization (GRPO) [40], a reinforcement learning approach enables the model to learn directly from outcome signals. Specifically, GRPO samples multiple candidate responses, evaluates them with a reward function, and updates the model’s policy to favor highreward outputs. To tailor this process to our ALR paradigm, we design a bespoke reward function beyond standard GRPO. Our EviGRPO reward jointly optimizes for format correctness, evidence grounding, and answer accuracy. The total reward, _R_ = _λ_ 1 _R_ format + _λ_ 2 _R_ evidence + _λ_ 3 _R_ answer, is a weighted sum of three rewards. First, a binary Format Reward _R_ format ensures the model’s output strictly adheres to the ALR template. Second, we introduce an evidence grounding Reward _R_ evidence to directly score the accuracy of the identified evidence pages. This is calculated using a weighted ( _β >_ 1) F1 score that prioritizes recall over precision. Finally, a standard Answer Reward measures the correctness of the final answer ( **Y** F) using the Average Normalized Levenshtein Similarity (ANLS) [43] metric. This multi-faceted reward signal guides the model to learn both localization and reasoning within the ALR paradigm in a more direct and efficient manner. 

**Evidence-Guided Resolution Allocation.** Training on long documents presents a significant challenge: processing full-resolution pages generates vast input visual tokens, resulting prohibitive out of memory on GPUs during training. To address this, we propose Evidence-Guided Resolution Allocation (EGRA), a simple yet effective training strategy employed during both the SFT and GRPO stages to support longer document inputs. The core idea is to apply a differentiated image resolution scheme: ground-truth evidence pages are maintained at a high resolution to preserve critical details, while for non-evidence pages, we randomly downsample a large portion (70%) to a lower resolution ( _ri_ from 1024 to 256), with the remaining ones (30%) retaining their original high resolution ( _ri_ = 1024). This approach significantly reduces the total input token count, enabling us to use longer documents for training. During inference, all 

pages are processed at their native high resolution, which is computationally feasible since inference has a lower memory footprint. Beyond mitigating GPU memory constraints, our experiments find this strategy can facilitate learning by increasing the signal-to-noise ratio in the training data, outperforming naive strategies such as fix resolution or directly removing a subset of non-evidence pages. 

## **4. Experiment** 

## **4.1. Experimental Setup** 

**Datasets.** Our raw training data is sourced from MPDocVQA [1] and DUDE [2] (both up to 20 pages). The evaluation is divided into two parts: (1) In-Domain evaluation, conducted on MP-DocVQA and DUDE; and (2) OOD evaluation on challenging benchmarks: MMLongBenchDoc [26] (up to 468 pages), LongDocURL [27] (avg. 30 pages), and SlideVQA [23] (up to 20 pages). 

**Data Distillation and Filtering.** Our SFT data preparation begins with pre-filtering to remove overly short documents, which yields a curated set of 19,386 long-document VQA samples. Subsequently, we employ Gemini-2.5-flash [41] as the teacher model to distill 13,986 high-quality ALRparadigm samples (8,676 from MP-DocVQA and 5,310 from DUDE) for the SFT stage. For the RL stage, we source data from the remaining training instances of MP-DocVQA and DUDE. To select valuable samples, we employ an _online filtering_ strategy, following [44], to filter out instances yield “zero advantage” for the policy update. 

**Training and Evaluation Configuration.** All training is conducted on 16 NVIDIA A800 GPUs. We adopt Qwen2.5VL-7B-Instruct [8] as our baseline. In the SFT stage, we train the model on our distilled data for 2 epochs with a learning rate of 1 _×_ 10 _[−]_[6] . In the RL stage, we train for 6 epochs with a learning rate of 1 _×_ 10 _[−]_[6] and a rollout group size of 16. The reward weights for format, evidence, and answer were set to 0.1, 0.3, and 0.6, respectively, and the _β_ for the weighted F1-score (evidence reward) was set to 2.0. For evaluation, we process each document page at a resolution of 1024 _×_ 784 to balance visual clarity with the capacity to accommodate longer document contexts. 

## **4.2. Main Results** 

Tab. 1 presents a comprehensive comparison of DocSeeker against state-of-the-art methods across five document understanding benchmarks. We compare with three categories of approaches: (1) End-to-end MLLMs that process full document pages as input, including InternVL3 [9] and mPLUG-DocOwl2 [10]; (2) Parsing-based methods that rely on OCR or PDF parsers to assist the visual language model, such as HiVT5 [1], CREAM [45], Docpilot [47], and DocVLM [48]; Retrieval-augmented methods, including M3DocRAG [46], Vis-RAG [17], SV-RAG [34], and 

5 

Table 1. Comparison of different methods on five document understanding benchmarks. The results are reported on DUDE (ANLS), MPDocVQA (ANLS), MMLongBench-doc (Acc), LongDocURL (Acc), and SlideVQA (F1). “Param.” denotes parameter scale; BaselineSFT (short-answer) refers to the baseline model fine-tuned on raw short-answer data; “Evi.” indicates evidence type (T = text, P = pages, I = images extracted from pages); “RAG” indicates whether retrieval is used; and “MMLong.“ and “LongDoc.“ are abbreviations for MMLongBench-doc and LongDocURL. The best and second-best results are highlighted in bold and underlined, respectively. 

|Method|Venue|Param.|Evi.|RAG|**DUDE**|**MPDocVQA**|**MMLong.**|**LongDoc.**|**SlideVQA**|
|---|---|---|---|---|---|---|---|---|---|
|_Open Source_||||||||||
|HiVT5 [1]|PR|0.3B|T+P|×|23.1|62.0|-|-|-|
|CREAM [45]|ACM MM 2024|7B|T+P|✓|52.5|74.3|-|-|-|
|mPLUG-DocOwl2 [10]|ACL 2025|8B|P|×|46.8|69.4|13.4|5.3|-|
|M3DocRAG [46]|Arxiv 2024|9B|P|✓|-|84.4|21.0|35.1|55.7|
|Vis-RAG [17]|ICLR 2025|4B|P|✓|-|70.9|18.8|41.9|50.7|
|SV-RAG [34]|ICLR 2025|4B|P|✓|45.0|71.0|23.0|-|-|
|VDocRAG [35]|CVPR 2025|8B|P|✓|44.0|62.6|18.4|39.8|42.0|
|Docopilot [47]|CVPR 2025|8B|T+I|×|-|81.3|28.8|-|-|
|DocVLM [48]|CVPR 2025|8B|T+P|×|47.4|84.5|-|-|-|
|InternVL3 [9]|Arxiv 2025|8B|P|×|47.4|80.8|24.1|38.7|54.4|
|_Closed-source commercial models_||||||||||
|Qwen-VL-Max [49]|-|-|P|×|-|-|-|49.5|-|
|GPT-4V|-|-|P|×|-|-|32.4|-|-|
|Gemini-1.5-Pro [50]|-|-|P|×|46.0|-|28.2|50.9|-|
|GPT-4o[42]|-|-|P|×|54.1|67.4|**42.8**|**64.5**|-|
|||||_Ours_||||||
|Baseline|-|7B|P|×|35.2|70.1|25.4|37.8|59.8|
|Baseline-SFT (short-answer)|-|7B|P|×|56.0|82.9|28.8|42.7|67.4|
|DocSeeker-SFT|-|7B|P|×|56.8|82.1|38.6|49.1|75.2|
|DocSeeker|-|7B|P|×|**57.4**|**86.2**|40.1|51.7|**77.1**|



VDocRAG [35]. We also include closed-source commercial models such as GPT-4o, GPT-4V, Qwen-VL-Max, and Gemini for reference. Detailed evaluation settings for all compared models are presented in Appendix E. We can draw the following key conclusions: 

**Performance and Generalization of DocSeeker.** In In-Domain evaluation, DocSeeker surpasses all existing open-source and commercial models on MPDocVQA and DUDE. More importantly, in OOD evaluation, DocSeeker establishes itself as the open-source state-of-the-art across all OOD benchmarks and proves highly competitive with advanced closed-source commercial models. 

**Limitations of SFT on Short-Answer Data.** SFT on shortanswer data boosts In-Domain performance but provides only marginal OOD generalization, suggesting the model is primarily memorizing answers rather than developing true long-document understanding capabilities. 

**The role of the ALR reasoning paradigm.** SFT on our high-quality, distilled data equips the model with the ALR reasoning paradigm. This acquired capability results in a massive performance leap across all benchmarks, demonstrating that the model receives effective supervision from our distilled data and that the ALR reasoning paradigm possesses strong generalization capabilities. 

**The role of the Evidence-aware GRPO.** Although the SFT stage achieved significant success, we observe that the introduction of RL stage brought further, consistent perfor- 

mance improvements across all five benchmarks. This indicates that the ALR capabilities injected during SFT still had room for optimization and our Evidence-aware GRPO provided this crucial refinement. 

We also provide extensive representative examples in Appendix F to corroborate our experimental conclusions. 

Table 2. Performance comparison of DocSeeker and the Baseline under two input conditions: Full-document input (Full-doc) and Evidence-only input (Evi-only). **Avg.** : Avg. input pages. 

|**Model**<br>Input<br>Avg.|**MMLong.**<br>Acc.<br>F1|
|---|---|
|Baseline<br>Evi-only<br>1.5|41.1<br>37.6|
|Baseline<br>Full-doc<br>43.4<br>DocSeeker<br>Full-doc<br>43.4|25.4(-15.7)<br>20.8(-16.8)<br>40.1(-1.0)<br>38.4(+0.8)|



## **4.3. Impact of Document Length on Performance** 

To investigate the reasons for DocSeeker’s superior performance, we conducted two experiments on MMLongBenchdoc to analyze the impact of document length. 

**Full-Document vs. Evidence-Only Input.** We hypothesize that DocSeeker’s advantage primarily stems from a robust evidence localization capability. To validate this, we compare performance under two distinct conditions: a controlled “evidence-only” input, which requires no localiza- 

6 

**==> picture [501 x 138] intentionally omitted <==**

**----- Start of picture text -----**<br>
tion and a “full-document” input, which demands localiza- Colqwen 2.5 + Qwen2.5-VL-7B Colqwen 2.5<br>Colqwen 2.5 + DocSeeker Colqwen 2.5 + DocSeeker<br>tion amidst noise. The results, presented in Tab 2, show 45 60<br>that feeding the entire document led to a significant declinein accuracy, highlighting the limitations of current MLLMs 40 40.1 k  = Inf 50 52.8 k  = Inf<br>in terms of document localization for long documents. In 40<br>contrast, Docseeker’s performance with full-document in- 35<br>put closely matched that with evidence page input. 30<br>30<br>20<br>Baseline DocSeeker (Ours) 25.4 k  = Inf 11.0 k  = Inf<br>40 36 34.6 33.7 33.9 25 1 5 10 15 20 25 30 35 40 45 50 55 60 10 1 5 10 15 20 25 30 35 40 45 50 55 60<br>29.5 31.8 k  (Number of retrieved pages) k  (Number of retrieved pages)<br>30 34.5 +7.1 (a) Acc across retrieved page counts (b) F1 score across retrieved page counts<br>Acc  F1<br>**----- End of picture text -----**<br>


**==> picture [221 x 105] intentionally omitted <==**

**----- Start of picture text -----**<br>
Baseline DocSeeker (Ours)<br>40 36 34.6<br>33.7 33.9<br>31.8<br>29.5<br>30 34.5 32.4 +7.1 +9.1<br>26.6 +20.0<br>26<br>20 22.7<br>Doc_length (MMLongbench-doc) SEC 13.9<br>10<br>10 20 30 40 50 60<br>Acc<br>**----- End of picture text -----**<br>


Figure 4. Performance comparison on MMLongBench-doc following integration with a retrievar Coilpail. **Acc** measures answer correctness, while **F1** quantifies evidence page retrieval. The k=Inf designation indicates the result for full-document input. 

Figure 3. Performance comparison on different document length 

**Analysis of Document Length on Performance.** To further investigate the impact of document length in a controlled manner, we conducted an experiment on a subset of MMLongBench-doc, using only documents originally longer than 60 pages. We fixed the question and ensured the evidence page was always included, while truncating the total context to different lengths. This allows us to isolate the total context length as the primary variable and precisely quantify its direct impact on performance. For this analysis, the "evidence-only" input condition serves as the theoretical upper bound for performance, representing perfect evidence localization. The results in Fig. 3 show that Baseline’s performance dramatically degrades as the document length increases, with its accuracy plummeting from 34.5% to 13.9%. In contrast, DocSeeker exhibits remarkable robustness, as its performance remains largely stable. This experiment re-confirms that our ALR paradigm enables the model to robustly localize evidence within long contexts, undeterred by the interference of irrelevant pages. 

## **4.4. Integration with RAG-based systems** 

Sec. 4.3 has demonstrated the strong evidence localization capabilities of DocSeeker, enabling its performance on full input to closely approach that of ideal retrieval ground truth input. This core capability for precise localization within long contexts makes our model an ideal choice for integration with visual retriever (e.g., ColPail [18]) in a RAG system to tackle multi-document or ultra-long document reasoning. This allows it to directly address a critical and wellknown flaw in RAG: model performance collapses when the number of retrieved pages (K) is too small, which risks missing key evidence, or too large, which introduces excessive noise and causes the SNR to plummet. 

Table 3. Ablation study on data types and size. The first group investigates different ALR CoT data configurations, while the second explores the effect of varying data size on performance. 

|**Data Config**|**Size**|**MMLong.**<br>Acc.<br>F1|**MMLong.**<br>Acc.<br>F1|
|---|---|---|---|
|_Data type ablations_<br>Baseline|–|25.4|20.8|
|Raw short-answer data|6.3k|27.4|27.6|
|Vanilla CoT|6.3k|31.3|32.4|
|ALR CoT data<br>6.3k<br>-w/o Page id<br>6.3k<br>_Data size ablations (ALR CoT data)_||**33.8**<br>30.4|**33.9**<br>31.1|
|20% of ALR CoT data<br>40% of ALR CoT data<br>60% of ALR CoT data|2.8k<br>5.6k<br>8.4k|32.7<br>35.8<br>36.5|31.7<br>33.8<br>34.9|
|80% of ALR CoT data|11k|38.2|36.5|
|All ALR CoT data|13k|**38.6**|**36.9**|



The experimental results clearly validate this phenomenon. As shown in Fig. 4 (a), the performance of baseline collapses dramatically as K increases, especially for values greater than 5. This is because as the retriever increases K to improve Recall, its Precision and F1 sharply declines, as shown in Fig. 4 (b). The baseline model, lacking the ability to perform localization in long and noisy contexts, is consequently overwhelmed by the noise. In contrast, DocSeeker effectively resists this noise interference. More importantly, integrating with the retrieval system further enhances DocSeeker’s performance. This indicates a strong synergy: the retriever performs coarse-grained filtering, while our model efficiently conducts fine-grained reading and localization within the still-noisy retrieved results. 

7 

## **4.5. Ablation Study** 

We conducted comprehensive experiments on the MMLongBench-doc (hereafter referred to as MMLong.), chosen for its OOD characteristics regarding both data and document length distribution, to validate the effectiveness of the proposed method. 

**Ablation on Reasoning Paradigm and Data Size.** To evaluate the impact of different reasoning paradigms and training data sizes on the model’s performance, we conduct an ablation study with data types including Raw Short-Answer Data, Vanilla CoT Data (with unstructured reasoning and answers also distilled from the same teacher model Gemini2.5-Flash), and our ALR CoT Data, trained for two epochs. The generation process for Vanilla CoT and ALR CoT Data is detailed in Appendix A. To ensure fair comparison, we select the intersection subset of the three datasets after correctness validation. Additionally, we also verify the impact of Page ID and data size in the ALR CoT Data. The corresponding In-domain evaluation results for different reasoning paradigms and data sizes are detailed in Appendix B. The experimental results in Tab. 3 indicate that: 1) Raw Short-Answer Data shows limited generalization, providing only marginal gains. 2) Vanilla CoT Data offers modest improvements, while our ALR CoT data provides a more significant boost. 3) Larger-scale ALR CoT data consistently enhances performance. Additionally, removing Page ID reduces performance, highlighting its crucial role in providing page-level context and aiding information retrieval. 

Table 4. Ablation study on training resolution strategies. “FixedRes” denotes fixed resolution for all pages, while EGRA adjusts page resolution during training based on evidence relevance. Tokens indicate the number of visual tokens used to encode an image. 

|**Method**<br>**Strategy**<br>**Tokens**|**MMLong.**<br>Acc.<br>F1|
|---|---|
|Baseline<br>-<br>-|25.4<br>20.8|
|Fixed-Res<br>Full Low-Res.<br>576<br>Truncated<br>1024|34.2<br>33.5<br>36.6<br>35.8|
|EGRA<br>Full<br>1024 or 256<br>-w/o Non-Evi. Hi-Res<br>1024 or 256<br>-w/o Low-Res<br>1024|**38.6**<br>**36.9**<br>35.5<br>33.8<br>34.5<br>33.4|



**Ablation on Resolution Allocation Strategy.** To investigate the effect of resolution allocation on model performance under a fixed token budget, we conduct an ablation study comparing our proposed EGRA strategies with other strategies. The Fixed Allocation strategy either employs higher resolution with truncated inputs or full-page inputs at lower resolution. EGRA (w/o Low-Res) refers to directly removing 70% of none-evidence pages (instead of retaining low resolution), while EGRA (w/o Non-Evi. Hi-Res) refers to reducing the resolution of all none-evidence pages. The results in Tab. 4 show that: 1) Fixed Allocation set- 

Table 5. Ablation on GRPO variants and reward weights. Vanilla GRPO optimizes answer correctness only; EviGRPO further includes localization-related rewards. 

|**Method**|**Weight**<br>(_λ_1_, λ_2_, λ_3)|**MMLong.**<br>Acc.<br>F1|**MMLong.**<br>Acc.<br>F1|
|---|---|---|---|
|SFT|–|38.6|36.9|
|Vanilla GRPO|(0_._1_,_ 0_._0_,_ 0_._9)|38.7|36.3|
||(0_._1_,_ 0_._1_,_ 0_._8)|39.4|37.8|
|EviGRPO|(0_._1_,_ 0_._3_,_ 0_._6)|**40.1**|**38.4**|
||(0_._1_,_ 0_._5_,_ 0_._4)|38.9|37.0|



tings underperform compared to EGRA. Because fixed resolution training presents a trade-off: higher resolution risks losing context, while lower resolution provides limited supervision, both leading to the loss of effective information during training; 2) Non-evidence pages, though not contributing to reasoning, retaining them at a low resolution is superior to discarding; 3) Selectively downsampling a random subset of non-evidence pages, rather than all of them, enhances model robustness by introducing high-resolution distractors, which also mitigates the resolution inconsistency between training and inference phases. EGRA effectively balances detail and context length, optimizing resolution and computational efficiency, leading to enhanced long-document performance and generalization. 

**Ablation on RL Configurations.** To evaluate the effect of GRPO variants and reward weights on DocSeeker’s performance, we conducted a series of ablation experiments. The experiment results presented in Tab. 5 suggest that The proposed Evidence Localization Reward effectively assists the Answer Reward, further improving the model’s performance and generalization in long-document understanding tasks. Furthermore A large Evidence Localization weight causes the model to overly focus on evidence localization, leading to recognition errors, where incorrect answers may receive high reward; whereas a small weight may cause the model to ignore evidence localization signals. 

## **5. Conclusion** 

In this work, we present DocSeeker, which adopts the novel ALR visual reasoning paradigm, a two-stage training framework and an effective resolution strategy to address the core challenges of low signal-to-noise ratio (SNR) and supervision scarcity in long-document understanding. DocSeeker learns fine-grained evidence localization and achieves robust long-document reasoning, even when trained on existing relatively short multi-page datasets. Experimental results demonstrate that our method achieves satisfactory performance and generalization on long document understanding. Furthermore, visual RAG integra- 

8 

tion experiments confirm that DocSeeker effectively alleviates the inherent limitations of visual RAG system, exhibiting remarkable robustness to retrieval noise and achieving strong synergy with the retriever. These results highlight DocSeeker as a promising foundation for building efficient and reliable RAG-based multimodal reasoning systems. 

## **Acknowledgments** 

This work was supported by NSFC (No. 62225603, No. 62576147, No. U25A20538, No. U25B2078). 

## **References** 

- [1] Rubèn Tito, Dimosthenis Karatzas, and Ernest Valveny. Hierarchical multimodal transformers for multipage docvqa. _Pattern Recognition_ , 144:109834, 2023. 1, 2, 5, 6, 19 

- [2] Jordy Van Landeghem, Rubèn Tito, Łukasz Borchmann, Michał Pietruszka, Pawel Joziak, Rafal Powalski, Dawid Jurkiewicz, Mickaël Coustaty, Bertrand Anckaert, Ernest Valveny, et al. Document understanding dataset and evaluation (dude). In _IEEE/CVF International Conference on Computer Vision_ , page 19471–19483, 2023. 2, 5, 12 

- [3] Wenjun Ke, Yifan Zheng, Yining Li, Hengyuan Xu, Dong Nie, Peng Wang, and Yao He. Large language models in document intelligence: A comprehensive survey, recent advances, challenges and future trends. _ACM Transactions on Information Systems_ , 2025. 1 

- [4] Bin Wang, Chao Xu, Xiaomeng Zhao, Linke Ouyang, Fan Wu, Zhiyuan Zhao, Rui Xu, Kaiwen Liu, Yuan Qu, Fukai Shang, et al. Mineru: An open-source solution for precise document content extraction. In _arXiv:2409.18839_ , 2024. 1 

- [5] Hao Feng, Shu Wei, Xiang Fei, Wei Shi, Yingdong Han, Lei Liao, Jinghui Lu, Binghong Wu, Qi Liu, Chunhui Lin, et al. Dolphin: Document image parsing via heterogeneous anchor prompting. In _arXiv:2505.14059_ , 2025. 

- [6] Zhang Li, Yuliang Liu, Qiang Liu, Zhiyin Ma, Ziyang Zhang, Shuo Zhang, Zidun Guo, Jiarui Zhang, Xinyu Wang, and Xiang Bai. Monkeyocr: Document parsing with a structure-recognition-relation triplet paradigm. In _arXiv:2506.05218_ , 2025. 1 

- [7] YL Liu, HL Li, X Bai, et al. A brief analysis of chatgpt: historical evolution current applications and future prospects [j]. _Journal of Image and Graphics_ , 28(04):893–902, 2023. 1 

- [8] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. In _arXiv:2502.13923_ , 2025. 1, 3, 5, 19 

- [9] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. In _arXiv:2504.10479_ , 2025. 3, 5, 6, 19, 20 

- [10] Anwen Hu, Haiyang Xu, Liang Zhang, Jiabo Ye, Ming Yan, Ji Zhang, Qin Jin, Fei Huang, and Jingren Zhou. mplugdocowl2: High-resolution compressing for ocr-free multi- 

   - page document understanding. In _Annual Meeting of the Association for Computational Linguistics_ , pages 5817–5834, 2025. 3, 5, 6, 19, 20 

- [11] Geewook Kim, Teakgyu Hong, Moonbin Yim, JeongYeon Nam, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, Sangdoo Yun, Dongyoon Han, and Seunghyun Park. Ocr-free document understanding transformer. In _European Conference on Computer Vision_ , pages 498–517. Springer, 2022. 

- [12] GLM-V Team, Wenyi Hong, Wenmeng Yu, Xiaotao Gu, Guo Wang, Guobing Gan, Haomiao Tang, Jiale Cheng, Ji Qi, Junhui Ji, Lihang Pan, Shuaiqi Duan, Weihan Wang, Yan Wang, Yean Cheng, Zehai He, Zhe Su, Zhen Yang, Ziyang Pan, Aohan Zeng, Baoxu Wang, Bin Chen, Boyan Shi, Changyu Pang, Chenhui Zhang, Da Yin, Fan Yang, Guoqing Chen, Jiazheng Xu, Jiale Zhu, Jiali Chen, Jing Chen, Jinhao Chen, Jinghao Lin, Jinjiang Wang, Junjie Chen, Leqi Lei, Letian Gong, Leyi Pan, Mingdao Liu, Mingde Xu, Mingzhi Zhang, Qinkai Zheng, Sheng Yang, Shi Zhong, Shiyu Huang, Shuyuan Zhao, Siyan Xue, Shangqin Tu, Shengbiao Meng, Tianshu Zhang, Tianwei Luo, Tianxiang Hao, Tianyu Tong, Wenkai Li, Wei Jia, Xiao Liu, Xiaohan Zhang, Xin Lyu, Xinyue Fan, Xuancheng Huang, Yanling Wang, Yadong Xue, Yanfeng Wang, Yanzi Wang, Yifan An, Yifan Du, Yiming Shi, Yiheng Huang, Yilin Niu, Yuan Wang, Yuanchang Yue, Yuchen Li, Yutao Zhang, Yuting Wang, Yu Wang, Yuxuan Zhang, Zhao Xue, Zhenyu Hou, Zhengxiao Du, Zihan Wang, Peng Zhang, Debing Liu, Bin Xu, Juanzi Li, Minlie Huang, Yuxiao Dong, and Jie Tang. Glm-4.5v and glm-4.1v-thinking: Towards versatile multimodal reasoning with scalable reinforcement learning. In _arXiv:2507.01006_ , 2025. 

- [13] Zhiyu Wu, Xiaokang Chen, Zizheng Pan, Xingchao Liu, Wen Liu, Damai Dai, Huazuo Gao, Yiyang Ma, Chengyue Wu, Bingxuan Wang, Zhenda Xie, Yu Wu, Kai Hu, Jiawei Wang, Yaofeng Sun, Yukun Li, Yishi Piao, Kang Guan, Aixin Liu, Xin Xie, Yuxiang You, Kai Dong, Xingkai Yu, Haowei Zhang, Liang Zhao, Yisong Wang, and Chong Ruan. Deepseek-vl2: Mixture-of-experts visionlanguage models for advanced multimodal understanding. In _arXiv:2412.10302_ , 2024. 

- [14] Yuliang Liu, Biao Yang, Qiang Liu, Zhang Li, Zhiyin Ma, Shuo Zhang, and Xiang Bai. Textmonkey: An ocr-free large multimodal model for understanding document. IEEE, 2026. 

- [15] Kenton Lee, Mandar Joshi, Iulia Raluca Turc, Hexiang Hu, Fangyu Liu, Julian Martin Eisenschlos, Urvashi Khandelwal, Peter Shaw, Ming-Wei Chang, and Kristina Toutanova. Pix2struct: Screenshot parsing as pretraining for visual language understanding. In _International Conference on Machine Learning_ , pages 18893–18912. PMLR, 2023. 

- [16] Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, et al. Minicpm-v: A gpt-4v level mllm on your phone. In _arXiv:2408.01800_ , 2024. 1 

- [17] Shi Yu, Chaoyue Tang, Bokai Xu, Junbo Cui, Junhao Ran, Yukun Yan, Zhenghao Liu, Shuo Wang, Xu Han, Zhiyuan Liu, et al. Visrag: Vision-based retrieval-augmented gener- 

9 

ation on multi-modality documents. In _arXiv:2410.10594_ , 2024. 1, 3, 5, 6, 19, 20 

- [18] Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani, Gautier Viaud, Céline Hudelot, and Pierre Colombo. Colpali: Efficient document retrieval with vision language models. In _arXiv:2407.01449_ , 2024. 3, 7, 19 

- [19] Yuanlei Zheng, Pei Fu, Hang Li, Ziyang Wang, Yuyi Zhang, Wenyu Ruan, Xiaojin Zhang, Zhongyu Wei, Zhenbo Luo, Jian Luan, et al. Doc-v*: Coarse-to-fine interactive visual reasoning for multi-page document vqa. In _arXiv:2604.13731_ , 2026. 1 

- [20] Jiabo Ye, Anwen Hu, Haiyang Xu, Qinghao Ye, Ming Yan, Yuhao Dan, Chenlin Zhao, Guohai Xu, Chenliang Li, Junfeng Tian, et al. mplug-docowl: Modularized multimodal large language model for document understanding. In _arXiv:2307.02499_ , 2023. 1 

- [21] Xudong Xie, Hao Yan, Liang Yin, Yang Liu, Jing Ding, Minghui Liao, Yuliang Liu, Wei Chen, and Xiang Bai. Pdfwukong: A large multimodal model for efficient long pdf reading with end-to-end sparse sampling. In _International Journal of Computer Vision_ , 2026. 19 

- [22] Wenwen Yu, Zhibo Yang, Yuliang Liu, and Xiang Bai. Docthinker: Explainable multimodal large language models with rule-based reinforcement learning for document understanding. In _IEEE/CVF International Conference on Computer Vision_ , pages 837–847, 2025. 1 

- [23] Ryota Tanaka, Kyosuke Nishida, Kosuke Nishida, Taku Hasegawa, Itsumi Saito, and Kuniko Saito. Slidevqa: A dataset for document visual question answering on multiple images. In _AAAI Conference on Artificial Intelligence_ , volume 37, pages 13636–13645, 2023. 2, 5 

- [24] Linjie Li, Yuxuan Wang, Rui Xu, Peiyi Wang, Xinyun Feng, Lingpeng Kong, and Qun Liu. MultimodalArXiv: A dataset for improving scientific comprehension of large vision-language models. In _Annual Meeting of the Association for Computational Linguistics_ , pages 14369–14387, 2024. 

- [25] Yihao Ding, Kaixuan Ren, Jiabin Huang, Siwen Luo, and Soyeon Caren Han. MVQA: A dataset for multimodal information retrieval in pdf-based visual question answering. In _arXiv:2404.12720_ , 2024. 2 

- [26] Yubo Ma, Yuhang Zang, Liangyu Chen, Meiqi Chen, Yizhu Jiao, Xinze Li, Xinyuan Lu, Ziyu Liu, Yan Ma, Xiaoyi Dong, et al. Mmlongbench-doc: Benchmarking long-context document understanding with visualizations. In _Advances in Neural Information Processing Systems_ , volume 37, pages 95963–96010, 2024. 3, 5 

- [27] Chao Deng, Jiale Yuan, Pi Bu, Peijie Wang, Zhong-Zhi Li, Jian Xu, Xiao-Hui Li, Yuan Gao, Jun Song, Bo Zheng, et al. Longdocurl: a comprehensive multimodal long document benchmark integrating understanding, reasoning, and locating. In _Annual Meeting of the Association for Computational Linguistics_ , pages 1135–1159, 2025. 3, 5 

- [28] Kuicai Dong, Yujing Chang, Derrick Goh Xin Deik, Dexun Li, Ruiming Tang, and Yong Liu. MMDocIR: Benchmarking multimodal retrieval for long documents. In _Conference on Empirical Methods in Natural Language Processing_ , pages 30971–31005, 2025. 3 

- [29] Yew Ken Chia, Liying Cheng, Hou Pong Chan, Maojia Song, Chaoqun Liu, Mahani Aljunied, Soujanya Poria, and Lidong Bing. M-LongDoc: A benchmark for multimodal super-long document understanding and a retrieval-aware tuning framework. In _Conference on Empirical Methods in Natural Language Processing_ , pages 9233–9250, 2025. 3 

- [30] Anni Zou, Wenhao Yu, Hongming Zhang, Kaixin Ma, Deng Cai, Zhuosheng Zhang, Hai Zhao, and Dong Yu. DocBench: A benchmark for evaluating LLM-based document reading systems. In _International Workshop on Knowledge-Augmented Methods for Natural Language Processing_ , pages 359–373, 2025. 3 

- [31] Iz Beltagy, Matthew E Peters, and Arman Cohan. Longformer: The long-document transformer. In _arXiv:2004.05150_ , 2020. 3, 19 

- [32] Zhaoqing Zhu, Chuwei Luo, Zirui Shao, Feiyu Gao, Hangdi Xing, Qi Zheng, and Ji Zhang. A simple yet effective layout token in large language models for document understanding. In _IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 14472–14482, 2025. 3, 19, 20 

- [33] Xueguang Ma, Sheng-Chieh Lin, Minghan Li, Wenhu Chen, and Jimmy Lin. Unifying multimodal retrieval via document screenshot embedding. In _Conference on Empirical Methods in Natural Language Processing_ , pages 6492–6505, 2024. 3 

- [34] Jian Chen, Ruiyi Zhang, Yufan Zhou, Tong Yu, Franck Dernoncourt, Jiuxiang Gu, Ryan A Rossi, Changyou Chen, and Tong Sun. Sv-rag: Lora-contextualizing adaptation of mllms for long document understanding. In _arXiv:2411.01106_ , 2024. 3, 5, 6, 19 

- [35] Ryota Tanaka, Taichi Iki, Taku Hasegawa, Kyosuke Nishida, Kuniko Saito, and Jun Suzuki. Vdocrag: Retrievalaugmented generation over visually-rich documents. In _IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 24827–24837, 2025. 3, 6, 19, 20 

- [36] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In _Advances in Neural Information Processing Systems_ , volume 30, 2017. 3 

- [37] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In _arXiv:2010.11929_ , 2020. 3 

- [38] Keith Stenning and Michiel Van Lambalgen. _Human reasoning and cognitive science_ . MIT Press, 2008. 3 

- [39] Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. In _arXiv:2412.16720_ , 2024. 3 

- [40] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. In _arXiv:2501.12948_ , 2025. 3, 5 

- [41] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: 

10 

Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. In _arXiv:2507.06261_ , 2025. 4, 5, 12 

attention recycling. In _Advances in Neural Information Processing Systems_ , 2025. 20 

- [42] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. In _arXiv:2410.21276_ , 2024. 4, 6 

- [43] Li Yujian and Liu Bo. A normalized levenshtein distance metric. _IEEE Transactions on Pattern Analysis and Machine Intelligence_ , 29(6):1091–1095, 2007. 5 

- [44] Fanqing Meng, Lingxiao Du, Zongkai Liu, Zhixiang Zhou, Quanfeng Lu, Daocheng Fu, Tiancheng Han, Botian Shi, Wenhai Wang, Junjun He, et al. Mm-eureka: Exploring the frontiers of multimodal reasoning with rule-based reinforcement learning. In _arXiv:2503.07365_ , 2025. 5 

- [45] Jinxu Zhang, Yongqi Yu, and Yu Zhang. Cream: coarse-tofine retrieval and multi-modal efficient tuning for document vqa. In _ACM International Conference on Multimedia_ , pages 925–934, 2024. 5, 6, 19 

- [46] Jaemin Cho, Debanjan Mahata, Ozan Irsoy, Yujie He, and Mohit Bansal. M3docrag: Multi-modal retrieval is what you need for multi-page multi-document understanding. In _arXiv:2411.04952_ , 2024. 5, 6, 19 

- [47] Yuchen Duan, Zhe Chen, Yusong Hu, Weiyun Wang, Shenglong Ye, Botian Shi, Lewei Lu, Qibin Hou, Tong Lu, Hongsheng Li, et al. Docopilot: Improving multimodal models for document-level understanding. In _IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 4026– 4037, 2025. 5, 6, 19 

- [48] Mor Shpigel Nacson, Aviad Aberdam, Roy Ganz, Elad Ben Avraham, Alona Golts, Yair Kittenplon, Shai Mazor, and Ron Litman. Docvlm: Make your vlm an efficient reader. In _IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 29005–29015, 2025. 5, 6, 19 

- [49] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. In _arXiv:2308.12966_ , 2023. 6 

- [50] Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. In _arXiv:2403.05530_ , 2024. 6 

- [51] Shitao Xiao, Zheng Liu, Peitian Zhang, and Niklas Muennighoff. C-Pack: Packaged resources to advance general chinese embedding. In _arXiv:2309.07597_ , 2023. 19 

- [52] Dongsheng Wang, Natraj Raman, Mathieu Sibue, Zhiqiang Ma, Petr Babkin, Simerjot Kaur, Yulong Pei, Armineh Nourbakhsh, and Xiaomo Liu. DocLLM: A layout-aware generative language model for multimodal document understanding. In _arXiv:2401.00908_ , 2023. 19, 20 

- [53] Yang Liu, Xudong Xie, Yuliang Liu, and Xiang Bai. Multiscenario overlapping text segmentation with depth awareness. In _IEEE/CVF International Conference on Computer Vision_ , pages 17454–17463, 2025. 20 

- [54] Liang Yin, Xudong Xie, Zhang Li, Xiang Bai, and Yuliang Liu. Mstar: Box-free multi-query scene text retrieval with 

11 

## **APPENDIX** 

## **A. Details of Data Distillation** 

## **A.1. Overview of the Distillation Pipeline** 

To address the issue of supervision scarcity in existing multi-page document VQA datasets, which typically provide only final short answers and evidence page indices without intermediate reasoning steps, we designed a rigorous data distillation pipeline. As illustrated in Figure 5, this pipeline transforms raw samples into high-quality structured ALR CoT data. The overall process consists of three key phases: 

**Data Filtering and Context Construction.** We strictly select long-document samples from MP-DocVQA and DUDE, applying a pre-filtering step to discard documents that are overly short. Subsequently, we construct a "Minimal Context" for each sample, composed exclusively of the evidence pages, corresponding Page IDs, and the question. This strategy significantly reduces distillation costs while substantially enhancing the quality of generated data. Quantitative analysis on a subset of 1,000 long documents (avg. 17.6 pages) demonstrates that this approach improves the distillation success rate from 20.4% (full-context input) to 67.3%. 

**ALR CoT Data Distillation.** We employ Gemini-2.5Flash [41] as the teacher model due to its strong multimodal reasoning capabilities. By utilizing a specialized prompt, we instruct the teacher to act as an expert annotator. The teacher generates a structured response that explicitly includes Question Analysis, Evidence Localization (citing specific Page IDs), and a Reasoning Process, strictly following our proposed ALR paradigm. 

**Secondary Verification.** To guarantee the correctness of the synthesized reasoning paths, we implement a robust two-stage verification mechanism. First, we apply an automated Exact Match (EM) filter to check if the generated final answer and evidence page IDs align perfectly with the ground truth. For samples that fail the strict EM check (e.g., due to paraphrasing), we employ GPT-4o as a semantic judge to validate the correctness of the answer . Only samples passing this verification are retained for the SFT stage. 

## **A.2. Prompt Templates** 

This section outlines the specific prompt templates integral to our data distillation pipeline. Figure 6 presents the ALR CoT Data Distillation Prompt, designed to extract high-quality structured reasoning from the teacher model. Furthermore, we provide the Vanilla CoT Data Distillation Prompt in Figure 7 for ablation studies; unlike the ALR approach, this prompt encourages the model to generate 

free-form reasoning chains without enforcing the structured constraints of question analysis or explicit evidence localization. Finally, Figure 8 illustrates the LLM-based Judge Prompt, employed during secondary verification to salvage factually correct samples initially discarded due to formatting mismatches. 

## **A.3. Failure Analysis** 

We visualize representative examples of samples that failed the initial automated Exact Match screening in Figure 9, which fall into two distinct categories: 

1) **Genuine Reasoning Failures.** These cases contain substantive quality defects. Despite attempting to follow the ALR paradigm, the model commits logical errors during the reasoning process, resulting in factually incorrect answers. These constitute actual noise and must be strictly discarded. 

2) **Correct Reasoning with Formatting Mismatches.** These samples exhibit correct analysis, localization, and reasoning process. The derived answers are semantically correct but fail the EM check solely due to discrepancies in formatting, e.g. 17th February 1916 vs 1916-02-17. Such samples also represent high-quality supervision signals that should be preserved to maximize data utilization. 

These observations underscore the necessity of the Secondary Verification mechanism introduced in Section A.1, which is essential for accurately distinguishing valid semantic matches from genuine reasoning errors. 

## **B. Additional Ablation Study Results** 

To delve deeper into the impact of distinct reasoning paradigms and data strategies on Generalization Capabilities, we supplemented the In-domain (DUDE [2]) evaluation in Table 6. SFT on Raw Short-Answer data yields In-domain gains comparable to other paradigms, yet improvement on OOD tasks remains negligible. This suggests the model primarily engages in Rote Memorization rather than mastering the general reasoning skills required for unseen long documents. In contrast, increasing the scale of ALR CoT data not only progressively enhances Indomain performance but, critically, achieves a synchronous leap in OOD tasks. This demonstrates that the structured "Analysis-Localization-Reasoning" paradigm instills transferable reasoning capabilities, enabling the model to transcend specific data distributions and actively adapt to unfamiliar long-document structures, thereby achieving true generalization. 

12 

**==> picture [472 x 231] intentionally omitted <==**

**----- Start of picture text -----**<br>
Minimal Context from<br>Long Document Verification<br>Question <think><br>\boxed{ Question Analysis }<br>Q : What is the date reported<br>Raw Data at the beginning of the  Short Answer The user is asking to identify the<br>document on page one? Generator date…<br>\boxed{ Evidence Localization }<br>Evidence Document I am reviewing the provided<br>𝑎 output image for  page 1 . At the very top<br>of the page, ……states  "March 3,<br>Evidence preprocess compare 2020 - Polling Place<br>filter recompare Locations"  ……<br>(doc length) Generation  \boxed{ Reasoning Process }<br>Prompt output ……The most prominent date at<br>the start of the document is<br>… Judger “March 3, 2020”……<br>Question Evidence Info discard </think><br>Short Answer Evidence Page<br>Page 1: <image> <answer>{evidence_pages: [1],  answer: 2020-03-03 }<br></answer><br>Short Answer<br>Data Filtering &  ALR CoT Data  Secondary<br>— Context Construction oo Distillation ee Verification<br>**----- End of picture text -----**<br>


Figure 5. Data Distillation Pipeline. 

Table 6. Ablation study on data types and training data size. The first group investigates different CoT data configurations, while the second explores the effect of varying data size on performance. 

|**Data Config**|**Size**|**MMLongbench.**|**MMLongbench.**|**DUDE**|
|---|---|---|---|---|
|||Acc.|F1|ANLS|
|Baseline|-|25.4|20.8|35.2|
|Raw short-answer data|6.3k|27.4|27.6|48.8|
|Vanilla CoT data|6.3k|31.3|32.4|48.7|
|ALR CoT data|6.3k|**33.8**|**33.9**|48.9|
|-w/o Page id|6.3k|30.4|31.1|47.5|
|20% of ALR CoT data|2.8k|32.7|31.7|46.5|
|40% of ALR CoT data|5.6k|35.8|33.8|52.3|
|60% of ALR CoT data|8.4k|36.5|34.9|54.5|
|80% of ALR CoT data|11k|38.2|36.5|55.9|
|All ALR CoT data|13k|**38.6**|**36.9**|**56.5**|



## **C. Efficiency and Accuracy Trade-offs** 

Although the proposed ALR paradigm enhances the model’s capability in long-document understanding, it inevitably introduces modest additional computational latency. To investigate the trade-off between inference latency and accuracy, we conducted experiments on MMLongBench-Doc, as shown in Table 7. Although the ALR paradigm nearly doubles output tokens, the end-toend latency increases only modestly from 19s to 25s. This is because the primary computational bottleneck in longdocument processing is the visual pre-fill stage rather than 

the text decoding stage. Ultimately, DocSeeker trades a 31.6% latency increase for a 14.7% absolute accuracy gain and enhanced interpretability. 

Table 7. Analysis of inference latency and accuracy trade-off on MMLongBench-Doc 

||Token|Latency|Acc|
|---|---|---|---|
|Baseline|202|19s|25.4|
|DocSeeker|401|25s|40.1|



13 

## **ALR CoT Data Distillation Prompt** 

## **Task Definition:** 

You are an AI assistant specialized in constructing high-quality question-answering datasets. Your core task is to generate a detailed reasoning process and a complete, natural answer based on the given question, the corresponding document evidence page, and the page number. 

## **Important Rule:** 

All reasoning and the final answer must be strictly derived from the provided _Document Page Content_ , without using external knowledge or unsupported guesses. 

## **Output Requirements:** 

Your output must strictly contain two sections, “<think>” and “<answer>” following the exact format below. 

## **Expected Output Format:** 

Reasoning Process ("<think>" section) This section must use "\boxed{}" to wrap the following three subhead -ings to clearly present your reasoning steps: <think> \boxed{Question Analysis} Analyze the intent of the question in depth. Clearly identify what entity, concept, quantity, or relationship the question is asking about. Break complex questions down if needed. \boxed{Evidence Localization} Identify which page contains the evidence, then locate the key content within the provided Document Page Content. \boxed{Reasoning Process} Derive the answer step by step using only localized evidence. Explain how the conclusion is formed. </think> Final Answer ("<answer>" section) This section must be a single JSON object containing: <answer> evidence_pages: A JSON list including the page numbers. answer: A concise string giving the direct answer without introductory phrases or full sentence structures. </answer> 

## **Example:** 

**Question:** mention the texts in brackets with bold fonts in paragraph 2 in page 5? **Document Page Number:** [5] 

<think> \boxed{Question Analysis} The user asks for bracketed text in paragraph 2 on page 5 that is also bolded. \boxed{Evidence Localization} Page 5 contains bracketed text but no bolded text. \boxed{Reasoning Process} No text meets all required conditions; therefore the question cannot be answered. </think> <answer> {"evidence_pages": [5], "answer": "Not answerable"} </answer> 

14 

Figure 6. ALR CoT Data Distillation Prompt Templete. 

## **Vanilla CoT Data Distillation Prompt** 

## **Task Definition:** 

You are an AI assistant specialized in constructing high-quality question-answering datasets. Your core task is to generate a detailed reasoning process and a complete, natural answer based on the given question, the corresponding document evidence page, and the page number. 

## **Requirements:** 

- All your reasoning and final answer must be strictly derived from the provided “Document Page Content”. 

- You are not allowed to use any external knowledge or make unsupported guesses. 

- Your output must strictly follow the format below with two required sections: “<think>” and “<answer>“. 

## **Expected Output:** 

<think> 

Let’s think step by step. 

... 

</think> 

<answer> 

A string that provides a direct and concise answer to the question without any introductory phrases or full sentence structures. 

</answer> 

Figure 7. Vanilla CoT Data Distillation Prompt Templete. 

Table 8. Stage-wise error breakdown on MMLongBench-Doc. 

|||**Acc**_≥_0_._5|**Acc**=|0|**Total**|
|---|---|---|---|---|---|
|**Recall**=|1|300|288||588|
|**Recall**_<_|1|121|373||494|
|**Total**||421|661||1082|



## **E. Details of Compared Methods** 

Table 9 details the specific evaluation configurations for the compared models across five benchmarks. For all baselines, we strictly adhered to the officially recommended or optimal settings to ensure a fair comparison. 

## **D. Stage-wise Error Breakdown** 

To systematically decompose the errors encountered in long-document understanding, we perform a stage-wise error breakdown on the samples from MMLongBench-Doc. As detailed in Table 8, we categorize these samples based on the correlation between evidence recall (Recall) and answer accuracy (Acc), which reveals three primary failure modes: 

(i) **Reasoning failures.** The model successfully retrieves all required evidence pages ( _Recall_ = 1) but fails to provide the correct final deduction 

(ii) **Localization failures.** The model fails to localize the evidence pages, leading to a lack of sufficient context for the model to perform downstream reasoning. 

(iii) **Exceptions.** The model provides correct answers despite incomplete localization ( _Recall <_ 1) by leveraging its internal global context awareness. 

## **F. Case Studies** 

In this section, we present qualitative visualizations of representative examples to demonstrate the superior capabilities of DocSeeker in handling complex long-document understanding tasks. 

## **F.1. Comparative Analysis of Reasoning Paradigms** 

Figure 10 presents a qualitative comparison of DocSeeker against alternative reasoning paradigms. Experimental results indicate that the traditional Baseline model, constrained by limited long-context processing capabilities, exhibits complete localization failure, whereas the ShortAnswer model yields "black-box" predictions devoid of logical support. Vanilla CoT successfully localizes evidence, it suffers from "reasoning drift" during the unstructured generation process, resulting in deviations within the logical chain. In contrast, DocSeeker leverages the structured ALR paradigm to strictly decouple evidence acquisition from 

15 

## **Judge Prompt** 

## **Task Definition:** 

You are an expert AI assistant for data validation and correction. Your core task is to compare a model-generated response with a ground-truth answer and determine the final, most accurate output based on a set of rules. 

## **Inputs:** 

You will be given three pieces of information: _question_ , _response_ , and _answer_ . 

## **Requirements:** 

Your output should be a single, clean string representing the corrected answer, or the word ‘Error’ if applicable. 

## **Rules:** 

1. **Formatting Mismatch:** If the response and answer contain the same core information but differ only in formatting (e.g., date separators like / vs. -, number separators like , vs. ., or case sensitivity), you must output the version from the ground-truth answer. 

2. **Overly Detailed Response:** If the response is factually correct and contains all the information from the answer, but includes additional conversational phrases or details (e.g., “The answer is X” vs. “X”), you should conservatively output the more concise ground-truth answer. 

3. **Overly Simplified Response:** If the response is correct but misses key details that are present in the answer, you must output the more complete ground-truth answer. 

4. **Correct Response, Flawed Answer:** If you determine that the response is factually or grammatically more correct than the ground-truth answer (e.g., the answer has a clear typo or factual error that the response corrects), you must output the model-generated response to ensure data quality. 

5. **Factually Incorrect/Contradictory:** If the response and answer are factually contradictory and irreconcilable (e.g., different names, numbers, dates), or if one provides a factual answer while the other states it’s not answerable, you must output the exact string Error. 

**Output Format:** 

output: <answer> 

Please read the following example and complete the output in the specified format based on the given information. 

**Example: Question:** WHAT IS THE APPROVED DATE FOR RELEASE? **Response:** 2001/07/27 **Answer:** 2001-07-27 **Output:** 2001-07-27 

Figure 8. LLM-based Judge Prompt Templete in Secondary Verification. 

logical deduction. This constraint mechanism effectively complex long-document understanding. mitigates reasoning instability and ensures the execution of rigorous inference steps based on precise evidence, thereby demonstrating the significant robustness of our paradigm in 

16 

**==> picture [482 x 96] intentionally omitted <==**

**----- Start of picture text -----**<br>
Failure Cases<br>Question Ground Truth<br>How many Adjudicatory Cases are<br>Answer:  12    Evidence_pages:  [3, 4, 5]   Evidence_sources:  [‘Table’]<br>there in the document?<br>**----- End of picture text -----**<br>


**==> picture [34 x 7] intentionally omitted <==**

**----- Start of picture text -----**<br>
<think><br>**----- End of picture text -----**<br>


## **\\boxed{Question Analysis}** 

The user is asking for the total count of "Adjudicatory Cases" present in the document. This requires identifying the section dedicated to adjudicatory cases and then counting each individual case listed within that section. **\\boxed{Evidence Localization}** I will examine the provided document pages (3, 4, and 5). On page 3, there is a clear heading \"ADJUDICATORY CASES\". Under this heading, cases are listed, each identified by a \"PCB\" number. I will count each distinct entry under this section across all provided pages. **\\boxed{Reasoning Process}** 

1. Locate the section: The \"ADJUDICATORY CASES\" section begins on page 3. 

**==> picture [483 x 191] intentionally omitted <==**

**----- Start of picture text -----**<br>
2. Count cases on Page 3:- PCB 21-3, PCB 21-5 (One combined entry)- PCB 21-29, -PCB 21-47  (page 3: 3 cases)<br>3. Count cases on Page 4:    - PCB 21-110    - PCB 21-114    - PCB 22-11… Flawed reasoning<br>5. Sum the counts: 3 (page 3) + 7 (page 4) + 1 (page 5) = 11. Therefore, there are 11 adjudicatory cases listed.<br></think><br><answer><br>{\"evidence_pages\":  [3, 4, 5] , \"answer\": \" 11 \"} Wrong Answer<br></answer><br>Question Ground Truth<br>What date is mentioned on page three? Answer: 1916-02-17 Evidence_pages:  [3]   Evidence_sources:  ['Chart’]<br><think><br>\\boxed{Question Analysis}<br>**----- End of picture text -----**<br>


The user is asking for a specific piece of information: a date. The question specifies that this date should be found on \"page three\". My task is to locate any date mentioned within the provided document content, which corresponds to **page 3.** 

**\\boxed{Evidence Localization}** 

I am reviewing the provided OCR text, which is confirmed to be from page 3. Near the top of the page, below the title and author, there is a line that reads: \" **Read 17th February 1916** .\" This is a clear and explicit date. **\\boxed{Reasoning Process}** The question asks for the date mentioned on page three. The evidence localization step identified **\"17th February 1916\"** as the only date present on the provided page content. Therefore, this is the direct answer to the question. **</think>** 

**<answer>** {\"evidence_pages\": [3],  \"answer\": \" **17th February 1916** \"} _**Answer correct but formatting mismatch**_ **</answer>"** 

Figure 9. Failure cases in ALR CoT Data Distillation. 

17 

**==> picture [399 x 169] intentionally omitted <==**

**----- Start of picture text -----**<br>
Case Study<br>Question Ground Truth<br>Assume that in a survey conducted after July 2020, the percentage of<br>Answer:  54%<br>Americans who believe China has done a "good" job dealing with the<br>coronavirus outbreak increased by 10 percentage points, then what  Evidence_pages:  [5]<br>percentage of  Americans would believe China has done a "bad" job<br>Evidence_sources:  ['Chart’]<br>(assuming the percentage of all the other options stays the same)?<br>China has done a __job dealing with the coronavirus<br>outbreak<br>toteinthe Spend = Bad<br>…<br>5 Good<br>Localization failure<br>**----- End of picture text -----**<br>


## **Baseline** 

Based on … if the percentage of ... a “good” job … increased by 10 percentage points, we can calculate the new percentage for those who believe China has done a “bad” job. Let’s assume the original percentage of  … a “good” job was ( x ).  After the increase, it becomes ( x + 10 ) … Therefore, the percentage of Americans who believe China has done a “bad” job is: [ 100% - (x + 10) = 90% - x ]. Without knowing the exact original value of ( x ), **we cannot provide a specific numerical answer.** However, if you provide …  I can calculate … 

## **Fine-tuned on short answer data** 

## _**Wrong Answer  & Missing Reasoning Process**_ 

31. 

**Vanilla CoT** 

## _**Correct localization & Incorrect reasoning**_ 

## **<think>** 

1.  Analyze  …; 2.  Scan the document: …3.  Locate the relevant chart: I see **a bar chart titled “Most Americans** f **see China as having dealt poorly with COVID-19”** shows …; 4.  Examine the chart's data: The chart has two categories: "Bad" and "Good."  … ; 5.  Extract the specific numbers: The question states … increased by 10 percentage points. Looking at the chart, the "Good" category for the "Total" group is **31%** .  … the original percentage as (x). The problem says (x + 10 = 31). **Solving for (x): (x = 31 - 10 = 21).** So, the original percentage …"good" job was **21%** .  6.  Calculate the percentage who believe China has done a "bad" job:  The total must be 100%. If 21% believe it has been good, then (100% - 21% = **79%** ) believe it has been bad… **</think>** 

**<answer>79%</answer>** 

## **DocSeeker-SFT** 

## **<think>** 

## _**Correct localization   Incomplete reasoning**_ 

**{Question Analysis}** : … **{Evidence Localization}** : I need to find the initial percentages for "Bad" and "Good" jobs from the provided document.- The document states: … This refers to blame, not ... However, there is a chart titled “…" which shows: … **-Bad: 64% - Good: 31%.  {Reasoning Process}** : … From the evidence, the initial percentage for "Good" was 31%. An increase of 10 percentage points means the new percentage for "Good" is 31% + 10% = 41%. Since the total must sum to 100%, the percentage for "Bad" will be 100% - 41% = **59%** ... 

## **</think>** 

**<answer>** {"evidence_pages": [5], "answer": "output: **59** "} **</answer>** 

## **DocSeeker** 

## _**Correct localization   Correct reasoning**_ 

## **<think>** 

**{Question Analysis}** : … **{Evidence Localization}** :… **the "Bad" shows 64% and the "Good" shows 31% … {Reasoning Process}** : The question states … increased by 10 percentage points. From the evidence, the initial percentage for "Good" is 31% … the new percentage for "Good" is 31% + 10% = 41%. 

Since the total percentage of all responses must sum to 100%, the percentage for "Bad" will be 100% - 41% = 59%. Therefore, if the "Good" percentage increases by 10 points, the "Bad" percentage will decrease by 10 points from its original value of 64%. The original "Bad" percentage was 64%, so the new "Bad" percentage is **54%.** 

## **</think>** 

**<answer>** { “ evidence_pages ” : [5], “ answer ” : “ output: **54** "} **</answer>** 

18 Figure 10. Qualitative Comparison of Reasoning Paradigms. 

Table 9. Detailed Configurations of Compared Methods on Long Document Understanding. 

|**Method**<br>**Model**<br>**OCR**|**Retrieval Confg**<br>**Backbone**<br>**Trained on Dataset**<br>**Retriever**<br>**Top-k**<br>**DUDE MPDocVQA SlideVQA**|**Trained on Dataset**|
|---|---|---|
|**RAG**<br>CREAM [45]<br>✓<br>bge-large [51]<br>3<br>Pix2Struct/LLaMa2<br>✓<br>✓<br>_×_<br>M3DocRAG [46]<br>_×_<br>Colpali [18]<br>4<br>Qwen2-VL<br>_×_<br>_×_<br>_×_<br>Vis-RAG [17]<br>_×_<br>VisRAG-Ret [17]<br>3<br>MiniCPM-V 2.6<br>_×_<br>_×_<br>_×_<br>SV-RAG [34]<br>_×_<br>SV-RAG-InternVL2 [34]<br>5<br>SV-RAG-InternVL2<br>_×_<br>_×_<br>✓<br>VDocRAG [35]<br>_×_<br>VDocRetriever [35]<br>3<br>VDocGenerator<br>✓<br>_×_<br>_×_|||
|**End-to-End**<br>HiVT5 [1]<br>✓<br>-<br>-<br>DiT/T5<br>_×_<br>✓<br>_×_<br>mPLUG-DocOwl2 [10]<br>_×_<br>-<br>-<br>ViT/LLaMa+MAM<br>✓<br>✓<br>_×_<br>Docopilot [47]<br>_×_<br>-<br>-<br>InternVL2<br>✓<br>✓<br>_×_<br>DocVLM [48]<br>✓<br>-<br>-<br>Qwen2-VL<br>_×_<br>_×_<br>_×_<br>InternVL3[9]<br>_×_<br>-<br>-<br>InternViT/Qwen2.5<br>_×_<br>_×_<br>_×_|||



Table 10. Comparison of method architectures and attributes, where "OCR" and "Retriever" denote dependency on external modules, "Training-free" indicates the absence of additional fine-tuning on multi-page documents, "Page-level Reasoning" refers to processing at page granularity, and "Evidence Localization" marks the capability for explicit source grounding. 

|**Method**|**OCR**|**Retriever**|**Backbone**|**Training-**<br>**free**|**Page-level**<br>**Reasoning**|**Evidence**<br>**Localization**|
|---|---|---|---|---|---|---|
|_OCR-based_|||||||
|Longformer [31]|✓|_×_|RoBERTa|✓|_×_|_×_|
|DocLLM [52]|✓|_×_|Falcon/LLaMa2|✓|_×_|_×_|
|LayTokenLLM[32]|✓|_×_|LLaMa3/LLaMa2/Qwen1.5|_×_|_×_|_×_|
|_End-to-End_|||||||
|HiVT5 [1]|✓|_×_|DiT/T5|_×_|_×_|_×_|
|mPLUG-DocOwl2 [10]|_×_|_×_|ViT/LLaMa+MAM|_×_|✓|_×_|
|Docopilot [47]|_×_|_×_|InternVL2|_×_|✓|_×_|
|DocVLM [48]|✓|_×_|Qwen2-VL|✓|_×_|_×_|
|InternVL3 [9]|_×_|_×_|InternViT/Qwen2.5|✓|✓|_×_|
|Qwen2.5VL[8]|_×_|_×_|QwenViT/Qwen2.5|✓|✓|_×_|
|_RAG_|||||||
|PDF-WuKong [21]|✓|bge-m3 [51]|XComposer2-4KHD|_×_|_×_|✓|
|CREAM [45]|✓|bge-large [51]|Pix2Struct/LLaMa2|_×_|_×_|✓|
|M3DocRAG [46]|_×_|Colpali [18]|Qwen2-VL|✓|✓|✓|
|Vis-RAG [17]|_×_|VisRAG-Ret [17]|MiniCPM-V 2.6|✓|✓|✓|
|SV-RAG [34]|_×_|SV-RAG-InternVL2 [34]|SV-RAG-InternVL2|_×_|✓|✓|
|VDocRAG[35]|_×_|VDocRetriever[35]|VDocGenerator|_×_|✓|✓|



## **F.2. Efficacy of the Two-Stage Training Framework** 

Figure 11 illustrates the evolution of model capabilities within the two-stage training framework. After SFT on ALR CoT data, DocSeeker-SFT successfully acquires the structural form of the ALR paradigm, strictly adhering to the "Analysis-Localization-Reasoning" workflow; however, it exhibits unstable grounding capabilities, evidenced by a incorrect page that results in an erroneous conclusion. This indicates that while SFT facilitates the imitation of reasoning patterns, it is insufficient for ensuring factual precision. The subsequent introduction of Evidence-aware GRPO effectively rectifies this limitation. Guided by localization- 

specific reward signals, the model achieves precise evidence grounding while retaining the structured reasoning path, demonstrating that the reinforcement learning stage is critical for optimizing localization accuracy and enhancing overall performance in long-document understanding. 

## **G. Related Work** 

To provide a more intuitive comparison of distinct technical approaches in the field of long-document understanding, we summarize related studies in Table 10. These methods can be primarily categorized into the following three paradigms: 

1) OCR-based Methods: These approaches,such as Lay- 

19 

**==> picture [471 x 359] intentionally omitted <==**

**----- Start of picture text -----**<br>
Case Study<br>Question<br>What position is the man with the red shirt in the figure on the last page, directly answer 'bottom', 'middle' or 'top'?<br>Cn<br>Ground Truth<br>Answer:  bottom<br>… … Evidence_pages:  [77]<br>Evidence_sources:  ['Figure’]<br>1 Page 1 Page 40 Page 76 Page 77<br>Baseline DocSeeker-SFT DocSeeker<br>Response:  Response:  Response:<br>The image shows a group of  {Question Analysis} {Question Analysis}<br>people posing for a photo.  The user is asking … This requires… The user is asking … This requires…<br>The text “Thank u very  {Evidence Localization} {Evidence Localization}<br>much.” appears at the top of  …  page 77 ...see a photo of several men.<br>… is  page 71 …see a group of people.<br>the image, which is likely a  In the front row, …a man wearing ... is<br>One individual in the front row, …<br>message of gratitude from  positioned centrally in the front row.<br>one person to another. This  Incorrect localization Precise localization<br>could be ....  {Reasoning Process} {Reasoning Process}<br>.... This individual is positioned  The man wearing the red shirt is located<br>———————<br>If you have any specific  centrally in the front row … suggests  in the lower half … specifically in the<br>questions about the image or  a prominent or middle position.  front row… the  lowest visible row  of<br>need further details,  feel  Therefore… is  in the middle . people …Therefore, … is  "bottom"<br>free to ask! Correct reasoning<br>{Answer}<br>Fail to answer {Answer}<br>a "evidence_pages": [71] 8<br>"evidence_pages": [77]<br>Answer:  Fail to answer "answer": "middle" "answer":  "bottom"<br>Score: 0.0 Score: 0.0 Score: 1.0<br>**----- End of picture text -----**<br>


Figure 11. Performance Comparison across Different Training Stages. 

TokenLLM [32] and DocLLM [52], rely on external OCR engines to extract textual information from documents, which then serves as input for the model. With the continuous advancement of OCR [53, 54], these methods are becoming increasingly accurate and efficient. 

2) RAG-based Methods. To address the limitations of MLLMs in processing ultra-long documents, these method- 

s, such as Vis-RAG [17] and VDocRAG [35], introduce a retriever to select the Top- _k_ most relevant pages for subsequent reasoning. 

3) End-to-End Methods. These approaches, such as mPLUG-DocOwl2 [10] and InternVL3 [9], directly encode document images into visual tokens for processing, thereby preserving complete visual features. 

20 

