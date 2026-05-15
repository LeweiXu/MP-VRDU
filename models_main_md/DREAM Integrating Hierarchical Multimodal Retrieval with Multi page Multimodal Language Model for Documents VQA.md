# **DREAM: Integrating Hierarchical Multimodal Retrieval with Multi-page Multimodal Language Model for Documents VQA** 

Jinxu Zhang Harbin Institute of Technology Harbin, China jxzhang@ir.hit.edu.cn 

Yongqi Yu Harbin Institute of Technology Harbin, China yqyu@ir.hit.edu.cn 

## **Abstract** 

Understanding the content of multi-page documents with rich layout information is a challenging task. Recent multimodal large language models (MLLMs) have made remarkable progress in understanding single-page document images. However, the understanding of multi-page documents remains insufficiently explored. This work proposes a **D** ocument **R** etrieval-enhanced, **E** xpert-guided, **A** ttention-aware **M** ultimodal Framework, dubbed DREAM. Specifically, we propose a confidence-based, high-level semantic, multimodal retrieval method. Then, we propose a machine learning algorithm to complement the result of confidence-based retrieval and multimodal embedding similarity retrieval to obtain the most queryrelevant set of document images. Subsequently, we designed a decoupled cross-page attention-aware multimodal language model for multi-page documents to interpret these retrieved images and produce the final answer. Experimental results demonstrate the effectiveness of the retrieval module within the framework, as well as the robust performance of the multimodal model in multi-page document comprehension. These findings offer a compelling solution for multi-page document comprehension and cross-page document visual question answering. 

## **CCS Concepts** 

• **Information systems** → **Document representation** ; **Question answering** ; • **Applied computing** → **Document analysis** . 

## **Keywords** 

Multi-page Document VQA, Multimodal Retrieval Augmented Generation, Multimodal Language Model, Mixture of Experts, Attention Mechanism 

∗Corresponding author 

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org. _MM ’25, Dublin, Ireland._ 

© 2025 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 979-8-4007-2035-2/2025/10 https://doi.org/10.1145/3746027.3755357 

Qiyuan Fan Harbin Institute of Technology Harbin, China qyfan@ir.hit.edu.cn 

Yu Zhang[∗] Harbin Institute of Technology Harbin, China zhangyu@ir.hit.edu.cn 

## **ACM Reference Format:** 

Jinxu Zhang, Qiyuan Fan, Yongqi Yu, and Yu Zhang. 2025. DREAM: Integrating Hierarchical Multimodal Retrieval with Multi-page Multimodal Language Model for Documents VQA. In _Proceedings of the 33rd ACM International Conference on Multimedia (MM ’25), October 27–31, 2025, Dublin, Ireland._ ACM, New York, NY, USA, 9 pages. https://doi.org/10.1145/3746027. 3755357 

## **1 Introduction** 

Documents are fundamental information carriers, including articles, web pages, reports, etc. They contain dense and diverse information, such as vision, text, layout, and other modalities. Therefore, understanding multi-page document content and conducting crosspage document visual question answering (DVQA) are challenging and practical tasks. 

At present, research methods in DVQA have evolved from smallscale multimodal models to MLLMs, and the relevant datasets have also transitioned from single-page document data [26–28] to multipage document data [25, 36, 37]. Research on small-scale document understanding models began with the LayoutLM series [16, 40, 41], which extracts corresponding text and coordinates information using OCR tools and sets pre-training tasks such as image-text alignment and layout-text alignment in conjunction with images. Subsequently, to avoid efficiency issues and error propagation introduced by OCR, Donut [19] and Pix2Struct [21] proposed end-toend architectures combining visual encoders and text decoders and designed pre-training tasks such as pseudo-OCR and image masking to enhance the model’s understanding of document content. Recently, the emergence of MLLMs has further promoted the development of document understanding. Due to the high information density of document images, models such as InternVL [8], QwenVL [38], mPLUG-DocOwl [15], and InternLM-XComposer2-4KHD [10] are based on existing multimodal large language architectures to improve the resolution of image processing by visual encoders, enabling the model to understand the entire document image by combining global and local features. 

However, current document understanding models still focus on single-page documents with simple layouts, and only a small number of models [8, 38] have the ability to process multi-page documents, adopting the method of splicing. Due to the limited number of tokens accepted by the model, it can only process a few document images while maintaining high resolution. As a result, 

4213 

MM ’25, October 27–31, 2025, Dublin, Ireland. 

Jinxu Zhang, Qiyuan Fan, Yongqi Yu and Yu Zhang 

**==> picture [232 x 137] intentionally omitted <==**

**----- Start of picture text -----**<br>
Question Previous Methods<br>Concatenated images Answer Limited Tokens<br>DVQA model<br>Relevant  Information loss<br>Paragraphs<br>Text Extraction<br>an.” and  LLM Answer<br>Retrieval<br>Efficient Effective<br>Our Method<br>Multimodal<br>Embedding Retrieval Multi-page<br>Attention-aware Answer<br>Multimodal Keywords   DVQA model<br>Guided Retrieval Unlimited Tokens<br>=e Integration : Effective information<br>**----- End of picture text -----**<br>


**Figure 1: Existing DVQA models struggle to comprehend multi-page documents, often concatenating all visual tokens into a single input or using OCR to extract text from document images, followed by retrieval-augmented generation methods for answering. However, due to the model’s token limitations, neither approach can effectively handle largescale document images and cross-page question answering. To address this challenge, we propose an efficient multimodal retrieval algorithm and develop a multi-page DVQA model incorporating a cross-page attention mechanism.** 

performance is seriously degraded, and the content of multi-page documents cannot be fully understood. 

In this work, we propose a new architecture consisting of a hierarchical multimodal retrieval module and a cross-page attentionaware DVQA module. For the retrieval module, we leverage the complementarity of multimodal retrieval at different semantic levels. Specifically, for lower-level semantic retrieval, we use a multimodal embedding similarity retrieval model, such as CoPali [11] and VisRAG [43], to extract representations of the document image and the query and find the most relevant document images by calculating their embedding similarity. For high-level semantics, we decompose the query into multiple keywords and input them into an MLLM to determine whether the specific content of the document image is related to these keywords. If relevant, the MLLM outputs “Yes,” and the confidence of its logits for “Yes” is used as the ranking basis. Finally, we propose a multi-modal retrieval combination reranking algorithm to effectively integrate the results based on different semantic levels. For the question-answering module, we propose a multi-page document understanding model, which can process multi-page documents while maintaining high-resolution capabilities, and there is no limit to the number of documents it can accept. Specifically, inspired by the Mixture of Experts, we use the query to guide the weighting of visual features from each page and use weighted averaging to filter out the most relevant visual information for the LLM to understand. Considering the complexity of document images and the bottlenecks of current models, our proposed method of multi-page document visual question answering, combined with multi-level semantic and multimodal retrieval, is effective and has promising applications. 

In summary, our contributions are outlined as follows: 

- We introduce a multi-page document visual question answering (DVQA) framework, termed DREAM, which integrates hierarchical multimodal retrieval and a multi-page attentionaware document understanding model. This framework can effectively process extensive document information and facilitate cross-page DVQA. 

- We propose a multimodal retrieval method based on confidence and design a multimodal retrieval combination ranking algorithm that integrates low-level embedding similarity retrieval with high-level keyword-based retrieval, thereby filtering out the most relevant document images. 

- We construct a multi-page VLM that can combine globaltoken attention and mixture-of-experts across the vision encoder to filter relevant visual information, enhance comprehension of multi-page documents, and further align multimodal information. 

- Our approach achieves robust results on three multipage document visual question-answer datasets and effectively addresses cross-page problems. Extensive experiments are conducted to demonstrate the effectiveness of our approach. 

## **2 Related Work** 

## **2.1 Visually Rich Document Understanding** 

VRDU means that given a question and a document image, the model needs to understand the document content related to the question and extract or infer the answer. The initial small-scale DVQA models can be categorized into two types. The first type is based on OCR. OCR tools are used to extract text information and corresponding coordinate information from the document image. The document understanding model integrates this text and coordinates information to enable understanding of the document layout. Simultaneously, image information is input to achieve alignment among the three modalities. For example, LayoutLMv3 [16] utilizes pre-training tasks such as Word-Patch Alignment (WPA), Masked Image Modeling (MIM), and others to promote alignment between text and layout and between text and image. Similarly, UDOP [34] uses Text-Layout Reconstruction (TLR) and Image Reconstruction (IR) to achieve this goal. The second type is based on the architecture of a visual encoder and text decoder, such as Donut [19] and Pix2Struct [21]. The visual encoder is required to be able to read document images. Therefore, pseudo-OCR, visual masking, and other pre-training tasks are designed. The emergence of MLLMs [8, 15, 42] has improved the performance of visual encoders by improving the understanding of document images. 

However, existing methods are limited to handling the simple layouts of single-page documents and are inadequate for processing multi-page document images or answering cross-page questions. To address this challenge, we propose an effective multi-page document visual question-answering model. 

## **2.2 Retrieval Augmented Generation (RAG)** 

RAG consists of two parts, namely retrieval and LLM generation, which can effectively address knowledge enhancement tasks, reduce the hallucination of large models, and provide effective external knowledge. Early retrieval methods mainly focused on plain text retrieval based on word frequency measures such as TF-IDF 

4214 

MM ’25, October 27–31, 2025, Dublin, Ireland. 

DREAM: Integrating Hierarchical Multimodal Retrieval with Multi-page Multimodal Language Model for Documents VQA 

[32] and BM25 [31]. Later, with the emergence of LLMs, embedding models are trained to map queries and document sets into a vector space. Relevant information can then be obtained by calculating the similarity between vector embeddings. The LLM can then generate a final answer based on what is retrieved [3, 14, 17]. During this period, multimodal retrieval [11, 24, 43] has also made progress; that is, contrastive learning is used to train a shared vector space for text and visual features. Subsequently, the trained retrieval model filters relevant documents and feeds them into an LLM for question-answering tasks [7, 9]. 

However, due to limited query information, identifying relevant documents solely via embedding similarity calculations remains challenging. Therefore, we propose an advanced method that integrates different level semantic retrieval methods to compensate for this shortcoming. 

## **3 Methodology** 

In the following sections, we introduce the specifics of hierarchical multimodal retrieval and the model architecture for multi-page DVQA. Figure 2 illustrates the design of all key modules. 

## **3.1 Hierarchical Multimodal Retrieval** 

Currently, document retrieval methods primarily rely on plain-text retrieval. For instance, CREAM [45] leverages OCR-based data to identify relevant documents within multi-page collections. However, this approach disregards information from other modalities and remains inefficient, making it challenging to scale to document images. In this paper, we propose a method that leverages the multimodal information in document images to identify relevant documents without requiring external tools, such as OCR, to extract and process plain text. 

**Multimodal Embedding Similarity Retrieval.** Building on the approach introduced in VisRAG [43], which employs a visionlanguage model (VLM) rather than an LLM to encode queries and document pages. By calculating similarity scores between the query and each document image individually, we can select the top K most relevant images. This process can be formalized as: 

**==> picture [204 x 12] intentionally omitted <==**

Where D represents all document pages, the top-K images with the highest inner product scores. 

However, there is still a representational mismatch in the embedding space, especially for highly information-dense document images that contain different structured information, such as tables and charts, as well as for abstract queries. Consequently, it is insufficient to rely solely on this lower-level semantic matching. 

**Multimodal Keyword-guided Retrieval.** To address the limitations of lower-level semantic similarity methods, we propose a high-level semantic retrieval method guided by keywords. Specifically, by engaging in an instruction-based dialogue, MLLM can assess the relevance of each keyword in the query _𝑞_ . Initially, we ask the model to determine whether the image is relevant to the specified keywords _𝐾𝑞_ = { _𝑘_ 1 _, ...,𝑘𝑛_ } and to respond with “Yes” or “No.” If the answer is “Yes”, a subsequent instruction is triggered: “Extract the context in the image that is relevant to the following 

keywords related to the query.” In addition, the model’s probability of responding ’Yes’ is quantified and used for the ranking of relevance. The overall procedure can be formalized as follows: 

**==> picture [221 x 64] intentionally omitted <==**

where y denotes the first token in the generated output, _𝑃𝐼𝑖_ represents the confidence value that the _𝑀𝐿𝐿𝑀_ round1 output is "Yes", _𝐶𝐼𝑖_ represents the context in the image _𝐼𝑖_ relevant to _𝑄_ , which is different from the plain text extracted by OCR, the information extracted by MLLM is well-structured and summarized, and the semantic information is more clear. 

**Algorithm 1** Multi-modal Retrieval Combination Reranking 

|**Algorithm 1**Multi-modal Retrieval Combination Reranking|**Algorithm 1**Multi-modal Retrieval Combination Reranking|
|---|---|
|**Require:** _𝑆_: A list of images and their similarity scores for a given||
||query|
|**Require:** _𝑁_: A list of images and their keywords relevant proba-||
||bilities for the same query|
|**Require:** _𝐾_: The number of top images from_𝑆_(e.g., 10)||
|**Require:** _𝑁_: The number of top images to return (e.g., 5)||
|**Ensure:** A ranked list of top-_𝑀_images||
||**Ofline Model Training:**:|
|1:|Use a set with ground-truth evidence pages|
|2:|Train a Learning-to-Rank model (e.g., LightGBM) on|
||(_𝑠𝑖𝑚𝑖𝑙𝑎𝑟𝑖𝑡𝑦_𝑠𝑐𝑜𝑟𝑒,𝑘𝑒𝑦𝑤𝑜𝑟𝑑_𝑝𝑟𝑜𝑏_) and labels|
|3:|Save the trained modelF for fusion|
||**Online Retrieval Fusion**:|
|4:|_𝐼𝑆_←TopK(_𝑆, 𝐾_)<br>{Select the top-_𝐾_images from_𝑆_}|
|5:|_𝑊_(_𝑖_) ←KeywordScoreMap(_𝑁_)|
|6:|**for**_𝑖𝑗_∈_𝐼𝑆_**do**|
|7:|_𝑠𝑗_←SimilarityScore(_𝑖𝑗_)|
|8:|_𝑤𝑗_←_𝑊_(_𝑖𝑗_)|
|9:|_𝑓𝑗_←F (_𝑠𝑗,𝑤𝑗_){Predict fused score using the trained model|
||F}|
|10:|**end for**|
|11:|_𝐼𝑆_←Sort(_𝐼𝑆_by _𝑓𝑗_in descending order)|
|12:|**return**the Top_𝑀_images of_𝐼𝑆_|



**Multi-modal Retrieval Combination Reranking.** The two retrieval methods discussed above are complementary: when there is a semantic match between the query and candidate document images but vector similarity is low, keyword-relevant scores can enhance the ranking of the items. Conversely, similarity scores support cases where the query is highly similar to the candidate vectors but lacks specific keywords. Therefore, the ranking result obtained by their integration will be better than a single low-level or high-level retrieval. To effectively integrate the two aforementioned retrieval methods, we utilize the MP-DocVQA training set and construct the corresponding training data for similarity scores, keyword-relevant probabilities, and evidence pages. Specifically, we employ a learning-to-rank model and the constructed training data to directly optimize the ranking indices based on Normalized 

4215 

MM ’25, October 27–31, 2025, Dublin, Ireland. 

Jinxu Zhang, Qiyuan Fan, Yongqi Yu and Yu Zhang 

Discounted Cumulative Gain (NDCG), enabling the model to learn the desired order as closely as possible to that specified by the evidence_pages label. The process is outlined in Algorithm 1. For a more detailed explanation, refer to Appendix A. 

For fine-grained interactions, we compute token-level similarities between each text token _𝑞 𝑗_ ∈ _𝑄_ and each image region _𝑣𝑖,𝑘_ ∈ _𝑉𝑖_ . 

**==> picture [170 x 12] intentionally omitted <==**

Apply softmax over image feature dimension _𝑘_ : 

## **3.2 Cross-page Attention-aware DVQA** 

Existing MLLMs lack the capability to comprehend large-scale document pages. They can only perform DVQA on single-page documents and rely on direct concatenation for handling multi-page documents. However, to effectively comprehend the dense information within a document image, these models enhance image resolution by splitting the image into smaller sub-images. This operation, however, increases the number of image tokens. Consequently, these models are restricted to processing a limited number of documents due to the inherent length constraints of their inputs. This frequently results in truncation, leading to information loss. Additionally, capturing connections between different document pages becomes challenging. To address these challenges, we propose a multi-page DVQA model that employs a cross-page attentionaware mechanism to effectively assign importance weights to visual features, thereby filtering out the visual information most relevant to the query. 

**Multi-page Vision Encoder.** Due to the limited length of the existing visual encoder processing images, a single image is divided into multiple batches of input visual encoder _𝐸_ to prevent information loss; similarly, for multi-page document images _𝐷_ , each page can be divided and finally divided into more batches for processing. 

**==> picture [202 x 28] intentionally omitted <==**

**Multi-page Attention-aware Connector.** Inspired by the mixtureof-experts paradigm, we can obtain the visual representation of each page { _𝑉𝑖_ } _𝑖[𝑀]_ =1[, where each] _[ 𝑉][𝑖]_[∈][R] _[𝑙]_[×] _[𝑑]_[(with] _[ 𝑙]_[being the length of] image features per page and _𝑑_ the feature dimension). Each page of the document images is treated as an individual expert. By incorporating the textual features of the query _𝑄_ ∈ R _[𝑛]_[×] _[𝑑]_ (n is the number of text tokens), we compute fine-grained attention scores for each document image and global attention across all images. This process can be formally expressed as follows: 

We first compute global representations for both text and images using an attention-pooling mechanism: 

**==> picture [197 x 58] intentionally omitted <==**

Where _𝑋_ ∈ R _[𝑁]_[×] _[𝐷]_ is the input feature matrix, and N is the length of features. _𝑤_ ∈ R _[𝐷]_ is a learnable weight vector. _𝛼𝑖_ is the attention weight of the _𝑖_ − _𝑡ℎ_ feature. 

**==> picture [177 x 30] intentionally omitted <==**

Compute the global similarity between each page and the query: 

**==> picture [175 x 14] intentionally omitted <==**

**==> picture [184 x 25] intentionally omitted <==**

Then, aggregate these similarities by averaging over all tokens and summing over the weighted image features: 

**==> picture [196 x 29] intentionally omitted <==**

Next, we employ a gating mechanism to integrate these two forms of attention, allowing the model to learn optimal weights during supervised training for the question-answering task. This process assigns appropriate importance weights to each page, and the weighted results are then fed into the LLM. In addition, similar to expert selection strategies, we can dynamically adjust the number of activated pages to enhance the efficiency of the question-answer process. This procedure can be formally expressed as follows: 

> We feed the combined normalized scores _𝑆_[ˆ] global( _𝑖_ ) and _𝑆_[ˆ] token( _𝑖_ ) into a gating function _𝐺_ (·) to produce a score for each page: 

**==> picture [178 x 11] intentionally omitted <==**

Each page _𝑉𝑖_ is treated as an expert. In order to highlight the most influential experts in the calculation, our objective is to select only the top _𝑃_ experts with the highest scores when calculating the weights, thereby allowing them to dominate the final representation. This strategy aids in filtering out irrelevant or noisy page features: 

**==> picture [190 x 9] intentionally omitted <==**

For the selected K experts, softmax normalization is performed according to their scores: 

**==> picture [175 x 33] intentionally omitted <==**

At this point, only the P experts with the highest score have a non-zero weight. 

An expert network _𝐸𝑖_ (·) processes _𝑉𝑖_ to produce an output representation _𝐸𝑖_ ( _𝑉𝑖_ ). Finally, we combine the outputs from all experts using the computed weights: 

**==> picture [157 x 23] intentionally omitted <==**

The resulting representation _𝑅_ integrates key information from multiple pages conditioned on the textual query, following the mixture-of-experts paradigm. 

**Answer Generation.** After the multi-page document images pass through our designed connector, we obtain weighted visual information highly relevant to the query, which can then be fed into the LLM to generate the final answer. In addition, the retrieved relevant contexts can be integrated as auxiliary information. This process can be formally described as follows: 

**==> picture [165 x 10] intentionally omitted <==**

4216 

MM ’25, October 27–31, 2025, Dublin, Ireland. 

DREAM: Integrating Hierarchical Multimodal Retrieval with Multi-page Multimodal Language Model for Documents VQA 

**==> picture [444 x 227] intentionally omitted <==**

**----- Start of picture text -----**<br>
Similarity Scores: Select highest-scoring images. Relevant Images: connector<br>0.51, 0.34, 0.78, 0.96…<br>Query<br>……<br>top K  Attention gate<br>Text  Document Vision<br>Encoder retriever Encoder<br>Vision<br>Encoder E1 E2 E3 … EM<br>Query: Document Images:<br>How many meetings  top P<br>did PepsiCo's Board of<br>Directors hold in 2005? Global attn Token attn<br>top N  x M<br>as<br>MLLM<br>Keywords  Round1: Relevant Images<br>Extractor<br>Whether the image is relevant<br>to the following keywords, output yes or no. yes optional LLM<br>Key Words: Round2: probs Relevant Contexts<br>Meeting(0.53), PepsiCo(0.37),  If yes, Extract the relevant<br>2005(0.33), …… context in the image. Answer<br>Hierarchical Multimodal Retrieval  Cross-page Attention-aware  DVQA<br>Rerank Algorithm<br>**----- End of picture text -----**<br>


**Figure 2: The overall framework of DREAM. The left portion represents the retrieval module, while the right portion corresponds to the multi-page DVQA component. The retrieval module integrates low-level semantic similarity retrieval with a high-level, keyword-based contextual retrieval method, subsequently applying a multi-modal retrieval combination reranking algorithm. After filtering, the top M document images are passed to the question-answering module, where relevant visual information is further refined through a cross-page attention mechanism. Finally, the model generates the answer jointly with the LLM.** 

## **4 Experiments** 

In this section, we evaluate our framework through a series of experiments and ablation studies. More detailed insights into our method are also provided. 

## **4.1 Experiment Setup** 

**Datasets.** We conducted experiments on three multi-page document datasets from various fields, such as papers, reports, and scanned documents, including MP-DocVQA [36], DUDE [37], and MMLongBench-Doc [25], each featuring multi-page document images tailored to specific query tasks. In MP-DocVQA and DUDE, all answers originate from a single page within the document set, whereas MMLongBench-Doc presents more complex, cross-page questions spanning a greater number of document pages. 

**Models.** We employ small-scale baseline models relevant to this task on MP-DocVQA and DUDE, including plain-text models such as BERT [18], T5 [30], Longformer [5], and Big Bird [44]; OCR toolbased models such as LayoutLMv3 [16] and HiVT5 [36], designed specifically for multi-page documents, employs two methods: (1) predicting the document page likely containing the answer, similar to the single-page document process, and (2) predicting the answer for each document page, ultimately selecting the most probable one. Thus, it fundamentally addresses single-page document scenarios. Pure visual language models, including InternVL2 [8], Qwen2VL [38] and GPT4o [29]. MMLong-Bench-Doc is primarily compared against MLLMs, including open source models [4, 8, 10, 13, 15, 20, 22, 23, 33, 39], RAG models [7, 9] and closed source models 

[1, 2, 29, 35]. The approach of these models to processing multi-page documents involves directly concatenating the document images into the model. This results in significant information redundancy, which hinders the identification of key information and is restricted by an upper input limit. 

**Evaluation Metrics.** We evaluated DREAM’s performance on both retrieval and question answering, assessing page-matching accuracy via Exact Match (EM) and evaluating Q&A accuracy using standard metrics such as Accuracy and F1 for MMLongBench-Doc and Average Normalized Levenshtein Similarity (ANLS) for MPDocVQA and DUDE. See Appendix C for details on how evaluation metrics are calculated. 

**Implementation Details.** For keyword extraction during the retrieval stage, we employ the KeyBERT [12] to generate query keywords along with their corresponding importance scores. The keyword retrieval model employs InternVL2-40B, while the learningto-rank model utilizes the LightGBM Ranker, trained using gold pages from MP-DocVQA. During multi-page DVQA training, the backbone model is InternVL2-4B, and we utilize the MP-DocVQA and DUDE training datasets, where the ranking results of DUDE’s Multi-modal Retrieval Combination Reranking are obtained from the MP-DOCVQA-trained ranking model. The model’s visual encoder remains frozen, trainable parameters include connector and LLM LoRA parameters. We fix the maximum number of document image patches per batch to ensure a uniform input structure for multiple images. We trained the model for 3 epochs with a batch 

4217 

MM ’25, October 27–31, 2025, Dublin, Ireland. 

Jinxu Zhang, Qiyuan Fan, Yongqi Yu and Yu Zhang 

|Retrieval Method|MP-DocVQA<br>Top1<br>Top3<br>Top5|MMLongBench-Doc|
|---|---|---|
|||Top1<br>Top3<br>Top5|
|Multimodal Similarity [43]<br>Keywords Retrieval (ours)<br>Combination Reranking (ours)|77.6<br>92.1<br>95.5<br>79.2<br>92.6<br>94.8<br>82.4<br>94.1<br>97.3|54.3<br>71.7/58.4<br>80.0/68.6/65.9<br>56.8<br>75.1/59.5<br>81.1/67.6/65.3<br>58.6<br>76.5/61.2<br>83.4/70.6/67.6|



**Table 1: Page hit ratio results for various retrieval methods are presented. For MP-DocVQA, each answer is associated with only a single page, whereas for MMLong, the answer may span multiple pages. Consequently, for MPDocVQA, Top1, Top3, and Top5 indicate the hit rate of the first 1, 3, and 5 retrieved pages, respectively. For MMLong, Top1 measures the hit rate of the first retrieved page for all instances where the answer is contained within 1 page. Top3 evaluates the hit rate of the first three retrieved pages for all answers contained within 1 page and** ≤ **3 pages. Similarly, Top5 considers the hit rate of the first five retrieved pages for all answers that span 1 page,** ≤ **3 pages, and** ≤ **5 pages.** 

size of 8, set r to 16 in LoRA, and used a learning rate of 4e-5. All experiments were conducted on two A100 80G GPUs. 

## **4.2 Main Results** 

**Hierarchical Multimodal Retrieval.** Table 1 presents the results of various retrieval methods. Since the DUDE dataset does not provide evidence pages, we only tested MP-DocVQA and MMLongBenchDoc. It can be observed that our proposed keyword-based retrieval method slightly outperforms the embedding-similarity-based approach on both top1 and top3 but is slightly inferior in top5; this indicates that the calculation method based on embedding similarity demonstrates stronger robustness in multi-evidence page retrieval. By integrating the two methods, our multi-modal retrieval combination reranking algorithm improves performance on both datasets, which shows that the two retrieval methods are complementary and proves that our algorithm can fully realize the complementary effect of the two. Clearly, selecting a greater number of relevant document images increases the likelihood of including the correct images, particularly in cross-page DVQA tasks. This underscores the necessity for the model to possess the capability to comprehend multi-page documents. 

|Model|#Param|Modality|MP-DocVQA|DUDE|
|---|---|---|---|---|
|BERT-base [18]|334M|T|53.47|25.48|
|Longformer [5]|148M|T|55.06|27.14|
|Big Bird [44]|131M|T|58.54|26.27|
|T5-base [30]|223M|T|41.80|41.80|
|LayoutLMv3 [16]|125M|T+L+V|55.13|20.31|
|HiVT5 [36]|316M|T+L+V|62.01|23.06|
|GRAM[6]|859M|T+L+V|80.32|51.15|
|InternVL2 [8]|8B|V|78.14|42.23|
|Qwen2VL [38]|7B|V|82.12|45.94|
|GPT4o[29]|-|V|79.33|54.12|
|DREAM(ours)|4B|V|82.48|48.32|
|DREAM†(ours)|4B|T+V|**85.72**|**52.14**|



**Table 2: DREAM’s performance results on MPDocVQA and DUDE, both datasets with answers sourced from a single page and evaluated by ANLS, where T represents text, L represents Layout, V represents Vision. The models marked with †indicate those associated with relevant contextual text.** 

**Multimodal Multi-page DVQA.** Table 2 presents the questionanswering results of DREAM on the MPDocVQA and DUDE datasets. DREAM achieves its best results on MPDocVQA and goes beyond the existing open-source models, even with the addition of relevant text context, achieving results close to GPT-4o on DUDE. However, when the retrieved text information is not used, DREAM’s results on DUDE lag behind GRAM [6], indicating that other modal information, such as text and layout, is also crucial to improve the model’s performance further, but the use of OCR for text extraction will lead to inefficiency. At the same time, compared to HiVT5 [36], GRAM has optimized the model architecture to enable it to capture key information in multi-page documents, resulting in significant performance improvements. Furthermore, the performance of DREAM on DUDE is inferior to that on MPDocVQA due to the more complex layouts and the diverse, intricate nature of the questions in DUDE. Therefore, existing vision encoders still exhibit limitations in comprehending complex-layout document images. 

Table 3 compares DREAM with MLLMs on the MMLongBenchDoc benchmark. The results indicate that DREAM outperforms all other larger open-source models using only the 4B parameter model and trails only the GPT-4 family of closed-source models, and it has more advantages than other methods that only use embedding similarity retrieval. Given the poor performance of all models on this dataset, we believe that existing models are still unable to effectively understand multi-page documents. Additionally, DREAM has not fully leveraged its potential advantages due to the lack of relevant training data, making it difficult for the model to learn the data from multiple evidence pages. For questions derived from different evidence sources, DREAM performs best with text-based evidence, while its comprehension of structured content (e.g., charts, tables, and figures) remains insufficient. This suggests significant room for improvement in the model’s ability to process complex document layouts. For questions requiring evidence across multiple pages, the model performs poorly in cross-page comprehension, as redundant information interferes with its ability to locate relevant content accurately. Moreover, for unanswerable questions, the model frequently generates responses, a phenomenon attributable to model hallucination. In summary, future work should incorporate sufficient multi-page document training data to improve the model’s performance in complex layouts, cross-page comprehension, and challenges while mitigating the hallucination of irrelevant outputs. 

4218 

MM ’25, October 27–31, 2025, Dublin, Ireland. 

DREAM: Integrating Hierarchical Multimodal Retrieval with Multi-page Multimodal Language Model for Documents VQA 

|Model<br>#Param|Evidence Source|Evidence Page|ACC<br>F1|
|---|---|---|---|
||TXT<br>LAY<br>CHA<br>TAB<br>IMG|SIN<br>MUL<br>UNA||
|DeepSeek-VL-Chat [23]<br>7.3B<br>Idfics2 [20]<br>8B<br>MiniCPM-Llama3-v2.5 [35]<br>8B<br>InternLM-XC2-4KHD [10]<br>8B<br>mPLUG-DocOwl 1.5 [15]<br>8.1B<br>Qwen-VL-Chat [4]<br>9.6B<br>Monkey-Chat [22]<br>9.8B<br>CogVLM2-LLaMA3-Chat [39]<br>19B<br>InternVL-Chat-v1.5 [8]<br>26B<br>EMU2-Chat[33]<br>37B|7.2<br>6.5<br>1.6<br>5.2<br>7.6<br>9.0<br>10.6<br>4.8<br>4.1<br>8.7<br>11.9<br>10.8<br>5.1<br>5.9<br>12.2<br>9.9<br>14.3<br>7.7<br>6.3<br>13.0<br>8.2<br>8.4<br>2.0<br>3.4<br>9.9<br>5.5<br>9.0<br>5.4<br>2.2<br>6.9<br>6.8<br>7.2<br>3.6<br>6.7<br>9.4<br>3.7<br>2.7<br>6.0<br>3.2<br>6.9<br>14.0<br>16.2<br>7.1<br>10.1<br>16.6<br>6.1<br>9.7<br>2.6<br>3.8<br>7.7|5.2<br>7.0<br>12.8<br>7.7<br>7.2<br>5.0<br>9.5<br>9.5<br>4.5<br>12.6<br>7.6<br>9.6<br>7.4<br>6.4<br>6.2<br>5.2<br>7.1<br>6.2<br>6.6<br>6.2<br>6.2<br>3.9<br>5.3<br>3.7<br>14.9<br>12.2<br>**17.5**<br>5.7<br>6.1<br>16.5|7.4<br>5.4<br>7.0<br>6.8<br>8.5<br>8.6<br>10.3<br>9.8<br>6.9<br>6.3<br>6.1<br>5.4<br>6.2<br>5.6<br>4.4<br>4.0<br>14.6<br>13.0<br>8.3<br>5.5|
|M3DocRAG [9]<br>7B<br>SV-RAG[7]<br>4B|30.0<br>**23.5**<br>18.9<br>20.1<br>20.8<br>26.3<br>22.1<br>25.0<br>20.7<br>25.2|32.4<br>14.8<br>5.8<br>34.0<br>10.6<br>15.7|21.0<br>22.6<br>23.0<br>24.2|
|DREAM(Ours)<br>4B<br>DREAM†(Ours)<br>4B|28.4<br>20.1<br>25.2<br>21.0<br>26.3<br>**32.4**<br>22.5<br>**26.8**<br>**22.3**<br>**26.5**|36.8<br>15.6<br>14.2<br>**39.6**<br>**18.2**<br>16.3|25.5<br>26.2<br>**27.3**<br>**28.6**|
|Claude-3 Opus [2]<br>-<br>Gemini-1.5-Pro [35]<br>-<br>GPT-4V [1]<br>-<br>GPT-4o [29]<br>-|24.9<br>24.7<br>14.8<br>13.0<br>17.1<br>21.0<br>17.6<br>6.9<br>14.5<br>15.2<br>34.4<br>28.3<br>28.2<br>32.4<br>26.8<br>46.3<br>46.0<br>45.3<br>50.0<br>44.1|25.6<br>13.8<br>7.6<br>21.1<br>11.1<br>69.2<br>36.4<br>27.0<br>31.2<br>54.5<br>41.5<br>20.2|17.4<br>18.1<br>28.2<br>20.6<br>32.4<br>31.2<br>42.8<br>44.9|



**Table 3: Performance of Various MLLMs on MMLongBench-Doc. Questions are categorized into two types: Evidence Source and Evidence Page. The former encompasses text (TXT), layout (LAY), chart (CHA), table (TAB), and image (IMG), while the latter includes single page (SIN), cross-page (MUL), and unanswerable (UNA). The models marked with †indicate those associated with relevant contextual text information.** 

## **4.3 Ablation Studies** 

**The Effect of Combination ReRanking Methods.** Given that the training features and labels we adopted are relatively simple, we attempt to employ relevant machine-learning models to capture the relationships between them and achieve better ranking results. The correlations are primarily categorized into three types: linear fusion, non-linear fusion, and ranking learning method fusion. Through experiments shown in Figure 3, we found that the ranki ~~ng~~ learning and linear fusion methods outperform nonlinear fusion. Additionally, ranking learning performs better than linear fusion on MMLong, while its performance on MPDocVQA remains similar. It is worth noting that ranking learning is more dependent on the distribution of the training set data. Since the training set includes only MPDocVQA, its ranking performance has not been fully developed for MMLong. Therefore, more data containing cross-page evidence, such as that in MMLong, are needed in the future. 

**The Effect of Different Page Experts and Selected Experts.** 

The question-answering module is essentially decoupled from the retrieval module. In theory, an arbitrary number of images can be processed, and irrelevant or noisy image information will be filtered out by the expert selection algorithm. However, due to insufficient training data, it remains unable to achieve satisfactory results when processing large-scale document images. Table 4 shows that after retrieval, we selected the top 8 and 5 document images as input and then chose the top 4 and 2 among them as the final query-related document image representations, which were subsequently input into the LLM to obtain the final answer. This design rationale is supported by the following evidence: Figure 1 (Appendix A) shows 

**==> picture [236 x 80] intentionally omitted <==**

**----- Start of picture text -----**<br>
Hit Rate Hit Rate<br>85 100<br>80 90<br>75 80<br>70<br>70<br>60<br>65 50<br>60 <=1 <=2 <=3 <=4 <=5 40 <=1 <=2 <=3 <=4 <=5<br>81.8 71 68.8 67.2 66.1 Linear 81.8 90.8 94.4 96 97.1<br>80.1 69.4 67.1 64.7 63.6 Non-Linear 78.6 89.7 93.4 95.1 96.5<br>-Rank 83.4 72.6 70.6 68.7 67.6 Learning-to-Rank 82.4 91.1 94.1 95.8 97.3<br>**----- End of picture text -----**<br>


**Figure 3: Different combinations of results for different retrieval methods, where the hit ratio represents the data associated with multiple pages in both datasets.** 

that 94% of Q&A pairs occur within 4 pages, while 87% are contained within just 2 pages. Additionally, Table 1 Table 2 (Appendix A) reveals that the first 5 pages contain nearly all relevant content, while subsequent pages primarily introduce redundancy with minimal relevance gains, until the 8th page without any promotion. Consequently, we established an 8-page threshold to assess the effect on model performance. The experimental results indicate that for MPDocVQA and DUDE, processing 5 pages of document images and activating 2 experts yields the best performance. This is because there is only 1 page of relevant evidence in these two datasets, and our designed strategy can accurately locate the pertinent information. For MMLong, processing 5 pages and activating 4 experts yields the best performance. This is because the dataset contains many cross-page questions, and Q&A tasks involving 2 to 4 pages of evidence account for 42% of the dataset. Activating 

4219 

MM ’25, October 27–31, 2025, Dublin, Ireland. 

Jinxu Zhang, Qiyuan Fan, Yongqi Yu and Yu Zhang 

|Page<br>Experts|Selected<br>Experts|MP-DocVQA|DUDE|MMLong|
|---|---|---|---|---|
|8|4|80.84|46.53|25.15|
|8|2|81.46|47.94|24.67|
|5|4|80.95|46.88|**25.52**|
|5|2|**82.48**|**48.32**|24.92|



**Table 4: Performance of input page count and active experts in the question-answering module.** 

**==> picture [134 x 92] intentionally omitted <==**

**----- Start of picture text -----**<br>
Which dataset used in this paper was proposed in 2022 and all of its<br>logical reasoning problems are multiple-choice questions with 5 options?<br>Reteieval pages:  [6, 4, 14, 19, 2]<br>Prediction:  AR-LSAT<br>Evidence page:  [6,14]<br>Ground Truth:  AR-LSAT<br>Evidence sources:  Table, Pure-text<br>When using GPT-4 as the base language model,<br>how many datasets does Logic-LM (without self-<br>refinement) outperform the two baseline models in?<br>ES Reteieval pages:  [7, 8, 15, 2, 10]<br>,  Prediction:Evidence page:  4 7<br>Ground Truth:  4<br>Evidence sources:  Table<br>**----- End of picture text -----**<br>


**Figure 4: Two examples of DREAM’s responses to multi-page documents, where areas relevant to the evidence are highlighted.** 

attention and token attention play important roles, with token attention demonstrating a more significant improvement. Notably, when neither global attention nor token attention is used (i.e., when multi-page documents are input to the model through simple concatenation), performance reaches its lowest level. The addition of global attention leads to slight performance improvements across all three datasets. This suggests the model can capture certain aspects of global layout information, while token attention further enhances the model’s ability to process fine-grained visual details. It shows that these two attention mechanisms have certain filtering effects on noise information. 

**Qualitative Analysis.** Figure 4 presents examples of DREAM in single-page document VQA and cross-page document VQA, respectively. In cross-page document VQA, DREAM can retrieve relevant pages based on questions, accurately locate the pertinent context within specific pages, and integrate information from different evidence sources to provide precise answers. In single-page document VQA, DREAM also demonstrates reasoning abilities, understands heterogeneous information within the document image, and provides accurate answers by leveraging a large language model. However, due to a lack of training data, DREAM does not perform well for cross-page documents where evidence spans more than two pages. On the one hand, fully identifying fine-grained information within relevant pages is challenging. On the other hand, integrating and reasoning over this relevant information is also difficult. The model may have problems such as information recognition errors and inference hallucinations. 

## **5 Conclusion** 

4 experts maximizes coverage of such data. When the input is 8 pages, the overall performance is lower than that of 5 images, indicating that the additional redundant information will lead to the degradation of model performance. Specifically, activating 2 experts for MPDocVQA and DUDE yields better results, possibly because activating 4 experts introduces additional noise, which affects the model’s judgment. Activating 4 experts for MMLong works even better. As the number of input pages increases, the model requires a stronger ability to recognize relevant information. 

|Global_attn|Token_attn|MP-DocVQA|DUDE|MMLong|
|---|---|---|---|---|
|✗|✗|80.36|46.68|23.0|
|✓|✗|81.14|46.97|23.82|
|✗|✓|82.05|47.78|24.65|
|✓|✓|82.48|48.32|25.52|



**Table 5: Results of cross-page attention mechanisms in the question-answer module. The first row represents the original baseline model. The setup input for MP-DocVQA and DUDE is 5 pages with 2 active experts, while the input for MMLongBench-Doc is 5 pages with 4 active experts.** 

This paper introduces an efficient and high-performance framework, dubbed DREAM, which initially proposes a confidence-based retrieval method and then employs a learning-to-rank algorithm to integrate the results of low-level embedding similarity-based multimodal retrieval and high-level keyword-guided semantic multimodal retrieval, thereby identifying the most relevant set of document images. Secondly, to tackle the complexity of interpreting visual information across multiple document pages, DREAM implements a joint cross-page coarse-grained global attention mechanism and a fine-grained token attention mechanism, along with an expert selection mechanism, visual information related to the query can be effectively filtered out, thereby enhancing the accuracy of question answering. Experimental results demonstrate the effectiveness of our proposed method, achieving more robust performance across multiple datasets and exhibiting strong practical applicability. 

Furthermore, the current limitation in multi-page document datasets prevents the model from fully leveraging its capabilities. Consequently, future research should focus on developing more comprehensive and higher-quality multi-page document training datasets to enhance multimodal models’ cross-page visual questionanswering capabilities. 

## **6 Acknowledgements** 

**The Effect of Cross-page Attention Module.** Table 5 compares the effectiveness of different cross-modal attention mechanisms used in the connector. The results indicate that both global 

We would like to thank the anonymous reviewers for their helpful comments. This work was supported by the National Natural Science Foundation of China (No.62476066) and (No.62277002). 

4220 

MM ’25, October 27–31, 2025, Dublin, Ireland. 

DREAM: Integrating Hierarchical Multimodal Retrieval with Multi-page Multimodal Language Model for Documents VQA 

## **References** 

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. 2023. Gpt-4 technical report. _arXiv preprint arXiv:2303.08774_ (2023). 

- [2] Anthropic. 2024. Introducing the next generation of claude. (2024). 

- [3] Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. 2023. Self-rag: Learning to retrieve, generate, and critique through self-reflection. _arXiv preprint arXiv:2310.11511_ (2023). 

- [4] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. 2023. Qwen-vl: A frontier large visionlanguage model with versatile abilities. _arXiv preprint arXiv:2308.12966_ (2023). 

- [5] Iz Beltagy, Matthew E Peters, and Arman Cohan. 2020. Longformer: The longdocument transformer. _arXiv preprint arXiv:2004.05150_ (2020). 

- [6] Tsachi Blau, Sharon Fogel, Roi Ronen, Alona Golts, Roy Ganz, Elad Ben Avraham, Aviad Aberdam, Shahar Tsiper, and Ron Litman. 2024. GRAM: Global reasoning for multi-page VQA. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ . 15598–15607. 

- [7] Jian Chen, Ruiyi Zhang, Yufan Zhou, Tong Yu, Franck Dernoncourt, Jiuxiang Gu, Ryan A Rossi, Changyou Chen, and Tong Sun. 2024. LoRA-Contextualizing Adaptation of Large Multimodal Models for Long Document Understanding. _arXiv preprint arXiv:2411.01106_ (2024). 

- [8] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. 2024. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. _arXiv preprint arXiv:2404.16821_ (2024). 

- [9] Jaemin Cho, Debanjan Mahata, Ozan Irsoy, Yujie He, and Mohit Bansal. 2024. M3docrag: Multi-modal retrieval is what you need for multi-page multi-document understanding. _arXiv preprint arXiv:2411.04952_ (2024). 

- [10] Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Bin Wang, Linke Ouyang, Songyang Zhang, Haodong Duan, Wenwei Zhang, Yining Li, et al. 2024. Internlmxcomposer2-4khd: A pioneering large vision-language model handling resolutions from 336 pixels to 4k hd. _arXiv preprint arXiv:2404.06512_ (2024). 

- [11] Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani, Gautier Viaud, Céline Hudelot, and Pierre Colombo. 2024. Colpali: Efficient document retrieval with vision language models. _arXiv preprint arXiv:2407.01449_ (2024). 

- [12] Maarten Grootendorst. 2020. KeyBERT: Minimal keyword extraction with BERT. doi:10.5281/zenodo.4461265 

- [13] Zonghao Guo, Ruyi Xu, Yuan Yao, Junbo Cui, Zanlin Ni, Chunjiang Ge, Tat-Seng Chua, Zhiyuan Liu, and Gao Huang. 2025. Llava-uhd: an lmm perceiving any aspect ratio and high-resolution images. In _European Conference on Computer Vision_ . Springer, 390–406. 

- [14] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Mingwei Chang. 2020. Retrieval augmented language model pre-training. In _International conference on machine learning_ . PMLR, 3929–3938. 

- [15] Anwen Hu, Haiyang Xu, Jiabo Ye, Ming Yan, Liang Zhang, Bo Zhang, Chen Li, Ji Zhang, Qin Jin, Fei Huang, et al. 2024. mplug-docowl 1.5: Unified structure learning for ocr-free document understanding. _arXiv preprint arXiv:2403.12895_ (2024). 

- [16] Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, and Furu Wei. 2022. Layoutlmv3: Pre-training for document ai with unified text and image masking. In _Proceedings of the 30th ACM International Conference on Multimedia_ . 4083–4091. 

- [17] Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense passage retrieval for opendomain question answering. _arXiv preprint arXiv:2004.04906_ (2020). 

- [18] Jacob Devlin Ming-Wei Chang Kenton and Lee Kristina Toutanova. 2019. Bert: Pre-training of deep bidirectional transformers for language understanding. In _Proceedings of naacL-HLT_ , Vol. 1. 2. 

- [19] Geewook Kim, Teakgyu Hong, Moonbin Yim, JeongYeon Nam, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, Sangdoo Yun, Dongyoon Han, and Seunghyun Park. 2022. Ocr-free document understanding transformer. In _European Conference on Computer Vision_ . Springer, 498–517. 

- [20] Hugo Laurençon, Léo Tronchon, Matthieu Cord, and Victor Sanh. 2024. What matters when building vision-language models? _arXiv preprint arXiv:2405.02246_ (2024). 

- [21] Kenton Lee, Mandar Joshi, Iulia Raluca Turc, Hexiang Hu, Fangyu Liu, Julian Martin Eisenschlos, Urvashi Khandelwal, Peter Shaw, Ming-Wei Chang, and Kristina Toutanova. 2023. Pix2struct: Screenshot parsing as pretraining for visual language understanding. In _International Conference on Machine Learning_ . PMLR, 18893–18912. 

- [22] Zhang Li, Biao Yang, Qiang Liu, Zhiyin Ma, Shuo Zhang, Jingxu Yang, Yabo Sun, Yuliang Liu, and Xiang Bai. 2024. Monkey: Image resolution and text label are important things for large multi-modal models. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ . 26763–26773. 

- [23] Haoyu Lu, Wen Liu, Bo Zhang, Bingxuan Wang, Kai Dong, Bo Liu, Jingxiang Sun, Tongzheng Ren, Zhuoshu Li, Hao Yang, et al. 2024. Deepseek-vl: towards realworld vision-language understanding. _arXiv preprint arXiv:2403.05525_ (2024). 

- [24] Xueguang Ma, Sheng-Chieh Lin, Minghan Li, Wenhu Chen, and Jimmy Lin. 2024. Unifying multimodal retrieval via document screenshot embedding. _arXiv preprint arXiv:2406.11251_ (2024). 

- [25] Yubo Ma, Yuhang Zang, Liangyu Chen, Meiqi Chen, Yizhu Jiao, Xinze Li, Xinyuan Lu, Ziyu Liu, Yan Ma, Xiaoyi Dong, Pan Zhang, Liangming Pan, YuGang Jiang, Jiaqi Wang, Yixin Cao, and Aixin Sun. 2024. MMLongBenchDoc: Benchmarking Long-context Document Understanding with Visualizations. arXiv:2407.01523 [cs.CV] https://arxiv.org/abs/2407.01523 

- [26] Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. 2022. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. _arXiv preprint arXiv:2203.10244_ (2022). 

- [27] Minesh Mathew, Viraj Bagal, Rubèn Tito, Dimosthenis Karatzas, Ernest Valveny, and CV Jawahar. 2022. Infographicvqa. In _Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision_ . 1697–1706. 

- [28] Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. 2021. Docvqa: A dataset for vqa on document images. In _Proceedings of the IEEE/CVF winter conference on applications of computer vision_ . 2200–2209. 

- [29] OpenAI. 2024. Hello gpt-4o. (2024). 

- [30] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. _The Journal of Machine Learning Research_ 21, 1 (2020), 5485–5551. 

- [31] S Robertson, Steve Walker, Susan Jones, and MHB GATFORD. 1994. Okapi at 3. In _Proceedings of the 3rd Text REtrieval Conference (-3)_ . 109–126. 

- [32] Karen Sparck Jones. 1972. A statistical interpretation of term specificity and its application in retrieval. _Journal of documentation_ 28, 1 (1972), 11–21. 

- [33] Quan Sun, Yufeng Cui, Xiaosong Zhang, Fan Zhang, Qiying Yu, Yueze Wang, Yongming Rao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. 2024. Generative multimodal models are in-context learners. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ . 14398–14409. 

- [34] Zineng Tang, Ziyi Yang, Guoxin Wang, Yuwei Fang, Yang Liu, Chenguang Zhu, Michael Zeng, Cha Zhang, and Mohit Bansal. 2023. Unifying vision, text, and layout for universal document processing. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ . 19254–19264. 

- [35] Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. 2024. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. _arXiv preprint arXiv:2403.05530_ (2024). 

- [36] Rubèn Tito, Dimosthenis Karatzas, and Ernest Valveny. 2023. Hierarchical multimodal transformers for multipage docvqa. _Pattern Recognition_ 144 (2023), 109834. 

- [37] Jordy Van Landeghem, Rubèn Tito, Łukasz Borchmann, Michał Pietruszka, Pawel Joziak, Rafal Powalski, Dawid Jurkiewicz, Mickaël Coustaty, Bertrand Anckaert, Ernest Valveny, et al. 2023. Document understanding dataset and evaluation (dude). In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ . 19528–19540. 

- [38] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. 2024. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. _arXiv preprint arXiv:2409.12191_ (2024). 

- [39] Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, et al. 2023. Cogvlm: Visual expert for pretrained language models. _arXiv preprint arXiv:2311.03079_ (2023). 

- [40] Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei, and Ming Zhou. 2020. Layoutlm: Pre-training of text and layout for document image understanding. In _Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining_ . 1192–1200. 

- [41] Yang Xu, Yiheng Xu, Tengchao Lv, Lei Cui, Furu Wei, Guoxin Wang, Yijuan Lu, Dinei Florencio, Cha Zhang, Wanxiang Che, et al. 2020. Layoutlmv2: Multimodal pre-training for visually-rich document understanding. _arXiv preprint arXiv:2012.14740_ (2020). 

- [42] Jiabo Ye, Anwen Hu, Haiyang Xu, Qinghao Ye, Ming Yan, Guohai Xu, Chenliang Li, Junfeng Tian, Qi Qian, Ji Zhang, et al. 2023. Ureader: Universal ocr-free visually-situated language understanding with multimodal large language model. _arXiv preprint arXiv:2310.05126_ (2023). 

- [43] Shi Yu, Chaoyue Tang, Bokai Xu, Junbo Cui, Junhao Ran, Yukun Yan, Zhenghao Liu, Shuo Wang, Xu Han, Zhiyuan Liu, et al. 2024. VisRAG: Vision-based Retrieval-augmented Generation on Multi-modality Documents. _arXiv preprint arXiv:2410.10594_ (2024). 

- [44] Manzil Zaheer, Guru Guruganesh, Kumar Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, et al. 2020. Big bird: Transformers for longer sequences. _Advances in neural information processing systems_ 33 (2020), 17283–17297. 

- [45] Jinxu Zhang, Yongqi Yu, and Yu Zhang. 2024. CREAM: Coarse-to-Fine Retrieval and Multi-modal Efficient Tuning for Document VQA. In _Proceedings of the 32nd ACM International Conference on Multimedia_ . 925–934. 

4221 

