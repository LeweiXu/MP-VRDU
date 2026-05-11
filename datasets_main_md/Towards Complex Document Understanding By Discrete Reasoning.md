# **Towards Complex Document Understanding By Discrete Reasoning** 

Fengbin Zhu[1] _[,]_[2] , Wenqiang Lei[3][∗] , Fuli Feng[4] , Chao Wang[2] , Haozhou Zhang[3] , Tat-Seng Chua[1] 

1National University of Singapore, Singapore 

26Estates Pte Ltd, Singapore 

3Sichuan University, China 

4University of Science and Technology of China, China 

{zhfengbin,wenqianglei,fulifeng93}@gmail.com,wangchao@6estates.com,dcscts@nus.edu.sg 

## **ABSTRACT** 

Document Visual Question Answering (VQA) aims to answer questions over visually-rich documents. In this work, we introduce a new Document VQA dataset, named TAT-DQA, which consists of 3,067 document pages comprising semi-structured table(s) and unstructured text as well as 16,558 question-answer pairs. The documents are sampled from financial reports and contain lots of numbers, which means discrete reasoning capability is demanded to answer the questions. Based on TAT-DQA, we further develop a novel model named MHST that takes into account the information in multi-modalities to intelligently address different types of questions with corresponding strategies, i.e., extraction or reasoning. The experiments show that MHST model significantly outperforms the baseline methods, demonstrating its effectiveness. However, the performance still lags far behind that of expert humans. We expect that our TAT-DQA dataset would facilitate the research on understanding of visually-rich documents, especially for scenarios that require discrete reasoning. Also, we hope the proposed model would inspire researchers to design more advanced Document VQA models in future. Our dataset will be publicly available for noncommercial use at https://nextplusplus.github.io/TAT-DQA/. 

## **CCS CONCEPTS** 

**Question** : What was the total cost in Wireless including spectrum license fee in 2019? **Derivation** : 1,320 + 1,731 = 3,051 **Scale** : Millions **Answer** : 3,051,000,000 

- **Computing methodologies** → **Natural language processing** ; 

- **Information systems** → _Question answering_ . 

## **KEYWORDS** 

**Figure 1: An example of TAT-DQA dataset. Given a question and a visually-rich document that contains both tabular and textual data, the machine is expected to derive the answer.** 

Question Answering, Visually-rich Document Understanding, Discrete Reasoning 

## **ACM Reference Format:** 

Fengbin Zhu[1] _[,]_[2] , Wenqiang Lei[3][∗] , Fuli Feng[4] , Chao Wang[2] , Haozhou Zhang[3] , Tat-Seng Chua[1] . 2022. Towards Complex Document Understanding By Discrete Reasoning. In _Proceedings of the 30th ACM International Conference_ 

## ∗Corresponding author. 

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than ACM must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org. _MM ’22, October 10–14, 2022, Lisboa, Portugal_ © 2022 Association for Computing Machinery. ACM ISBN 978-1-4503-9203-7/22/10...$15.00 https://doi.org/10.1145/3503161.3548422 

_on Multimedia (MM ’22), October 10–14, 2022, Lisboa, Portugal._ ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/3503161.3548422 

## **1 INTRODUCTION** 

Document understanding and analysis are indispensable in businesses of diverse domains like finance, legal, medical, etc [36]. Such work is mostly performed manually, which is labor-intensive and time-consuming with low scalability [9]. Intelligent Document Understanding (IDU) emerges as an important research area in multimedia, which spans Natural Language Processing (NLP) and Computer Vision (CV) [9]. It aims to automatically read and understand business documents. Many IDU tasks have been proposed, including Document Layout Analysis [26, 49], Table Detection and Recognition [24, 35, 48], Key Information Extraction (KIE) [14, 20, 21], Document Visual Question Answering (VQA) [28, 29, 37], etc. 

MM ’22, October 10–14, 2022, Lisboa, Portugal 

Zhu, Fengbin, et al. 

Among these tasks, Document VQA is a high-level document understanding task wherein given a visually-rich document and a relevant question in natural language (Figure 1), the model is required to give the correct answer to the question based on the document [9]. In Document VQA, the model needs to effectively exploit and harness the textual and layout information of the document besides its image information, compared to traditional VQA tasks [3, 13, 34] where image information is the focus. For this task, we are particularly interested in handling those documents with semi-structured tables that usually contain numbers in addition to unstructured text. This document type is informative and very pervasive in real-world businesses, however with only a few prior efforts [7, 50] on auto-understanding them. To facilitate the research on these problems, we construct a new **D** ocument VQA dataset by extending the **TAT-QA** [50], called **TAT-DQA** dataset. 

The documents in TAT-DQA dataset are sampled from real-world high-quality financial reports and each document contains both tabular and textual data. Furthermore, these documents contain lots of numbers, which means discrete reasoning capability, such as counting, sorting, comparison, addition, subtraction, multiplication, division and the compositions of them, is demanded to answer questions. The average length of the documents in TAT-DQA is significantly larger than all existing Document VQA datasets. To our best knowledge, TAT-DQA is the first Document VQA dataset that is constructed based on real-world high-quality business documents. Also, this work is the first one that attempts to understand the documents with multiple pages in literature on document understanding, which is more challenging compared to the understanding tasks over single-page documents. 

Based on TAT-DQA, we further propose a novel multi-modal Document VQA model, named **MHST** . To represent the question and document, the MHST employs a multi-modal Transformer encoder to take the question as well as the document text, layout and visual image information as input. After that, to infer the answer, it first adopts a “ **M** ulti- **H** ead” classifier to predict the answer type, i.e., _Span_ , _Spans_ , _Counting_ and _Arithmetic_ , based on which different prediction strategies are applied. For the answer type of _Span_ , a Span Answer Predictor [10, 31] is applied to predict the starting and ending positions of the answer span; for the other three answer types, a Sequence Tagging Module is applied to extract the evidences for deriving the answer. After obtaining the evidences, an Expression Tree Generator following “ **S** eq2 **T** ree” architecture is used to generate an expression tree to infer the final answers for arithmetical questions; for _Spans_ and _Counting_ questions the final predictions are obtained by collecting or counting all non-contiguous spans in the evidences. Experiments show that MHST model significantly outperforms baseline methods, demonstrating effectiveness. 

We expect that our TAT-DQA dataset would facilitate future research on the deep understanding of complex real-world documents combining vision and language, especially for scenarios requiring discrete reasoning, and that our MHST model would inspire the community to develop more advanced Document VQA models. 

## **2 RELATED WORK** 

In this section we briefly review previous research in Intelligent Document Understanding, the datasets for Document VQA, as well 

as discrete reasoning, with special attention to those works that are most related to ours. 

## **2.1 Intelligent Document Understanding** 

Intelligent Document Understanding (IDU) is to enable a machine to automatically read, understand, and analyze business documents. This research area is very practically meaningful and much demanded, which spans both the Natural Language Processing (NLP) and the Computer Vision (CV) fields [9]. Thanks to the wide success of Transformer [10, 38] in addressing NLP and CV problems, Transformer-based models have been popularly applied to solving document understanding tasks. The documents mainly involve three modalities of information: text, layout, and visual information. Some works [12, 17, 23, 44] combine document layout information with text information in the Transformer encoder. LayoutLM [44] and StructuralLM [23] incorporate document layout information as new positional embeddings into the embedding layer. BROS [17] and LAMBERT [12] take into account the spatial distance between tokens when computing the attention weights and bias in the selfattention layers. Recently, some works [2, 42, 45] propose to model all three modalities of information, which has become a defacto approach for almost all document understanding tasks. In this work, we also adopt a Transformer-based model to encode document text, layout and visual image information for our task. 

## **2.2 Document VQA Datasets** 

Document VQA is a high-level document understanding task wherein a model is required to answer a question in natural language given a visually-rich document [9]. To date, there are only a few datasets that have been proposed specially for Document VQA, to our best knowledge, including DocVQA [29], VisualMRC [37], InfographicVQA [28] and WebSRC [6]. Among them, InfographicVQA focuses on infographic instead of documents, and WebSRC does not provide its statistics in the original paper. DocVQA is constructed using various types of industry documents for extractive question answering, where answers can always be extracted from the text in the document. In DocVQA, the documents are mostly from the 19602000 period, with low-resolution, and besides some high-quality documents with printed or digital-born text, there are also some with typewritten and handwritten text. The VisualMRC [37] dataset is built for abstractive question answering, where answers cannot be directly extracted from the text in the document. The documents in VisualMRC and WebSRC are screenshots of web pages. In this work, we build a new and complex TAT-DQA dataset with realworld high-quality business documents, hoping to facilitate future research in the community. 

## **2.3 Discrete Reasoning** 

Discrete reasoning is key to solving many NLP tasks, like Math Word Problems (MWPs) [19, 22, 40, 43, 47] and Question Answering (QA) over text [11, 51], tables [30] and both [7, 25, 50]. In a Math Word Problem, it is required to answer a mathematical query according to a textual description and has been studied since the 1960s [4]. The MVPs are often solved by generating an expression tree to derive the final answer implicitly [8, 39] or explicitly [27, 43, 47]. For discrete reasoning in textual QA like the DROP [11] dataset, 

Towards Complex Document Understanding By Discrete Reasoning 

|**Statistic**|**Train**|**Dev**|**Test**|
|---|---|---|---|
|# of questions|13,251|1,645|1,662|
|# of documents|2,207|274|277|
|# of documents (1 page)|2,004|229|219|
|# of documents (>1 page)|203|45|58|
|Avg. length of words (per document)|539.90|579.89|603.57|
|Avg. length of words (per page)|494.43|496.53|496.42|



MM ’22, October 10–14, 2022, Lisboa, Portugal 

|**Answer Type**|**Train**|**Dev**|**Test**|**Total**|
|---|---|---|---|---|
|Span|5,737|690|714|7,141|
|Spans|1,656|216|210|2,082|
|Counting|305|32|40|377|
|Arithmetic|5,553|706|699|6,958|
|Total|13,251|1,645|1,662|16,558|



**Table 2: Question distribution regarding different answer types of each split in TAT-DQA.** 

**Table 1: Basic statistics of each split in TAT-DQA** 

researchers usually employ a “Multi-Head” classifier to predict the answer type first and then perform arithmetic operations to derive the final answer [1, 5, 18, 32, 33]. For discrete reasoning over tables only or a combination of the table and text, recent works incorporate the table structure feature in the positional encoding layer [16, 41], or the attention layer [41, 46] to jointly train tables and text. To our best knowledge, no prior work attempts to develop models that are capable of performing discrete reasoning over real-world business documents. 

## **3 PROPOSED DATASET: TAT-DQA** 

In this section, we present the definition of the Document VQA task, the construction of the TAT-DQA dataset, and the statistical analysis of the dataset. 

## **3.1 Task Definition** 

Consider a visually-rich document _𝐷_ of one or more pages, which contains both tabular and text contents. Each page is converted to an image plus a list of words using the PDF/OCR converter. Given a question _𝑄_ , the model F is required to predict the answer _𝑎_ according to the document _𝐷_ as demonstrated in Figure 1. Formally, the task is formulated as 

**==> picture [145 x 9] intentionally omitted <==**

In TAT-DQA, the answer value _𝑎_ may be either extracted from the given document, or generated by performing discrete reasoning such as addition, subtraction, multiplication, division, counting, comparison/sorting, and their compositions. 

To solve this task, a multi-modal Document VQA model usually needs to take into account the document text content, layout information, and visual image information in order to derive the final answer. In this process, the capability of discrete reasoning over the visually-rich document is much demanded. 

## **3.2 Dataset Construction** 

The dataset TAT-DQA is built upon a previous TAT-QA [50] dataset. In TAT-QA, each _hybrid context_ (i.e. data) comprises a well-structured table and some relevant texts. They are all selected and sorted manually by human experts from financial reports in PDF format. With human interference, each hybrid context contains only one table and the corresponding texts are sure to semantically correlate with the table. Compared with TAT-QA, the TAT-DQA better aligns with real scenarios. Each document in TAT-DQA may contain one or more tables, and the texts inside this document may correlate with the table(s), or may have nothing to do with the table(s). This task setting is much challenging than that on TAT-QA. 

_3.2.1 Collection of Question-Answer Pairs._ To construct the new TAT-DQA dataset, we borrow the question-answer (Q-A) pairs from the previous TAT-QA dataset, which are generated by human experts in finance. On this basis, we ask human experts in finance to generate some more Q-A pairs, and meanwhile remove a few pairs with errors we have found during data preparation. In total, we get 16 _,_ 558 Q-A pairs for TAT-DQA. The same as TAT-QA, TATDQA offers four types of answers: 

- _Span_ : The answer is a continuous text in the document [31]; 

- _Spans_ : This type of answer is also called “Multi-span" and is a set of non-contiguous spans in the document [11]; 

- _Counting_ : The answer is an integer that is computed by performing counting; 

- _Arithmetic_ : The answer is a numerical value that is obtained by performing arithmetic operations such as addition, subtraction, multiplication, division and their compositions. 

Also, the scale of the answer in TAT-DQA can have five values, including None, Thousands, Millions, Billions, Percent. To facilitate development of discrete reasoning capability of Document VQA models, we also keep the corresponding derivations for _Counting_ and _Arithmetic_ questions in the same formats with TAT-QA. 

_3.2.2 Collection of Document Pages._ To construct the new TATDQA dataset, we filter out and keep the document pages corresponding to the above acquired Q-A pairs from the raw financial reports. These raw financial reports are real-world ones, mostly dated between 2018 and 2020, which are downloaded from the web[1] based on the file names provided by the authors of TAT-QA. Each Q-A pair corresponds to one document, and a document may contain at most three pages. We process each retaining document page after filtering to obtain the text with a bounding box by using Apache PDFBox[2] for the readable PDF files or a commercial OCR engine for the images. Finally, each document page is converted to a list of text blocks and each text block has a list of words, with every block or word framed by its own bounding box. In total, we obtain 2 _,_ 758 documents consisting of 3 _,_ 067 pages. The document itself and its text content with the corresponding bounding box will be released together in TAT-DQA. 

## **3.3 Statistics and Analysis** 

In total, the TAT-DQA dataset includes 2 _,_ 758 documents, consisting of 3 _,_ 067 document pages from 182 financial reports, and 16 _,_ 558 question-answer pairs. These documents are randomly split into a training set (80%), a development set (10%) and a test set (10%); all 

1https://www.annualreports.com/ 

> 2https://pdfbox.apache.org/ 

MM ’22, October 10–14, 2022, Lisboa, Portugal 

Zhu, Fengbin, et al. 

|**Property**|**DocVQA**|**VisualMRC**|**TAT-DQA**|
|---|---|---|---|
|Document Type|Industry<br>document|Web<br>pages|Finance<br>reports|
|Document Period|1960 - 2000|Jan - Mar 2020|2018 - 2020|
|Avg len of document<br>Avg no. of pages|182.75<br>1|151.46<br>1|550.29<br>1.33|
|Answer Type|Ext.|Abs.|Ext. + Abs.|



**Table 3: The comparison among the three Document VQA datasets, i.e., DocVQA, VisualMRC and TAT-DQA. The length of the document is counted in terms of OCR words.** _**Ext**_ **. and** _**Abs**_ **denote** _**extractive**_ **and** _**abstractive**_ **respectively.** 

the questions about a particular document belong to only one of the splits. We summarize the basic statistics of each split in Table 1, and the question distribution regarding the answer type in Table 2. 

A comparison of our new TAT-DQA dataset with the two existing document VQA datasets DocVQA [29] and VisualMRC [37] is summarized in Table 3. In particular, for **document type** , the documents in TAT-DQA stem from real-world high-quality financial reports between 2018 and 2020, and each document must contain both tabular and textual data. Comparably, the documents of DocVQA are from the UCSF Industry Documents Library, which are mostly within the 1960-2000 period, of low-resolution and include some typewritten and handwritten text; VisualMRC is built with the screenshots of web pages instead of real-world documents. For **document length** , the average length of documents in TATDQA (550.29 words) is significantly larger than that of DocVQA (182.75 words) and VisualMRC (151.46 words), which makes TATDQA more complex and challenging. For the **number of pages per document** , each document in DocVQA or VisualMRC has only 1 page; in contrast, TAT-DQA has at most 3 pages in the document, and its average page number per document is 1.33 and over 11.0% documents in TAT-DQA have more than 1 page. We also compare the **answer type** of the three datasets: TAT-DQA consists of extractive (i.e., _Span_ and _Spans_ ) and abstractive (i.e., _Counting_ and _Arithmetic_ ) answers that need to be generated with discrete reasoning; DocVQA only has SQuAD-like extractive and short answers while VisualMRC focuses on long abstractive answers. 

To the best of our knowledge, the TAT-DQA dataset is the first document VQA dataset that is constructed on top of real-world highquality business documents. It is also the most complex document VQA dataset till now, with more than one page per document. We promise to release this dataset in near future, hoping to facilitate the research on document understanding techniques in the community. 

## **4 PROPOSED METHOD: MHST** 

The data type in the new TAT-DQA dataset is very common in real-world business scenarios, i.e. a document of one or more pages containing one or more tables as well as several texts. To effectively auto-understand such documents, we introduce a novel multimodal QA model, named MHST, which has two stages, i.e., **Question and Document Representation** , and **Answer Extraction and Reasoning** . To represent the question and the document, the 

MHST employs a multi-modal Transformer architecture to encode the question, document text, layout and visual image in the input. To infer the final answer, the MHST adopts a “ **M** ulti- **H** ead” classifier to predict the answer type, i.e., _Span_ , _Spans_ , _Counting_ and _Arithmetic_ . For the answer type of _Span_ , a Span Answer Predictor [10, 31] is applied to predict the starting and ending positions of the answer span; for the other three answer types, a Sequence Tagging Module is applied to extract the evidences for deriving the answer. After obtaining the evidences, an Expression Tree Generator following “ **S** eq2 **T** ree” architecture is applied to generate an expression tree to infer the final answers for arithmetical questions; for _Spans_ and _Counting_ questions the final predictions are obtained by collecting or counting all non-contiguous spans in the evidences. An overall architecture of the MHST model is illustrated in Figure 2. 

## **4.1 Question and Document Representation** 

The MHST model adopts LayoutLMv2LARGE [45] to generate the representations of the question and the document, which is a recent popular multi-modal Transformer model for document understanding. The MHST model takes as input a question as well as the text content, layout, and visual image information of the document. 

The document text and layout information in the input can be obtained as follows. Each document in TAT-DQA contains both tabular and textual data. We empirically find that the performance would be better if we handle them separately, as is verified in the experiment of Section 5.3.3. To identify the table(s) vs. the text in the document, we propose to apply a heuristic method to identify the text blocks that belong to the table in each document page. Then we apply TF-IDF to sort the rest non-table text blocks by estimating the similarity score of each block with respect to the question, considering that the average length of the document in TAT-DQA is quite long, as demonstrated in Table 1. 

To obtain the visual information of the document for the input, we transform the document page[3] in PDF to an image, which is resized to 224 x 224 and then fed into the image encoder proposed in [45] to obtain the visual embedding. Finally, the embeddings of the question, table blocks, non-table blocks and the image are input sequentially to the LayoutLMv2LARGE model to obtain question token representations _𝑄_ and document token representations _𝐷_ . 

## **4.2 Answer Extraction and Reasoning** 

After obtaining the representations of the question and document, MHST further identifies the answer type of the question and applies the corresponding strategies to derive the final answer. 

_4.2.1 Multi-Head Predictor._ We design a Multi-Head Predictor to predict the answer type, i.e., _Span_ , _Spans_ , _Counting_ and _Arithmetic_ , as explained in Section 3.2.1. This Multi-Head Predictor is essentially a multi-class classifier. In particular, we take the vector of [CLS] as input to compute the probability of each answer type: 

**==> picture [177 x 12] intentionally omitted <==**

where FFN denotes a two-layer feed-forward network with the GELU activation [15]. For the answer type of _Span_ , a Span Answer Predictor is applied to predict the starting and ending positions of 

3We select the document page that has the major table for the multi-page documents. 

MM ’22, October 10–14, 2022, Lisboa, Portugal 

Towards Complex Document Understanding By Discrete Reasoning 

**==> picture [418 x 244] intentionally omitted <==**

**----- Start of picture text -----**<br>
Answer Extraction & Reasoning 3,051 millions<br>millions Expression Tree<br>+<br>Counting Count Scale Predictor<br>1,320 1,731<br>Sequence Tagging<br>Spans Module Multiple spans<br>Multi-Head<br>Predictor Expression Tree<br>Arithmetic 1,320 1,731<br>Generator<br>Span Answer<br>Span Start&End position<br>Predictor<br>Question & Document Representation<br>[CLS] what was ... 2019 [SEP] capital expend ... with 2018 [SEP] V1 V2 ... Vn<br>Transformer Layers<br>[CLS] what was ... 2019 [SEP] capital expend ... with 2018 [SEP] V1 V2 ... Vn<br>Question Embeddings Document Text Embeddings Document Visual Embeddings<br>**----- End of picture text -----**<br>


**Figure 2: The overall architecture of the model MHST. Take the question in Figure 1 as an example. MHST employs a multimodal Transformer architecture to take the text, layout and visual embeddings as input. The visual embeddings (i.e., V1, V2,...,Vn) are obtained by leveraging a CNN-based visual encoder. Then it adopts a “Multi-Head” classifier to predict the answer type, i.e.,** _**Span**_ **,** _**Spans**_ **,** _**Counting**_ **and** _**Arithmetic**_ **. An Expression Tree Generator following “Seq2Tree” architecture is applied to generate an expression tree with the selected numbers by Sequence Tagging Module to derive the final answers for the arithmetical questions.** 

the answer span following the typical SQuAD-like QA [10, 31]. For the other three answer types, we apply a Sequence Tagging Module to extract the corresponding evidences of the answer inspired by [50], but use Beginning-Inside–Outside (BIO) tagging [33] instead of Inside–Outside (IO) tagging to better handle multi-span answers. After obtaining the evidences, the final prediction for _Spans_ and _Counting_ questions are obtained by collecting or counting all non-contiguous spans in the evidences following TagOp [50]. 

_4.2.2 Expression Tree Generator._ For the question whose answer type is predicted as _Arithmetic_ , we apply an Expression Tree Generator to generate an expression tree to compute the answer [39, 43, 47]. The Expression Tree Generator in our MHST is implemented with the Goal-driven Tree Structure (GTS) [43]. GTS is a tree structured neural network that generates expression trees in a goal-driven manner, demonstrating noticeable effectiveness in solving Mathematical Word Problems (MWPs). However, a typical MWP involves only a handful of numbers, while in TAT-DQA, one document usually contains much more numbers, which significantly overwhelms the capacity of GTS. To address this issue, we propose to select several most relevant numbers with the Sequence Tagging Module, and feed them into the GTS. 

The expression trees generated by the Generator contain three kinds of nodes: the arithmetical operators _𝑉𝑜𝑝_ (i.e., +-*/), the constant numbers _𝑉𝑐𝑜𝑛_ , and the numbers _𝑉𝑡𝑎𝑔_ that are identified by 

the Sequence Tagging Module in the document _𝐷_ , which form the target vocabulary _𝑉[𝑑𝑒𝑐]_ of the document _𝐷_ . The constant numbers _𝑉𝑐𝑜𝑛_ and the numbers _𝑉𝑡𝑎𝑔_ selected from the document are always set to be in leaf nodes positions. The operators _𝑉𝑜𝑝_ will always occupy the non-leaf nodes positions, and each operator node must have two child nodes. The construction of the tree starts from producing the topmost operator, followed by the left child node, which will be repeated until the leaf node is produced. Then, the right child nodes are generated recursively. As such, the Generator generates an equation following the pre-order traversal ordering. 

To start tree generation process of GTS, our model initializes the topmost root node vector using the vector of [CLS]. For each token t in target vocabulary _𝑉[𝑑𝑒𝑐]_ , its embedding e( _𝑡_ | _𝐷_ ) is defined as 

**==> picture [183 x 39] intentionally omitted <==**

The representations of the numbers in _𝑉𝑡𝑎𝑔_ are document-dependent; i.e., they will take the corresponding _ℎ𝑙𝑜𝑐_ ( _𝑡, 𝐷_ ) from _𝐷_ . However, the representations of the operators and the constant numbers are independent of the document, which are obtained by two independent embedding matrices M _𝑜𝑝_ and M _𝑐𝑜𝑛_ respectively. 

_4.2.3 Scale Predictor._ If the answer type is _Arithmetic_ , the scale predictor takes as input the concatenation of the vector of the 

MM ’22, October 10–14, 2022, Lisboa, Portugal 

Zhu, Fengbin, et al. 

[ _𝐶𝐿𝑆_ ] token and the mean of the representations of all predicted numerical values to compute the probability of each scale: 

**==> picture [183 x 12] intentionally omitted <==**

where FFN denotes a two-layer feed-forward network with GELU activation and _ℎ_[N] is obtained by computing the mean of all representations of the predicted numerical tokens by the Sequence Tagging Module. For the types of _Span_ , _Spans_ and _Counting_ , we adopt the same method in TagOp, which takes the vector of [ _𝐶𝐿𝑆_ ] only as input. After obtaining the scale, the final answer is derived by multiplying or concatenating the answer value with the scale, depending on whether the answer value is a number or a string. 

## **4.3 Training and Inference** 

To train our model, the objective is to minimize the negative loglikelihood loss which is the sum of the losses of all above classification tasks depending on the answer types, i.e., Multi-Head Predictor, Sequence Tagging Module, Span Answer Predictor, Scale Predictor and Expression Tree Generator. Note that the ground-truths of the sequence tagging predictions are extracted from the annotated answers and derivations. For the arithmetic type, if the numerical values that are used to generate the ground-truth expression tree are not predicted, we will also add them in the ground truth in order to train the Expression Tree Generator. 

During inference, our model first chooses the answer type and then performs the corresponding prediction strategies. For the _Span_ type, the span with the maximum probability is attained as the final prediction among all the valid spans. If the answer type is _Spans_ or _Counting_ , we collect or count all the predicted non-contiguous spans as the final prediction. Following [43], we apply a beam search to select the best expression tree and execute it to obtain the final prediction for the arithmetical questions. 

## **5 EXPERIMENTS** 

In this section, we report and analyze the extensive experimental results to demonstrate the validity of our new TAT-DQA dataset and also the promising effectiveness of our proposed Document VQA model MHST by comparing it with two baseline models. 

## **5.1 Baseline Models** 

To the best of our knowledge, there are very limited models that have been proposed to effectively solve QA tasks over the documents containing both tabular data and textual data, where discrete reasoning is particularly demanded. We choose two state-of-the-art QA models that have demonstrated promising discrete reasoning capability for comparison, i.e. NumNet+ V2 [32] and TagOp [50]. 

NumNet+ V2 [32] is a textual QA model with impressive performance on DROP [11] dataset that requires discrete reasoning over the textual data. It constructs a numerically-aware graph neural network, which takes all the numbers in the given question and passage as nodes and builds edges via numerical comparison, and performs discrete reasoning over the graph to infer the final answer. To adapt the model to TAT-DQA, we apply TF-IDF between the question and each text block to sort the text blocks and convert them to a sequence as the input to the model, which is similar with the method introduced in Section 4.1. 

|**Method**|**Dev**<br>EM<br>F1|**Test**<br>EM<br>F1|
|---|---|---|
|**Human Experts**<br>**Baselines**<br>NumNet+ V2<br>TagOp<br>**Text Only**<br>MHST (RoBERTaLARGE)<br>**Text + Image**<br>MHST (LayoutLMv2LARGE)|-<br>-<br>28.1<br>36.6<br>32.3<br>39.6<br>37.1<br>43.6<br>**39.1**<br>**47.4**|84.1<br>90.8<br>30.6<br>40.1<br>33.7<br>42.5<br>39.8<br>47.6<br>**41.5**<br>**50.7**|



**Table 4: Performance of our model and baseline models on dev and test set of TAT-DQA. Best results are marked in bold.** 

The other model, TagOp [50], achieves the state-of-the-art results on the TAT-QA dataset that requires discrete reasoning over a well-structured table and its relevant text to derive the answer. It first employs a sequence tagging module to identify relevant cells from the table and spans from the text, and then applies a set of aggregation operators (e.g., addition, subtraction, multiplication, division, counting,etc) over them to infer the final answer. To enable the model to work on the new TAT-DQA dataset, we omit the processing of the table, apply TF-IDF to sort the text blocks in the document and feed them to the model sequentially. 

## **5.2 Evaluation Metrics** 

We adopt the same evaluation metrics used in [50] to measure the model performance on the TAT-DQA dataset, i.e. the Exact Match (EM) and the numeracy-focused (macro-averaged) F1 score, taking into account the scale and the plus-minus of a numerical value. Both Exact-Match and numeracy-focused (macro-averaged) F1 score measure the overlap between a bag-of-words representation of the gold and predicted answers. Note that the numeracy-focused F1 score is set to 0 unless the predicted number multiplied by the predicted scale equals exactly the ground truth. 

## **5.3 Results and Analysis** 

In experiments, we first compare our MHST model with two baseline models, NumNet+ V2 and TagOp, by testing their performance on understanding the documents in TAT-DQA dataset via Document VQA tasks. We then analyze effects of our MHST model over each answer type in TAT-DQA. Also, we experiment on TATQA dataset [50] to further verify the effectiveness of our proposed model. Finally, based on all experimental results, we analyze the challenges of our new TAT-DQA dataset to reveal its properties. 

_5.3.1 Comparison with Baselines on TAT-DQA._ The experimental results of our proposed model MHST and the baseline models are shown in Table 4. We train two different variants of our MHST model by using different modalities in **Question and Document Representation** stage. The first variant is the full model MHST(LayoutLMv2LARGE), which adopts LayoutLMv2LARGE as the encoder and takes the question as well as document text, layout, and visual image as input. The other one is MHST (RoBERTaLARGE), 

MM ’22, October 10–14, 2022, Lisboa, Portugal 

Towards Complex Document Understanding By Discrete Reasoning 

|**Answer Type**|**Dev**<br>EM<br>F1|**Test**<br>EM<br>F1|
|---|---|---|
|Span<br>Spans<br>Counting<br>Arithmetic|39.0<br>53.5<br>26.9<br>43.0<br>46.3<br>46.3<br>42.5<br>42.5|41.1<br>58.3<br>25.7<br>43.3<br>43.2<br>43.2<br>42.7<br>42.7|



**Table 5: Performance of MHST for different answer types on TAT-DQA dev and test set.** 

**==> picture [153 x 95] intentionally omitted <==**

**----- Start of picture text -----**<br>
Baseline +Table block detection<br>50.00%<br>40.00%<br>30.00%<br>20.00%<br>NumNet+ V2 TagOp<br>**----- End of picture text -----**<br>


**Figure 3: The F1 score on TAT-DQA test set after we add the same table block detection module in NumNet+ V2 and TagOp from MHST.** 

|**Method**|**Dev**<br>EM<br>F1|**Test**<br>EM<br>F1|
|---|---|---|
|NumNet+ V2<br>TagOp<br>MHST (RoBERTaLARGE)|38.1<br>48.3<br>55.2<br>62.7<br>**68.2**<br>**76.8**|37.0<br>46.9<br>50.1<br>58.0<br>**63.6**<br>**72.7**|



**Table 6: Performance of our model and baseline models on dev and test set of TAT-QA. Best results are marked in bold.** 

_5.3.3 The Challenges of TAT-DQA._ The new TAT-DQA dataset differs from TAT-QA in two aspects: 1) in TAT-DQA there is no well-structured table and instead, all contents are organized with text blocks; 2) the texts in TAT-DQA are not necessarily associated to the table, which are with longer contents and more complex (as verified experimentally above). Hereby, we try to reveal the challenges of TAT-DQA by answering following three questions: 

- **Q1** : How do the tabular data in the document affect the model performance? 

which employs RoBERTaLARGE as the encoder instead to represent the document with textual features only. From Table 4, we can observe that both the two variants significantly outperform the baseline methods, demonstrating the effectiveness of our MHST. The superior performance of our full model MHST(LayoutLMv2LARGE) highlights the importance of the fusion of vision and language features to successfully answering the questions in TAT-DQA. But, there is still a big gap compared to the human performance. 

The TAT-DQA dataset offers four different answer types. To better reveal the effects of our model, we analyze the performance of the MHST(LayoutLMv2LARGE) on each of these answer types. The results are summarized in Table 5. From the table, we can see that the metric F1 on test set for the answer type of _Span_ shows the best results, while the results on _Counting_ and _Spans_ are similar, which is probably because these two types use similar prediction strategies in our model, i.e. generating the answer directly after obtaining the evidence using sequence tagging. Comparably, the _Arithmetic_ type has the worst results of F1, indicating that the discrete reasoning capability still demands further enhancement on complex real-world documents. 

_5.3.2 Results of MHST model on TAT-QA._ To further test the effectiveness of our MHST model, we also test it on the TAT-QA dataset. In particular, we adapt the variant MHST(RoBERTaLARGE) as there are only well-structured tables and relevant text and no document images in TAT-QA. 

Following TagOp, MHST(RoBERTaLARGE) takes the question, the flattened table by row and the associated paragraphs sequentially as input instead. From Table 6, we can observe that our MHST model significantly outperforms the state-of-the-art method on TAT-QA. It is worth mentioning that our model performs better on TATQA compared with its performance on the new TAT-DQA dataset. This indicates that TAT-DQA is more complex and challenging than TAT-QA, which will be analyzed in detail at below. 

- **Q2** : What is the difference in the model performance between multi-page documents and one-page documents? 

- **Q3** : What is the model performance on the documents with more content compared to those with less content? 

For **Q1** , to evaluate the effect of tabular data in the document on the performance, we apply the same table block detection module in Section 4.1 to NumNet+ V2 and TagOp models respectively. The results are shown in Figure 3. We can observe that both NumNet+ V2 and TagOp models gain better performance on TAT-DQA test set after adding the table block detection module. Particularly, the TagOp model achieves 3 _._ 9% higher F1 score than original version, which is a significant increase. This indicates that it is favorable to recognize tables in the document first and then process separately. 

To answer **Q2** , we compare the performance of three models, NumNet+ V2, TagOp and MHST (RoBERTaLARGE), over multi-page documents and one-page documents, respectively. We here do not choose MHST(LayoutLMv2LARGE) as it takes as input the visual image, which is converted from the document page with the major table, meaning using only one page and hence unfair for comparison with other models. Figure 4 shows the comparison results on the test set of TAT-DQA. We can observe that the performance on multi-page documents is much worse than that on one-page documents for all three models. Among them, TagOp and MHST(RoBERTaLARGE) perform significantly better on documents with only one page. The results indicate that multi-page documents are more challenging than one-page ones. 

Then let us consider **Q3** . As shown in Table 1, the average number of OCR words per page is around 500 in TAT-DQA, which is much larger than previous document VQA datasets DocVQA [29] and VisualMRC [37]. We investigate the model performance with respect to the increase of document length for NumNet+ V2, TagOp and MHST(RoBERTaLARGE). For fairness, we only use the documents with one single page in the TAT-DQA test set for experiments here. We divide these one-page documents into two halves by the median 

MM ’22, October 10–14, 2022, Lisboa, Portugal 

Zhu, Fengbin, et al. 

**==> picture [164 x 102] intentionally omitted <==**

**----- Start of picture text -----**<br>
# of pages  > 1 # of pages = 1<br>50.00%<br>40.00%<br>30.00%<br>20.00%<br>NumNet+ V2 TagOp MHST<br>**----- End of picture text -----**<br>


**Figure 4: Performance comparison in F1 score between one-page documents and multi-page documents on TAT-DQA test set. Here MHST denotes the variant of MHST(RoBERTaLARGE).** 

**==> picture [156 x 97] intentionally omitted <==**

**----- Start of picture text -----**<br>
Long documents Short documents<br>50.00%<br>40.00%<br>30.00%<br>20.00%<br>NumNet+ V2 TagOp MHST<br>**----- End of picture text -----**<br>


**Figure 5: Performance comparison in F1 score between Long documents and Short document on TAT-DQA test set. Here MHST denotes the variant of MHST (RoBERTaLARGE).** 

length, with the half containing fewer words named as “Short documents” and the other containing more words as “Long documents”. The results are shown in Figure 5. We can see that the performance on “Short documents” is significantly better than that on “Long documents”. Long documents are still a big challenge for document understanding tasks, where further research efforts are demanded. 

_5.3.4 Error Analysis._ To further investigate our MHST model, we randomly sample 100 error instances on the test set and analyze the reasons. As shown in Table 7, the errors occurred to four modules, Span Answer Predictor (SAP), Expression Tree Generator (ETG), Sequence Tagging (ST) module and Scale Predictor (SP), which are classified into seven categories (Col. 2 in Table 7), each with an example. We can observe that, 1) _SAP module_ (37%): 37% errors are due to inaccurate predictions of starting and ending positions for Span questions, i.e., 30% predictions overlapping but not exactly matching with ground truth, and 7% predictions having zero overlap with ground truth; 2) _ETG module_ (34%): 34% of all errors are caused by ETG generating wrong expressions for the input of correct evidences from ST module, among which 19% are wrong number signs (i.e., positive/negative) and 15% are other wrong expressions; 3) _ST module_ (25%): 25% errors are due to ST predicting wrong taggings, where interestingly, in 12% error cases, ST predicts a single string instead of identifying multiple answers from it for multi-span questions; 4) _SP module_ (4%): 4% errors are due to SP module failing to predict the scale of the answer. We will further improve our model based on these error analysis findings. 

|**Module**|**Error(%)**|**Example**|
|---|---|---|
|**SAP**|Ofset<br>Error<br>(30%)<br>**Q**: How is the fair value of fnancial instruments that<br>are not traded in active markets determined?<br>**G**:using valuation techniques<br>**P**:using valuation<br>||
||No<br>Overlap<br>(7%)|**Q**: In 2018, why did the revenues grew across all re-<br>gions?<br>**G**:mainly due to growth in Imaging and Automotive.<br>**P**:increase.|
|**ETG**|Wrong<br>Sign<br>(19%)|**Q**: What was the change in raw materials between<br>2018 and 2019?<br>**G**: 36,987 - 45,333 =-8346<br>**P**: 45,333 - 36,987 =8346|
||Wrong<br>Expressions<br>(15%)|**Q**: What is the percentage change in the short term<br>investments between 2018 and 2019?<br>**G**:(17,779 - 11,303)/11,303<br>**P**:(11,303 - 17,779 )/17,779|
|**ST**|Wrong<br>Taggings<br>(13%)|**Q**: What was the average Reductions for prior year tax<br>positions from 2017-2019?<br>**G**: ( -14 +0+0)/3<br>**P**: (2019- 14 - 14)/3|
||Failed<br>Segment<br>(12%)|**Q**: What are the three operating and reportable seg-<br>ments?<br>**G**: “PSG”, “ASG”, “ISG”<br>**P**: “PSG, ASG and ISG. The”|
|**SP**|Wrong<br>Scale<br>(4%)|**Q**: What was the diference between total Term Loans<br>and total Future Lease Commitments?<br>**G**:1,500 - 1,202 = 298millions<br>**P**: 1,500 - 1,202 = 298thousands|



**Table 7: Examples of errors and corresponding percentage in each module. SAP, ETG, ST and SP are the abbreviations of Span Answer Predictor, Expression Tree Generator, Sequence Tagging Module and Scale Predictor respectively. Q, G and P denote question, ground truth and prediction.** 

## **6 CONCLUSION AND FUTURE WORKS** 

In this work, we propose a new challenging Document VQA dataset, named TAT-DQA, constructed based on real-world high-quality financial reports, in which each document must have both tabular and textual data. With the new dataset, we further propose a novel MHST model for addressing Document VQA tasks. From our experiments, we have gained an insight that processing tabular data separately would significantly improve model performance. This inspires us to explore the approaches of detecting and recognizing all tables in the given document and processing them differently to enhance the Document VQA models. There are usually plentiful numbers in the document, which would overwhelm the capability of almost all existing discrete reasoning models. One promising solution to such an issue is to differentiate the relevant numbers to the question from lots of numbers in the document before feeding them to the discrete reasoning module to derive the final answer, like what we have done in this work as an inspiring work. In the future, we would like to continue exploring other potential methods to effectively identify the relevant numbers to a given question to enhance the discrete reasoning over complex business documents. 

## **ACKNOWLEDGMENTS** 

The authors gratefully thank all the anonymous reviewers for their positive feedback. This research is supported by the NExT Research Center, Singapore. 

MM ’22, October 10–14, 2022, Lisboa, Portugal 

Towards Complex Document Understanding By Discrete Reasoning 

## **REFERENCES** 

- [1] Daniel Andor, Luheng He, Kenton Lee, and Emily Pitler. 2019. Giving BERT a Calculator: Finding Operations and Arguments with Reading Comprehension. In _EMNLP-IJCNLP_ . ACL, 5947–5952. 

- [2] Srikar Appalaraju, Bhavan Jasani, Bhargava Urala Kota, Yusheng Xie, and R. Manmatha. 2021. DocFormer: End-to-End Transformer for Document Understanding. In _Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)_ . 993–1003. 

- [3] Ali Furkan Biten, Ruben Tito, Andres Mafla, Lluis Gomez, Marçal Rusinol, Ernest Valveny, CV Jawahar, and Dimosthenis Karatzas. 2019. Scene text visual question answering. In _Proceedings of the IEEE/CVF international conference on computer vision_ . 4291–4301. 

- [4] Daniel G Bobrow. 1964. Natural language input for a computer problem solving system. (1964). 

- [5] Kunlong Chen, Weidi Xu, Xingyi Cheng, Zou Xiaochuan, Yuyu Zhang, Le Song, Taifeng Wang, Yuan Qi, and Wei Chu. 2020. Question Directed Graph Attention Network for Numerical Reasoning over Text. In _EMNLP-IJCNLP_ . ACL, 6759–6768. 

- [6] Xingyu Chen, Zihan Zhao, Lu Chen, JiaBao Ji, Danyang Zhang, Ao Luo, Yuxuan Xiong, and Kai Yu. 2021. WebSRC: A Dataset for Web-Based Structural Reading Comprehension. In _Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing_ . Association for Computational Linguistics, 4173– 4185. 

- [7] Zhiyu Chen, Wenhu Chen, Charese Smiley, Sameena Shah, Iana Borova, Dylan Langdon, Reema Moussa, Matt Beane, Ting-Hao Huang, Bryan Routledge, and William Yang Wang. 2021. FinQA: A Dataset of Numerical Reasoning over Financial Data. In _Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing_ . Association for Computational Linguistics, Online and Punta Cana, Dominican Republic, 3697–3711. 

- [8] Ting-Rui Chiang and Yun-Nung Chen. 2019. Semantically-Aligned Equation Generation for Solving and Reasoning Math Word Problems. In _Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)_ . Association for Computational Linguistics, 2656–2668. 

- [9] Lei Cui, Yiheng Xu, Tengchao Lv, and Furu Wei. 2021. Document AI: Benchmarks, Models and Applications. _CoRR_ abs/2111.08609 (2021). 

- [10] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In _Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)_ . 4171–4186. 

- [11] Dheeru Dua, Yizhong Wang, Pradeep Dasigi, Gabriel Stanovsky, Sameer Singh, and Matt Gardner. 2019. DROP: A Reading Comprehension Benchmark Requiring Discrete Reasoning Over Paragraphs. In _Proc. of NAACL_ . 

- [12] Lukasz Garncarek, Rafal Powalski, Tomasz Stanislawek, Bartosz Topolski, Piotr Halama, and Filip Gralinski. 2020. LAMBERT: Layout-Aware language Modeling using BERT for information extraction. _CoRR_ abs/2002.08087 (2020). arXiv:2002.08087 

- [13] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. 2017. Making the V in VQA Matter: Elevating the Role of Image Understanding in Visual Question Answering. In _Conference on Computer Vision and Pattern Recognition (CVPR)_ . 

- [14] Filip Gralinski, Tomasz Stanislawek, Anna Wróblewska, Dawid Lipinski, Agnieszka Kaliska, Paulina Rosalska, Bartosz Topolski, and Przemyslaw Biecek. 2020. Kleister: A novel task for Information Extraction involving Long Documents with Complex Layout. _CoRR_ abs/2003.02356 (2020). arXiv:2003.02356 

- [15] Dan Hendrycks and Kevin Gimpel. 2016. Bridging Nonlinearities and Stochastic Regularizers with Gaussian Error Linear Units. _CoRR_ abs/1606.08415 (2016). arXiv:1606.08415 

- [16] Jonathan Herzig, Pawel Krzysztof Nowak, Thomas Müller, Francesco Piccinno, and Julian Eisenschlos. 2020. TaPas: Weakly Supervised Table Parsing via Pretraining. In _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics_ . ACL, 4320–4333. 

- [17] Teakgyu Hong, DongHyun Kim, Mingi Ji, Wonseok Hwang, Daehyun Nam, and Sungrae Park. 2021. {BROS}: A Pre-trained Language Model for Understanding Texts in Document. https://openreview.net/forum?id=punMXQEsPr0 

- [18] Minghao Hu, Yuxing Peng, Zhen Huang, and Dongsheng Li. 2019. A Multi-Type Multi-Span Network for Reading Comprehension that Requires Discrete Reasoning. In _Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)_ . Association for Computational Linguistics, 1596–1606. 

- [19] Danqing Huang, Shuming Shi, Chin-Yew Lin, and Jian Yin. 2017. Learning FineGrained Expressions to Solve Math Word Problems. In _Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing_ . ACL, 805–814. 

- [20] Zheng Huang, Kai Chen, Jianhua He, Xiang Bai, Dimosthenis Karatzas, Shijian Lu, and C. V. Jawahar. 2019. ICDAR2019 Competition on Scanned Receipt OCR and Information Extraction. In _2019 International Conference on Document Analysis_ 

_and Recognition (ICDAR)_ . 1516–1520. 

- [21] Guillaume Jaume, Hazim Kemal Ekenel, and Jean-Philippe Thiran. 2019. FUNSD: A Dataset for Form Understanding in Noisy Scanned Documents. _CoRR_ abs/1905.13538 (2019). arXiv:1905.13538 

- [22] Nate Kushman, Yoav Artzi, Luke Zettlemoyer, and Regina Barzilay. 2014. Learning to Automatically Solve Algebra Word Problems. In _Proceedings of the 52nd Annual Meeting of the Association for Computational Linguistics_ . ACL, 271–281. 

- [23] Chenliang Li, Bin Bi, Ming Yan, Wei Wang, Songfang Huang, Fei Huang, and Luo Si. 2021. StructuralLM: Structural Pre-training for Form Understanding. In _Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)_ . Association for Computational Linguistics, 6309–6318. 

- [24] Minghao Li, Lei Cui, Shaohan Huang, Furu Wei, Ming Zhou, and Zhoujun Li. 2020. TableBank: Table Benchmark for Image-based Table Detection and Recognition. In _Proceedings of the 12th Language Resources and Evaluation Conference_ . European Language Resources Association, 1918–1925. 

- [25] Moxin Li, Fuli Feng, Hanwang Zhang, Xiangnan He, Fengbin Zhu, and Tat-Seng Chua. 2022. Learning to Imagine: Integrating Counterfactual Thinking in Neural Discrete Reasoning. In _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ . Association for Computational Linguistics, 57–69. 

- [26] Minghao Li, Yiheng Xu, Lei Cui, Shaohan Huang, Furu Wei, Zhoujun Li, and Ming Zhou. 2020. DocBank: A Benchmark Dataset for Document Layout Analysis. In _Proceedings of the 28th International Conference on Computational Linguistics_ . International Committee on Computational Linguistics, 949–960. 

- [27] Qianying Liu, Wenyv Guan, Sujian Li, and Daisuke Kawahara. 2019. Treestructured Decoding for Solving Math Word Problems. In _Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)_ . Association for Computational Linguistics, 2370–2379. 

- [28] Minesh Mathew, Viraj Bagal, Rubèn Pérez Tito, Dimosthenis Karatzas, Ernest Valveny, and C. V Jawahar. 2021. InfographicVQA. arXiv:2104.12756 [cs.CV] 

- [29] Minesh Mathew, Dimosthenis Karatzas, R. Manmatha, and C. V. Jawahar. 2020. DocVQA: A Dataset for VQA on Document Images. _CoRR_ abs/2007.00398 (2020). arXiv:2007.00398 

- [30] Panupong Pasupat and Percy Liang. 2015. Compositional Semantic Parsing on Semi-Structured Tables. In _Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing_ . ACL, 1470–1480. 

- [31] Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016. SQuAD: 100,000+ Questions for Machine Comprehension of Text. In _Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing_ . Association for Computational Linguistics, 2383–2392. 

- [32] Qiu Ran, Yankai Lin, Peng Li, Jie Zhou, and Zhiyuan Liu. 2019. NumNet: Machine Reading Comprehension with Numerical Reasoning. In _EMNLP-IJCNLP_ . 2474– 2484. 

- [33] Elad Segal, Avia Efrat, Mor Shoham, Amir Globerson, and Jonathan Berant. 2020. A Simple and Effective Model for Answering Multi-span Questions. In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ . Association for Computational Linguistics, 3074–3080. 

- [34] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. 2019. Towards VQA Models that can Read. _CoRR_ abs/1904.08920 (2019). arXiv:1904.08920 

- [35] Brandon Smock, Rohith Pesala, and Robin Abraham. 2021. PubTables-1M: Towards comprehensive table extraction from unstructured documents. In _CVPR 2022_ . 

- [36] Nishant Subramani, Alexandre Matton, Malcolm Greaves, and Adrian Lam. 2020. A survey of deep learning approaches for ocr and document understanding. _arXiv preprint arXiv:2011.13534_ (2020). 

- [37] Ryota Tanaka, Kyosuke Nishida, and Sen Yoshida. 2021. VisualMRC: Machine Reading Comprehension on Document Images. In _AAAI_ . 

- [38] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is All you Need. In _Advances in Neural Information Processing Systems_ , I Guyon, U Von Luxburg, S Bengio, H Wallach, R Fergus, S Vishwanathan, and R Garnett (Eds.), Vol. 30. Curran Associates, Inc. 

- [39] Lei Wang, Yan Wang, Deng Cai, Dongxiang Zhang, and Xiaojiang Liu. 2018. Translating a Math Word Problem to a Expression Tree. In _Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing_ . Association for Computational Linguistics, 1064–1069. 

- [40] Yan Wang, Xiaojiang Liu, and Shuming Shi. 2017. Deep Neural Solver for Math Word Problems. In _Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing_ . Association for Computational Linguistics, 845– 854. 

- [41] Zhiruo Wang, Haoyu Dong, Ran Jia, Jia Li, Zhiyi Fu, Shi Han, and Dongmei Zhang. 2021. TUTA: Tree-Based Transformers for Generally Structured Table Pre-Training. In _Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining_ . Association for Computing Machinery, 1780–1790. 

MM ’22, October 10–14, 2022, Lisboa, Portugal 

Zhu, Fengbin, et al. 

- [42] Te-Lin Wu, Cheng Li, Mingyang Zhang, Tao Chen, Spurthi Amba Hombaiah, and Michael Bendersky. 2021. LAMPRET: Layout-Aware Multimodal PreTraining for Document Understanding. _CoRR_ abs/2104.08405 (2021). arXiv:2104.08405 

- [43] Zhipeng Xie and Shichao Sun. 2019. A Goal-Driven Tree-Structured Neural Model for Math Word Problems.. In _IJCAI_ . 5299–5305. 

- [44] Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei, and Ming Zhou. 2020. LayoutLM: Pre-Training of Text and Layout for Document Image Understanding. In _Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining_ . Association for Computing Machinery, 1192–1200. 

- [45] Yang Xu, Yiheng Xu, Tengchao Lv, Lei Cui, Furu Wei, Guoxin Wang, Yijuan Lu, Dinei Florencio, Cha Zhang, Wanxiang Che, Min Zhang, and Lidong Zhou. 2021. LayoutLMv2: Multi-modal Pre-training for Visually-rich Document Understanding. In _Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)_ . Association for Computational Linguistics, 2579–2591. 

- [46] Pengcheng Yin, Graham Neubig, Wen-tau Yih, and Sebastian Riedel. 2020. TaBERT: Pretraining for Joint Understanding of Textual and Tabular Data. In _ACL_ . ACL, 8413–8426. 

- [47] Jipeng Zhang, Lei Wang, Roy Ka-Wei Lee, Yi Bin, Yan Wang, Jie Shao, and EePeng Lim. 2020. Graph-to-Tree Learning for Solving Math Word Problems. In _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics_ . Association for Computational Linguistics, 3928–3937. 

- [48] Xu Zhong, Elaheh ShafieiBavani, and Antonio Jimeno Yepes. 2020. Image-Based Table Recognition: Data, Model, and Evaluation. In _Computer Vision – ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XXI_ . Springer-Verlag, 564–580. https://doi.org/10.1007/978-3-030-58589-1_34 

- [49] Xu Zhong, Jianbin Tang, and Antonio Jimeno Yepes. 2019. PubLayNet: largest dataset ever for document layout analysis. In _2019 International Conference on Document Analysis and Recognition (ICDAR)_ . IEEE, 1015–1022. 

- [50] Fengbin Zhu, Wenqiang Lei, Youcheng Huang, Chao Wang, Shuo Zhang, Jiancheng Lv, Fuli Feng, and Tat-Seng Chua. 2021. TAT-QA: A Question Answering Benchmark on a Hybrid of Tabular and Textual Content in Finance. In _Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)_ . Association for Computational Linguistics, 3277–3287. 

- [51] Fengbin Zhu, Wenqiang Lei, Chao Wang, Jianming Zheng, Soujanya Poria, and Tat-Seng Chua. 2021. Retrieving and Reading: A Comprehensive Survey on Open-domain Question Answering. _CoRR_ abs/2101.00774 (2021). 

