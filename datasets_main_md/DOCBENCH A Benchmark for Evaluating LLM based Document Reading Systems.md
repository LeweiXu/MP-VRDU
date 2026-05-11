# **DOCBENCH: A Benchmark for Evaluating LLM-based Document Reading Systems** 

**Anni Zou**[1] _[,]_[2] _[∗]_ **, Wenhao Yu**[2][�] **, Hongming Zhang**[2] **, Kaixin Ma**[2] **, Deng Cai**[2] **, Zhuosheng Zhang**[1] **, Hai Zhao**[1] **, Dong Yu**[2] 

1Shanghai Jiao Tong University 2Tencent AI Lab `anni0103zou@gmail.com` , 

> � `wenhaowyu@global.tencent.com` (corresponding author) 

## **Abstract** 

Recently, there has been a growing interest among large language model (LLM) developers in LLM-based document reading systems, which enable users to upload their own documents and pose questions related to the document contents, going beyond simple reading comprehension tasks. Consequently, these systems have been carefully designed to tackle challenges such as file parsing, metadata extraction, multi-modal information understanding and long-context reading. However, no current benchmark exists to evaluate their performance in such scenarios, where a raw file and questions are provided as input, and a corresponding response is expected as output. In this paper, we introduce DOCBENCH, a new benchmark designed to evaluate LLM-based document reading systems. Our benchmark involves a meticulously crafted process, including the recruitment of human annotators and the generation of synthetic questions. It includes 229 real documents and 1,102 questions, spanning across five different domains and four major types of questions. We evaluate both proprietary LLM-based systems accessible via web interfaces or APIs, and a parse-then-read pipeline employing open-source LLMs. Our evaluations reveal noticeable gaps between existing LLM-based document reading systems and human performance, underscoring the challenges of developing proficient systems. To summarize, DOCBENCH aims to establish a standardized benchmark for evaluating LLM-based document reading systems under diverse real-world scenarios, thereby guiding future advancements in this research area.[2] 

## **1 Introduction** 

The emergence of large language models (LLMs) has marked a significant milestone in the field of natural language processing, revolutionizing the way we approach a variety of tasks [2, 3, 7, 35, 37, 40, 50]. Existing LLMs such as GPT-4 [2], Llama-3 [37], and Claude-3 [3] have shown exceptional abilities in following human instructions to perform tasks such as answering questions, translating languages and summarizing texts. These tasks are typically characterized by straightforward inputoutput interactions, where the models generate responses solely based on the provided text. However, many real-world applications require more complex interactions involving user-provided documents. For instance, financial analysts might need to query comprehensive financial reports to inform their investment decisions [25, 42, 45]. Legal professionals often search through extensive legal documents to find relevant case law [8, 10, 22]. Similarly, scientific researchers frequently sift through academic papers to identify related works and extract key findings [5, 11]. 

> _∗_ This work was done during internship at Tencent AI Lab, Seattle. 

> 2 Data and code will be released at `https://github.com/Anni-Zou/DocBench` . 

Preprint. Under review. 

**==> picture [227 x 143] intentionally omitted <==**

**----- Start of picture text -----**<br>
ChatGPT840 Dense Passage Retrieval for Open-.Z User: User:  Upload the PDF file of DPR paperWho is most cited person in the paper ta@:<br>[Some file loading process is omitted … ] us with thethere. referencesdocument,the informationandsection howsion we manyneed.andand analyze timesanalyze  areit forit  theyor citationett cited?counts.te Th This | Black-box document reading systems<br>references section and count the citations for each author. 6-)<br>Step 1: parse the uploaded document<br>Step 2: extract the reference section<br>= =text[start_references:]text. find( )if start_references != ~ Step 3: extract author names from references<br>Step 4: count the occurrences of each person<br>(:2 0 0] Step 5: respond with the most cited person<br>Asai, Kazuma Hashimoto, Hannaneh Haj ishirzi,\nRi<br> section the citations for has been each successfully author to extracted, identify theNow, most| will frequently parse thiscited | (ahs72) System:  ws____ic______. Ming-Wei Chang, with 4 citations<br>**----- End of picture text -----**<br>


Figure 1: An example of OpenAI’s GPT-4 based document reading system. Unlike standalone LLMs, recent proprietary LLM-based document reading systems employ a carefully designed approach (e.g., file parsing, code execution) to answer user questions related to document contents. 

When users pose queries based on their provided documents, the situation becomes more intricate and challenging [23]. Unlike standalone LLMs that are primarily trained to process and respond to textual inputs (or images in the case of Vision LLMs), handling user-provided documents necessitates a more sophisticated approach that stretches beyond the capabilities of a single LLM. In order to provide accurate responses, an LLM-based document reading system should not only comprehend natural language queries, but also excel in a range of processing skills, including parsing and interpreting user documents and layouts, navigating complex formatting structures, extracting relevant metadata, and managing long textual contexts along with any embedded images. Mastery of these diverse skills is essential for generating precise and contextually relevant responses. 

At the same time, recent advancements in proprietary LLM developers such as OpenAI and Anthropic have provoked the release of several LLM-based document reading systems. Figure 1 illustrates an example of OpenAI’s GPT-4-based document reading system. Despite widespread claims of effectiveness and efficiency in various online public blogs[34] , **the absence of a standardized benchmark** makes it difficult to objectively evaluate and compare the document reading performance across these systems, thereby leaving a critical gap in fairly assessing these capabilities in a fine-grained manner. 

To fill this gap, our paper introduces DOCBENCH, a novel benchmark specifically designed to evaluate LLM-based document reading systems. DOCBENCH is developed to mirror real-world scenarios where each input consists of a document paired with one or multiple associated questions, and each question is annotated with a golden answer. Our benchmark undergoes a meticulous development process, incorporating human annotation and synthetic question generation. To the end, DOCBENCH features 229 real-world documents and 1,102 questions spanning 5 diverse domains: _Academia, Finance, Government, Laws, and News_ . Besides, the benchmark involves 4 question categories, including _text-only, multi-modal (i.e., tables and figures), meta-data, and unanswerable_ , ensuring comprehensive coverage of various document reading capabilities. 

Based upon DOCBENCH, we evaluate several proprietary LLM-based systems that are accessible via web interfaces or APIs. However, these proprietary systems are close-sourced, thus leading to the limited disclosure of their detailed operational strategies. As a result, we additionally assess a straightforward parse-then-read pipeline employing a series of open-source LLMs. Our evaluations reveal noticeable gaps between existing LLM-based document reading systems and human performance, underscoring the challenges of developing proficient systems. 

In summary, DOCBENCH serves as the first standardized benchmark to evaluate LLM-based document reading systems within real-world scenarios, where the systems take a document file paired with one or multiple related questions as input and generate textual responses as output. Moreover, our benchmark is carefully designed to encompass 5 diverse domains and 4 distinct question types, 

> 3Blog: Claude can now use tools `https://www.anthropic.com/news/tool-use-ga` 

> 4Blog: Talk with documents using LlamaIndex `https://codemaker2016.medium.com/ talk-with-documents-using-llamaindex-3952c76bd511` 

2 

**==> picture [375 x 174] intentionally omitted <==**

**----- Start of picture text -----**<br>
Q: What was the total non-operating<br>o= Multi-modal Info Instruction Prompts income for Amazon in 2021?  a)<br>¢a@aaqad Online Resources m» 8os n ba. Text-only A: $13,272 million. [Evidence]<br>Multimodal Q: Is SenseBERT a model mentioned<br>Based on the above figure and  in the provided text?<br>text, please design three QA pairs...These questions require locating  A: Yes. [Evidence]<br>Government Page Text:We introduce a new  the specific information, simple orcomplex calculations, comparisons, finding the maximum or minimum...<br>Finance Laws language model that...<br>A. Ld €-) Auto Filtering<br>rs ‘ ES<br>ey — 20 Manual Review<br>, (sioY &) pos, 7) 82 Expert Curation<br>Q: What is the average sales...<br>Academia i Multiple News 4 bane Large A: $10,537 million. [Evidence] y<br>» : Domains _ : & Language 9 Q: According to Figure 2, what is ... Q: What was the total non-operating  ,<br>ea <$o~ H: Models: A: Yes. [Evidence] income for Amazon in 2021?  ce<br>A: $13,272 million. [Evidence]<br>Q: On which page does the report<br>| ' 925 Human A: Page 5. Q: Is SenseBERT a model mentioned<br>= : <> Annotators e in the provided text?<br>Q: What does BERT...A: Not mentioned. A: Yes. [Evidence]<br>(a) Document Collection (b) QA-pair Generation (c) Quality Check<br><Text-only><br><Multimodal><br><Meta-data><br><Unanswerable><br>**----- End of picture text -----**<br>


Figure 2: Construction pipeline of DOCBENCH. (a) Document Collection: gathering `PDF` files from five different domains; (b) QA-pair Generation: creating diverse and comprehensive QA pairs through a combination of LLMs and human effort; (c) Quality Check: ensuring data quality through a multi-step process that includes auto filtering, manual review, and expert curation. 

ensuring a nuanced and thorough assessment. By facilitating fair comparisons across different systems, DOCBENCH highlights current limitations and paves the way for future advancements. 

## **2 The DOCBENCH** 

DOCBENCH is a benchmark that takes raw `PDF` files and accompanying questions as inputs, with the objective of generating corresponding textual answers. In this section, we will introduce the pipeline used to construct the dataset, present detailed statistics, and explain the evaluation method. 

## **2.1 Dataset Construction** 

Our dataset construction pipeline consists of three phases. First, we crawl documents across various domains from publicly accessible online resources ( _§_ 2.1.1). Second, we generate corresponding QA pairs with the help of GPT-4 and a team of human annotators ( _§_ 2.1.2). Finally, we conduct auto filtering followed by a manual review to validate the quality of the generated instances ( _§_ 2.1.3). 

## **2.1.1 Document Collection** 

To establish a practical and constructive benchmark for document reading, we concentrate on scenarios where it is crucial to read documents. We standardize the documents to `PDF` format due to its high compatibility and stability. We identify five domains where documents are frequently utilized: _Academia_ , _Finance_ , _Government_ , _Laws_ , _News_ . For Academia, papers are downloaded from arXiv within the range of top- _k_ citations in the field of natural language processing on Google Scholar.[5] For _Finance_ , we crawl the annual reports of companies with top- _k_ global market capitalization up to `2024-02-23` from AnnualReports.[6] For _Government_ , we manually download official governmental reports in 2023 from the U.S. Department of State and GovInfo.[7] For _Laws_ , files are gathered from an official online collection of publications from the Library of Congress, within the years ranging from 2020 to 2024.[8] For _News_ , we collect front-page scanned documents of the New York Times, covering dates from `2022-02-22` to `2024-02-22` .[9] We set _k_ = 100 in the initial crawling process for academic and financial documents. After skipping the unobtainable or damaged documents, we eventually obtained 229 `PDF` files, with 49 for academia, 40 for finance, 44 for government, 46 for laws, and 50 for news. Detailed statistics are shown in Table 1. 

> 5 `https://scholar.google.com/` ; `https://arxiv.org/` . 

> 6 `https://companiesmarketcap.com` ; `http://www.annualreports.com` . 

> 7 `https://www.state.gov/department-reports/` ; `https://www.govinfo.gov/` . 

> 8 `https://www.loc.gov/collections/publications-of-the-law-library-of-congress` . 

> 9 `https://static01.nyt.com/images/` . 

3 

Table 1: Overview statistics of DOCBENCH. All documents are in `PDF` format. We extract text content and calculate the corresponding _#Tokens_ of documents. 

**==> picture [396 x 199] intentionally omitted <==**

**----- Start of picture text -----**<br>
Questions. Documents.<br>Category<br>#Num #Tokens #Num #Pages #Size(KB) #Tokens<br>Aca. 303 16.8 49 11 847 11,123<br>Fin. 288 16.8 40 192 6,594 149,409<br>Gov. 148 14.1 44 69 2,183 36,105<br>Laws 191 15.4 46 58 969 32,339<br>News 172 13.5 50 1 3,095 2,909<br>Total/Avg. 1,102 15.7 229 66 2,738 46,377<br>Distribution of Question Token Counts. Distribution of QA pairs per Document. Distribution of Document Token Counts.<br>60<br>50<br>200 50<br>40<br>150 40<br>30<br>30<br>100<br>20 20<br>50 10 10<br>0 5 10 15 20 25 30 35 40 0 2 4 6 8 10 12 14 16 0 0 10 20 30 >40<br>#Tokens #QA pairs #Tokens(k)<br>#Questions #Documents #Documents<br>**----- End of picture text -----**<br>


Figure 3: Overview of Questions and Documents: distribution of question token counts (left); distribution of QA pairs per document (middle); distribution of document token counts (right). 

## **2.1.2 QA-pair Generation** 

The generation procedure revolves around two aspects: diversity and comprehensiveness. On one hand, as the document itself inherently abounds with multi-dimensional and multi-modal information including texts, tables, figures, and meta-data, we leverage the `fitz` library[10] to parse out the distinct modalities within the `PDF` files. Afterward, we deliver plain texts to GPT-4 ( `gpt-4-0125-preview` ) for generating _text-only_ QA pairs and resort to GPT-4V ( `gpt-4-1106-vision-preview` ) for yielding multi-modal ones based on tables, figures, and their related textual descriptions. On the other hand, we further request a set of human annotators to manually elaborate 350 QA pairs based on the given document files. Their primary task is to focus on types that are rarely covered in the previous generation stage but are frequent in daily usage, such as meta-data and unanswerable instances. Details and additional analysis of instruction prompts are attached in Appendix A. 

## **2.1.3 Quality Check** 

We begin by instructing GPT-4 to automatically filter out questions that are excessively lengthy, unnatural, or impractical. We then conduct a manual review following the automatic filtering to ensure both the quality of questions and the accuracy of answers. To further align our data with real-world user scenarios, we engage 7 practitioners from distinct domains to review and refine the data within their areas of expertise. In this way, our data quality is validated from multiple perspectives. 

## **2.2 Dataset Statistics** 

DOCBENCH comprises a total of 229 `PDF` documents sourced from publicly accessible online repositories along with 1,102 questions, spanning across 5 domains: Academia, Finance, Government, Law, and News. As shown in Table 1, we conduct comprehensive statistical analysis across various angles, encompassing the number of questions, documents, and average token counts within each. Given the unique nature of our task input, which involves processing `PDF` files, we additionally include information such as page count and file size. Moreover, Figure 3 presents distributions depicting the counts of question tokens, document tokens[11] , and QA pairs per document. Notably, we constrain the number of QA pairs per document to a maximum of 20, with its range spanning from 1 to 16, aiming to better emulate real-world usage scenarios. As for the token counts of questions and documents, the minimum and maximum values are (6 _||_ 40) and (1 _,_ 300 _||_ 598 _,_ 302) respectively. 

> 10 `https://pypi.org/project/fitz/` 

> 11We utilize the tokenizer of `gpt-4-turbo` for token measurement. 

4 

Table 2: Examples of instances from DOCBENCH, with multiple labels indicating our data diversity. 

**==> picture [389 x 331] intentionally omitted <==**

**----- Start of picture text -----**<br>
||||||||||||||||
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Question|Answer|Labels|Document|
|Why|does the model not per-|Due to its|complex mor-|<Aca.><Why>|When and Why are Pre-trained Word|
|form as well in German com-|phology and compound|<Text-only>|Embeddings|Useful|for|Machine|
|pared to Spanish and Dutch?|words|...|<Textual>|Translation [clickable file link]|
|By|how much|did the num-|The number increased by|<Fin.><How>|Bank|of|America|Annual|Report|
|ber|of|Erica|users|increase|5.5 million|...|<Multimodal>|2020 [clickable file link]|
|from 2018 to 2019?|<Numerical>|
|What|is the primary focus of|The report|does not con-|<Gov.>|<Wh->|Governmental|report|from|Secre-|
|Bureau Objective 3.4?|tain|such objective.|<Unanswerable>|tary’s|Office|of|Global|Women’s|Is-|
|<Others>|sues|2022 [clickable file link]|
|How|many|times|does|the|The report mentions "sci-|<Laws><How>|Report|on|Regulation|of|Stem|Cell|
|report|mention|"scientific|entific ethics"|11|times.|<Meta-data>|Research|from Library of Congress|
|ethics"?|<Numerical>|2023 [clickable file link]|
|Is|the article about Hurricane|Yes|,|the|article|is|about|<News><Y/N>|New|York|Times|front|page|on|
|Ian’s impact in Florida writ-|Hurrican Ian’s impace in|<Meta-data>|2022-09-30|[clickable file link]|
|ten by multiple authors?|Florida...|<Boolean>|
|Domain|27.5%|26.1%|13.4%|17.3%|15.6%|agit:|
|Aca.|Fin.|Gov.|Laws|~|News|8%|
|QA-pair|37.4%|27.9%|23.4%|11.3%|ss|wi58.6%|
|Text-onli“|Multimodal|Meta-data|Una.|=oa|Total|~~|
|noire|ge|
|Question|%|
|What|/|/|58.6%|/|/|22.1%|/|18.8% 0.5%|wee|How...|Ra|
|Who Where When|Which|YN|How|Why|%|“oy,ZFog.1%&|oh3aN|
|Answer|37.4%|35.7%|17.3%|9.6%|g|Bs|
|Numerical|Textual|Boolean|Others|
|ey|as|.|.|.|we|(|b|)|Detailed|data|distribution|
|(|a|)|Data|distribution|based on|different classification|criteria.|based on question types|

**----- End of picture text -----**<br>


Figure 4: Data distribution of DOCBENCH: (a) proportion(%) of various data groups based on four distinct classification criteria; (b) detailed data analysis based on question types. 

## **2.3 Dataset Analysis** 

Figure 4 illustrates the data distribution in DOCBENCH based on different classification criteria. 

**QA-pair Type** The types of QA pairs can be mainly divided into four groups: _text-only_ (37.4%), _multimodal_ (27.9%), _meta-data_ (23.4%), and _unanswerable_ (11.3%). The _text-only_ and _multimodal_ types collectively account for over half (65.3%), centering on the abilities to comprehend long contexts and interpret information from different modalities. Besides, we incorporate approximately one-third (34.7%) of questions to more closely fit the actual scenarios as well as assess the robustness of the document reading systems, including 23.4% inquiring about metadata (e.g., page numbers, word counts) and 11.3% that cannot be answered based on the given document. 

**Question Type** The types of questions can be primarily separated into four categories according to the inquiry focus: _what / who / where / when / which_ (58.6%), _Y/N_ (22.1%), _how_ (18.8%), and _why_ (0.5%). These categories respectively demand specific information or details, straightforward _yes_ or _no_ responses, methods or degrees, and the underlying reasons behind actions or phenomena. Figure 4(b) delineates a detailed data distribution based on question types. The interrogative _what_ holds a dominant proportion at 40.8%, which is reasonable as users commonly seek precise information when confronted with a document. 

**Answer Type** The types of answers can be partitioned into four classes: _numerical_ (37.4%), _textual_ (35.7%), _boolean_ (17.3%), and _others_ (9.6%). Within the _numerical_ class, 69% originate from the domains of _academia_ and _finance_ , as these documents naturally require extensive use of numbers to convey information, such as performance metrics in academic papers and figures in financial reports. 

5 

Table 3: The GPT-4 automatic evaluator shows a 98% agreement with human annotators. We randomly sample 40 questions and answers from five systems, asking human annotators to assess their accuracy. We then employ string matching (StrMatch), GPT-3.5, and GPT-4 as automatic evaluators. Finally, we measure the agreement between the human and these automatic evaluators. 

|**Sources**|**# Correct / Wrong by different evaluators**|**Agreement (human and automatic evaluators)**|
|---|---|---|
||**Human**<br>**GPT-4**<br>**GPT-3.5**<br>**StrMatch**|**GPT-4**<br>**GPT-3.5**<br>**StrMatch**|
|KimiChat<br>Qwen-2.5<br>Gemma (7B)<br>Mixtral (7B)<br>Llama-3 (70B)|24 / 16<br>23 / 17<br>33 / 7<br>0 / 40<br>17 / 23<br>18 / 22<br>31 / 9<br>0 / 40<br>19 / 21<br>18 / 22<br>18 / 22<br>0 / 40<br>14 / 26<br>14 / 26<br>26 / 14<br>0 / 40<br>16 / 24<br>15 / 25<br>28 / 12<br>0 / 40|97.5%<br>75.0%<br>40.0%<br>97.5%<br>57.5%<br>57.5%<br>97.5%<br>75.0%<br>52.5%<br>100.0%<br>65.0%<br>65.0%<br>97.5%<br>62.5%<br>60.0%|
|Total|90 / 110<br>88 / 112<br>136 / 64<br>0 / 200|98.0%<br>67.0%<br>55.0%|



## **2.4 Evaluation Setup** 

**Evaluation Process** Our dataset diversity poses two major evaluation challenges: (i) The evaluation methods vary depending on the answer type. For example, for boolean or numerical answers, a fair evaluator only needs to verify the correctness of a binary _yes/no_ response or a specific number using simple techniques like string matching or number extraction. In contrast, textual responses require more nuanced standards such as natural language generation (NLG) metrics. Thus, accurately determining the appropriate evaluation method becomes complex when the answer type is unknown. (ii) Different LLMs and systems exhibit substantial variations in the organization and style of their outputs, potentially leading to biases in traditional evaluation approaches. Therefore, we capitalize on the prowess of LLMs that have proven to be decent evaluators and can be easily adapted to the assessment of various answer types [14, 24, 39]. Inspired by Liu et al. [24], we clearly define the evaluation criteria for various types within the instruction prompt and then instruct GPT-4 to assign a score of 0 (incorrect) or 1 (correct). After evaluating 200 examples by both human evaluators and GPT-4, we found that the GPT-4 automatic evaluator shows a 98% agreement with human annotators, significantly exceeding the traditional string matching approach. Details of this experiment is shown in Table 3, and details of evaluation instruction prompts are attached in Appendix A. 

**Metrics** As mentioned above, we instruct GPT-4 to assign a score of 0 (incorrect) or 1 (correct), thus using Accuracy (abbreviated as Acc _._ ) to measure system performance. We report accuracy across all instances, as well as for each domain and QA-pair type in Table 4. 

## **3 Experiments and Analysis** 

## **3.1 Experimental Setup** 

We conduct a comprehensive evaluation of 22 LLM-based document reading systems, encompassing both proprietary systems that support document uploads and a series of _parse-then-read_ pipelines. For _parse-then-read_ pipelines, we leverage the `fitz` package to extract text and image blocks from `PDF` files. We retain the original texts and line breaks for text chunks while we denote the _i_ -th image as _[image i]_ for images. Our selection for the proprietary systems includes GPT-4 and GPT-4o [2] from OpenAI, GLM-4[12] from ZhipuAI, Kimi[13] from Moonshot AI, Claude-3[14] from Anthropic, Qwen2.5[15] from Alibaba Cloud, and ERNIE-3.5[16] from Baidu. In the case of the _parse-then-read_ pipelines, we assess 15 prominent LLMs as base models, featuring those from the GPT [2, 31], Llama [37], Mistral [17], Yi [48], InternLM [6], Phi-3 [1], Gemma [36], ChatGLM3 [12], and Command-R [9] families. The selection of base open-sourced LLMs adheres to three guiding principles: (i) official release with _instruct_ or _chat_ versions that are supported by vLLM [20] framework; (ii) model sizes ranging from 7B to 70B to accommodate GPU memory constraints; (iii) availability of the longest context length and the latest version. 

> 12 `https://chatglm.cn/main/doc` 

> 13 `https://kimi.moonshot.cn` 

> 14 `https://claude.ai/chats` 

> 15 `https://tongyi.aliyun.com/qianwen` 

> 16 `https://yiyan.baidu.com` 

6 

Table 4: Results on DOCBENCH across various types and domains. _Ver./Size_ stands for the model version or size; _File_ denotes the maximum uploaded file size; _Cxt._ refers to model’s context length. 

|**Methods**<br>**Form**<br>**Ver.**<br>**/Size**<br>**File**<br>**/Cxt.**|**Domain**<br>**Aca. Fin. Gov. Laws News **|**Type**<br>**Overall**Acc_._<br> **Text. Multi. Meta. Una.**|**Type**<br>**Overall**Acc_._<br> **Text. Multi. Meta. Una.**|
|---|---|---|---|
|Human<br>-<br>-<br>-|83.0 82.2 77.8<br>75.0<br>86.4|81.4<br>83.3<br>77.5<br>82.2|81.2|
||_LLM-based systems_|||
|||||
|GPT-4<br>_API_<br>`0409 100M`<br>GPT-4o<br>_API_<br>`0513 100M`<br>GLM-4<br>_Web_<br>-<br>`20M`<br>KimiChat<br>_Web_<br>-<br>`100M`<br>Claude-3<br>_Web_<br>`Opus`<br>`10M`<br>Qwen-2.5<br>_Web_<br>-<br>`150M`<br>ERNIE-3.5<br>_Web_<br>-<br>`10M`|65.7<br>**65.3** 75.7<br>69.6<br>79.6<br>56.4 56.3 73.0<br>65.5<br>75.0<br>55.8 35.4 61.5<br>62.8<br>82.0<br>62.4 61.8<br>**77.0**<br>78.5<br>**87.2**<br>**73.9** 40.6 70.3<br>**79.1**<br>86.6<br>42.9 29.9 51.4<br>55.5<br>69.2<br>56.4 37.5 54.7<br>58.1<br>58.1|**87.9**<br>**74.7**<br>50.8<br>37.1<br>69.8<br>85.0<br>62.7<br>50.4<br>17.7<br>63.1<br>73.1<br>50.3<br>48.8<br>33.1<br>56.5<br>87.6<br>65.3<br>50.4<br>**71.8**<br>**70.9**<br>80.8<br>64.6<br>**54.3**<br>58.9<br>67.6<br>61.7<br>31.8<br>36.0<br>58.1<br>46.9<br>63.6<br>47.7<br>36.8<br>54.0<br>51.8||
||_Parse-then-Read Pipelines_|||
|||||
|GPT-4<br>_API_<br>`0409 128k`<br>GPT-3.5<br>_API_<br>`0125`<br>`16k`<br>ChatGLM3<br>_Open_<br>`6B`<br>`128k`<br>Gemma<br>_Open_<br>`7B`<br>`8k`<br>Mixtral<br>_Open_<br>`7B`<br>`32k`<br>InternLM2<br>_Open_<br>`7B`<br>`32k`<br>Llama-3<br>_Open_<br>`8B`<br>`8k`<br>Yi-1.5<br>_Open_<br>`9B`<br>`16k`<br>Llama-2<br>_Open_<br>`13B`<br>`4k`<br>Phi-3<br>_Open_<br>`14B`<br>`128k`<br>InternLM2<br>_Open_<br>`20B`<br>`32k`<br>Yi-1.5<br>_Open_<br>`34B`<br>`16k`<br>Command-R<br>_Open_<br>`35B`<br>`128k`<br>Mixtral-8x7B _Open_<br>`47B`<br>`32k`<br>Llama-3<br>_Open_<br>`70B`<br>`8k`|**70.0 47.9 68.9**<br>**70.7**<br>**93.6**<br>**79.1**<br>**63.3**<br>**54.3**<br>70.2<br>49.8 24.0 58.8<br>50.3<br>83.7<br>65.0<br>37.0<br>42.6<br>44.4<br>34.7 41.7 58.1<br>51.3<br>58.1<br>70.4<br>40.3<br>31.0<br>12.1<br>34.3 12.5 43.2<br>34.0<br>65.1<br>43.0<br>17.2<br>21.3<br>**77.4**<br>42.6 29.2 58.8<br>50.3<br>82.0<br>71.8<br>33.8<br>38.4<br>30.6<br>38.6 27.1 52.0<br>46.1<br>65.7<br>63.3<br>28.9<br>35.3<br>25.8<br>44.6 23.6 61.5<br>54.5<br>86.6<br>68.0<br>29.2<br>45.0<br>49.2<br>40.6 26.4 58.1<br>52.4<br>83.1<br>66.0<br>33.8<br>45.7<br>27.4<br>20.8 18.4 29.7<br>23.6<br>55.2<br>43.4<br>15.9<br>21.7<br>12.9<br>50.2 44.4<br>65.5<br>64.4<br>76.7<br>77.4<br>45.8<br>45.3<br>44.4<br>43.2 28.5 59.5<br>54.5<br>80.8<br>73.3<br>33.4<br>43.0<br>22.6<br>47.2 27.1 59.5<br>56.5<br>78.5<br>68.2<br>39.0<br>49.2<br>19.4<br>49.5 38.9 66.2<br>64.4<br>80.8<br>78.4<br>50.0<br>49.6<br>13.7<br>48.5 31.9 60.1<br>59.2<br>81.4<br>76.0<br>42.9<br>46.9<br>12.1<br>52.1<br>25.3 68.2<br>59.2<br>90.7<br>69.2<br>38.6<br>49.2<br>56.5||**67.9**<br>49.6<br>46.2<br>34.6<br>48.7<br>42.9<br>49.6<br>47.9<br>27.2<br>57.4<br>49.4<br>50.1<br>56.4<br>52.7<br>54.5|



## **3.2 Results and Discussion** 

Table 4 showcases the performance of various document reading systems on DOCBENCH. Our findings reveal substantial variations in document reading capabilities among these systems, driven by differences in their foundational models, context length limitations, diverse design and implementation approaches, and etc. In this section, we will provide further discussions to delve deeper into the pros and cons of existing systems, as well as uncover the core challenges posed by DOCBENCH. 

## **3.2.1 Interpreting Multi-modal and Metadata Information** 

Figure 5 presents a case study illustrating the unique challenge of answering multi-modal questions in DOCBENCH. We observe that **leading proprietary LLM-based systems often fail due to errors in one of the steps in the** _**Location** →_ _**Extraction** →_ _**Calculation**_ **sequence.** Take the first case study as an example, in the first step, KimiChat fails to locate the relevant chart on page 17. In the extraction phase, Claude-3 misidentifies the data as _288 & 348_ , instead of the correct _326 & 390_ . Finally, while GPT-4 locates and extracts the correct information, it errs in calculating the percentage change, demonstrating the complexity of these questions. Interestingly, parse-then-read pipelines can achieve reasonable performance on multi-modal questions (e.g., 63.3% for GPT-4). This is likely because the 

7 

|Question:  What is the percentage change in the number of offshore stores for the company from FY18 to FY19?|Question:  What is the percentage change in the number of offshore stores for the company from FY18 to FY19?|Question:  What is the percentage change in the number of offshore stores for the company from FY18 to FY19?|Question:  What is the percentage change in the number of offshore stores for the company from FY18 to FY19?|Question:  What is the percentage change in the number of offshore stores for the company from FY18 to FY19?<br>The percentage change in the number of offshore stores from|Question:  What is the percentage change in the number of offshore stores for the company from FY18 to FY19?<br>The percentage change in the number of offshore stores from|
|---|---|---|---|---|---|
|Location|Extraction||KimiChat|The percentage change in the number of offshore stores from||
|"<br>Ei!<br>®<br>Fl|Calculation<br>MARKETS CONTINUED<br>10 GROW.<br>”,||GPT-4<br>Claude-3|The percentage change in the number of offshore stores from<br>FY18 to FY19 isnot providedin the document.<br>The number of offshore stores increased by21%from 288 in<br>FY18 to 348 in FY19.<br>bd<br>x||
||(390-326)/326 = 19.63%|||FY18 to FY19 is12.8%(326 stores in FY18 to 390 in FY19) .|x|
|Question:      Which region has the highest number of employees of the company and how many?|Question:      Which region has the highest number of employees of the company and how many?|||||
|Location|Extraction||KimiChat|The region with the highest number of employees of the||
|||||company is Asia, with 22,674 employees. This information is|company is Asia, with 22,674 employees. This information is|
|||||provided in the text: "Asia 22,674.",||
||||Claude-3  Head Genentech Research & Early Development (|Claude-3  Head Genentech Research & Early Development (gRED)||
|oe|Calculation|S|GPT-4|Europehas the highest number of employees at Roche,||
|=|max(43181, 22674, 678...)|||with a total of43,181employees.<br>Jf||
|Question:     Which domain has the highest number of dialogues in the test set?|Question:     Which domain has the highest number of dialogues in the test set?|Question:     Which domain has the highest number of dialogues in the test set?||||
|Location|Extraction||KimiChat|The "Taxi" domain has the highest number of dialogues in the|The "Taxi" domain has the highest number of dialogues in the|
|||||test set, with 195 dialogues.||
||||Claude-3|The restaurant domain has the highest number of dialogues<br>(437) in the test set.||
||Calculation<br>max(394,494, 395, 437...)<br>~~—~~||GPT-4|The "Restaurant" domain has the highest number of dialogues<br>in the test set, with 437 dialogues.<br>xK||



Question:      Which region has the highest number of employees of the company and how many? 

Question:     Which domain has the highest number of dialogues in the test set? 

Figure 5: To address multi-modal questions in DOCBENCH, it is essential to: (i) identify the relevant figure/table (Location); (ii) extract specific data (Extraction); (iii) perform necessary calculations (Calculation). In the first case study, KimiChat fails to locate the figure, Claude-3 retrieves incorrect data, and GPT-4, despite succeeding in the first two steps, struggles with the calculation. 

parsing process captures certain table information, and documents often include textual descriptions of figures. Meanwhile, for metadata-related questions, **current methods generally lack attention to global information** , resulting in relative low performances (below 55%). 

## **3.2.2 Handling Lengthy Documents** 

**Handling lengthy documents is demanding, especially in real-world scenarios where document size can be virtually unlimited.** Proprietary LLM-based systems struggle with uploading extensive files, while the parse-thenread pipelines with open-sourced LLMs are constrained by their maximum context length, leading to varying degrees of information loss. As shown in Figure 6, both methods perform poorly in the fnance domain but achieve higher performance in the news domain. This discrepancy arises because financial documents are typically longer and contain richer information, whereas news files are limited to single front pages with 

**==> picture [178 x 104] intentionally omitted <==**

**----- Start of picture text -----**<br>
Aca. Fin. Gov. Laws News<br>80 76.8 77.5<br>66.2 67.1<br>60 59.1 57.9<br>52.8<br>46.7 44.5<br>40<br>29.8<br>20<br>0<br>LLM-based Systems Parse-then-Read Pipelines<br>Average Accuracy (%)<br>**----- End of picture text -----**<br>


Figure 6: Average accuracy (%) of two methods under five different domains. 

fewer messages. Furthermore, certain strong models with relatively short context lengths may excel with smaller files, but context length becomes a crucial factor when it comes to large files. For instance, the _8k_ Llama-3 family performs exceptionally well in the news domain, but is outperformed 

8 

by all the _128k_ models in the fnance domain. Besides, we discover that KimiChat and Command-R, which are specifically enhanced for long-context and Retrieval-Augmented Generation (RAG) capabilities, achieve decent results on _text-only_ questions. **Therefore, a key challenge lies in adapting these systems to handle documents of varying lengths while balancing the foundational model’s capabilities and context length constraints.** 

## **3.2.3 Faithfulness to User-provided Documents** 

**Most existing document reading systems falter when faced with unanswerable questions based on the provided document, exhibiting a lack of fidelity.** Remarkably, Gemma and KimiChat perform better in such scenarios, which represents a crucial capability since users often expect systems to answer questions strictly based on given files. Intriguingly, despite the commonly-shared base model on GPT-4, there is a notable performance gap between the system and the parse-thenread pipeline in handling unanswerable questions (i.e., 37.1% and 70.2 % for system and pipeline, respectively). We analyze that this may be due to: (i) the proprietary LLM-based system have undergone optimizations on the base model, potentially causing overfitting; (ii) GPT-4 tends to adhere more closely to the in-context learning information. **Such phenomenon thus underscores a critical challenge for future document reading systems on enhancing fidelity to the given documents.** 

## **4 Related Works** 

## **4.1 Recent Advances of LLMs and LLM-based Systems** 

The latest generation of LLMs, such as GPT-4 [2], Llama-3 [37] and Claude-3 [3], have significantly extended the capabilities of language models [7, 40, 50]. These models are pre-trained on vast amounts of web-scale data, enabling them to perform a wide range of human-instructed tasks with impressive performance. Despite their remarkable performance, standalone LLMs may not be sufficient for many real-world applications. For example, LLMs lack access to real-time information and may struggle with tasks that require up-to-date knowledge [38]. Moreover, real-world applications often require non-text inputs parsing, code execution, API calling and interaction with external environments [15, 18, 21, 23, 44, 52]. The overall task completion usually requires multiple reasoning, execution and reflection steps that cannot be accomplished in a simple input-output manner [33, 41, 47]. To overcome the limitations of standalone LLMs, recent efforts have incorporated additional components and sophisticated system design. These systems, such as Microsoft’s Co-Pilot[17] and OpenAI’s GPT-4 all-in-one[18] , aim to provide more comprehensive and practical solutions for real-world applications. Other pioneering efforts on designing LLM-based systems include web agents [16, 26, 51], software agents [21, 46] and computer agents [43] that can interact with external resources (e.g., websites, search engine, code repositories or computers) and perform multi-step tasks. The success of these systems relies on integrating powerful LLMs with well-designed architectures and components that enable them to handle complex tasks effectively. 

## **4.2 Document reading: Datasets and Methods** 

Document reading is a critical area where LLM-based systems have demonstrated significant advancements. Proprietary developers such as OpenAI[19] and Anthropic[20] have introduced advanced systems that can take a user-provided document as input, parse its structure, extract relevant metadata, and handle long texts and images to provide accurate responses. While these systems build upon the fundamental capabilities of their underlying LLMs [2–4, 49], they differ in their design and implementation, with some systems excelling in long-context reading and others focusing on retrieval-augmented methods to improve document reading ability. Despite claims of effectiveness and efficiency in online public blogs, the absence of a standardized benchmark makes it difficult to objectively evaluate and compare the document reading performance across these systems. Existing benchmarks relevant to document reading are unable to adequately reflect the real performance of these systems. Datasets focusing on document understanding such as Doc2Dial [13], ConditionalQA [34] and those specifically focusing on long-context reading like NarrativeQA [19] and 

> 17 `https://copilot.microsoft.com` 

> 18 `https://chat.openai.com` 

> 19OpenAI’s ChatGPT: `https://chat.openai.com` 

> 20Anthropic’s Claude: `https://claude.ai/chats` 

9 

QuALITY [32], primarily use text as input only, ignoring the complex nature of document structure and multi-modal information. On the other hand, multi-modal document reading datasets like DocVQA [29], ChartQA [27], OCR-VQA [30], and InfoVQA [28] include multi-modal inputs and preserve the original document structure and layout. However these datasets often capture only parts of document (e.g. tables or figures) and ignored substantial amount of textual content. Different from previous works, DocBench requires systems to process the full documents as intact files and covers different types of questions targeting various abilities, which can more accurately evaluate the capabilities of LLM-based document reading systems in real-world scenarios. 

## **5 Conclusion** 

In this paper, we introduce DOCBENCH, a novel benchmark created to assess LLM-based document reading systems in a comprehensive and fine-grained manner. DOCBENCH consists of 229 documents and 1,102 questions, spanning 5 domains and 4 question types, developed with the help of human annotators and synthetic questions. We evaluate both proprietary LLM systems, accessible via web interfaces or APIs, and a parse-then-read approach using open-source LLMs. Our findings reveal significant disparities in document reading capabilities among these systems, highlighting current limitations, presenting potential challenges, and thus driving forward progress in this research field. 

## **References** 

- [1] Marah Abdin, Sam Ade Jacobs, Ammar Ahmad Awan, Jyoti Aneja, Ahmed Awadallah, Hany Awadalla, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Harkirat Behl, et al. 2024. Phi-3 technical report: A highly capable language model locally on your phone. _arXiv preprint arXiv:2404.14219_ (2024). 

- [2] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. 2023. Gpt-4 technical report. _arXiv preprint arXiv:2303.08774_ (2023). 

- [3] Anthropic. 2024. Claude 3 haiku: our fastest model yet. (2024). `https://www.anthropic. com/news/claude-3-haiku.` 

- [4] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. 2023. Qwen technical report. _arXiv preprint arXiv:2309.16609_ (2023). 

- [5] Abeba Birhane, Atoosa Kasirzadeh, David Leslie, and Sandra Wachter. 2023. Science in the age of large language models. _Nature Reviews Physics_ 5, 5 (2023), 277–280. 

- [6] Zheng Cai, Maosong Cao, Haojiong Chen, Kai Chen, Keyu Chen, Xin Chen, Xun Chen, Zehui Chen, Zhi Chen, Pei Chu, et al. 2024. Internlm2 technical report. _arXiv preprint arXiv:2403.17297_ (2024). 

- [7] Yupeng Chang, Xu Wang, Jindong Wang, Yuan Wu, Linyi Yang, Kaijie Zhu, Hao Chen, Xiaoyuan Yi, Cunxiang Wang, Yidong Wang, et al. 2024. A survey on evaluation of large language models. _ACM Transactions on Intelligent Systems and Technology_ 15, 3 (2024), 1–45. 

- [8] Zhiyu Zoey Chen, Jing Ma, Xinlu Zhang, Nan Hao, An Yan, Armineh Nourbakhsh, Xianjun Yang, Julian McAuley, Linda Petzold, and William Yang Wang. 2024. A Survey on Large Language Models for Critical Societal Domains: Finance, Healthcare, and Law. _arXiv preprint arXiv:2405.01769_ (2024). 

- [9] CohereAI. 2024. Introducing Command R. (2024). `https://docs.cohere.com/docs/ command-r` 

- [10] Jiaxi Cui, Zongjian Li, Yang Yan, Bohua Chen, and Li Yuan. 2023. Chatlaw: Opensource legal large language model with integrated external knowledge bases. _arXiv preprint arXiv:2306.16092_ (2023). 

10 

- [11] Pradeep Dasigi, Kyle Lo, Iz Beltagy, Arman Cohan, Noah A Smith, and Matt Gardner. 2021. A dataset of information-seeking questions and answers anchored in research papers. _arXiv preprint arXiv:2105.03011_ (2021). 

- [12] Zhengxiao Du, Yujie Qian, Xiao Liu, Ming Ding, Jiezhong Qiu, Zhilin Yang, and Jie Tang. 2021. Glm: General language model pretraining with autoregressive blank infilling. _arXiv preprint arXiv:2103.10360_ (2021). 

- [13] Song Feng, Hui Wan, Chulaka Gunasekara, Siva Patel, Sachindra Joshi, and Luis Lastras. 2020. doc2dial: A Goal-Oriented Document-Grounded Dialogue Dataset. In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , Bonnie Webber, Trevor Cohn, Yulan He, and Yang Liu (Eds.). Association for Computational Linguistics, Online, 8118–8128. `https://doi.org/10.18653/v1/2020.emnlp-main.652` 

- [14] Jinlan Fu, See-Kiong Ng, Zhengbao Jiang, and Pengfei Liu. 2023. GPTScore: Evaluate as You Desire. arXiv:2302.04166 [cs.CL] 

- [15] Siyuan Guo, Cheng Deng, Ying Wen, Hechang Chen, Yi Chang, and Jun Wang. 2024. DSAgent: Automated Data Science by Empowering Large Language Models with Case-Based Reasoning. arXiv:2402.17453 [cs.LG] 

- [16] Hongliang He, Wenlin Yao, Kaixin Ma, Wenhao Yu, Yong Dai, Hongming Zhang, Zhenzhong Lan, and Dong Yu. 2024. WebVoyager: Building an End-to-End Web Agent with Large Multimodal Models. _arXiv preprint arXiv:2401.13919_ (2024). 

- [17] Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, et al. 2024. Mixtral of experts. _arXiv preprint arXiv:2401.04088_ (2024). 

- [18] Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. 2023. SWE-bench: Can Language Models Resolve Real-World GitHub Issues? arXiv:2310.06770 [cs.CL] 

- [19] Tomáš Koˇcisky, Jonathan Schwarz, Phil Blunsom, Chris Dyer, Karl Moritz Hermann, Gábor` Melis, and Edward Grefenstette. 2018. The narrativeqa reading comprehension challenge. _Transactions of the Association for Computational Linguistics_ 6 (2018), 317–328. 

- [20] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient memory management for large language model serving with pagedattention. In _Proceedings of the 29th Symposium on Operating Systems Principles_ . 611–626. 

- [21] Cognition Labs. 2024. Devin, AI software engineer. (2024). `https://www.cognition.ai/ blog/introducing-devin` 

- [22] Jinqi Lai, Wensheng Gan, Jiayang Wu, Zhenlian Qi, and Philip S Yu. 2023. Large language models in law: A survey. _arXiv preprint arXiv:2312.03718_ (2023). 

- [23] Kuang-Huei Lee, Xinyun Chen, Hiroki Furuta, John Canny, and Ian Fischer. 2024. A Human-Inspired Reading Agent with Gist Memory of Very Long Contexts. _arXiv preprint arXiv:2402.09727_ (2024). 

- [24] Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang Zhu. 2023. G-Eval: NLG Evaluation using Gpt-4 with Better Human Alignment. In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_ , Houda Bouamor, Juan Pino, and Kalika Bali (Eds.). Association for Computational Linguistics, Singapore, 2511–2522. `https://doi.org/10.18653/v1/2023.emnlp-main.153` 

- [25] Zhuang Liu, Degen Huang, Kaiyu Huang, Zhuang Li, and Jun Zhao. 2021. Finbert: A pretrained financial language representation model for financial text mining. In _Proceedings of the twenty-ninth international conference on international joint conferences on artificial intelligence_ . 4513–4519. 

11 

- [26] Kaixin Ma, Hongming Zhang, Hongwei Wang, Xiaoman Pan, Wenhao Yu, and Dong Yu. 2023. LASER: LLM Agent with State-Space Exploration for Web Navigation. arXiv:2309.08172 [cs.CL] 

- [27] Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. 2022. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. _arXiv preprint arXiv:2203.10244_ (2022). 

- [28] Minesh Mathew, Viraj Bagal, Rubèn Tito, Dimosthenis Karatzas, Ernest Valveny, and CV Jawahar. 2022. Infographicvqa. In _Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision_ . 1697–1706. 

- [29] Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. 2021. Docvqa: A dataset for vqa on document images. In _Proceedings of the IEEE/CVF winter conference on applications of computer vision_ . 2200–2209. 

- [30] Anand Mishra, Shashank Shekhar, Ajeet Kumar Singh, and Anirban Chakraborty. 2019. Ocrvqa: Visual question answering by reading text in images. In _2019 international conference on document analysis and recognition (ICDAR)_ . IEEE, 947–952. 

- [31] OpenAI. 2022. Introducing chatgpt. (2022). `https://openai.com/blog/chatgpt.` 

- [32] Richard Yuanzhe Pang, Alicia Parrish, Nitish Joshi, Nikita Nangia, Jason Phang, Angelica Chen, Vishakh Padmakumar, Johnny Ma, Jana Thompson, He He, et al. 2022. QuALITY: Question Answering with Long Input Texts, Yes!. In _Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies_ . 5336–5358. 

- [33] Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. 2023. Reflexion: Language Agents with Verbal Reinforcement Learning. arXiv:2303.11366 [cs.AI] 

- [34] Haitian Sun, William Cohen, and Ruslan Salakhutdinov. 2022. ConditionalQA: A Complex Reading Comprehension Dataset with Conditional Answers. In _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , Smaranda Muresan, Preslav Nakov, and Aline Villavicencio (Eds.). Association for Computational Linguistics, Dublin, Ireland, 3627–3637. `https://doi.org/10.18653/v1/2022.acl-long.253` 

- [35] Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. 2023. Gemini: a family of highly capable multimodal models. _arXiv preprint arXiv:2312.11805_ (2023). 

- [36] Gemma Team, Thomas Mesnard, Cassidy Hardin, Robert Dadashi, Surya Bhupatiraju, Shreya Pathak, Laurent Sifre, Morgane Rivière, Mihir Sanjay Kale, Juliette Love, et al. 2024. Gemma: Open models based on gemini research and technology. _arXiv preprint arXiv:2403.08295_ (2024). 

- [37] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. _arXiv preprint arXiv:2307.09288_ (2023). 

- [38] Tu Vu, Mohit Iyyer, Xuezhi Wang, Noah Constant, Jerry Wei, Jason Wei, Chris Tar, Yun-Hsuan Sung, Denny Zhou, Quoc Le, et al. 2023. Freshllms: Refreshing large language models with search engine augmentation. _arXiv preprint arXiv:2310.03214_ (2023). 

- [39] Jiaan Wang, Yunlong Liang, Fandong Meng, Zengkui Sun, Haoxiang Shi, Zhixu Li, Jinan Xu, Jianfeng Qu, and Jie Zhou. 2023. Is ChatGPT a Good NLG Evaluator? A Preliminary Study. In _Proceedings of the 4th New Frontiers in Summarization Workshop_ , Yue Dong, Wen Xiao, Lu Wang, Fei Liu, and Giuseppe Carenini (Eds.). Association for Computational Linguistics, Singapore, 1–11. `https://doi.org/10.18653/v1/2023.newsum-1.1` 

- [40] Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, et al. 2024. A survey on large language model based autonomous agents. _Frontiers of Computer Science_ 18, 6 (2024), 1–26. 

12 

- [41] Xingyao Wang, Yangyi Chen, Lifan Yuan, Yizhe Zhang, Yunzhu Li, Hao Peng, and Heng Ji. 2024. Executable Code Actions Elicit Better LLM Agents. In _ICML_ . arXiv:2402.01030 

- [42] Shijie Wu, Ozan Irsoy, Steven Lu, Vadim Dabravolski, Mark Dredze, Sebastian Gehrmann, Prabhanjan Kambadur, David Rosenberg, and Gideon Mann. 2023. Bloomberggpt: A large language model for finance. _arXiv preprint arXiv:2303.17564_ (2023). 

- [43] Zhiyong Wu, Chengcheng Han, Zichen Ding, Zhenmin Weng, Zhoumianze Liu, Shunyu Yao, Tao Yu, and Lingpeng Kong. 2024. OS-Copilot: Towards Generalist Computer Agents with Self-Improvement. arXiv:2402.07456 [cs.AI] 

- [44] Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Toh Jing Hua, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, et al. 2024. OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments. _arXiv preprint arXiv:2404.07972_ (2024). 

- [45] Hongyang Yang, Xiao-Yang Liu, and Christina Dan Wang. 2023. Fingpt: Open-source financial large language models. _arXiv preprint arXiv:2306.06031_ (2023). 

- [46] John Yang, Carlos E Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik Narasimhan, and Ofir Press. 2024. SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering. _arXiv preprint arXiv:2405.15793_ (2024). 

- [47] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and Yuan Cao. 2023. ReAct: Synergizing Reasoning and Acting in Language Models. In _The Eleventh International Conference on Learning Representations_ . 

- [48] Alex Young, Bei Chen, Chao Li, Chengen Huang, Ge Zhang, Guanwei Zhang, Heng Li, Jiangcheng Zhu, Jianqun Chen, Jing Chang, et al. 2024. Yi: Open foundation models by 01. ai. _arXiv preprint arXiv:2403.04652_ (2024). 

- [49] Aohan Zeng, Xiao Liu, Zhengxiao Du, Zihan Wang, Hanyu Lai, Ming Ding, Zhuoyi Yang, Yifan Xu, Wendi Zheng, Xiao Xia, et al. 2022. Glm-130b: An open bilingual pre-trained model. _arXiv preprint arXiv:2210.02414_ (2022). 

- [50] Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, et al. 2023. A survey of large language models. _arXiv preprint arXiv:2303.18223_ (2023). 

- [51] Boyuan Zheng, Boyu Gou, Jihyung Kil, Huan Sun, and Yu Su. 2024. GPT-4V(ision) is a Generalist Web Agent, if Grounded. arXiv:2401.01614 [cs.IR] 

- [52] Shuyan Zhou, Frank F Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Yonatan Bisk, Daniel Fried, Uri Alon, et al. 2023. Webarena: A realistic web environment for building autonomous agents. _arXiv preprint arXiv:2307.13854_ (2023). 

## **A Instruction Prompts** 

## **A.1 Response Evaluation** 

Detailed instruction prompts for response evaluation are shown in Table 5. 

## **A.2 QA-pair Generation** 

Details of instruction prompts for generating QA pairs are attached in Table 6. We discover that simply passing diagrams to GPT-4V leads to subpar question quality. This issue likely stems from the fact that figures or tables without accompanying text descriptions typically lack sufficient information, thus causing the generated QA pairs to deviate from their intended meanings. In addition, we observe that adding difficulty settings for QA generation (e.g., _Easy_ , _Medium_ , _Hard_ ) in the instruction prompt can result in higher quality. We analyze that this may be due to the model being able to favor higher generation quality in potential comparisons. 

13 

Table 5: Instruction Prompts in Response Evaluation. 

```
SystemContent:
```

You are a helpful evaluator. 

## `Prompt:` 

## **Task Overview:** 

You are tasked with evaluating user answers based on a given question, reference answer, and additional reference text. Your goal is to assess the correctness of the user answer using a specific metric. 

## **Evaluation Criteria:** 

1. Yes/No Questions: Verify if the user’s answer aligns with the reference answer in terms of a "yes" or "no" response. 

2. Short Answers/Directives: Ensure key details such as numbers, specific nouns/verbs, and dates match those in the reference answer. 

3. Abstractive/Long Answers: The user’s answer can differ in wording but must convey the same meaning and contain the same key information as the reference answer to be considered correct. 

## **Evaluation Process:** 

1. Identify the type of question presented. 

2. Apply the relevant criteria from the Evaluation Criteria. 

3. Compare the user’s answer against the reference answer accordingly. 

4. Consult the reference text for clarification when needed. 

5. Score the answer with a binary label 0 or 1, where 0 denotes wrong and 1 denotes correct. NOTE that if the user answer is 0 or an empty string, it should get a 0 score. 

**Question:** `{{question}}` **User Answer:** `{{sys_ans}}` **Reference Answer:** `{{ref_ans}}` **Reference Text:** `{{ref_text}}` 

## **Evaluation Form (score ONLY):** 

- Correctness: 

## **B Performance Comparison** 

Figure 7 demonstrates the relative performance of LLM-based systems and parse-then-read pipelines against the best on DOCBENCH. For LLM-based systems, KimiChat consistently scores high across various metrics, demonstrating balanced performance. Notably, GPT-4 performs poorly in the unanswerable category, indicating potential overfitting in optimized GPT-4 file systems, which leads to decreased fidelity to given documents. Additionally, Claude-3 excels in the meta-data category, highlighting its superior ability to comprehend high-level metadata information. For parse-then-read pipelines, we select models with the highest overall accuracy for comparison. Unlike LLM-based systems, GPT-4 demonstrates consistently high and balanced performance across all aspects within this pipeline. Notably, significant discrepancies arise in handling multi-modal and unanswerable questions, where GPT-4 and Gemma exhibit clear distinctions from the remaining methods. 

## **C Analysis of Input Sources** 

Table 7 presents the impact of different input sources on model performance. We provide questions to GPT-4 and GPT-4o, both with and without attached files. Remarkably, even without files, the models correctly answer a portion of the questions (19.1% for GPT-4 and 21.7% for GPT-4o). Our analysis reveals that the correctly answered questions are predominantly textual and are largely associated with government, law, and news domains. This trend suggests that the models’ underlying training 

14 

Table 6: Instruction Prompts in QA-pair Generation. 

```
SystemContent:
```

You are a helpful assistant that can generate question-answer pairs. 

## `Text-only QA:` 

- Based on the above text, please design three question-answer pairs with different levels of 

- difficulty: Easy, Medium, Hard. 

- The questions should be close-ended and should be answered based on the provided text. The answer form should be as diverse as possible, including [Yes/No, Short Answer, Long 

- Answer, Abstractive Answer]. 

You should provide the reference in the text and the answer form if possible. 

The output should be formalized as: ”’Q: | A: | Reference: | Difficulty Level: | Answer Form:”’ 

## `Multimodal QA (w/table+text):` 

Based on the above table and text, please design three question-answer pairs with different levels of difficulty: Easy, Medium, Hard. 

- The text provided is text related to the table, which can provide more reference for question 

- generation, but the focus is still on the table itself. 

- These questions require locating the specific information, simple or complex calculations, 

- comparisons, finding the maximum and minimum, reading across rows and columns, etc. Note that these questions also need to be realistic. You should provide the reason if possible. The output should be formalized as: ”’Q: | A: | Reference: | Difficulty Level: | Answer Form:”’ 

## `Multimodal QA (w/figure+text):` 

- Based on the above figure and text, please design three question-answer pairs with different 

- levels of difficulty: Easy, Medium, Hard. 

- The text provided is text related to the figure, which can provide more reference for question 

- generation, but the focus is still on the figure itself. 

- These questions require a deep reading of the meaning of the image. 

- Note that these questions also need to be realistic. You should provide the reason if possible. The output should be formalized as: ”’Q: | A: | Reason: | Difficulty Level: | ”’ 

## `Multimodal QA (w/table):` 

Based on the above image, please design three question-answer pairs with different levels of difficulty: Easy, Medium, Hard. 

- These questions require locating the specific information, simple or complex calculations, 

- comparisons, finding the maximum and minimum, reading across rows and columns, etc. Note that these questions also need to be realistic. You should provide the reason if possible. The output should be formalized as: ”’Q: | A: | Reason: | Difficulty Level: | ”’ 

## `Multimodal QA (w/figure):` 

- Based on the above image, please design three question-answer pairs with different levels of 

- difficulty: Easy, Medium, Hard. 

- These questions require a deep reading of the meaning of the image. Note that these 

- questions also need to be realistic. You should provide the reason if possible. The output should be formalized as: ”’Q: | A: | Reason: | Difficulty Level: | ”’ 

data is heavily skewed towards these categories, enabling them to answer some questions accurately without additional files. Moreover, as GPT-4o is an optimized version of GPT-4, it likely benefits from a broader and more extensive training data. 

15 

**==> picture [366 x 211] intentionally omitted <==**

**----- Start of picture text -----**<br>
LLM-based systems Parse-then-Read Pipelines<br>Overall Overall<br>Unans- Text- Unans- Text-<br>werable only werable only<br>Meta- Multi- Meta- Multi-<br>data modal data modal<br>— GPT-4 — Phi-3 — Gemma<br>— Kimi — GPT-4 — Claude-3 — ChatGLM-6B — Mixtral-8x7B<br>— Llama-3-70B — Yi-1.5-34B<br>— GLM-4 — ERNIE-3.5 — Qwen-2.5 — InternLM2-20B — Command-R-35B<br>**----- End of picture text -----**<br>


Figure 7: Performance (Relative) of two major methods on DOCBENCH against the best. 

Table 7: Analyzing the Influence of Input Sources: We deliver questions with attached files and without files to GPT-4 and GPT-4o for evaluation, respectively. 

|**Methods**|||**Domain**||||**Type**|**Type**||**Overall**Acc_._|
|---|---|---|---|---|---|---|---|---|---|---|
||**Aca.**|**Fin.**|**Gov.**|**Laws**|**News**|**Text.**|**Multi.**|**Meta.**|**Una.**||
|GPT-4|||||||||||
|w/ file|65.7|65.3|75.7|69.6|79.6|87.9|74.7|50.8|37.1|69.8|
|w/o file|10.9|10.8|23.0|29.3|32.6|40.8|8.1|1.6|10.5|19.1|
|GPT-4o|||||||||||
|w/ file|56.4|56.3|73.0|65.5|75.0|85.0|62.7|50.4|17.7|63.1|
|w/o file|11.2|13.5|29.1|31.9|36.0|46.6|10.7|2.3|6.5|21.7|



16 

