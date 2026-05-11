# **FeTaQA: Free-form Table Question Answering** 

**Linyong Nan**[1] **Chiachun Hsieh**[3] **Ziming Mao**[1] **Xi Victoria Lin**[2] _[∗]_ **Neha Verma**[1] **Rui Zhang**[4] **Wojciech Kry´sci´nski**[2] **Nick Schoelkopf**[1] **Riley Kong**[5] **Xiangru Tang**[1] **Murori Mutuma**[1] **Ben Rosand**[1] **Isabel Trindade**[1] **Renusree Bandaru**[4] **Jacob Cunningham**[4] **Caiming Xiong**[2] **Dragomir Radev**[1] _[,]_[2] 1 Yale University 2 Salesforce Research 3 The University of Hong Kong 

4 Penn State University 5 Archbishop Mitty High School 

_{_ linyong.nan, ziming.mao _}_ @yale.edu, hsiehcc@connect.hku.hk 

## **Abstract** 

Existing table question answering datasets contain abundant factual questions that primarily evaluate the query and schema comprehension capability of a system, but they fail to include questions that require complex reasoning and integration of information due to the constraint of the associated short-form answers. To address these issues and to demonstrate the full challenge of table question answering, we introduce FeTaQA, a new dataset with 10K Wikipediabased _{table_ , _question_ , _free-form answer_ , _supporting table cells}_ pairs. FeTaQA yields a more challenging table question answering setting because it requires generating free-form text answers after retrieval, inference, and integration of multiple discontinuous facts from a structured knowledge source. Unlike datasets of generative QA over text in which answers are prevalent with copies of short text spans from the source, answers in our dataset are human-generated explanations involving entities and their high-level relations. We provide two benchmark methods for the proposed task: a pipeline method based on semantic parsingbased QA systems and an end-to-end method based on large pretrained text generation models, and show that FeTaQA poses a challenge for both methods. 

## **1 Introduction** 

Question Answering (QA) is the task of producing answers to natural language questions based on knowledge resources (Burke et al., 1997; Yao and Van Durme, 2014; Chen et al., 2017). One of the primary goals of QA is to allow users to directly and efficiently interact with large-scale and heterogeneous knowledge sources. In the real world, knowledge sources take a variety of forms, including unstructured texts (documents, passages, 

> _∗_ Now at Facebook AI. 

or conversations), structured knowledge bases or databases, and semi-structured tables, each requiring dedicated modeling approaches. 

For QA over text, a sequence modeling approach is usually adopted to encode the query and the context, and answers are either categorical (Lai et al., 2017), extractive (Rajpurkar et al., 2016; Yang et al., 2018) or abstractive/generative (Kocisky et al.´ , 2017; Nguyen et al., 2016; Fan et al., 2019; Kwiatkowski et al., 2019). 

For table-based QA, a common approach is to apply semantic parsing on the query and the table schema to generate a logical form (e.g. a SQL-like database query) that can be executed to retrieve the answer from the relevant portion of the table (Pasupat and Liang, 2015; Iyyer et al., 2017; Zhong et al., 2017; Yu et al., 2018). The answers are extracted facts/entities in the table, therefore usually in short-form. 

Though existing datasets have enabled significant progress for table QA, their limitations prevent them from reflecting the full challenge of the task. Users of QA systems tend to ask complex questions which require elaborate answers, often containing explanations, while existing datasets are limited to simple short-form answers. 

To address these shortcomings we present FeTaQA, a **F** r **e** e-form **Ta** ble Question Answering dataset which includes long, informative, and freeform answers. FeTaQA challenges QA systems with the following tasks: 1) retrieving multiple entities from tables based on the query; 2) reasoning over relations of these entities that are pertinent to the query and integrating these pieces of information into a coherent answer 3) aggregating associated information into an explanation when the query is abstract or ambiguous; and 4) generating an informative, relevant, and faithful answer to the query. In addition, with tables sourced from Wikipedia, FetaQA covers a diverse set of top- 

Figure 1: Examples of **FeTaQA** instances. Only part of the original table is shown for better visualization. These examples are referred as (a), (b), (c), (d) from upper left to bottom right in the paper. 

|**Dataset**|Wikipedia articles|**Knowledge Source**<br>Stories, books, movie scripts<br>Online forum texts|**Knowledge Source**<br>Stories, books, movie scripts<br>Online forum texts|Wikipedia tables|**Answer Format**|**Avg # Words**<br>**in Answer**|
|---|---|---|---|---|---|---|
|SQuAD (Rajpurkar et al.,2016)|wv||||Text-span|3.2|
|HotpotQA (Yang et al.,2018)|wv||||Short-form entity|2.2|
|NarrativeQA (Koˇcisky et al.,2018)||a|||Free-form text|4.7|
|ELI5 (Fan et al.,2019)|||Va||Free-form text|130.6|
|WikiTableQuestions (Pasupat and Liang,2015)||||a|Short-form entity|1.7|
|SequenceQA (Saha et al.,2018)||||a|Short-form entity|1.2|
|HybridQA (Chen et al.,2020e)|wv|||a|Short-form entity|2.1|
|**FeTaQA**||||wv|Free-form text|18.9|



Table 1: Comparison of **FeTaQA** with other QA datasets. 

ics and includes semi-structured tables containing un-normalized text, including numbers, dates, and phrases. FeTaQA examples are presented in Figure 1 and differences between FeTaQA and other QA datasets are described in Table 1. 

We formulate generative table question answering as a Sequence-to-Sequence learning problem to evaluate the state-of-the-art models’ performances on FeTaQA. We propose two benchmark methods and provide experiment results for them. The first one is an end-to-end model that integrates query and table comprehension, logical reasoning, and language generation by adapting T5 (Raffel et al., 2019). The other is a pipeline model that achieves content selection and surface realization in separate modules involving TAPAS (Herzig et al., 2020). 

Through human studies, we evaluate answers generated by our proposed models as well as the reference answer based on fluency, correctness, adequacy (informativeness), and faithfulness. The results indicate the challenging nature of FeTaQA and that there is much room for improvement in QA systems. We make the dataset available online.[1] 

1https://github.com/Yale-LILY/FeTaQA 

## **2 Dataset** 

Here we introduce FeTaQA and describe the process and criteria for collecting the tables, questions and answers. Some statistics of FeTaQA are shown in § 2.4. 

## **2.1 Desiderata** 

We frame generative table question answering as the problem of generating an answer _a_ to a question _q_ based on a semi-structured table _T_ and its metadata _m_ . Our goal was to construct a table QA dataset _{_ ( _qi, ai, Ti, mi_ ) _|i_ = 1 _. . . n}_ that includes a large number of tables on diverse topics. The tables should be intelligible, well-formed, and moderately sized to make retrieval challenging yet plausible. Each table pairs a question with an answer sentence. The question should require retrieval and reasoning over multiple sources of information in the table, and the answer should integrate both facts and inferences into a coherent sentence that answers the question. Both questions and answers should be natural and fully grounded in the context of the entire table and its metadata such as the title. 

## **2.2 Data Collection Method** 

We start building the dataset by collecting data instances from ToTTo (Parikh et al., 2020), a recent large-scale Table-to-Text dataset that contains tables and table-grounded sentences obtained from a diverse variety of Wikipedia pages. Additionally, ToTTo comes with annotations of table cells that support the sentence: a sentence is supported by the cell contents if it is directly stated or can be logically inferred by them. ToTTo applied several heuristics to sample the tables and the candidate sentences from Wikipedia pages, and their annotators are asked to revise sentences and highlight the corresponding table regions so that the sentences still have the varied language and structure found in natural sentences while being grounded to the table. 

Sampling examples from the ToTTo dataset was conducted in multiple steps. We first sample tables whose sizes are within 3 to 34 rows long and 3 to 7 columns wide (up to 75th percentile of all ToTTo table sizes) to avoid truncation of sequence of linearized table for transformer-based models, whose default maximum input sequence length is 512. To ensure sentences contain several table entities, we further select tables whose annotation of highlighted regions covers multiple rows. We also collect a subset of single-row highlighted regions which span multiple rows or columns in content. Following this sampling procedure, we were able to obtain 16,576 _{table, metadata, highlighted region, sentence}_ instances with which we conduct the annotation procedure as described below. The flowchart of the sampling process is found in Figure 7 of the Appendix. 

We adopted these table-grounded sentences as the answers in our new QA dataset since they are long, natural sentences containing rich information and inferences over the corresponding table. We also exploit ToTTo’s annotations of table cells (the highlighted table region) as the weak supervision (denotations) for training models and labels for evaluating model retrieval competency. We parsed the tables (originally in HTML format) into a 2- dimensional array, where the first row corresponds to the table header. We also processed merged cells by copying the cell content and cell highlighted region to all the individual cells that compose the original merged cell. 

## **2.2.1 Question Annotation** 

Question annotations were collected with the help of human judges in two phases: an internal phase conducted by on-site expert annotators, and an external phase conducted by crowd workers on Amazon Mechanical Turk. To streamline the process, we built a custom web interface to visualize table HTML and metadata, augmented with web widgets that allow table region highlighting and sentence editing. A screenshot of the annotation interface is shown in Figure 8 of the Appendix. 

Provided the necessary context, the annotators were asked to write a question whose answer is the provided ToTTo sentence. The annotators were given the option to modify the sentence, the table cell content, and the highlighted region to better match the associated question. 

**Internal Annotations** In the first phase of annotation, we enrolled 15 internal annotators who were provided with preliminary guidelines. In addition to the annotation task, they were asked to provide feedback regarding the task instructions and the user experience of the website, based on which we iteratively modified the guideline and the website design. 

**External Annotations** For external annotations, we hired MTurk workers who have completed at least 500 HITs, have 97% approval rate, and are from English-speaking regions. To ensure that the MTurk annotators understand our task, we provided an instruction video for the interactive annotation tool usage, FAQs that clarify the annotations we desire, along with good vs. bad annotation examples. We also created a Slack channel for crowdsourced workers to ask questions and clarify doubts. 

**Annotation Evaluation** To ensure FeTaQA is of high quality, we evaluate crowdsourced annotations as follows. First we auto-rejected questions that fall outside the length range (4 to 25) or convoluted questions that contain more than two interrogatives (259 examples in total). For the remaining annotations, we built another web interface for evaluation and asked internal evaluators to label an annotation as “approve”, “reject” or “questionable” and score the annotation based on its fluency, faithfulness, and the extent to which the question needs the full sentence as the answer. Internal evaluators were also asked to modify the question annotations that were not approved. Our final dataset anno- 

tators contribution is distributed as follows: we have 3,039 (30%) instances from internal annotators, 7,291 (70%) from MTurk workers. In total, our dataset contains 10,330 instances. 

## **2.3 Dataset Split** 

Randomly splitting the dataset may make train, development, and test splits contain tables with similar contents (Finegan-Dollak et al., 2018; Lewis et al., 2020). Therefore, to increase the generalization challenge, we calculated the Jaccard similarity of two instances based on the set of tokens shown in table headers and questions, and split the dataset in such a way that models are evaluated on test split instances that are least similar to those used for training. We first sampled 800 instances randomly as a seed split. Then we add those that have Jaccard similarities greater than 0.465 to the seed split. This process generates two splits of 70% and 30% of all instances, the former becomes the train split and the latter is randomly divided with a ratio of 1:2 to form the development and test splits. This results in 7,326/1,001/2,003 instances in the train/dev/test splits, respectively. 

## **2.4 Data Analysis and Statistics** 

Basic statistics of FeTaQA are shown in Table 2, and human evaluation scores and inter-evaluator agreements are reported in Table 3. A quantitative and qualitative analysis of FeTaQA shows it contains abundant complex questions that require retrieval of multiple entities in the context, as shown by the human evaluation score for question complexity, and that the median number of highlighted cells (denotations) is 6, which is twice as much as the corresponding number for ToTTo. These denotations are correct and adequate as indicated by the corresponding high evaluation scores. The freeform answers have a median of 18 tokens in length, and are grounded to the table and the denotations, also suggested by the high evaluation scores. 

**Topics** Similar to ToTTo, we use Wikimedia Foundation’s topic categorization model (Asthana and Halfaker, 2018) to investigate the topics distribution of FeTaQA. Although our dataset is limited to topics presented in ToTTo, we are able to sample instances that have evenly distributed topics, as shown in Figure 2. We found that most of the instances are related to biography, sports and geographical regions. There are also abundant instances related to media, politics and government. 

|**Property**|**Value**|
|---|---|
|Unique Tables|10,330|
|Question Length (Median/Avg)|12 / 13.2|
|Answer Length (Median/Avg)|18 / 18.9|
|Rows per Table (Median/Avg)|12 / 13.8|
|Columns per Table (Median/Avg)|5 / 5.9|
|No. of Highlighted Cell (Median/Avg)|6 / 8.0|
|Percentage of Cells Highlighted (Median/Avg)|10.7% / 16.2%|
|Page Title Length (Median/Avg)|2 / 3.3|
|Section Title Length (Median/Avg)|2 / 1.9|
|Training Set Size|7,326|
|Development Set Size|1,001|
|Test Set Size|2,003|



Table 2: FeTaQA Core Statistics 

|**Annotation Quality**|**Score****_>_= 4**<br>**(%)**|**% Agreement**|**Randolph’s Kappa**<br>**/ 95% CI**|
|---|---|---|---|
|Question Complexity|52.6|0.65|0.48 / [0.41, 0.55]|
|Denotation Correctness|89.0|0.88|0.82 / [0.76, 0.88]|
|Denotation Adequacy|91.6|0.89|0.83 / [0.77, 0.89]|
|Answer Fluency|95.0|0.92|0.89 / [0.84, 0.94]|
|Answer Correctness|92.4|0.91|0.86 / [0.80, 0.92]|
|Answer Adequacy|90.6|0.88|0.82 / [0.76, 0.88]|
|Answer Faithfulness|95.6|0.93|0.89 / [0.84, 0.94]|



Table 3: Human evaluation over 100 samples of FeTaQA. 5 internal evaluators are asked to rate the samples on a scale of 1 to 5. We report % of samples that have score _≥_ 4 to show high quality of FeTaQA, and report percent agreement and Randolph’s Kappa (Randolph, 2005) (with 95% CI) to show that our human evaluation has high inter-annotator agreement. 

Figure 2: FeTaQA Topics Distribution. 

**Question Types** FeTaQA has diverse and complex questions, as illustrated in Figure 3. Comparison of question type distributions with other table QA datasets is shown in Figure 9 of the Appendix. We found that in FeTaQA, a large percentage of _what_ questions are asking entities in plural, or abstract entity such as _outcome, result, margin, percentage_ . In addition, there is a higher percentage of _how_ questions that are not _how many/much_ , compared to existing table QA datasets. 

Figure 3: FeTaQA questions by top 5 most frequent starting words, where box size represents frequency. 

## **3 Models** 

To quantify the challenge posed by FeTaQA for state-of-the-art models, we used two modeling approaches that have been shown to be effective for the existing table question answering datasets, with some modifications made to adjust to our task. Model configurations are shown in Figure 4. 

## **3.1 Pipeline Model** 

Question answering over tables is usually seen as a semantic parsing task. Based on the table schema, a semantic parser maps the question to a logical form that can be used to retrieve the result from the table. The answer is usually a single entity that is either a table cell value or the aggregation result of multiple cell values (aggregation operators include COUNT, MAX, SUM, etc.). A table semantic parser is trained using logical forms as supervised examples, but due to its high annotation cost, an alternative approach is to use denotations for weak supervision. The denotations are usually the targets of the existing table QA tasks. With this approach, the parser is able to retrieve denotations directly. 

However, in our task, targets are generated texts instead of retrieved denotations, suggesting that we also need a generator to integrate the retrieved information into a cogent sentence. Therefore, we propose a pipeline model with two separately trained modules, described below. 

## **Weakly Supervised Table Semantic Parsing** 

The first module adopts a table semantic parser that is pre-trained with weak supervision. We use TAPAS (Herzig et al., 2020), a state-of-the-art model for table QA, to start with. We fine-tune it on FeTaQA with our annotated denotations (highlighted table regions). We believe fine-tuning is 

crucial for our task because TAPAS is pre-trained on questions that require retrieval of limited denotations (single entity or homogeneous entities that can be aggregated with COUNT, SUM, or AVG operation), while FeTaQA questions require retrieval of multiple entities and complex aggregation operations. Details of experiment results are provided in Section 4.3. Note that besides denotations, TAPAS also predicts an aggregation operation (choose from COUNT, SUM, AVG, NONE) applied to the predicted denotations to obtain the final answer. However, we use NONE as the aggregation operation label for fine-tuning due to the lack of annotations, therefore leaving the inference of aggregation operation to the second module. 

**Data-to-Text** As shown in Figure 5, we fine-tune T5 (Raffel et al., 2019) on DART (Nan et al., 2021) to obtain a Data-to-Text model as the second module of the pipeline to perform surface realization of table cells (denotations in our case). We first convert the denotation prediction into the triple-set format with the following scheme: for each table cell in the highlighted region, we generate the following triple: [[TABLECONTEXT], column ~~h~~ eader, cell ~~v~~ alue], where column ~~h~~ eader is the cell’s corresponding column name. Similar to DART, we use [TABLECONTEXT] as a special token for converting a table cell into a triple. We then incorporate the metadata into triples by replacing column ~~h~~ eader with the field name (TABLE ~~T~~ ITLE, PAGE ~~T~~ ITLE) and cell ~~v~~ alue with the metadata content (table title text, page title text). We end up with a triple-set containing all highlighted table cells and the metadata (table title and title of the Wikipedia page that includes the table). We further fine-tune the Datato-Text model on ToTTo instances so that it adapts 

Figure 4: Pipeline model and End-to-End model diagrams. 

to our formation of triple-set inputs. To avoid exposure to FeTaQA test instances, we fine-tune with a sample of 8K ToTTo instances that are not used for creating FeTaQA. 

Figure 6: Table linearization in end-to-end model. 

## **4 Experiments** 

In this section, we explain the experiment settings and report the automatic and human evaluations on model outputs. 

Figure 5: Weakly supervised fine-tuning of table semantic parser on FeTaQA. We choose a checkpoint of TAPAS-base fine-tuned on WikiTableQuestions to start with. After fine-tuning, the table semantic parser predicts denotations, which are then converted to triples and sent to the Data-to-Text module. 

## **3.2 End-to-End Model** 

In this approach, we model the task as a sequenceto-sequence learning problem by linearizing table _T_ appended to question _q_ as the source sequence, and treating the free-form answer _a_ as the target sequence. We propose a simple linearization scheme as a baseline: table rows are concatenated with [SEP] tokens in between, and cells in each row are separated by spaces. Since the input sequence length may exceed the model limit, we prepend _q_ to table linearization _T_[�] , using [CLS] tokens as prefixes for separation. We fine-tune models from the T5-family on the FeTaQA train set. The linearization scheme is visualized in Figure 6. 

## **4.1 Experiment Setup** 

We first experiment with the pipeline model in a zero-shot setting, that is, without any fine-tuning on FeTaQA. We use a checkpoint of TAPAS-base that is fine-tuned on WikiTableQuestions (Pasupat and Liang, 2015) to perform table semantic parsing implicitly in order to produce a set of denotations, which is then converted to a triple-set as described in 3.1. We then employ a T5-large model (Raffel et al., 2019) that goes through two fine-tuning stages: in the first stage it is fine-tuned on the downstream Data-to-Text task with DART (Nan et al., 2021); in the second stage it is further fine-tuned on ToTTo instances to adapt to the triple-set formulation we proposed. We denote this setting as Pipeline - zeroshot in Table 4. Next we experiment with the pipeline model by fine-tuning the table semantic parser on FeTaQA. We further fine-tune the TAPAS-base checkpoint (WTQ finetuned) on FeTaQA train set and select models based on their performance on the development set. We 

use the same Data-to-Text model as described in the zero-shot setting. 

For the End-to-End model, we adapt Hugging Face’s implementation (Wolf et al., 2020) of T5 (Raffel et al., 2019) for our task. We use a standard T5-tokenizer with additional [CLS] and [SEP] tokens and the model vocabulary is resized accordingly. Since we expect the input sequence to be significantly longer than the target, we fine-tuned the models using T5’s “summarize: ” prefix. The motivation behind this is to avoid simple extraction from the table since abstractive summarization is supposed to rephrase important details in the source. T5-small is trained on 4 Tesla K80 GPUs with per-device batch size of 16 for 30 epochs (about 6,900 steps). T5-base is trained on 4 Tesla K80 with per-device batch size of 4 (due to GPU memory constraints) for 80 epochs (about 36,640 steps). As for T5-large, we distributed the layers across 8 Tesla K80 to train with a batch size of 4 for 80 epochs (about 80k steps). 

## **4.2 Evaluation Metrics** 

We use a variety of automatic metrics and human evaluation (Section 4.4) to evaluate the quality of the generated answers. We report sacreBLEU (Post, 2018), ROUGE- _{_ 1, 2, L _}_ (Lin, 2004), and METEOR (Banerjee and Lavie, 2005) that evaluate the n-gram match between generated and reference answers. Considering the limitations of these measures in evaluating the semantic meanings of sentences, we also report BERTScore (Zhang et al., 2020) and BLEURT (Sellam et al., 2020) that incorporate semantics using contextual embeddings. To evaluate the retrieval competency of table semantic parsers, we applied various set similarity metrics to the predicted and reference denotation lists. Specifically, we report Jaccard similarity, Overlap, Cosine similarity, and Dice similarity. 

## **4.3 Results and Discussions** 

Our experimental results on the FeTaQA test set are summarized in Table 4. The T5-large model using an End-to-End modeling approach achieves the highest performance scores in almost all evaluation metrics. Also, we observe a large performance gap between pipeline models and End-to-End models, even though the latter only adopt a simple linearization strategy for encoding tables. 

We also see that after fine-tuning on FeTaQA with denotations as weak supervisions, the pipeline model improves by almost 2 BLEU points. To fur- 

ther examine the source of this improvement, we report the evaluation of table semantic parser performance in Table 5, from which we also observe an improvement in retrieval capability. However, we note that compared with the reference denotations that have a median of six table cells being highlighted (shown in 2), our table semantic parser is only able to predict two table cells on average before fine-tuning on FeTaQA, and three table cells on average after. This indicates a large space for improvement. We suspect that the low performance of denotation predictions and the loss of relational information between denotations lead to the inadequate performance of pipeline models. 

## **4.4 Human Evaluation** 

To further evaluate the quality of the answers generated by different models comparing to the references, we conduct our human evaluation based on four criteria: (1) _fluency_ if an answer is natural and grammatical; (2) _correctness_ if an answer is correct; (3) _adequacy_ if an answer contains all the information that is asked; (4) _faithfulness_ if an answer is faithful and grounded to the contents of the table and the highlighted region. Each evaluator is asked to examine an answer given the question and the full context (table, highlighted region, and metadata) and give a score on a scale of 1 to 5 for each of the criteria. We ask five internal annotators to evaluate 100 samples of FeTaQA instances. Each sample is paired with 3 answers: the reference, the pipeline model result, and the End-to-End model result. 

Table 6 attests to the high quality of our annotations and the challenging nature of FeTaQA. Similar to the evaluation result of the automatic metrics, we observe a large gap between the pipeline model and the End-to-End model, with the latter one significantly outperforming its counterpart in terms of answer correctness, adequacy, and faithfulness. Comparing the best performing End-to-End model outputs to human references, we see that there is room for improvement in the future. 

## **5 Related Work** 

**Generative QA** Generative question answering datasets such as NarrativeQA (Kociskˇ y et al.` , 2018), CoQA (Reddy et al., 2019), TriviaQA (Joshi et al., 2017), and MS MARCO (Nguyen et al., 2016) all have free-form answers that are generated based on the contexts of Wikipedia articles, books, movie 

||**sacreBLEU**1|**ROUGE-1**|**ROUGE-2**|**ROUGE-L**|**METEOR**|**BERTScore**|**BLEURT**|
|---|---|---|---|---|---|---|---|
|Pipeline - zeroshot|9.16|0.38|0.20|0.33|0.22|0.88|-0.79|
|Pipeline - fne-tuned|11.00|0.40|0.22|0.35|0.24|0.91|-0.35|
|End-to-End - T5-small|21.60|0.55|0.33|0.47|0.40|0.94|0.08|
|End-to-End - T5-base|28.14|0.61|0.39|0.51|0.47|0.96|0.31|
|End-to-End - T5-large|**30.54**|**0.63**|**0.41**|**0.53**|**0.49**|**0.96**|**0.57**|



Table 4: Experiment results on the test split of FeTaQA. 

||**Jaccard**|**Overlap**<br>**Coff.**|**Cosine**|**Dice**|
|---|---|---|---|---|
|Zeroshot|0.065|0.300|0.140|0.109|
|Fine-tuned|0.101|0.311|0.184|0.161|



Table 5: Evaluation of denotation prediction on the test split of FeTaQA. We report performance of TAPAS in zero-shot and fine-tuned with weak supervision. 

|**Source**|**Fluent**<br>**(%)**|**Correct**<br>**(%)**|**Adequate**<br>**(%)**|**Faithful**<br>**(%)**|
|---|---|---|---|---|
|Pipeline<br>End-to-End<br>Reference|85.2<br>94.6<br>95.0|25.4<br>54.8<br>92.4|8.4<br>48.4<br>90.6|23.6<br>50.4<br>95.6|



Table 6: Human evaluation over 100 samples of model outputs and references. We report the percentage of outputs that have scores of 4 or 5. 

scripts, dialogues or web documents. These responses are mostly crowd-sourced and are reported to mostly contain copies of short text spans from the source. By contrast, ELI5 (Fan et al., 2019) is a long form question answering dataset containing a diverse set of complex questions, each paired with a paragraph-long answer and 100 relevant _web source_ documents (Petroni et al., 2020; Krishna et al., 2021). FeTaQA is the first dataset for generative question answering over tables. Unlike the existing generative QA datasets that assess multi-documents retrieval and abstraction capability, FeTaQA poses new challenges in the reasoning and integration capability of a system given a structured knowledge source. 

**QA over Tables and Semantic Parsing** Several datasets have been proposed to apply semantic parsing on tables, including WikiTableQuestions (Pasupat and Liang, 2015), SequentialQA (Iyyer et al., 2017), WikiSQL (Zhong et al., 2017), Spider (Yu et al., 2018). With the development of pre-trained 

> 1SacreBLEU signature: BLEU+case.lc+numrefs.1+smooth.exp+tok.13a+version.1.3.7 

language models, recent work (Yin et al., 2020; Herzig et al., 2020; Eisenschlos et al., 2020; Iida et al., 2021) jointly learns representations for natural language sentences and structured tables, and Yu et al. (2020, 2021) use pre-training approach for table semantic parsing. HybridQA (Chen et al., 2020e) and OTT-QA (Chen et al., 2020a) have contexts of both structured tables and unstructured text. MultiModalQA (Talmor et al., 2021) contains complex questions over text, tables and images. These datasets define a table QA task that is extractive in nature by restricting their answers to be short-form, while FeTaQA frames table QA as a generation task. 

**Data-to-text generation** Recent neural end-toend models tested on the WebNLG 2017 dataset (Gardent et al., 2017) have focused on incorporating pre-training and fine-tuning for specific generation tasks (Chen et al., 2020c; Kale, 2020) to improve performance and strengthen generalization ability. However, recent models featuring separate content-planning and surface realization stages have exhibited improvements (Moryossef et al., 2019; Iso et al., 2020) over comparable baselines. TabFact (Chen et al., 2020d) is composed of Wikipedia tables coupled with statements labeled as either “ENTAILED” or “REFUTED” by the table. LogicNLG (Chen et al., 2020b) features statements logically entailed from tables. ToTTo (Parikh et al., 2020) is a largescale open-domain dataset consisting of Wikipedia tables with a set of highlighted table cells and a sentence description of those highlighted cells. DART (Nan et al., 2021) is an open-domain Data-to-Text dataset that contains table-ontologypreserving data samples with diverse predicate set occurred in Wikipedia tables. 

## **6 Conclusion** 

In this paper, we introduced the task of generative table question answering with FeTaQA, a table QA dataset consisting of complex questions that require 

free-form, elaborate answers. We also proposed two modeling approaches: (1) a pipeline model that incorporates a table semantic parser and (2) a Datato-Text generator, and an End-to-End model that includes query comprehension, reasoning and text generation. Our experimental results indicate that the End-to-End model with a simple table encoding strategy achieves much higher scores than the pipeline model that requires table semantic parsing. Furthermore, we show that FeTaQA introduces new challenges for table question answering that call for innovative model designs in the future. 

## **References** 

- Sumit Asthana and Aaron Halfaker. 2018. With few eyes, all hoaxes are deep. _Proc. ACM Hum.-Comput. Interact._ , 2(CSCW). 

- Satanjeev Banerjee and Alon Lavie. 2005. METEOR: An automatic metric for MT evaluation with improved correlation with human judgments. In _Proceedings of the ACL Workshop on Intrinsic and Extrinsic Evaluation Measures for Machine Translation and/or Summarization_ , pages 65–72, Ann Arbor, Michigan. Association for Computational Linguistics. 

- Robin D Burke, Kristian J Hammond, Vladimir Kulyukin, Steven L Lytinen, Noriko Tomuro, and Scott Schoenberg. 1997. Question answering from frequently asked question files: Experiences with the faq finder system. _AI magazine_ , 18(2):57–57. 

- Danqi Chen, Adam Fisch, Jason Weston, and Antoine Bordes. 2017. Reading Wikipedia to answer opendomain questions. In _Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 1870– 1879, Vancouver, Canada. Association for Computational Linguistics. 

- Wenhu Chen, Ming-Wei Chang, Eva Schlinger, W. Wang, and William W. Cohen. 2020a. Open question answering over tables and text. _ArXiv_ , abs/2010.10439. 

- Wenhu Chen, Jianshu Chen, Y. Su, Zhiyu Chen, and William Yang Wang. 2020b. Logical natural language generation from open-domain tables. In _ACL_ . 

- Wenhu Chen, Yu Su, X. Yan, and W. Wang. 2020c. Kgpt: Knowledge-grounded pre-training for data-totext generation. In _EMNLP_ . 

- Wenhu Chen, Hongmin Wang, Jianshu Chen, Yunkai Zhang, Hong Wang, LI SHIYANG, Xiyou Zhou, and William Yang Wang. 2020d. Tabfact: A largescale dataset for table-based fact verification. _ArXiv_ , abs/1909.02164. 

- Wenhu Chen, Hanwen Zha, Zhiyu Chen, Wenhan Xiong, Hong Wang, and William Wang. 2020e. Hybridqa: A dataset of multi-hop question answering over tabular and textual data. _Findings of EMNLP 2020_ . 

- Julian Eisenschlos, Syrine Krichene, and Thomas M¨uller. 2020. Understanding tables with intermediate pre-training. In _Findings of the Association for Computational Linguistics: EMNLP 2020_ , pages 281–296, Online. Association for Computational Linguistics. 

- Angela Fan, Yacine Jernite, Ethan Perez, David Grangier, Jason Weston, and Michael Auli. 2019. ELI5: Long form question answering. In _Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics_ , pages 3558–3567, Florence, Italy. Association for Computational Linguistics. 

- Catherine Finegan-Dollak, Jonathan K. Kummerfeld, Li Zhang, Karthik Ramanathan, Sesh Sadasivam, Rui Zhang, and Dragomir Radev. 2018. Improving text-to-SQL evaluation methodology. In _ACL 2018_ . Association for Computational Linguistics. 

- Claire Gardent, Anastasia Shimorina, Shashi Narayan, and Laura Perez-Beltrachini. 2017. The WebNLG challenge: Generating text from RDF data. In _Proceedings of the 10th International Conference on Natural Language Generation_ , pages 124–133, Santiago de Compostela, Spain. Association for Computational Linguistics. 

- Jonathan Herzig, Pawel Krzysztof Nowak, Thomas M¨uller, Francesco Piccinno, and Julian Eisenschlos. 2020. TaPas: Weakly supervised table parsing via pre-training. In _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics_ , pages 4320–4333, Online. Association for Computational Linguistics. 

- Hiroshi Iida, June Thai, Varun Manjunatha, and Mohit Iyyer. 2021. Tabbie: Pretrained representations of tabular data. In _NAACL_ . 

- Hayate Iso, Yui Uehara, Tatsuya Ishigaki, Hiroshi Noji, Eiji Aramaki, Ichiro Kobayashi, Yusuke Miyao, Naoaki Okazaki, and Hiroya Takamura. 2020. Learning to select, track, and generate for data-to-text. _Journal of Natural Language Processing_ , 27(3):599–626. 

- Mohit Iyyer, Wen-tau Yih, and Ming-Wei Chang. 2017. Search-based neural structured learning for sequential question answering. In _Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 1821–1831, Vancouver, Canada. Association for Computational Linguistics. 

- Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke Zettlemoyer. 2017. Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension. _arXiv preprint arXiv:1705.03551_ . 

- Mihir Kale. 2020. Text-to-text pre-training for data-totext tasks. _arXiv preprint arXiv:2005.10433_ . 

- Tom´as Kocisk´y, Jonathan Schwarz, Phil Blunsom, Chris Dyer, Karl Moritz Hermann, G´abor Melis, and Edward Grefenstette. 2017. The narrativeqa reading comprehension challenge. _CoRR_ , abs/1712.07040. 

- Tom´aˇs Koˇcisk`y, Jonathan Schwarz, Phil Blunsom, Chris Dyer, Karl Moritz Hermann, G´abor Melis, and Edward Grefenstette. 2018. The narrativeqa reading comprehension challenge. _Transactions of the Association for Computational Linguistics_ , 6:317–328. 

- Kalpesh Krishna, Aurko Roy, and Mohit Iyyer. 2021. Hurdles to progress in long-form question answering. In _NAACL_ . 

- Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Matthew Kelcey, Jacob Devlin, Kenton Lee, Kristina N. Toutanova, Llion Jones, Ming-Wei Chang, Andrew Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natural questions: a benchmark for question answering research. _Transactions of the Association of Computational Linguistics_ . 

- Guokun Lai, Qizhe Xie, Hanxiao Liu, Yiming Yang, and Eduard Hovy. 2017. RACE: Large-scale ReAding comprehension dataset from examinations. In _Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing_ , pages 785–794, Copenhagen, Denmark. Association for Computational Linguistics. 

- Patrick Lewis, Pontus Stenetorp, and Sebastian Riedel. 2020. Question and answer test-train overlap in open-domain question answering datasets. _arXiv preprint arXiv:2008.02637_ . 

- Chin-Yew Lin. 2004. ROUGE: A package for automatic evaluation of summaries. In _Text Summarization Branches Out_ , pages 74–81, Barcelona, Spain. Association for Computational Linguistics. 

- Amit Moryossef, Yoav Goldberg, and Ido Dagan. 2019. Step-by-step: Separating planning from realization in neural data-to-text generation. 

- Linyong Nan, Dragomir Radev, Rui Zhang, Amrit Rau, Abhinand Sivaprasad, Chiachun Hsieh, Xiangru Tang, Aadit Vyas, Neha Verma, Pranav Krishna, Yangxiaokang Liu, Nadia Irwanto, Jessica Pan, Faiaz Rahman, Ahmad Zaidi, Murori Mutuma, Yasin Tarabar, Ankit Gupta, Tao Yu, Yi Chern Tan, Xi Victoria Lin, Caiming Xiong, Richard Socher, and Nazneen Fatema Rajani. 2021. DART: Opendomain structured data record to text generation. In _NAACL_ . 

- Tri Nguyen, Mir Rosenberg, Xia Song, Jianfeng Gao, Saurabh Tiwary, Rangan Majumder, and Li Deng. 2016. Ms marco: A human generated machine reading comprehension dataset. In _CoCo@ NIPS_ . 

- Ankur Parikh, Xuezhi Wang, Sebastian Gehrmann, Manaal Faruqui, Bhuwan Dhingra, Diyi Yang, and Dipanjan Das. 2020. ToTTo: A controlled table-totext generation dataset. In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , pages 1173–1186, Online. Association for Computational Linguistics. 

- Panupong Pasupat and Percy Liang. 2015. Compositional semantic parsing on semi-structured tables. In _Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)_ , pages 1470–1480, Beijing, China. Association for Computational Linguistics. 

- Fabio Petroni, Aleksandra Piktus, Angela Fan, Patrick Lewis, Majid Yazdani, Nicola De Cao, James Thorne, Yacine Jernite, Vassilis Plachouras, Tim Rockt¨aschel, et al. 2020. Kilt: a benchmark for knowledge intensive language tasks. _arXiv preprint arXiv:2009.02252_ . 

- Matt Post. 2018. A call for clarity in reporting BLEU scores. In _Proceedings of the Third Conference on Machine Translation: Research Papers_ , pages 186– 191, Brussels, Belgium. Association for Computational Linguistics. 

- Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. 2019. Exploring the limits of transfer learning with a unified text-to-text transformer. _CoRR_ , abs/1910.10683. 

- Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016. SQuAD: 100,000+ questions for machine comprehension of text. In _Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing_ , pages 2383–2392, Austin, Texas. Association for Computational Linguistics. 

- Justus J. Randolph. 2005. Free-marginal multirater kappa (multirater k[free]): An alternative to fleiss’ fixed-marginal multirater kappa. 

- Siva Reddy, Danqi Chen, and Christopher D Manning. 2019. Coqa: A conversational question answering challenge. _Transactions of the Association for Computational Linguistics_ , 7:249–266. 

- Amrita Saha, Vardaan Pahuja, Mitesh Khapra, Karthik Sankaranarayanan, and Sarath Chandar. 2018. Complex sequential question answering: Towards learning to converse over linked question answer pairs with a knowledge graph. In _AAAI 2018_ . 

- Thibault Sellam, Dipanjan Das, and Ankur P. Parikh. 2020. Bleurt: Learning robust metrics for text generation. 

- Alon Talmor, Ori Yoran, Amnon Catav, Dan Lahav, Yizhong Wang, Akari Asai, Gabriel Ilharco, Hannaneh Hajishirzi, and Jonathan Berant. 2021. MultimodalQA: complex question answering over text, 

tables and images. In _International Conference on Learning Representations_ . 

- Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Remi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander Rush. 2020. Transformers: State-of-the-art natural language processing. In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations_ , pages 38–45, Online. Association for Computational Linguistics. 

   - Victor Zhong, Caiming Xiong, and Richard Socher. 2017. Seq2sql: Generating structured queries from natural language using reinforcement learning. _arXiv preprint arXiv:1709.00103_ . 

- Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. 2018. Hotpotqa: A dataset for diverse, explainable multi-hop question answering. _CoRR_ , abs/1809.09600. 

- Xuchen Yao and Benjamin Van Durme. 2014. Information extraction over structured data: Question answering with Freebase. In _Proceedings of the 52nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 956–966, Baltimore, Maryland. Association for Computational Linguistics. 

- Pengcheng Yin, Graham Neubig, Wen-tau Yih, and Sebastian Riedel. 2020. TaBERT: Pretraining for joint understanding of textual and tabular data. In _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics_ , pages 8413– 8426, Online. Association for Computational Linguistics. 

- Tao Yu, Chien-Sheng Wu, Xi Victoria Lin, Bailin Wang, Yi Chern Tan, Xinyi Yang, Dragomir Radev, Richard Socher, and Caiming Xiong. 2020. Grappa: Grammar-augmented pre-training for table semantic parsing. _arXiv preprint arXiv:2009.13845_ . 

- Tao Yu, Rui Zhang, Oleksandr Polozov, Christopher Meek, and Ahmed Hassan Awadallah. 2021. Score: Pre-training for context representation in conversational semantic parsing. In _ICLR_ . 

- Tao Yu, Rui Zhang, Kai Yang, Michihiro Yasunaga, Dongxu Wang, Zifan Li, James Ma, Irene Li, Qingning Yao, Shanelle Roman, Zilin Zhang, and Dragomir Radev. 2018. Spider: A largescale human-labeled dataset for complex and crossdomain semantic parsing and text-to-SQL task. In _Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing_ , pages 3911–3921, Brussels, Belgium. Association for Computational Linguistics. 

- Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, and Yoav Artzi. 2020. Bertscore: Evaluating text generation with bert. 

## **A Appendix** 

The Appendix contains the following contents: 

- Flowchart of ToTTo instances sampling process. (Figure 7) 

- Screenshot of FeTaQA annotation interface. (Figure 8) 

- Question type distribution comparison between FeTaQA and other Table QA datasets. (Figure 9) 

Figure 7: Flowchart of ToTTo filtering process 

Figure 8: FeTaQA annotation interface 

Figure 9: Question type distribution comparison between different Table QA datasets 

