# **Kleister: A novel task for Information Extraction involving Long Documents with Complex Layout** 

Filip Grali´nski[1,2] , Tomasz Stanisławek[1,4] , Anna Wróblewska[1,4] , Dawid Lipi´nski[1] , Agnieszka Kaliska[1,2] , Paulina Rosalska[1,3] , Bartosz Topolski[1] , Przemysław Biecek[4,5] 

1 Applica.ai, 15 Zaj˛ecza, Warsaw, 00351, Poland, firstname.lastname@applica.ai 

2 Adam Mickiewicz University, 1 Wieniawskiego, Poznan, 61712, Poland, firstname.lastname@amu.edu.pl 

3 Nicolaus Copernicus University, 11 Gagarina, Torun, 87100, Poland, firstname.lastname@umk.pl 

4 Warsaw University of Technology, Koszykowa 75, Warsaw, Poland, firstnameletter.lastname@pw.edu.pl 

5 Samsung R&D Institute Poland, Plac Europejski 1, Warsaw, Poland, firstnameletter.lastname@samsung.com 

## **Abstract** 

State-of-the-art solutions for Natural Language Processing (NLP) are able to capture a broad range of contexts, like the sentence-level context or document-level context for short documents. But these solutions are still struggling when it comes to longer, real-world documents with the information encoded in the spatial structure of the document, such as page elements like tables, forms, headers, openings or footers; complex page layout or presence of multiple pages. 

To encourage progress on deeper and more complex Information Extraction (IE) we introduce a new task (named _Kleister_ ) with two new datasets. Utilizing both textual and structural layout features, an NLP system must find the most important information, about various types of entities, in long formal documents. We propose _Pipeline_ method as a text-only baseline with different Named Entity Recognition architectures (Flair, BERT, RoBERTa). Moreover, we checked the most popular PDF processing tools for text extraction (pdf2djvu, Tesseract and Textract) in order to analyze behavior of IE system in presence of errors introduced by these tools. 

## **1 Introduction** 

Information Extraction (IE) requires quick but careful skimming through the whole document. We often have to not only search for pieces of information, but also to generate final output for specific entity type (e.g. aggregate multiple occurrences of organization names into one). In practice, this means that the results should be presented in an appropriate form (e.g. data points such as addresses normalized to a standard form). It should also be explained why certain information has been correlated. This may take the form of an indication in the input text. The process can be tedious and difficult for humans to do. Thus, we need automated systems to cope with multiple documents and to extract the required information in a simple and efficient way. 

However, the disparity between what can be done with the state of the art in IE and what is required by real-world business use cases is still large. From the point of view of business users, systems that automatically gather information about individuals, their roles, significant dates, addresses, and amounts from invoices, companies reports and contracts, would be useful (Holt and Chisholm, 2018; Katti et al., 2018; Wróblewska et al., 2018; Sunder et al., 2019). Furthermore, the systems should be reliable and should reliably assess their own certainty about extracted entities. 

However, as far as the state of the art is concerned, there are many machine learning models which must be trained for general named entities to be robust (Peters et al., 2018; Akbik et al., 2018; Devlin et al., 2018). To further increase training efficiency, we can use the documents of a previously defined layout, so that the models could learn how to extract a particular piece of information (Zhao et al., 2019; Denk and Reisswig, 2019; Liu et al., 2019a; Sarkhel and Nandi, 2019). On the other hand, more general extractors are still needed to deal with a variety of information. 

In this paper, we describe two novel datasets for Information Extraction from long documents with complex layouts. We will begin by explaining the need for the dataset that would contain authentic scenarios to provide a review of similar tasks and datasets in the next step (Section 2). Then, we describe 

characteristics of datasets in details (Section 3). Subsequently, we describe baseline methods (using only textual information, without relying directly on 2D information) and their results (with different PDF processing tools) applied to cope with the task with the _Pipeline_ approach described in Section 4. Finally, we discuss challenges in the process to extract the proper entities (Section 5). 

Figure 1: Three different examples of layouts from the _Kleister-charity_ dataset. 

## **2 Review of Information Extraction Datasets** 

Our main idea for preparing a new dataset was to develop a strategy to deal with the main challenge we face in business conditions, which means overcoming such difficulties as: complex layout, specific business logic (the way the content is formulated), OCR quality, document-level extraction and normalization. 

The most similar dataset to our approach regarding the NLP field is the WikiReading dataset and related challenges (Hewlett et al., 2016). This dataset is a large-scale natural language understanding task with 18 million entities and 4.7 million documents. The goal of the task is to predict textual values from the structured knowledge base, Wikidata, by reading the text of the corresponding Wikipedia articles. Some entities can be extracted from the given text, but some of them have to be inferred. Thus, similarly as in our assumptions, the task contains a rich variety of challenging extraction sub-tasks and is also well-suited for end-to-end models. In both sets there are also challenges with output data normalization, e.g. dates, names. 

However, our datasets are even more difficult to process, because they comprise documents with complex layout, noisy OCR-ed input (made by an Optical Character Recognition system) and they are much longer than an average Wikipedia article. These are the main issues that distinguish it from WikiReading and which justify why our task is not only about understanding the language. 

A list of challenges similar to some degree to our goal is also available at the International Conference on Document Analysis and Recognition ICDAR 2019[1] . However, the authors focus mainly on understanding tables and a limited range of document layouts, not extracting particular information from the data. There is a dataset called Form Understanding in Noisy Scanned Documents (FUNSD) (Guillaume Jaume, 2019). FUNSD aims at extracting and structuring the textual content of forms. Unfortunately, the dataset comprises only 200 scanned and annotated forms and the annotations are too general, i.e. question, answer, header. 

1http://icdar2019.org/competitions-2/ 

Another interesting dataset from ICDAR 2019 is a set of scanned receipts. The authors prepared 1000 whole scanned receipt images with annotations of company name, date, address and the total payment amount[2] . Of course, receipts are short documents, and have quite a uniform layout and information structure (they start with the company name, date and invoice number etc.). 

Finally, there are datasets with the information extraction task based on invoices, which are not publicly available to the community (Holt and Chisholm, 2018; Katti et al., 2018). Documents of this kind contain common entities like ‘Invoice date’, ‘Invoice number’, ‘Net amount’ and ‘Vendor Name’ which are extracted using a combination of Natural Language Processing and Computer Vision techniques. This is because spatial information is important to properly understand such documents. However, since they are usually short, there is rather no repetition of the same information, so there is no need to understand the context. 

## **3 Kleister: New Datasets** 

The main goal of the gathered datasets introduced in this paper is to emphasize business value and focus more on problems related to layout analysis and Information Extraction, as well as Natural Language Understanding (several entities should be inferred from the whole document context). Thus, it can be performed as well as an end-to-end task that can be used in real life use cases for robotic process automation of information extraction from documents of complex layout. 

We collected datasets of long formal documents that are US non-disclosure agreements ( _KleisterNDA_ ) and annual financial reports of charitable foundations in the UK ( _Kleister-Charity_ ). The datasets (training and development sets as well as the input to the main test set)[3] are available at https://github.com/applicaai/kleister-nda and https://github.com/ applicaai/kleister-charity. 

_Kleister_ datasets have a multi-modal input (i.e. text versions were obtained from OCR-ed noisy documents, some of which contain illustrations and some were scans) and a list of entities to be found. The list of reference findings is not indicated in the input documents. This is not a NER task, in which we would be interested in determining where a given piece of information or entity is in the text. We are interested in the information itself. Moreover, we assume that in our datasets some documents may be missing some entities, or some entities may have more than one gold value. The input of the dataset comprises: PDF files and text versions of the documents (we used most popular tools for extraction text from PDF files). 

These two datasets have been gathered in different ways because of their repository structures. Also, the reasons why they were published on the Internet were different. The most important difference between them is that the NDA dataset was born-digital but the Charity dataset needed to be OCR-ed. The detailed information about aforementioned open datasets (which are the most popular ones in the domain) and our _Kleister_ datasets are presented in Table 1. 

## **3.1 NDA Dataset** 

The NDA Dataset contains Non-disclosure Agreements, also known as Confidential Agreements. They are legally binding contracts between two or more parties, through which the parties agree not to disclose information covered by the agreement. The NDAs were collected from the Electronic Data Gathering, Analysis and Retrieval system (EDGAR) via Google search engine. Then, a list of entities was established (see Table 2) and documents were manually annotated by a team of linguists, which ensured excellent quality of the annotation ( _κ_ = 0 _._ 971[4] ). 

In Figure 2, there is an example of a problematic entity in a Non-Disclosure Agreement: effective date. It is the date on which the contract enters into force. In general, it coincides with the date of the contract or the date it was signed. It happens, however, that these dates are different and then the date 

> 2https://rrc.cvc.uab.es/?ch=13 

> 3We are planning to set up a shared-task platform where submissions could be evaluated for the test set as well. 

> 4This is the average result for all entities — Cohen’s kappa coefficient was calculated for each entities on the basis of double annotation of 100 random documents from the entire NDA Dataset. A detailed description of the data collection method and annotation procedure can be found in the Appendix. 

of entry into force of the contract is specially marked, e.g. as ‘Effective date’. However, none of dates in the figure is specified in this way. Most NDAs contain a special clause that indicates the date of entry into force of the contract. Usually it is immediately before the signatures of the parties. In this case, the correct answer is November 20, 2008, because in this agreement there is a clause: ‘IN WITNESS WHEREOF, he parties hereto have executed this Agreement on the date first written above.’ 

Table 1: Summary of the existing datasets and the _Kleister_ sets. 

|Dataset name|CoNLL 2003<br>WikiReading<br>FUNSD IC-<br>DAR 2019<br>SROIE<br>IC-<br>DAR 2019|Kleister-<br>NDA<br>Kleister-<br>Charity|
|---|---|---|
||||
|Source<br>Annotation<br>Documents<br>Entities|Reuters news<br>WikiData/<br>Wikipedia<br>scanned<br>forms<br>scanned re-<br>ceipts<br>manual<br>automatic<br>manual<br>manual<br>1 393<br>4.7M<br>199<br>973<br>35 089<br>18M<br>9 743<br>3 892|EDGAR<br>UK<br>Charity<br>Commission<br>manual<br>semi-automatic<br>540<br>2 778<br>2 160<br>21 612|
||||
|train docs<br>dev docs<br>test docs|946<br>16.03M<br>149<br>626<br>216<br>1.89M<br>—<br>—<br>231<br>0.95M<br>50<br>347|254<br>1 729<br>83<br>440<br>203<br>609|
||||
|Entity classes|4<br>867 (top 20 cover<br>75%)<br>3<br>4|4<br>8|
||||
|Mean pages/doc<br>Mean words/doc<br>Mean<br>enti-<br>ties/doc|—<br>1/Wikipedia article<br>1<br>1/receipt<br>216.4<br>489.2<br>158.2<br>45<br>25.2<br>5.31<br>49.0<br>4|5.98<br>22.19<br>2540<br>5149<br>4.0<br>7.8|
||||
|Complex layout|N<br>N<br>Y<br>Y|Y/N<br>Y/N|



Table 2: Summary of the entities in the NDA and Charity datasets. 

|Entities|Description|Total|% all entities|
|---|---|---|---|
||_NDA_dataset|||
|party|parties appearing in the agreement (each of them is treated as|1035|47.9|
||a separate entity)|||
|jurisdiction|state or country whose law governs the agreement|531|24.6|
|effective_date|date on which the contract becomes legally binding|400|18.5|
|term|duration of the agreement|194|9.0|
||_Charity_dataset|||
|address__post_town|post town (part of a charity address)|2692|12.5|
|address__postcode|postcode (part of a charity address)|2717|12.6|
|address__street_line|street with the house number (part of a charity address)|2414|11.1|
|charity_name|name of the charitable organization|2778|12.9|
|charity_number|identifcation number in the charity register|2763|12.8|
|report_date|date of reporting|2776|12.8|
|income_annually|annual income in British pounds (GBP)|2741|12.7|
|spending_annually|annual spending in British pounds (GBP)|2731|12.6|



## **3.2 Charity Dataset** 

The Charity dataset consists of annual financial reports that all charities registered in England and Wales are required to submit to the Charity Commission for England and Wales. Then, the Commission makes them publicly available via its website.[5] Charity reports were collected from the UK Charity Commission website, just like annotations to these documents. The entity list was established on the basis of information that we were able to automatically obtain from the tables on the page describing the content of the reports[6] (see Table 2). 

The quality of automatically obtained entities was checked by a team of annotators based on 100 random reports. After analyzing these documents, the following annotations were corrected: the names of 

> 5https://apps.charitycommission.gov.uk/showcharity/registerofcharities/ RegisterHomePage.aspx 

> 6A detailed description of the data collection method can be found in the Appendix. 

Figure 2: Examples of problematic entities in documents from the Kleister-NDA and Charity datasets. 

the organizations (normalization of _Ltd._ ) and amounts (we fixed entities by adding a decimal part of the value) in a part of the development set and in the whole test set (development and test sets are important in context of measuring model actual performance). Then we repeated the annotation check based on 200 random documents from train and development sets (we assume that the annotation of the test set is excellent— _κ_ = 0 _._ 848[7] ). Our preliminary and final results of the quality control procedure are presented in Table 2. The results for the _train_ dataset are definitely lower, but at the same time this set is four times larger than the two others and, unlike them, only a small part of it was manually annotated. 

Table 3: Results of the manual verification of Charity dataset. 

|Entities|Correct initial annotations[%]|Correct final annotations[%]|Correct final annotations[%]|Correct final annotations[%]|
|---|---|---|---|---|
||entire dataset|train|dev|test (_κ_)|
|address(as a whole)|23|55|93|0.920|
|address__post_town|—|83|99|0.889|
|address__postcode|—|78|98|1.000|
|address_street_line|—|67|93|0.871|
|charity_name|86|81|92|0.904|
|charity_number|99|95|100|0.492|
|charity_date|99|98|100|1.000|
|income_annually|82|90|91|0.906|
|spending_annually|78|86|92|0.725|



Figure 2 shows problems with two entities in reports of the charitable organization: charity address and number. Both can co-occur in many variants for the same organization and in the same document. In these cases, it was necessary to refer to the business logic, so the correct answers are “Registered address” and charity number for England and Wales. 

7Cohen’s kappa coefficient was calculated on the basis of double annotation of 100 random documents from the set test. 

## **4 Baseline models** 

Figure 3: Our process of preparing Kleister datasets and training baselines. Initially, we gathered PDF documents and required entities’ values; an important part of the process is the OCR. Then,based on only textual data we prepare pipeline solutions. The pipeline process is illustrated in the second frame and consists of the following stages: auto-tagging, standard NER, text normalization, final selection of the values of entities. 

Kleister datasets for information extraction are challenging tasks and do not exactly match any existing solutions in the current NLP world. In this paper, our aim is to produce strong baselines based on text treated as a sequence, without using additional spatial information. We propose _Pipeline_ technique to solve extraction problems. Our baseline _Pipeline_ method is a chain of processes with a named entity recognition (NER) model as a crucial one to indicate a given entity in the text, then to normalize the entities to canonical forms and finally to aggregate all results into one, adequate to the given entity type. Contextual String Embeddings (“Flair”) (Akbik et al., 2018), BERT-base (Devlin et al., 2018) and RoBERTa-base (Liu et al., 2019b) models are used for this. Moreover, we tried different PDF processing tools for text extraction from PDF documents to check importance of text quality to final score of the system. 

## **4.1 Pipeline** 

The core idea of this method is to select specific parts of the text in a document that denote the objects that we are looking for. The whole process is presented in Fig. 3 with the following stages: 

1. **Auto-tagging** : this stage involves extracting all the fragments that refer to the same or different entities by using sets of regular expressions combined with a gold-standard value for each general entity type (date, organization, amount, etc.), e.g. when we try to detect a report_date entity, we must handle different date formats: ‘November 29, 2019’, ‘11/29/19’ or ‘11-29-2019’. This step is performed only during training (to get data on which a NER model can be trained). 

2. **Named Entity Recognition** : using the auto-tagged dataset, we train a NER model[8] and then, at the evaluation stage, we use it for the detection of all occurrences of entities in the text being processed. 

3. **Normalization** : at this stage objects are normalized to the canonical form which we have defined in the _Kleister_ datasets. We use almost the same regular expression as during auto-tagging, e.g. all detected report_date occurrences are normalized from: ‘November 29, 2019’, ‘11/29/19’ and ‘11-29-2019’ into ‘2019-11-29’. 

4. **Aggregation** : we produce a single output from multiple candidates detected by the NER model. In our case, the technique is simple: we return the object with the maximum summarized scores grouped by the normalized forms of the extracted entities. 

Certainly, almost each stage of the above process can be done with a wide range of techniques, from regular expressions to more advanced machine learning models and deep neural networks. 

> 8Note that this is not a general NER model for addresses, dates, amounts, etc., but rather for more specific data-point types: charity addresses, report dates, incomes, etc. 

## **4.1.1 Pipeline based on Flair** 

The Flair model (based on stacked char-Bi-LSTM language model and GloVe word embeddings (Pennington et al., 2014)) is used as an encoder and Bi-LSTM with a CRF layer—as an output decoder.[9] Based on many experiments on the NDA and Charity development datasets, we found out the best setup for parameters, which is: _learning rate_ = 0 _._ 1, _batch size_ = 32, _hidden size_ = 256, _epoch_ = 30 _/_ 15 (NDA/Charity), _patience_ = 3, _anneal factor_ = 0 _._ 5 and with a CRF layer on the top. Moreover, each document was split into chunks of 300 words with overlapping parts of 15 words. Results from overlapping parts were normalized into one by using the mean of probabilities for each word from both overlapping parts. 

## **4.1.2 Pipeline based on BERT/RoBERTa** 

The BERT/RoBERTa models are fine-tuning approaches based on the Bidirectional Encoder Representations from Transformers language model.[10] We found out the best experimental setup, which is: _learning rate_ = 2 _e −_ 5, _batch size_ = 8, _epoch_ = 20, _patience_ = 2 after many experiments on the NDA and Charity datasets for both models. Moreover, each document was split into chunks of 510 tokens (plus two special tokens: [CLS] and [SEP]) with overlapping parts of 100 tokens. Results from overlapping parts were normalized into one by using the mean of probabilities for each token from both overlapping parts. 

## **4.2 PDF processing tools** 

PDF documents are input for Kleister challenges, from which we must extract text for further processing. Thus, an important role in whole system and final score of the _Pipeline_ approach will be accuracy of the tool that we use. Therefore, we checked performance on three PDF processing tools and one combination of them: 

1. Tesseract (Smith, 2020) in version 4.1.1-rc1-7-gb36c[11] . This is the most popular free OCR engine currently available. 

2. Textract with API version from March 1, 2020[12] . One of the most recognizable OCR tools and an open-source competitor of Tesseract. 

3. pdf2djvu/djvu2hocr in version 0.9.8[13] (later we will call that method _pdf2djvu_ ). Free tool for object extraction (and text extraction) from born-digital PDF files. 

4. pdf2djvu+Tesseract is a combination of pdf2djvu/djvu2hocr and Tesseract. Documents are processed with both tools, by default we take the text from pdf2djvu/djvu2hocr, unless the text returned by Tesseract is 1000 characters longer. This is a simple and efficient way to merge PDF processing solutions for extracting text from scans and born-digital PDF files. 

## **4.3 Results** 

## **4.3.1 Text extraction methods** 

Results of the performance of different PDF processing tools are presented in Table 4. The general conclusion is that by using software with the best text extraction methods we could achieve much better results, but we still cannot resolve all problems connected to information extraction task. 

The best tool for born-digital documents in _Kleister-NDA_ challenge for all models is a pdf2djvu tool. We should expect it, thus that documents are without any text errors normally caused by using OCR engines which are not perfect yet. For _Kleister-Charity_ challenge the best tool is Textract with huge advantage on all models, especially flair based. In the next sub-section, we describe in detail baseline results based on the most accurate PDF processing tool for each Kleister task. 

> 9We used implementation from the Flair library (Akbik et al., 2018) in version 0.4.5. 

> 10We used implementation from _pytorch-transformers_ (Hugging Face, 2019) library. 

> 11We ran it with --oem 2 -l eng --dpi 300 flags (meaning both new and old OCR engines were used simultaneously, and language and pixel density were forced for better results) 

> 12https://aws.amazon.com/textract/ 

> 13 http://jwilk.net/software/pdf2djvu, https://github.com/jwilk/ocrodjvu 

Table 4: Performance of different PDF processing tools checked on _Kleister_ challenges test-sets. All results are f-scores over 3 runs with standard deviation. (*) pdf2djvu does not work on scans, so we have empty 54/24/21 documents in train/dev/test sets. 

|_Kleister-NDA_dataset (born-digital PDF fles)|_Kleister-NDA_dataset (born-digital PDF fles)|_Kleister-NDA_dataset (born-digital PDF fles)|
|---|---|---|
|PDF processing tool<br>**Flair**<br>**BERT-base**<br>**RoBERTa-base**|||
||||
|pdf2djvu<br>**77.70**_±_**0.01**<br>Tesseract<br>75.17_±_0.12<br>Textract<br>75.63_±_0.17|**72.17**_±_**1.07**<br>68.30_±_0.41<br>70.43_±_1.11|**77.07**_±_**1.61**<br>74.20_±_1.90<br>76.20_±_0.64|
|_Kleister-Charity_dataset (mixture of born-digital and scans PDF fles)|||
||||
|pdf2djvu (*)<br>71.80_±_0.35<br>Tesseract<br>72.87_±_0.81<br>Textract<br>**80.10**_±_**0.35**<br>pdf2djvu+Tesseract<br>74.00_±_1.28|67.53_±_0.71<br>71.37_±_1.25<br>**73.30**_±_**0.43**<br>70.47_±_0.26|72.50_±_0.49<br>75.70_±_0.57<br>**79.87**_±_**0.65**<br>75.63_±_0.68|



## **4.3.2 NER models** 

The results for the two Kleister datasets obtained with the _Pipeline_ method based on Flair, BERT and RoBERTa models are shown in Table 5. The differences in F-score between Flair and RoBERTa models are not substantial in both challenges. Moreover, the RoBERTa model is much better as far as amounts (income_annually and spending_annually entities in Charity dataset) and organization names (party entity in NDA dataset and charity_name entity in Charity dataset) are concerned. 

The most challenging problems for all models are types related to the business reasoning (e.g. in NDA term or address__*), the visual features (e.g. in Charity income and spending) and finally hard normalization (e.g. in Charity charity_name). We can also observe that the entities appearing in the sequential contexts achieve a higher F-score. Moreover, after analyzing the model results, we prepared a list of common problems in models, which we grouped into specific problem categories (see Table 6). 

Table 5: Results of our baselines for _Kleister_ challenges test-sets. Results for all models are f-scores over 3 runs with standard deviation. Human baseline is a percentage of annotators agreements for 100 random documents. 

|Table 5: Results of our baselines for _Kleister_ challenges test-sets. Results for all models are f-scores<br>over 3 runs with standard deviation. Human baseline is a percentage of annotators agreements for 100<br>random documents.|Table 5: Results of our baselines for _Kleister_ challenges test-sets. Results for all models are f-scores<br>over 3 runs with standard deviation. Human baseline is a percentage of annotators agreements for 100<br>random documents.|Table 5: Results of our baselines for _Kleister_ challenges test-sets. Results for all models are f-scores<br>over 3 runs with standard deviation. Human baseline is a percentage of annotators agreements for 100<br>random documents.|Table 5: Results of our baselines for _Kleister_ challenges test-sets. Results for all models are f-scores<br>over 3 runs with standard deviation. Human baseline is a percentage of annotators agreements for 100<br>random documents.|
|---|---|---|---|
|_Kleister-NDA_dataset (pdf2djvu)||||
|Entity type<br>**Flair**<br>**BERT-base**<br>**RoBERTa-base**<br>**Human baseline**||||
|||||
|effective_date<br>**82.03**_±_**1.72**<br>party<br>70.13_±_0.11<br>jurisdiction<br>**93.80**_±_**0.42**<br>term<br>**60.82**_±_**26.7**|78.90_±_0.86<br>68.50_±_1.92<br>92.07_±_0.61<br>40.23_±_1.01|78.53_±_2.70<br>**78.47**_±_**0.58**<br>92.87_±_0.90<br>42.03_±_4.41|100 %<br>98 %<br>100 %<br>95 %|
|||||
|ALL<br>**77.70**_±_**0.01**|72.17_±_1.07|77.07_±_1.61|97.86%|
|_Kleister-Charity_dataset (Textract)||||
|||||
|address__post_town<br>**83.30**_±_**3.81**<br>address__postcode<br>**82.63**_±_**0.54**<br>address__street_line<br>**68.17**_±_**4.36**<br>charity_name<br>72.40_±_0.98<br>charity_number<br>**96.73**_±_**0.12**<br>income_annually<br>70.93_±_0.43<br>report_date<br>**95.67**_±_0.00<br>spending_annually<br>68.50_±_0.26|73.57_±_2.49<br>79.00_±_0.65<br>61.33_±_2.74<br>73.53_±_3.16<br>96.43_±_0.52<br>69.97_±_1.68<br>94.97_±_0.38<br>67.30_±_1.36|77.70_±_1.27<br>82.57_±_0.56<br>63.80_±_3.27<br>**76.87**_±_**0.37**<br>96.13_±_0.45<br>**73.20**_±_**0.64**<br>95.53_±_0.12<br>**71.27**_±_**0.84**|98 %<br>100 %<br>96 %<br>99 %<br>98 %<br>97 %<br>100 %<br>92 %|
|||||
|ALL<br>**80.10**_±_0.35|77.30_±_0.43|79.87_±_0.65|97.45 %|



## **5 Discussion and Challenges** 

In Table 1, we gathered the most important information about open datasets, especially we outlined the difference between our datasets and other sets. Additionally, we prepared descriptions of problems related to Kleister tasks (see Table 6). Thus, the _Kleister_ datasets appear to be more focused on the 

real life examples, where layout, document-level context, OCR quality, business logic and normalization problems need to be resolved for obtaining good results. 

Summing up, the proposed datasets are useful for testing real life applications to solve the challenge of the robotic process automation tackled by machine learning techniques. 

Table 6: Common problems in _Kleister_ datasets with examples. 

||Table 6: Commonproblems in_Kleister_datasets with examples.|
|---|---|
||**Normalization**: Differences in the way entities are given in expected values and documents.|
|_NDA_|effective_date: October 24, 2012, 10/24/12 or 24th day of October, 2012|
||term: 2 years, 24 months, two (2) years, two years or second anniversary|
|_Charity_|charity_name: 1. Ltd vs Limited: King’s Schools Taunton LTD [expected] vs King’s Schools Taunton|
||Limited [document]; 2. The vs non-The: The League of Friends of the Exmouth Hospital [expected] vs|
||League of Friends of Exmouth Hospital [document]|
||**Layout**: understand complex layout properly|
|_NDA_|all entities: four types of layout: 1. Simple layout (one column), 2. Simple layout (two columns),|
||3. E-mail, 4. Plain text. See Fig. 4 in the Appendix.|
|_Charity_|all entities: three types of layouts: 1. Simple document, 2. Report with tables, graphic elements|
||and pictures, 3. Form. See Fig. 1.|
||**Document-level context**: understand document as a whole|
|_NDA_|term: The term informs about the duration of the contract. Information on this is generally found in the|
||“Term” chapter. However, this section may also include other periods of validity of certain provisions of|
||the contract.|
||Example: “Term. This Agreement will be effective for a period of one (1) year after the Effective Date.|
||The restrictions on use and disclosure of the Discloser’s Confdential Information by the Recipient shall|
||survive any expiration or termination of this Agreement and shall continue in full force and effect for a|
||period of fve (5) years thereafter.”|
|_Charity_|income_annually,spending_annually: Co-occurrence of exact and rounded values in one doc-|
||ument. See Fig. 7 in the Appendix|
||**Business logic**: apply some rules in a case of ambiguity|
|_NDA_|term: Co-occurrence of two terms in one document. In such a case, the one constituting the duration of|
||the renewed contract was considered inappropriate.|
||Example: ‘Term; Termination. The term of the employment agreement set forth in this shall be for a|
||period commencing at the Effective Date and continuing for three (3) years thereafter (the “Scheduled|
||Term”). Following the Scheduled Term, the Agreement shall automatically renew for successive one-year|
||terms (each a “Renewal Term”).’|
|_Charity_|address__*: Co-occurrence of different addresses (e.g. Principal address, Registered offce, Adminis-|
||trative address, etc.) next to each other in one document, or the lack of a clear identifcation of the charity’s|
||address. In such a case the Registered address was considered to be the main one. See Fig. 2.|
||**OCR quality**: process scan documents|
|_NDA_|N/A — born-digital documents.|
|_Charity_|all entities: Handwriting in the document, pages upside down or poor scan quality.|



As described above, working with the proposed datasets can be compared to challenges dealing with Information Retrieval and Natural Language Understanding, including challenges related to page layout understanding (i.e. tables, rich graphics, etc.). To solve these challenges, we presented the _Pipeline_ approach that will help to deal with specific problems. 

Most of these stages are described in the process of building baselines and are shown in Fig. 3. 

Using the presented challenges we are also able to study the impact of each stage of the full process on the final results. It is useful in the production environment where we can have a baseline, and then we can assess what should be done with the highest priority to improve final results. 

## **6 Conclusions** 

_Kleister_ datasets have been prepared to challenge the business usability of Information Extraction models and processes. In this article, we described in detail how they were prepared (i.e. manually or automatically — for _Kleister-NDA_ and _Kleister-Charity_ respectively). Due to their multi-modal nature, we had to face various problems and needed to develop methods to improve the quality of data sets. 

We consider our datasets and tasks will help the community to extend the understanding of documents with substantial length, various reasoning problems, complex layouts and OCR quality problems. Moreover, the community can use our methodology to extend the datasets or prepare similar sets. 

In addition, we prepared baseline solutions on the basis of text data generated by different PDF processing tools from the datasets (see Table 4). This benchmark shows weakness of the models working on a pure text (i.e. input is a sequence of words) without using any visual information. 

## **References** 

- Alan Akbik, Duncan Blythe, and Roland Vollgraf. 2018. Contextual string embeddings for sequence labeling. In _Proceedings of the 27th International Conference on Computational Linguistics_ , pages 1638–1649, Santa Fe, New Mexico, USA, August. Association for Computational Linguistics. 

- Timo I. Denk and Christian Reisswig. 2019. {BERT}grid: Contextualized embedding for 2d document representation and understanding. In _Workshop on Document Intelligence at NeurIPS 2019_ . 

- Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. BERT: pre-training of deep bidirectional transformers for language understanding. _CoRR_ , abs/1810.04805. 

- Jean-Philippe Thiran Guillaume Jaume, Hazim Kemal Ekenel. 2019. FUNSD: A Dataset for Form Understanding in Noisy Scanned Documents. In _Accepted to ICDAR-OST_ . 

- Daniel Hewlett, Alexandre Lacoste, Llion Jones, Illia Polosukhin, Andrew Fandrianto, Jay Han, Matthew Kelcey, and David Berthelot. 2016. WikiReading: A novel large-scale language understanding task over Wikipedia. In _Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 1535–1545, Berlin, Germany, August. Association for Computational Linguistics. 

- Xavier Holt and Andrew Chisholm. 2018. Extracting structured data from invoices. In _Proceedings of the Australasian Language Technology Association Workshop 2018_ , pages 53–59, Dunedin, New Zealand, December. 

- Hugging Face. 2019. Transformers. https://github.com/huggingface/transformers (accessed November 27, 2019). 

- Anoop R. Katti, Christian Reisswig, Cordula Guder, Sebastian Brarda, Steffen Bickel, Johannes Höhne, and Jean Baptiste Faddoul. 2018. Chargrid: Towards Understanding 2D Documents. _CoRR_ , abs/1809.08799. 

- Xiaojing Liu, Feiyu Gao, Qiong Zhang, and Huasha Zhao. 2019a. Graph convolution for multimodal information extraction from visually rich documents. _CoRR_ , abs/1903.11279. 

- Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019b. Roberta: A robustly optimized bert pretraining approach. _ArXiv_ , abs/1907.11692. 

- Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. GloVe: Global vectors for word representation. In _In EMNLP_ . 

- Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, and Luke Zettlemoyer. 2018. Deep contextualized word representations. _CoRR_ , abs/1802.05365. 

- Ritesh Sarkhel and Arnab Nandi. 2019. Visual segmentation for information extraction from heterogeneous visually rich documents. In _Proceedings of the 2019 International Conference on Management of Data_ , SIGMOD ’19, pages 247–262, New York, NY, USA. ACM. 

- Ray Smith. 2020. Tesseract Open Source OCR Engine. 

- Vishal Sunder, Ashwin Srinivasan, Lovekesh Vig, Gautam Shroff, and Rohit Rahul. 2019. One-shot information extraction from document images using neuro-deductive program synthesis. _CoRR_ , abs/1906.02427. 

- Anna Wróblewska, Tomasz Stanisławek, Bartłomiej Prus-Zaj ˛aczkowski, and Łukasz Garncarek. 2018. Robotic process automation of unstructured data with machine learning. In _Position Papers of the 2018 Federated Conference on Computer Science and Information Systems, FedCSIS 2018, Pozna´n, Poland, September 9-12, 2018_ , pages 9–16. 

- Xiaohui Zhao, Zhuo Wu, and Xiaoguang Wang. 2019. CUTIE: learning to understand documents with convolutional universal text information extractor. _CoRR_ , abs/1903.12363. 

## **Appendix** 

In this supplement we describe more precisely our datasets and the annotation processes in Section A and Section B, respectively. 

## **A NDA Dataset** 

## **A.1 Data Detailed Description** 

The NDA agreements prevent the disclosure of confidential information by one of the parties to a third party. Such agreements, even in oral form, are often found in everyday life (e.g. in the patient-doctor relationship). In business, they usually have a written form, signed by a representative of the legal profession and another person (legal or natural). In our database, we have collected business contracts, but without differentiating them, either by their form (these are both independent contracts and contracts annexed to other contracts), or by the way they were concluded (all contracts were concluded in writing, some of them by e-mail) or because of the number of parties (the dataset contains unilateral, bilateral and multilateral agreements). 

The NDAs can take various forms (contract attachments, emails, etc.), but they all generally have a similar structure. First, the circumstances of the contract are determined, i.e. the parties to the contract are presented and the date from which the contract becomes effective is provided. Then they usually contain the following elements: 

- a definition of confidential information, including exceptions to this definition; 

- description of the disclosure procedure (also during court and administrative proceedings); 

- procedures related to non-compliance with confidentiality obligations; 

- term of the contract (termination date); 

- the period during which the information remains confidential (confidential period); 

- information about the jurisdiction to which the contract is subject; 

- information about the possibility of making legally binding copies of the contract; 

- due to the fact that confidential information can be used to recruit new employees or contractors of one party by another, the NDA often also includes non-compete clauses in force for a certain period of time. 

## **A.2 Data Collection Method** 

During the collection of the NDAs, we focused on contracts concluded by public companies in the United States. All public companies (i.e. those with shareholders) in the US are supervised by the United States Securities and Exchange Commission (SEC). Companies are required to submit a number of reports and forms, the attachments of which are often contracts concluded by these companies, including NDAs. This is done through the Electronic Data Gathering, Analysis and Retrieval system (EDGAR), which is also a public database of these documents (these documents must be made public)[14] . As a result, EDGAR is a huge NDA base. Unfortunately, NDAs are usually attachments to other contracts or forms submitted to EDGAR, as a result of which it is not possible to simply aggregate them from this database. Thus, the process of gathering the dataset had to be manual, with a weak model supervision. 

The NDAs were collected with the help of the Google search engine. Two collections were created—the first contained 170 contracts and the second 330 contracts, except that 117 duplicates were found, so that ultimately the dataset counted a total of 383 documents. After the first tests on the already annotated dataset, it turned out that machine learning models achieve quite poor results for information 

14https://www.sec.gov/edgar.shtml 

Figure 4: Four different examples of layouts from the _NDA-charity_ dataset. 

on jurisdiction. Analysis of the dataset showed that this was due to the under-representation of documents that were prepared in accordance with non-US law (e.g. China, India or Israel). Since no more such documents were obtained, the 68 previously obtained ones were removed from the dataset, which reduced it to 315 documents. In the next step, the collection was supplemented with an additional 127 documents consistent with the others in terms of applicable law (i.e. US law). 

The original files were HTML documents, but they were transformed into PDF files to keep processing 

simple and similar to how other datasets were created. Transformation was made using the puppeteer library, which in turn used the “Print to PDF” functionality present in the Chrome web browser. Subsequently, the transformed PDFs were processed with the Tesseract OCR engine. 

## **A.3 Annotation Procedure** 

The whole dataset was annotated in two ways. Its first part, i.e. 315 documents, was annotated by linguists, except that only selected contexts, preselected by an in-house system based on semantic similarity, were taken into account (to make the annotation easier and faster). The second, i.e. 127 documents, was entirely annotated by hand. When preparing the dataset, we wanted to find out if the semantic similarity methods could be used to limit the time it would take to perform annotation procedures (this solution saved about 50% of the time compared to fully manual annotation). 

The annotation of the dataset consisted of listing the extracted entities. The entities themselves may appear repeatedly in the document, but this did not matter for the annotation procedure (contrary to NER, we are not interested in the exact location(s) of an entity). The following entities have been normalized according to standards adopted by us: (a) parties — commas have been removed before acronyms referring to organization types, and the format has been unified, e.g. _LHA LONDON LTD_ ; (b) effective date — the format has been standardized according to ISO 8601, i.e. YYYY-MM-DD; (c) terms — standardized to the following format: number of units followed by a unit, e.g. _2 years_ ; (d) jurisdiction and counterparts did not require standardization. Then the annotations were checked by the super-annotator on 45 random documents (10% of the whole dataset). All the super-annotated entities were correct and did not need to be changed. 

## **B Charity Dataset** 

## **B.1 Data Detailed Description** 

There is no rule about how such a charity report should look. Therefore, some take the form of reports richly illustrated with photos and charts, where financial information constitutes a small part of the entire report, while others have only a few pages, where only basic data on revenues and expenses in a given calendar year are given (see Figure 5). However, each of these reports should contain at least the following information (although there may be exceptions to this rule): 

- organization’s address, name and number; 

- the date of submission of the report; 

- total income in the reporting year; 

- total expenditure in the reporting year. 

## **B.2 Data Collection Method** 

The decision to create a dataset from the financial reports of British charities was driven by the following goal: to find a publicly available collection of English-language and multi-page documents on the Internet, which would be accompanied by easy-to-extract information about data contained in these documents (e.g. as a separate XML file or a table on a website). We decided that the database of financial reports of British charity organizations would be the best of all the options considered. It is not just that the Charity Commission website actually has a database of all the charity organizations registered in England and Wales, but also that each of these organizations has a separate subpage on the Commission’s website and it is easy to find the most important information about them (see Fig. 5): 

- Charity’s name and number; 

- main activities; 

- current address parts (post town, postcode and street line); 

Figure 5: Organization’s page on the Charity Commission’s website (left: organization whose annual income is between 25k and 500k GBP, right: over 500k). Information on the website has a different layout, and within documents there is also the case. Entities are underlined in red and names of entities are circled. 

- a list of the current trustees of the organization; 

- basic financial data for the past year, i.e. income and expenditure (these data are more detailed in the case of organizations with revenues of over 500,000 GBP a year); 

- the date of submission of the report. 

This information partly overlaps with what the reports actually contain (although it might happen that some entities are not to be found in the reports, e.g. a list of trustees is given on the website, but it does not have to be included in the report). For this reason, we decided to extract only those entities which also appear in the form of a brief description on the website. 

The reports can be found on the website as PDF files (but this does not apply to organizations with income below 25,000 GBP a year, as they are required to submit a condoned financial report). Therefore, the information available on the website and the documents attached to it made the database of these documents perfectly fit the objectives outlined above. In this way, 3414 documents were obtained. 

During the analysis of the documents, it turned out that several reports are in Welsh. As we are interested in the English language only, all documents in other languages were found and removed from the collection. In addition, documents, that contained reports for more than one organization, were handwritten, or the quality of their OCR was low, were deleted. As a result, the collection has 2778 documents. 

## **B.3 Annotation Procedure** 

There was no need to manually annotate documents, because basic information about the reporting organizations could be obtained directly from the website where these documents were located. 

Only a random sample of 100 documents was manually checked (see Table 7). The permissible error limit for a given entity was set at 15%. These results were exceeded for charity name (18% of errors and minor differences) and for charity address (76% of errors and minor differences). However, as a result of detailed analysis, it turned out that there are few erroneous entities (respectively 5% and 9%), while the rest is rather due to differences in the way the data is presented on the page and in the document. These minor differences have been corrected manually and automatically, as described below. 

Table 7: Comparison of data on the Charity Commission’s website and in charity reports. 

|Entities|Correct|Minor|Error|
|---|---|---|---|
||[%]|differ-|[%]|
|||ences||
|||[%]||
|charity_name|82|13|5|
|charity_address|24|67|9|
|charity_number|98|0|2|
|report_date|99|0|1|
|income_annually|86|315|11|
|spending_annually|86|316|11|



Hence, the charity’s name on the website and in the documents could be noted once with the term _Limited_ (shortened to _LTD_ ), and once not. This problem was eliminated by the manual annotation of all documents in which the name of the charity organization co-occurred with the word _Limited_ or _LTD_ . As a result, 366 documents were analyzed manually in this way. 

In the case of the charity’s address the most problematic were the names of counties, districts as well as the names of towns and cities, which were once specified on the website, but not in the documents, other times—the other way round. This problem was solved by splitting address data into the three separate entities that we considered the most important—postcode, postal town name and street or road name. The postal code was used as the key element of the address, on the basis of which the city name and street name could be determined[17] . 

Other problems show Fig. 6 and 7. On the first of them we have two different values for income_annually and spending_annually, because the values in the table are rounded and in the text are accurate. In the second picture there is no total for all expenses, so we can not extract the value for spending_annually. 

Figure 6: No value for spending_annually. 

> 15Including two cases of non-rounding of the amount and one filling in the amount in USD instead of GBP. 16As above. 

> 17Postal codes in the UK were aggregated from a website: streetlist.co.uk 

Figure 7: Different values for income_annually and spending_annually. 

