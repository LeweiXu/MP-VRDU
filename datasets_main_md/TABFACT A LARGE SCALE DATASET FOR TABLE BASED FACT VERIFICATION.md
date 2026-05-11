Published as a conference paper at ICLR 2020 

## - - TABFACT: A LARGE SCALE DATASET FOR TABLE BASED FACT VERIFICATION 

**Wenhu Chen, Hongmin Wang, Jianshu Chen, Yunkai Zhang, Hong Wang, Shiyang Li, Xiyou Zhou, William Yang Wang** University of California, Santa Barbara, CA, USA Tencent AI Lab, Bellevue, WA, USA 

_{_ wenhuchen,hongmin ~~w~~ ang,yunkai ~~z~~ hang,hongwang600,william _}_ @ucsb.edu 

_{_ shiyangli,xiyou _}_ @cs.ucsb.edu jianshuchen@tencent.com 

## ABSTRACT 

The problem of verifying whether a textual hypothesis holds based on the given evidence, also known as fact verification, plays an important role in the study of natural language understanding and semantic representation. However, existing studies are mainly restricted to dealing with unstructured evidence (e.g., natural language sentences and documents, news, etc), while verification under structured evidence, such as tables, graphs, and databases, remains under-explored. This paper specifically aims to study the fact verification given semi-structured data as evidence. To this end, we construct a large-scale dataset called TabFact with 16k Wikipedia tables as the evidence for 118k human-annotated natural language statements, which are labeled as either ENTAILED or REFUTED. TabFact is challenging since it involves both soft linguistic reasoning and hard symbolic reasoning. To address these reasoning challenges, we design two different models: Table-BERT and Latent Program Algorithm (LPA). Table-BERT leverages the state-of-the-art pre-trained language model to encode the linearized tables and statements into continuous vectors for verification. LPA parses statements into programs and executes them against the tables to obtain the returned binary value for verification. Both methods achieve similar accuracy but still lag far behind human performance. We also perform a comprehensive analysis to demonstrate great future opportunities. The data and code of the dataset are provided in https://github.com/wenhuchen/Table-Fact-Checking. 

## 1 INTRODUCTION 

Verifying whether a textual hypothesis is entailed or refuted by the given evidence is a fundamental problem in natural language understanding (Katz & Fodor, 1963; Van Benthem et al., 2008). It can benefit many downstream applications like misinformation detection, fake news detection, etc. Recently, the first-ever end-to-end fact-checking system has been designed and proposed in Hassan et al. (2017). The verification problem has been extensively studied under different natural language tasks such as recognizing textual entailment (RTE) (Dagan et al., 2005), natural language inference (NLI) (Bowman et al., 2015), claim verification (Popat et al., 2017; Hanselowski et al., 2018; Thorne et al., 2018) and multimodal language reasoning (NLVR/NLVR2) (Suhr et al., 2017; 2019). RTE and NLI view a premise sentence as the evidence, claim verification views passage collection like Wikipedia[1] as the evidence, NLVR/NLVR2 views images as the evidence. These problems have been previously addressed using a variety of techniques including logic rules, knowledge bases, and neural networks. Recently large-scale pre-trained language models (Devlin et al., 2019; Peters et al., 2018; Yang et al., 2019; Liu et al., 2019) have surged to dominate the other algorithms to approach human performance on several textual entailment tasks (Wang et al., 2018; 2019). 

However, existing studies are restricted to dealing with unstructured text as the evidence, which would not generalize to the cases where the evidence has a highly structured format. Since such structured evidence (graphs, tables, or databases) are also ubiquitous in real-world applications like 

> 1https://www.wikipedia.org/ 

1 

Published as a conference paper at ICLR 2020 

**==> picture [377 x 122] intentionally omitted <==**

**----- Start of picture text -----**<br>
United States House of Representatives Elections, 1972<br>District Incumbent Party Result Candidates<br>California 3 John E. Moss democratic re-elected John E. Moss  (d) 69.9% John Rakus (r) 30.1%<br>California 5 Phillip Burton democratic re-elected Phillip Burton (d) 81.8% Edlo E. Powell (r) 18.2%<br>California 8 George Paul Miller democratic lost renomination democratic hold Pete Stark (d) 52.9% Lew M. Warden , Jr. (r) 47.1%<br>California 14 Jerome R. Waldie republican re-elected Jerome R. Waldie (d) 77.6% Floyd E. Sims (r) 22.4%<br>California 15 John J. Mcfall republican re-elected John J. Mcfall (d) unopposed<br>Entailed Statement Refuted Statement<br>1. John E. Moss and Phillip Burton are both re-elected in the  1. John E. Moss and George Paul Miller are both re-elected in the house<br>of representative election.<br>2.3. house of representative election.John J. Mcfall is unopposed during the re-election.There are three different incumbents from democratic. 2.3. John J. McfallThere are five candidates in total failed to be re-elected , two of them though being unopposed.are democrats and<br>three of them are republicans.<br>**----- End of picture text -----**<br>


Figure 1: Examples from the TABFACT dataset. The top table contains the semi-structured knowledge facts with caption ”United...”. The left and right boxes below provide several entailed and refuted statements. The error parts are highlighted with red font. 

database systems, dialog systems, commercial management systems, social networks, etc, we argue that the fact verification under structured evidence forms is an equivalently important yet underexplored problem. Therefore, in this paper, we are specifically interested in studying fact verification with semi-structured Wikipedia tables (Bhagavatula et al., 2013)[2] as evidence owing to its structured and ubiquitous nature (Jauhar et al., 2016; Zhong et al., 2017; Pasupat & Liang, 2015). To this end, we introduce a large-scale dataset called TABFACT, which consists of 118K manually annotated statements with regard to 16K Wikipedia tables, their relations are classified as ENTAILED and REFUTED[3] . The entailed and refuted statements are both annotated by human workers. With some examples in Figure 1, we can clearly observe that unlike the previous verification related problems, TABFACT combines two different forms of reasoning in the statements, (i) _Linguistic Reasoning_ : the verification requires semantic-level understanding. For example, “John J. Mcfall failed to be re-elected though being unopposed.” requires understanding over the phrase “lost renomination ...” in the table to correctly classify the entailment relation. Unlike the existing QA datasets (Zhong et al., 2017; Pasupat & Liang, 2015), where the linguistic reasoning is dominated by paraphrasing, TABFACT requires more linguistic inference or common sense. (ii) _Symbolic Reasoning_ : the verification requires symbolic execution on the table structure. For example, the phrase “There are three Democrats incumbents” requires both condition operation (where condition) and arithmetic operation (count). Unlike question answering, a statement could contain compound facts, all of these facts need to be verified to predict the verdict. For example, the ”There are ...” in Figure 1 requires verifying three QA pairs (total count=5, democratic count=2, republic count=3). The two forms of reasoning are interleaved across the statements making it challenging for existing models. 

In this paper, we particularly propose two approaches to deal with such mixed-reasoning challenge: (i) _Table-BERT_ , this model views the verification task completely as an NLI problem by linearizing a table as a premise sentence _p_ , and applies state-of-the-art language understanding pre-trained model to encode both the table and statements _h_ into distributed representation for classification. This model excels at linguistic reasoning like paraphrasing and inference but lacks symbolic reasoning skills. (ii) _Latent Program Algorithm_ , this model applies lexical matching to find linked entities and triggers to filter pre-defined APIs (e.g. argmax, argmin, count, etc). We adopt bread-first-search with memorization to construct the potential program candidates, a discriminator is further utilized to select the most “consistent” latent programs. This model excels at the symbolic reasoning aspects by executing database queries, which also provides better interpretability by laying out the decision rationale. We perform extensive experiments to investigate their performances: the best-achieved accuracy of both models are reasonable, but far below human performance. Thus, we believe that the proposed table-based fact verification task can serve as an important new benchmark towards the goal of building powerful AI that can reason over both soft linguistic form and hard symbolic forms. To facilitate future research, we released all the data, code with the intermediate results. 

> 2In contrast to the database tables, where each column has strong type constraint, the cell records in our semi-structured tables can be string/data/integer/floating/phrase/sentences. 

> 3we leave out NEUTRAL due to its low inter-worker agreement, which is easily confused with REFUTED. 

2 

Published as a conference paper at ICLR 2020 

## 2 TABLE FACT VERIFICATION DATASET 

First, we follow the previous Table-based Q&A datasets (Pasupat & Liang, 2015; Zhong et al., 2017) to extract web tables (Bhagavatula et al., 2013) with captions from WikiTables[4] . Here we filter out overly complicated and huge tables (e.g. multirows, multicolumns, latex symbol) and obtain 18K relatively clean tables with less than 50 rows and 10 columns. 

For crowd-sourcing jobs, we follow the human subject research protocols[5] to pay Amazon Mechanical Turk[6] workers from the native English-speaking countries “US, GB, NZ, CA, AU” with approval rates higher than 95% and more than 500 accepted HITs. Following WikiTableQuestion (Pasupat & Liang, 2015), we provide the annotators with the corresponding table captions to help them better understand the background. To ensure the annotation quality, we develop a pipeline of “positive two-channel annotation” _→_ “negative statement rewriting” _→_ “verification”, as described below. 

## 2.1 POSITIVE TWO-CHANNEL COLLECTION & NEGATIVE REWRITING STRATEGY 

To harvest statements of different difficulty levels, we design a two-channel collection process: **Low-Reward Simple Channel** : the workers are paid 0.45 USD for annotating one Human Intelligent Task (HIT) that requires writing five statements. The workers are encouraged to produce plain statements meeting the requirements: (i) corresponding to a single row/record in the table with unary fact without involving compound logical inference. (ii) mention the cell values without dramatic modification or paraphrasing. The average annotation time of a HIT is 4.2 min. 

**High-Reward Complex Channel** : the workers are paid 0.75 USD for annotating a HIT (five statements). They are guided to produce more sophisticated statements to meet the requirements: (i) involving multiple rows in the tables with higher-order semantics like argmax, argmin, count, difference, average, summarize, etc. (ii) rephrase the table records to involve more semantic understanding. The average annotation time of a HIT is 6.8 min. The data obtained from the complex channel are harder in terms of both linguistic and symbolic reasoning, the goal of the two-channel split is to help us understand the proposed models can reach under different levels of difficulty. 

As suggested in (Zellers et al., 2018), there might be annotation artifacts and conditional stylistic patterns such as length and word-preference biases, which can allow shallow models (e.g. bag-ofwords) to obtain artificially high performance. Therefore, we design a negative rewriting strategy to minimize such linguistic cues or patterns. Instead of letting the annotators write negative statements from scratch, we let them rewrite the collected entailed statements. During the annotation, the workers are explicitly guided to modify the words, phrases or sentence structures but retain the sentence style/length to prevent artificial cues. We disallow naive negations by adding “not, never, etc” to revert the statement polarity in case of obvious linguistic patterns. 

## 2.2 QUALITY CONTROL 

To control the quality of the annotation process, we review a randomly sampled statement from each HIT to decide whether the whole annotation job should be rejected during the annotation process. Specifically, a HIT must satisfy the following criteria to be accepted: (i) the statements should contain neither typos nor grammatical errors. (ii) the statements do not contain vague claims like might, few, etc. (iii) the claims should be explicitly supported or contradicted by the table without requiring the additional knowledge, no middle ground is permitted. After the data collection, we re-distribute all the annotated samples to further filter erroneous statements, the workers are paid 0.05 USD per statement to decide whether the statement should be rejected. The criteria we apply are similar: no ambiguity, no typos, explicitly supported or contradictory. Through the post-filtering process, roughly 18% entailed and 27% refuted instances are further abandoned due to poor quality. 

> 4http://websail-fe.cs.northwestern.edu/wikiTables/about/ 

> 5https://en.wikipedia.org/wiki/Minimum_wage_in_the_United_States 

> 6https://www.mturk.com/ 

3 

Published as a conference paper at ICLR 2020 

**==> picture [357 x 84] intentionally omitted <==**

**----- Start of picture text -----**<br>
Proportion of different Higher-order Operations<br>40<br>Simple Complex Overall<br>20<br>0<br>AGGREGATION NEGATE SUPERLATIVE COUNT COMPATIVE ORDINAL UNIQUE ALL<br>Percent<br>**----- End of picture text -----**<br>


Figure 2: Proportion of different higher-order operations from the simple/complex channels. 

|Channel<br>#Sentence<br>#Table<br>Len(Ent)<br>Len(Ref)|Split<br>#Sentence<br>Table<br>Row<br>Col|
|---|---|
|||
|Simple<br>50,244<br>9,189<br>13.2<br>13.1<br>Complex<br>68,031<br>7,392<br>14.2<br>14.2<br>Total<br>118,275<br>16,573<br>13.8<br>13.8|Train<br>92,283<br>13,182<br>14.1<br>5.5<br>Val<br>12,792<br>1,696<br>14.0<br>5.4<br>Test<br>12,779<br>1,695<br>14.2<br>5.4|



Table 1: Basic statistics of the data collected from the simple/complex channel and the division of Train/Val/Test Split in the dataset, where “Len” denotes the averaged sentence length. 

2.3 DATASET STATISTICS 

**Inter-Annotator Agreement** : After the data collection pipeline, we merged the instances from two different channels to obtain a diverse yet clean dataset for table-based fact verification. We sample 1000 annotated (table, statement) pairs and re-distribute each to 5 individual workers to re-label them aset al.,either2015)ENTAILEDto adopt theor REFUTEDFleiss Kappa. We (Fleiss,follow¯ the1971)previousas an indicator,works (Thornewhereet¯Fleissal., 2018; _κ_ = Bowman _p_ ¯1 _c−−pp_ ¯¯ _ee_[is] computed from from the observed agreement _pc_ and the agreement by chance _pe_ . We obtain a Fleiss _κ_ = 0 _._ 75, which indicates strong inter-annotator agreement and good-quality. 

**Dataset Statistics** : As shown in Table 1, the amount of data harvested via the complex channel slightly outnumbers the simple channel, the averaged length of both the positive and negative samples are indistinguishable. More specifically, to analyze to which extent the higher-order operations are included in two channels, we group the common higher-order operations into 8 different categories. As shown in Figure 2, we sample 200 sentences from two different channels to visualize their distribution. We can see that the complex channel overwhelms the simple channel in terms of the higher-order logic, among which, count and superlatives are the most frequent. We split the whole data roughly with 8:1:1 into train, validation[7] , and test splits and shows their statistics in Table 1. Each table with an average of 14 rows and 5-6 columns corresponds to 2-20 different statements, while each cell has an average of 2.1 words. In the training split, the positive instances slightly outnumber the negative instances, while the validation and test split both have rather balanced distributions over positive and negative instances. 

## 3 MODELS 

With the collected dataset, we now formally define the table-based fact verification task: the dataset is comprised of triple instances ( **T** _, S, L_ ) consisting of a table **T** , a natural language statement _S_ = _s_ 1 _, · · · , sn_ and a verification label _L ∈{_ 0 _,_ 1 _}_ . The table **T** = _{Ti,j|i ≤ RT , j ≤ CT }_ has _RT_ rows and _CT_ columns with the _Tij_ being the content in the ( _i, j_ )-th cell. _Tij_ could be a word, a number, a phrase, or even a natural language sentence. The statement S describes a fact to be verified against the content in the table **T** . If it is entailed by **T** , then _L_ = 1, otherwise the label _L_ = 0. Figure 1 shows some entailed and refuted examples. During training, the model and the learning algorithm are presented with _K_ instances like ( **T** _, S, L_ ) _[K] k_ =1[from the training split.][In the] testing stage, the model is presented with ( **T** _, S_ ) _[K] k_ =1 _[′]_[and][supposed][to][predict][the][label][as] _[L]_[ˆ][.][We] measure the performance by the prediction accuracy _Acc_ = _K_ 1 _[′]_ � _K_ 1 _[′]_ I( _L_[ˆ] _k_ = _Lk_ ) on the test set. Before building the model, we first perform entity linking to detect all the entities in the statements. Briefly, we first lemmatize the words and search for the longest sub-string matching pairs between statements and table cells/captions, where the matched phrases are denoted as the linked entities. To focus on statement verification against the table, we do not feed the caption to the model and simply 

> 7We filter roughly 400 sentences from abnormal tables including hyperlinks, math symbols, etc 

4 

Published as a conference paper at ICLR 2020 

mask the phrases in the statements which link to the caption with placeholders. The details of the entity linker are listed in the Appendix. We describe our two proposed models as follows. 

## 3.1 LATENT PROGRAM ALGORITHM (LPA) 

In this approach, we formulate the table fact verification as a program synthesis problem, where the latent program algorithm is not given in TABFACT. Thus, it can be seen as a weakly supervised learning problem as discussed in Liang et al. (2017); Lao et al. (2011). Under such a setting, we propose to break down the verification into two stages: (i) latent program search, (ii) discriminator ranking. In the first program synthesis step, we aim to parse the statement into programs to represent its semantics. We define the plausible API set to include roughly 50 different functions like _min, max, count, average, filter, and_ and realize their interpreter with Python-Pandas. Each API is defined to take arguments of specific types ( _number, string, bool, and view (e.g sub-table)_ ) to output specifictype variables. During the program execution, we store the generated intermediate variables to different-typed caches _N , R, B, V_ (Num, Str, Bool, View). At each execution step, the program can fetch the intermediate variable from the caches to achieve semantic compositionality. In order to shrink the search space, we follow NSM (Liang et al., 2017) to use trigger words to prune the API set and accelerate the search speed. The definitions of all API, trigger words can be found in the Appendix. The comprehensive the latent program search procedure is summarized in Algorithm 1, 

## **Algorithm 1** Latent Program Search with Comments 

1: Initialize Number Cache _N_ , String Cache _R_ , Bool Cache _B_ , View Cache _V →∅_ 2: Push linked numbers, strings from the given statement _S_ into _N , R_ , and push **T** into _V_ 3: Initialize the result collector _P →∅_ and an empty program trace _P_ = _∅_ 4: Initialize the Queue _Q_ = [( _P, N , R, B, V_ )], we use _Q_ to store the intermediate states 5: Use trigger words to find plausible function set _F_ , for example, _more_ will trigger _Greater_ function. 6: **while** loop over time _t_ = 1 _→_ MAXSTEP **do** : 7: **while** ( _P, N , R, B, V_ ) = _Q.pop_ () **do** : 8: **while** loop over function set _f ∈F_ **do** : 9: **if** arguments of _f_ are in the caches **then** 10: Pop out the required arguments _arg_ 1 _, arg_ 2 _, · · · , argn_ for different cachess. 11: Execute _A_ = _f_ ( _arg_ 1 _, · · · , argn_ ) and concatenate the program trace _P_ . 12: **if** Type(A)=Bool **then** 13: **if** _N_ = _S_ = _B_ = _∅_ **then** 14: _P.push_ (( _P, A_ )) # The program _P_ is valid since it consumes all the variables. 15: _P_ = _∅_ # Collect the valid program _P_ into set _P_ and reset _P_ 16: **else** 17: _B.push_ ( _A_ ) # The intermediate boolean value is added to the bool cache 18: _Q.push_ (( _P, N , R, B, V_ )) # Add the refreshed state to the queue again 19: **if** Type(A) _∈{_ Num, Str, View _}_ **then** 20: **if** _N_ = _S_ = _B_ = _∅_ **then** 21: _P_ = _∅_ ;break # The program ends without consuming the cache, throw it. 22: **else** 23: push _A_ into _N_ or _S_ or _V_ # Add the refreshed state to the queue for further search 24: _Q.push_ (( _P, N , R, B, V_ )) 25: Return the triple ( **T** _, S, P_ ) # Return (Table, Statement, Program Set) 

## and the searching procedure is illustrated in Figure 3. 

After we collected all the potential program candidates _P_ = _{_ ( _P_ 1 _, A_ 1) _, · · · ,_ ( _Pn, An_ ) _}_ for a given statement _S_ (where ( _Pi, Ai_ ) refers to _i_ -th candidate), we need to learn a discriminator to identify the “appropriate” traces from the set from many erroneous and spurious traces. Since we do not have the ground truth label about such discriminator, we use a weakly supervised training algorithm by viewing all the label-consistent programs as positive instances _{Pi|_ ( _Pi, Ai_ ); _Ai_ = _L}_ and the label-inconsistent program as negative instances _{Pi|_ ( _Pi, Ai_ ); _Ai_ = _L}_ to minimize the cross-entropy of discriminator _pθ_ ( _S, P_ ) with the weakly supervised label. Specifically, we build our discriminator with a Transformer-based two-way encoder (Vaswani et al., 2017), where the statement encoder encodes the input statement _S_ as a vector _Enc[S]_ ( _S_ ) _∈_ R _[n][×][D]_ with dimension _D_ , while the program encoder encodes the program _P_ = _p_ 1 _, · · · , pm_ as another vector _Enc[P]_ ( _P_ ) _∈_ R _[m][×][D]_ , we concatenate these two vectors and feed it into a linear projection layer 

5 

Published as a conference paper at ICLR 2020 

**==> picture [357 x 125] intentionally omitted <==**

**----- Start of picture text -----**<br>
There are more  democrats  than  republicans  in the election.<br>Feature-based Entity Linking String incumbentincumbent democraticrepublican pop<br>V1=Filter(T, incumbent==democratic)) View Sub V1 incumbent republican<br>V2=Filter(T, incumbent==republican)) View Sub V2 pop<br>Sub V1<br>LISP Engine Sub V1<br>3=Count(V1) Num Count 3 Sub V2<br>2=Count(V2) Num Count 2<br>Count 3 pop<br>Table<br>Greater(3, 2) Bool Bool True Entailed<br>Search<br>**----- End of picture text -----**<br>


Figure 3: The program synthesis procedure for the table in Figure 1. We link the entity (e.g. _democratic_ , _republican_ ), and then composite functions on the fly to return the values from the table. 

**==> picture [377 x 118] intentionally omitted <==**

**----- Start of picture text -----**<br>
Label Game Date Opponent Score<br>51 February 3 , 2009 Florida 3-4<br>12-Layer BERT-Base Model<br>52 February 4 , 2009 Buffalo 0-5<br>53 February 7 , 2010 Montreal 5-2<br>Word [CLS] 51 [SEP] February 3 , 2009 [SEP] Florida is playing [SEP]<br>Position 0 1 0 1 2 3 4 0 1 2 3 0<br>Type TOK game TOK date date date date TOK S S S TOK<br>Word [CLS] row one game is 51 ; date is February 3 2019 ; Florida [SEP]<br>Position 0 1 2 3 4 5 6 7 8 9 10 11 12 13 20<br>Concat<br>Template<br>**----- End of picture text -----**<br>


Figure 4: The diagram of Table-BERT with horizontal scan, two different linearizations are depicted. 

to compute _pθ_ ( _S, P_ ) = _σ_ ( _vp[T]_[[] _[Enc][S]_[(] _[S]_[);] _[ Enc][P]_[ (] _[P]_[)])][ as the relevance between S and] _[ P]_[with weight] _vp ∈_ R _[D]_ . At test time, we use the discriminator _pθ_ to assign confidence _pθ_ ( _S, P_ ) to each candidate _P ∈P_ , and then either aggregate the prediction from all hypothesis with the confidence weights or rank the highest-confident hypothesis and use their outputs as the prediction. 

## 3.2 TABLE-BERT 

In this approach, we view the table verification problem as a two-sequence binary classification problem like NLI or MPRC (Wang et al., 2018) by linearizing a table **T** into a sequence and treating the statement as another sequence. Since the linearized table can be extremely long surpassing the limit of sequence models like LSTM, Transformers, etc. We propose to shrink the sequence by only retaining the columns containing entities linked to the statement to alleviate such a memory issue. In order to encode such sub-table as a sequence, we propose two different linearization methods, as is depicted in Figure 4. (i) Concatenation: we simply concatenate the table cells with [SEP] tokens in between and restart position counter at the cell boundaries; the column name is fed as another type embedding to the input layer. Such design retains the table information in its machine format. (ii) Template: we adopt simple natural language templates to transform a table into a “somewhat natural” sentence. Taking the horizontal scan as an example, we linearize a table as “row one’s game is 51; the date is February; ..., the score is 3.4 (ot). row 2 is ...”. The isolated cells are connected with punctuations and copula verbs in a language-like format. 

After obtaining the linearized sub-table **T[˜]** , we concatenate it with the natural language statement S and prefix a [CLS] token to the sentence to obtain the sequence-level representation _H_ = _fBERT_ ([ **T[˜]** _, S_ ]), with _H ∈_ R[768] from pre-trained BERT (Devlin et al., 2019). The representation is further fed into multi-layer perceptron _fMLP_ to obtain the entailment probability _pθ_ ( **T[˜]** _, S_ ) = _σ_ ( _fMLP_ ( _H_ )), where _σ_ is the sigmoid function. We finetune the model _θ_ (including the parameters of BERT and MLP) to minimize the binary cross entropy _L_ ( _pθ_ ( **T[˜]** _, S_ ) _, L_ ) on the training set. At test time, we use the trained BERT model to compute the matching probability between the (table, statement) pair, and classify it as ENTAILED statement when _pθ_ ( **T[˜]** _, S_ ) is greater than 0.5. 

6 

Published as a conference paper at ICLR 2020 

## 4 EXPERIMENTS 

In this section, we aim to evaluate the proposed methods on TABFACT. Besides the standard validation and test sets, we also split the test set into a simple and a complex partition based on the channel from which they were collected. This facilitates analyzing how well the model performs under different levels of difficulty. Additionally, we also hold out a small test set with 2K samples for human evaluation, where we distribute each (table, statement) pair to 5 different workers to approximate human judgments based on their majority voting, the results are reported in Table 2. 

|Model|Val|Test|Test (simple)|Test (complex)|Small Test|
|---|---|---|---|---|---|
|BERT classifer w/o Table|50.9|50.5|51.0|50.1|50.4|
|Table-BERT-Horizontal-F+T-Concatenate|50.7|50.4|50.8|50.0|50.3|
|Table-BERT-Vertical-F+T-Template|56.7|56.2|59.8|55.0|56.2|
|Table-BERT-Vertical-T+F-Template|56.7|57.0|60.6|54.3|55.5|
|Table-BERT-Horizontal-F+T-Template|66.0|65.1|79.0|58.1|67.9|
|Table-BERT-Horizontal-T+F-Template|**66.1**|**65.1**|**79.1**|**58.2**|**68.1**|
|NSM w/ RL (Binary Reward)|54.1|54.1|55.4|53.1|55.8|
|NSM w/ LPA-guided ML + RL|63.2|63.5|77.4|56.1|66.9|
|LPA-Voting w/o Discriminator|57.7|58.2|68.5|53.2|61.5|
|LPA-Weighted-Voting|62.5|63.1|74.6|57.3|66.8|
|LPA-Ranking w/ Discriminator|**65.2**|65.0|78.4|**58.5**|68.6|
|LPA-Ranking w/ Discriminator (Caption)|65.1|**65.3**|**78.7**|**58.5**|**68.9**|
|Human Performance|-|-|-|-|**92.1**|



Table 2: The results of different models, the numbers are in percentage. T+F means table followed by fact, while F+T means fact followed by table. NSM is modified from Liang et al. (2017). 

**NSM** We follow Liang et al. (2017) to modify their approach to fit the setting of TABFACT. Specifically, we adopt an LSTM as an encoder and another LSTM with copy mechanism as a decoder to synthesize the program. However, without any ground truth annotation for the intermediate programs, directly training with reinforcement learning is difficult as the binary reward is underspecified, which is listed in Table 2 as ”NSM w/ RL”. Further, we use LPA as a teacher to search the top programs for the NSM to bootstrap and then use reinforcement learning to finetune the model, which achieves reasonable performance on our dataset listed as ”NSM w/ ML + RL”. 

**Table-BERT** We build Table-BERT based on the open-source implementation of BERT[8] using the pre-trained model with 12-layer, 768-hidden, 12-heads, and 110M parameters trained in 104 languages. We use the standard BERT tokenizer to break the words in both statements and tables into subwords and join the two sequences with a [SEP] token in between. The representation corresponding to [CLS] is fed into an MLP layer to predict the verification label. We finetune the model on a single TITAN X GPU with a mini-batch size of 6. The best performance is reached after about 3 hours of training (around 10K steps). We implement and compare the following variants of the Table-BERT model including (i) Concatenation vs. Template: whether to use natural language templates during linearization. (ii) Horizontal vs. Vertical: scan direction in linearization. 

**LPA** We run the latent program search in a distributed fashion on three 64-core machines to generate the latent programs. The search terminates once the buffer has more than 50 traces or the path length is larger than 7. The average search time for each statement is about 2.5s. For the discriminator model, we design two transformer-based encoders (3 layers, 128-dimension hidden embedding, and 4 heads at each layer) to encode the programs and statements, respectively. The variants of LPA models considered include (i) Voting: assign each program with equal weight and vote without the learned discriminator. (ii) Weighted-Voting: compute a weighted-sum to aggregate the predictions of all latent programs with the discriminator confidence as the weights. (iii) Ranking: rank all the hypotheses by the discriminator confidence and use the top-rated hypothesis as the output. (Caption) means feeding the caption as a sequence of words to the discriminator during ranking. 

**Preliminary Evaluation** In order to test whether our negative rewriting strategy eliminates the artifacts or shallow cues, we also fine-tune a pre-trained BERT (Devlin et al., 2019) to classify the statement _S_ without feeding in table information. The result is reported as “BERT classifier w/o 

> 8https://github.com/huggingface/pytorch-pretrained-BERT 

7 

Published as a conference paper at ICLR 2020 

Table” in Table 2, which is approximately the majority guess and reflects the effectiveness of the rewriting strategy. Before presenting the experiment results, we first perform a preliminary study to evaluate how well the entity linking system, program search, and the statement-program discriminator perform. Since we do not have the ground truth labels for these models, we randomly sample 100 samples from the dev set to perform the human study. For the entity linking, we evaluate its accuracy as the number of correctly linked sentences / total sentences. For the latent program search, we evaluate whether the “true” programs are included in the candidate set _P_ as recall score. 

**Results** We report the performance of different methods as well as human performance in Table 2. First of all, we observe that the naive serialized model fails to learn anything effective (same as the Majority Guess). It reveals the importance of template when using the pre-trained BERT (Devlin et al., 2019) model: the “natural” connection words between individual cells is able to unleash the power of the large pre-trained language model and enable it to perform reasoning on the structured table form. Such behavior is understandable given the fact that BERT is pre-trained on purely natural language corpora. In addition, we also observe that the horizontal scan excels in the vertical scan because it better captures the convention of human expression. Among different LPA methods, we found that LPA-Ranking performs the best since it can better suppress the spurious programs than the voting-based algorithm. Overall, the LPA model is on par with Table-BERT on both simple and test split without any pre-training on external corpus, which reflects the effectiveness of LPA to leverage symbolic operations in the verification process. 

Through our human evaluation, we found that only 58% of sentences have been correctly linked without missing-link or over-link, while the systematic search has a recall of 51% under the cases where the sentence is correctly linked. With that being said, the chance for LPA method to cover the correct program (rationale) is roughly under 30%. After the discriminator’s re-ranking step, the probability of selecting these particular oracle program is even much lower. However, we still observe a final overall accuracy of 65%, which indicates that the spurious problem is quite severe in LPA, where the correct label is predicted based on the wrong reason. 

Through our human evaluation, we also observe that Table-BERT exhibits poor consistency as it can misclassify simple cases but correctly-classify hard cases. These two major weaknesses are yet to be solved in future studies. In contrast, LPA behaves much more consistently and provides a clear latent rationale for its decision. But, such a pipeline system requires laborious handcrafting of API operations and is also very sensitive to the entity linking accuracy. Both methods have pros and cons; how to combine them still remains an open question. 

**Program Annotation** To further promote the development of different models in our dataset, we collect roughly 1400 human-annotated programs paired with the original statements. These statements include the most popular logical operations like superlative, counting, comparison, unique, etc. We provide these annotations in Github[9] , which can either be used to bootstrap the semantic parsers or provide the rationale for NLI models. 

## 5 RELATED WORK 

**Natural Language Inference & Reasoning:** Modeling reasoning and inference in human language is a fundamental and challenging problem towards true natural language understanding. There has been extensive research on RTE in the early years (Dagan et al., 2005) and more recently shifted to NLI (Bowman et al., 2015; Williams et al., 2017). NLI seeks to determine whether a natural language hypothesis _h_ can be inferred from a natural language premise _p_ . With the surge of deep learning, there have been many powerful algorithms like the Decomposed Model (Parikh et al., 2016), Enhanced-LSTM (Chen et al., 2017) and BERT (Devlin et al., 2019). Besides the textual evidence, NLVR (Suhr et al., 2017) and NLVR2 (Suhr et al., 2019) have been proposed to use images as the evidence for statement verification on multi-modal setting. Our proposed fact verification task is closely related to these inference tasks, where our semi-structured table can be seen as a collection of “premises” exhibited in a semi-structured format. Our proposed problem hence could be viewed as the generalization of NLI under the semi-structured domain. 

**Table Question Answering:** Another line of research closely related to our task is the table-based 

> 9https://github.com/wenhuchen/Table-Fact-Checking/tree/master/bootstrap 

8 

Published as a conference paper at ICLR 2020 

**==> picture [359 x 204] intentionally omitted <==**

**----- Start of picture text -----**<br>
United States House of Representatives Elections, 1972<br>(1) District Incumbent Party Result Candidates<br>California 3 John E. Moss democratic re-elected John E. Moss  (d) 69.9% John Rakus (r) 30.1%<br>California 5 Phillip Burton democratic re-elected Phillip Burton (d) 81.8% Edlo E. Powell (r) 18.2%<br>California 8 George Paul Miller democratic lost renomination democratic hold Pete Stark (d) 52.9% Lew M. Warden , Jr. (r) 47.1%<br>California 14 Jerome R. Waldie republican re-elected Jerome R. Waldie (d) 77.6% Floyd E. Sims (r) 22.4%<br>California 15 John J. Mcfall republican re-elected John J. Mcfall (d) unopposed<br>There are five candidates in total, two of them are democrats and three of them are republicans.<br>Conjunctive<br>Question: How many of candidates in total?Answer: 5 ∧ Question: How many democrats are there?Answer: 2 ∧ Question: How many republicans are there?Answer: 3<br>Eq(Count(T), 5)=T ∧ Eq(Count(Filter(T, party=‘dem..’)), 2)=F ∧ Eq(Count(Filter(T, party=‘rep..’)), 3)=F<br>(2) Jordi Arrese<br>outcome date tournament surface partner opponents in the final score in the final<br>runner - up 1985 bologna , italy clay alberto tous paolo canè simone colombo 5 - 7 , 4 - 6<br>winner 1986 bordeaux , france clay david de miguel ronald agénor mansour bahrami 7 - 5 , 6 - 4<br>winner 1989 prague , czechoslovakia clay horst skoff petr korda tomáš šmíd 6 - 4 , 6 - 4<br>1986: Winner 1986: Runner-up 7 - 5 , 6 - 4<br>Linguistic Inference Mathematic Inference<br>Jordi Arrese achieves better score in 1986 than in 1985. Jordi Arrese won both of the final games in 1986.<br>**----- End of picture text -----**<br>


Figure 5: The two uniqueness of Table-based fact verification against standard QA problems. 

question answering, such as MCQ (Jauhar et al., 2016), WikiTableQuestion (Pasupat & Liang, 2015), Spider (Yu et al., 2018), Sequential Q&A (Iyyer et al., 2017), and WikiSQL (Zhong et al., 2017), for which approaches have been extended to handle large-scale tables from Wikipedia (Bhagavatula et al., 2013). However, in these Q&A tasks, the question types typically provide strong signals needed for identifying the type of answers, while TABFACT does not provide such specificity. The uniqueness of TABFACT lies in two folds: 1) a given fact is regarded as a false claim as long as any part of the statement contains misinformation. Due to the conjunctive nature of verification, a fact needs to be broken down into several sub-clauses or (Q, A) pairs to separate evaluate their correctness. Such a compositional nature of the verification problem makes it more challenging than a standard QA setting. On one hand, the model needs to recognize the multiple QA pairs and their relationship. On the other hand, the multiple sub-clauses make the semantic form longer and logic inference harder than the standard QA setting. 2) some facts cannot even be handled using semantic forms, as they are driven by linguistic inference or common sense. In order to verify these statements, more inference techniques have to be leveraged to enable robust verification. We visualize the above two characteristics of TABFACT in Figure 5. 

**Program Synthesis & Semantic Parsing:** There have also been great interests in using program synthesis or logic forms to solve different natural language processing problems like question answering (Liang et al., 2013; Berant et al., 2013; Berant & Liang, 2014), visual navigation (Artzi et al., 2014; Artzi & Zettlemoyer, 2013), code generation (Yin & Neubig, 2017; Dong & Lapata, 2016), SQL synthesis (Yu et al., 2018), etc. The traditional semantic parsing papers (Artzi et al., 2014; Artzi & Zettlemoyer, 2013; Zettlemoyer & Collins, 2005; Liang et al., 2013; Berant et al., 2013) greatly rely on rules, lexicon to parse natural language sentences into different forms like lambda calculus, DCS, etc. More recently, researchers strive to propose neural models to directly perform end-to-end formal reasoning like Theory Prover (Riedel et al., 2017; Rockt¨aschel & Riedel, 2017), Neural Turing Machine (Graves et al., 2014), Neural Programmer (Neelakantan et al., 2016; 2017) and Neural-Symbolic Machines (Liang et al., 2017; 2018; Agarwal et al., 2019). The proposed TABFACT serves as a great benchmark to evaluate the reasoning ability of different neural reasoning models. Specifically, TABFACT poses the following challenges: 1) spurious programs (i.e., wrong programs with the true returned answers): since the program output is only a binary label, which can cause serious spurious problems and misguide the reinforcement learning with the under-specified binary rewards. 2) decomposition: the model needs to decompose the statement into sub-clauses and verify the sub-clauses one by one, which normally requires the longer logic inference chains to infer the statement verdict. 3) linguistic reasoning like inference and paraphrasing. 

**Fact Checking** The problem of verifying claims and hypotheses on the web has drawn significant attention recently due to its high social influence. Different fact-checking pioneering studies have been 

9 

Published as a conference paper at ICLR 2020 

performed including LIAR (Wang, 2017), PolitiFact (Vlachos & Riedel, 2014), FEVER (Thorne et al., 2018) and AggChecker (Jo et al., 2019), etc. The former three studies are mainly based on textual evidence on social media or Wikipedia, while AggChecker is closest to ours in using relational databases as the evidence. Compared to AggChecker, our paper proposes a much larger dataset to benchmark the progress in this direction. 

## 6 CONCLUSION 

This paper investigates a very important yet previously under-explored research problem: semistructured fact verification. We construct a large-scale dataset and proposed two methods, TableBERT and LPA, based on the state-of-the-art pre-trained natural language inference model and program synthesis. In the future, we plan to push forward this research direction by inspiring more sophisticated architectures that can perform both linguistic and symbolic reasoning. 

## REFERENCES 

- Rishabh Agarwal, Chen Liang, Dale Schuurmans, and Mohammad Norouzi. Learning to generalize from sparse and underspecified rewards. _International Conference of Machine Learning_ , 2019. 

- Yoav Artzi and Luke Zettlemoyer. Weakly supervised learning of semantic parsers for mapping instructions to actions. _Transactions of the Association for Computational Linguistics_ , 1:49–62, 2013. 

- Yoav Artzi, Dipanjan Das, and Slav Petrov. Learning compact lexicons for ccg semantic parsing. In _Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , pp. 1273–1283, 2014. 

- Jonathan Berant and Percy Liang. Semantic parsing via paraphrasing. In _Proceedings of the 52nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pp. 1415–1425, 2014. 

- Jonathan Berant, Andrew Chou, Roy Frostig, and Percy Liang. Semantic parsing on freebase from question-answer pairs. In _Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing_ , pp. 1533–1544, 2013. 

- Chandra Sekhar Bhagavatula, Thanapon Noraset, and Doug Downey. Methods for exploring and mining tables on wikipedia. In _Proceedings of the ACM SIGKDD Workshop on Interactive Data Exploration and Analytics_ , pp. 18–26. ACM, 2013. 

- Samuel R Bowman, Gabor Angeli, Christopher Potts, and Christopher D Manning. A large annotated corpus for learning natural language inference. In _Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing_ , pp. 632–642, 2015. 

- Qian Chen, Xiaodan Zhu, Zhen-Hua Ling, Si Wei, Hui Jiang, and Diana Inkpen. Enhanced lstm for natural language inference. In _Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pp. 1657–1668, 2017. 

- Ido Dagan, Oren Glickman, and Bernardo Magnini. The pascal recognising textual entailment challenge. In _Machine Learning Challenges Workshop_ , pp. 177–190. Springer, 2005. 

- Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. _Proceedings of NAACL-HLT_ , 2019. 

- Li Dong and Mirella Lapata. Language to logical form with neural attention. In _Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pp. 33–43, 2016. 

- Joseph L Fleiss. Measuring nominal scale agreement among many raters. _Psychological bulletin_ , 76(5):378, 1971. 

- Alex Graves, Greg Wayne, and Ivo Danihelka. Neural turing machines. _arXiv preprint arXiv:1410.5401_ , 2014. 

10 

Published as a conference paper at ICLR 2020 

- Andreas Hanselowski, Hao Zhang, Zile Li, Daniil Sorokin, Benjamin Schiller, Claudia Schulz, and Iryna Gurevych. Ukp-athene: Multi-sentence textual entailment for claim verification. _arXiv preprint arXiv:1809.01479_ , 2018. 

- Naeemul Hassan, Gensheng Zhang, Fatma Arslan, Josue Caraballo, Damian Jimenez, Siddhant Gawsane, Shohedul Hasan, Minumol Joseph, Aaditya Kulkarni, Anil Kumar Nayak, et al. Claimbuster: the first-ever end-to-end fact-checking system. _Proceedings of the VLDB Endowment_ , 10 (12):1945–1948, 2017. 

- Mohit Iyyer, Wen-tau Yih, and Ming-Wei Chang. Search-based neural structured learning for sequential question answering. In _Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , volume 1, pp. 1821–1831, 2017. 

- Sujay Kumar Jauhar, Peter Turney, and Eduard Hovy. Tables as semi-structured knowledge for question answering. In _Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , volume 1, pp. 474–483, 2016. 

- Saehan Jo, Immanuel Trummer, Weicheng Yu, Xuezhi Wang, Cong Yu, Daniel Liu, and Niyati Mehta. Aggchecker: A fact-checking system for text summaries of relational data sets. _Proceedings of the VLDB Endowment_ , 12(12), 2019. 

- Jerrold J Katz and Jerry A Fodor. The structure of a semantic theory. _language_ , 39(2):170–210, 1963. 

- Ni Lao, Tom Mitchell, and William W Cohen. Random walk inference and learning in a large scale knowledge base. In _Proceedings of the Conference on Empirical Methods in Natural Language Processing_ , pp. 529–539. Association for Computational Linguistics, 2011. 

- Chen Liang, Jonathan Berant, Quoc Le, Kenneth D Forbus, and Ni Lao. Neural symbolic machines: Learning semantic parsers on freebase with weak supervision. _International Conference of Machine Learning_ , 2017. 

- Chen Liang, Mohammad Norouzi, Jonathan Berant, Quoc V Le, and Ni Lao. Memory augmented policy optimization for program synthesis and semantic parsing. In _Advances in Neural Information Processing Systems_ , pp. 9994–10006, 2018. 

- Percy Liang, Michael I Jordan, and Dan Klein. Learning dependency-based compositional semantics. _Computational Linguistics_ , 39(2):389–446, 2013. 

- Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized bert pretraining approach. _arXiv preprint arXiv:1907.11692_ , 2019. 

- Arvind Neelakantan, Quoc V Le, and Ilya Sutskever. Neural programmer: Inducing latent programs with gradient descent. _International Conference on Learning Representation_ , 2016. 

- Arvind Neelakantan, Quoc V Le, Martin Abadi, Andrew McCallum, and Dario Amodei. Learning a natural language interface with neural programmer. _International Conference on Learning Representation_ , 2017. 

- Ankur Parikh, Oscar T¨ackstr¨om, Dipanjan Das, and Jakob Uszkoreit. A decomposable attention model for natural language inference. In _Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing_ , pp. 2249–2255, 2016. 

- Panupong Pasupat and Percy Liang. Compositional semantic parsing on semi-structured tables. In _Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)_ , volume 1, pp. 1470–1480, 2015. 

- Matthew E Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, and Luke Zettlemoyer. Deep contextualized word representations. In _Proceedings of NAACL-HLT_ , pp. 2227–2237, 2018. 

11 

Published as a conference paper at ICLR 2020 

- Kashyap Popat, Subhabrata Mukherjee, Jannik Str¨otgen, and Gerhard Weikum. Where the truth lies: Explaining the credibility of emerging claims on the web and social media. In _Proceedings of the 26th International Conference on World Wide Web Companion_ , pp. 1003–1012. International World Wide Web Conferences Steering Committee, 2017. 

- Sebastian Riedel, Matko Bosnjak, and Tim Rockt¨aschel. Programming with a differentiable forth interpreter. _ICML_ , 2017. 

- Tim Rockt¨aschel and Sebastian Riedel. End-to-end differentiable proving. In _Advances in Neural Information Processing Systems_ , pp. 3788–3800, 2017. 

- Alane Suhr, Mike Lewis, James Yeh, and Yoav Artzi. A corpus of natural language for visual reasoning. In _Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)_ , pp. 217–223, 2017. 

- Alane Suhr, Stephanie Zhou, Ally Zhang, Iris Zhang, Huajun Bai, and Yoav Artzi. A corpus for reasoning about natural language grounded in photographs. In _Proceedings of the Annual Meeting of the Association for Computational Linguistics_ , 2019. 

- James Thorne, Andreas Vlachos, Christos Christodoulopoulos, and Arpit Mittal. Fever: a largescale dataset for fact extraction and verification. In _Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)_ , volume 1, pp. 809–819, 2018. 

- Johan Van Benthem et al. _A brief history of natural logic_ . LondonCollege Publications9781904987444, 2008. 

- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In _Advances in neural information processing systems_ , pp. 5998–6008, 2017. 

- Andreas Vlachos and Sebastian Riedel. Fact checking: Task definition and dataset construction. In _Proceedings of the ACL 2014 Workshop on Language Technologies and Computational Social Science_ , pp. 18–22, 2014. 

- Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R Bowman. Glue: A multi-task benchmark and analysis platform for natural language understanding. _EMNLP 2018_ , pp. 353, 2018. 

- Alex Wang, Yada Pruksachatkun, Nikita Nangia, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R Bowman. Superglue: A stickier benchmark for general-purpose language understanding systems. _arXiv preprint arXiv:1905.00537_ , 2019. 

- William Yang Wang. liar, liar pants on fire: A new benchmark dataset for fake news detection. In _Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)_ , pp. 422–426, 2017. 

- Adina Williams, Nikita Nangia, and Samuel R Bowman. A broad-coverage challenge corpus for sentence understanding through inference. _arXiv preprint arXiv:1704.05426_ , 2017. 

- Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, and Quoc V Le. Xlnet: Generalized autoregressive pretraining for language understanding. _Advances in neural information processing systems_ , 2019. 

- Pengcheng Yin and Graham Neubig. A syntactic neural model for general-purpose code generation. _arXiv preprint arXiv:1704.01696_ , 2017. 

- Tao Yu, Rui Zhang, Kai Yang, Michihiro Yasunaga, Dongxu Wang, Zifan Li, James Ma, Irene Li, Qingning Yao, Shanelle Roman, et al. Spider: A large-scale human-labeled dataset for complex and cross-domain semantic parsing and text-to-sql task. In _Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing_ , pp. 3911–3921, 2018. 

12 

Published as a conference paper at ICLR 2020 

- Rowan Zellers, Yonatan Bisk, Roy Schwartz, and Yejin Choi. Swag: A large-scale adversarial dataset for grounded commonsense inference. In _Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing_ , pp. 93–104, 2018. 

- Luke S Zettlemoyer and Michael Collins. Learning to map sentences to logical form: structured classification with probabilistic categorial grammars. In _Proceedings of the Twenty-First Conference on Uncertainty in Artificial Intelligence_ , pp. 658–666. AUAI Press, 2005. 

- Victor Zhong, Caiming Xiong, and Richard Socher. Seq2sql: Generating structured queries from natural language using reinforcement learning. _arXiv preprint arXiv:1709.00103_ , 2017. 

13 

Published as a conference paper at ICLR 2020 

## A APPENDIX 

## A.1 FUNCTION DESCRIPTION 

We list the detailed function description in Figure 6. We also visualize the functionality of the most 

|Name|Arguments|Output|Comment|
|---|---|---|---|
|Count|View|Number|Return the number of rows in the View|
|Within|View, Header String, Cell<br>String/Number|Bool|Return whether the cell string/number exists under the Header Column of the given<br>view|
|Without|View, Header String, Cell<br>String/Number|Bool|Return whether the cell string/number does not exist under the Header Column of the<br>given view|
|None|String|Bool|Whether the string represents None, like “None”, “No”, “-”, “No information provided”|
|Before/After|Row, Row|Row|Returns whether row1 is before/after row2|
|First/Second/Third/Fourth|View, Row|Bool|Returns whether the row is in the first/second/third position  of  the view|
|Average/Sum/Max/Min|View, Header String|Number|Returns the average/summation/max/min value under the Header Column of the given<br>view|
|Argmin/<br>Argmax|View, Header String|Row|Returns the row with the maximum/minimum value under the Header Column of the<br>given view|
|Hop|Row, Header String|Number/<br>String|Returns the cell value under the Header Column of the given row|
|Diff/Add|Number, Number|Number|Perform arithmetic operations on two numbers|
|Greater/Less|Number, Number|Bool|Returns whether the first number is greater/less than the second number|
|Equal/<br>Unequal|String, String/<br>Number, Number|Bool|Compare two numbers or strings to see whether they are the same|
|Filter_eq/<br>Filter_greater/<br>Filter_less/<br>Filter_greater_or_equal/<br>Filter_less_or_equal|View, Header String,<br>Number|View|Returns the subview of the given with the cell values under the Header column<br>greater/less/eq/… against the given number|
|All_eq/All_greater/<br>All_less/All_greater_or_equa<br>l/All_less_or_equal|View, Header String,<br>Number|Bool|Returns the whether all of the cell values under the Header column are<br>greater/less/eq/… against the given number|
|And/Or|Bool, Bool|Bool|Returns the Boolean operation results of two inputs|
|||||



Figure 6: The function definition used in TabFact. 

typical functions and their input/output examples in Figure 7. 

**==> picture [396 x 211] intentionally omitted <==**

**----- Start of picture text -----**<br>
Name Age Name Age<br>Count ( ) = 2 within ( A , Name, A) = 𝑇𝑟𝑢𝑒<br>B<br>Name Age<br>without ( A , Name, A) = 𝐹𝑎𝑙𝑠𝑒 None ( -/Not Given/NA ) =𝑇𝑟𝑢𝑒<br>B<br>Before ( Name A Age ,  Name B Age ) = 𝑇𝑟𝑢𝑒 First ( Name A Age ) =𝑇𝑟𝑢𝑒<br>Name Age Argmin ( Name A Age 2 , Age) = Name Age<br>Avg ( A 2 , Age) = 3 A 2<br>B 4<br>B 4<br>Name Age<br>Filter_eq ( Name A Age 2 , Age, 2) = Name Age All_eq ( A 2 , Age, 2) = True<br>A 2 B 2<br>B 4<br>Hop ( Name A Age , Name) = 𝐴 Diff (2, 1) = 1 Greater (2, 1) = 𝑇𝑟𝑢𝑒 Equal (2, 2) = 𝑇𝑟𝑢𝑒<br>**----- End of picture text -----**<br>


Figure 7: The visualization of different functions. 

14 

Published as a conference paper at ICLR 2020 

We list all the trigger words for different functions in Figure 8 

|Trigger|Function|
|---|---|
|'average'|average|
|'difference', 'gap', 'than', 'separate'|diff|
|'sum', 'summation', 'combine', 'combined', 'total', 'add', 'all', 'there are'|ddd, sum|
|'not', 'no', 'never', "didn't", "won't", "wasn't", "isn't,"haven't", "weren't", "won't", 'neither', 'none', 'unable,<br>'fail', 'different', 'outside', 'unable', 'fail'|not_eq, not_within, Filter_not_eq, none|
|'not', 'no', 'none'|none|
|'first', 'top', 'latest', 'most'|first|
|'last', 'bottom', 'latest', 'most'|last|
|'RBR', 'JJR', 'more', 'than', 'above', 'after'|filter_greater, greater|
|'RBR', 'JJR', 'less', 'than', 'below', 'under'|filter_less, less|
|'all', 'every', 'each'|all_eq, all_less, all_greater,|
|['all', 'every', 'each'], ['not', 'no', 'never', "didn't", "won't", "wasn't"]|all_not_eq|
|'at most', 'than'|all_less_eq, all_greater_eq|
|'RBR', 'RBS', 'JJR', 'JJS'|max, min|
|'JJR', 'JJS', 'RBR', 'RBS', 'top', 'first'|argmax, argmin|
|'within', 'one', 'of', 'among'|within|
|'follow', 'following', 'followed', 'after', 'before', 'above', 'precede'|before|
|'follow', 'following', 'followed', 'after', 'before', 'above', 'precede'|after|
|’most’|most_freq|
|ordinal|First, second, third, fourth|



Figure 8: The trigger words used to shrink the search space. 

## B HIGHER-ORDER OPERATIONS 

1. Aggregation: the aggregation operation refers to sentences like “the averaged age of all ....”, “the total amount of scores obtained in ...”, etc. 

2. Negation: the negation operation refers to sentences like “xxx did not get the best score”, “xxx has never obtained a score higher than 5”. 

3. Superlative: the superlative operation refers to sentences like “xxx achieves the highest score in”, “xxx is the lowest player in the team”. 

4. Comparative: the comparative operation refers to sentences like “xxx has a higher score than yyy”. 

5. Ordinal: the ordinal operation refers to sentences like “the first country to achieve xxx is xxx”, “xxx is the second oldest person in the country”. 

6. Unique: the unique operation refers to sentences like “there are 5 different nations in the tournament, ”, “there are no two different players from U.S” 

7. All: the for all operation refers to sentences like “all of the trains are departing in the morning”, “none of the people are older than 25.” 

8. None: the sentences which do not involve higher-order operations like “xxx achieves 2 points in xxx game”, “xxx player is from xxx country”. 

## C ERROR ANALYSIS 

Before we quantitatively demonstrate the error analysis of the two methods, we first theoretically analyze the bottlenecks of the two methods as follows: 

**Symbolic** We first provide a case in which the symbolic execution can not deal with theoretically in Figure 9. The failure cases of symbolic are either due to the entity link problem or function coverage problem. For example, in the given statement below, there is no explicit mention of ”7-5, 6-4” cell. Therefore, the entity linking model fails to link to this cell content. Furthermore, even 

15 

Published as a conference paper at ICLR 2020 

though we can successfully link to this string, there is no defined function to parse ”7-5, 6-5” as ”won two games” because it requires linguistic/mathematical inference to understand the implication from the string. Such cases are the weakness of symbolic reasoning models. 

**==> picture [396 x 141] intentionally omitted <==**

**----- Start of picture text -----**<br>
Jordi Arrese<br>outcome date tournament surface partner opponents in the final score in the final<br>runner - up 1985 Bologna , Italy clay Alberto Tous Paolo Canè Simone Colombo 5 - 7 , 4 - 6<br>winner 1986 Bordeaux , France clay David De Miguel Ronald Agénor Mansour Bahrami 7 - 5 , 6 - 4<br>winner 1989 Prague , Czechoslovakia clay Horst Skoff Petr Korda Tomáš šmíd 6 - 4 , 6 - 4<br>1986: Winner 1986: Runner-up<br>Linguistic Inference<br>Jordi Arrese achieves better score in 1986 than in 1985.<br>7 - 5 , 6 - 4 Mathematic Inference<br>Jordi Arrese won both of the final games in 1986.<br>**----- End of picture text -----**<br>


Figure 9: The error case of symbolic reasoning model 

**BERT** In contrast, Table-BERT model seems to have no coverage problem as long as it can feed the whole table content. However, due to the template linearization, the table is unfolded into a long sequence as depicted in Figure 10. The useful information, ”clay” are separated in a very long span of unrelated words. How to grasp such a long dependency and memorize the history information poses a great challenge to the Table-BERT model. 

**==> picture [397 x 136] intentionally omitted <==**

**----- Start of picture text -----**<br>
Jordi Arrese<br>outcome date tournament surface partner opponents in the final score in the final<br>runner - up 1985 Bologna , Italy clay Alberto Tous Paolo Canè Simone Colombo 5 - 7 , 4 - 6<br>winner 1986 Bordeaux , France clay David De Miguel Ronald Agénor Mansour Bahrami 7 - 5 , 6 - 4<br>winner 1989 Prague , Czechoslovakia clay Horst Skoff Petr Korda Tomáš šmíd 6 - 4 , 6 - 4<br>Template<br>Given the table titled “Jordi Arrese”, in row one, the outcome is runner-up, the date is 1985, … , the surface is clay …. …… ,<br>In row two, the outcome is … , the surface is clay. In row three, the outcome is …, … the surface is clay.<br>Long Dependency The three “Clay” are separated by more over 20 words<br>Jordi Arrese played all of his games on clay surface.<br>**----- End of picture text -----**<br>


Figure 10: The error case of BERT NLI model 

**Statistics** Here we pick 200 samples from the validation set which only involve single semantic and divide them into different categories. We denote the above-mentioned cases as ”linguistic inference”, and the sentences which only describe information from one row as ”Trivial”, the rest are based on their logic operation like Aggregation, Superlative, Count, etc. We visualize the accuracy of LPA and Table-BERT in Figure 11. From which we can observe that the statements with linguistic inference are much better handled with the BERT model, while LPA achieves an accuracy barely higher than a random guess. The BERT model can deal with trivial cases well as it uses a horizontal scan order. In contrast, the LPA model outperforms BERT on higher-order logic cases, especially when the statement involves operations like Count and Superlative. 

16 

Published as a conference paper at ICLR 2020 

**Error Analysis of LPA/Table-BERT** 

**==> picture [303 x 93] intentionally omitted <==**

**----- Start of picture text -----**<br>
80<br>Table-BERT LPA<br>75<br>70<br>65<br>60<br>55<br>50<br>45<br>Linguistic Trivial Aggregation Superlative Count Compare Negation<br>**----- End of picture text -----**<br>


Figure 11: The error analysis of two different models 

## D REASONING DEPTH 

Given that our LPA has the breadth to cover a large semantic space. Here we also show the reasoning depth in terms of how many logic inference steps are required to tackle verify the given claims. We visualize the histogram in Figure 12 and observe that the reasoning steps are concentrated between 4 to 7. Such statistics indicate the difficulty of fact verification in our TABFACT dataset. 

**==> picture [278 x 116] intentionally omitted <==**

**----- Start of picture text -----**<br>
Reasoning Depth Statistics in LPA<br>450000<br>400000<br>350000<br>300000<br>250000<br>200000<br>150000<br>100000<br>50000<br>0<br>1 2 3 4 5 6 7<br>**----- End of picture text -----**<br>


Figure 12: The histogram of reasoning steps required to verify the claims 

## E WHETHER TO KEEP WIKIPEDIA CONTEXT 

Before crowd-sourcing the annotation for the tables, we observed that the previous WikiTableQuestion Pasupat & Liang (2015) provides context (Wikipedia title) during annotation while the WikiSQL Zhong et al. (2017) does not. Therefore, we particularly design ablation annotation tasks to compare the annotation quality between w/ and w/o Wikipedia title as context. We demonstrate a typical example in Figure 13, where a Wiki table[10] aims to describe the achievements of a tennis player named Dennis, but itself does not provide any explicit hint about “Tennis Player Dennis”. Unsurprisingly, the sentence fluency and coherence significantly drop without such information. Actually, a great portion of these Wikipedia tables requires background knowledge (like sports, celebrity, music, etc) to understand. We perform a small user study to measure the fluency of annotated statements. Specifically, we collected 50 sentences from both annotation w/ and w/o title context and randomly shuffle them as pairs, which are distributed to the 8 experts without telling them their source to compare the language fluency. It turns out that the experts ubiquitously agree that the statements with Wikipedia titles are more human-readable. Therefore, we argue that such a context is necessary for annotators to understand the background knowledge to write more fluent sentences. On the other end, we also hope to minimize the influence of the textual context in the table-based verification task, therefore, we design an annotation criterion: the Wikipedia title 

> 10https://en.wikipedia.org/wiki/Dennis_Ralston 

17 

Published as a conference paper at ICLR 2020 

is provided to the workers during the annotation, but they are explicitly banned from bringing any unrelated background information other than the title into the annotation. As illustrated in Figure 13, the title only acts as a placeholder in the statements to make it sound more natural. 

|**outcome**<br>**year**<br>**championship**<br>**surface**<br>**partner**|**outcome**<br>**year**<br>**championship**<br>**surface**<br>**partner**|**outcome**<br>**year**<br>**championship**<br>**surface**<br>**partner**|
|---|---|---|
|winner<br>1960<br>Wimbledon championships<br>grass<br>Rafael Osuna|||
|winner<br>1961<br>US Championships<br>grass<br>Chuck Mckinley|||
|runner - up<br>1962<br>US Championships<br>grass<br>Chuck Mckinley|||
|winner<br>1963<br>US Championships (2)<br>grass<br>Chuck Mckinley|||
|Context<br>(Title)|_Richard Dennis Ralston (born July 27, 1942,_<br>_an American former tennis player_|_No Information is provided_|
|Annotate|From 1960 to 1969, Ralston won five major<br>double championships.|Winner is on the grass surface.<br>Rafael Osuna is partner in the Wimbeldon|



Figure 13: Comparison of worker annotation w/ and w/o Wikipedia title as context 

## F ENTITY LINKING 

Here we propose to use the longest string match to find all the candidate entities in the table, when multiple candidates coexist, we select the one with the minimum edit distances. The visualization is demonstrated in Figure 14. 

**==> picture [357 x 110] intentionally omitted <==**

**----- Start of picture text -----**<br>
District Incumbent Party Result Candidates<br>California 3 John E. Moss democratic re-elected John E. Moss  (d) 69.9% John Rakus (r) 30.1%<br>California 5 Phillip Burton democratic re-elected Phillip Burton (d) 81.8% Edlo E. Powell (r) 18.2%<br>california 8 George Paul Miller democratic lost renomination democratic hold Pete Stark (d) 52.9% Lew M. Warden , Jr. (r) 47.1%<br>California 14 Jerome R. Waldie republican re-elected Jerome R. Waldie (d) 77.6% Floyd E. Sims (r) 22.4%<br>California 15 John J. Mcfall republican re-elected John J. Mcfall (d) unopposed<br>Statement:  John E. Moss  is a democratic who is from California 3 district<br>John E. Moss (d) 69.9% john rakus (r) 30.1% John E. Moss (d) 69.9% john rakus (r) 30.1%<br>John<br>John John E. Moss E. Moss John E. Moss is<br>John J. Mcfall<br>**----- End of picture text -----**<br>


Figure 14: Entity Linking System. 

## G THE PROGRAM CANDIDATES 

Here we demonstrate some program candidates in Figure 15, and show how our proposed discriminator is designed to compute the matching probability between the statement and program. Specifically, we employ two transformer-based encoder Vaswani et al. (2017), the left one is aimed to encode the program sequence and the right one is aimed to encode the statement sequence. Their output from [CLS] position is concatenated and fed into an MLP to classify the verification label. 

## H HIT INTERFACE 

We provide the human intelligent task interface on AMT in the following. Very detailed instructions on what are trivial statements and what are non-trivial statements. Comprehensive examples have been given to guide the Turkers to write well-formed while logically plausible statements. In order to harvest fake statements without statistical cues, we also provide detailed instructions on how to re-write the ”fake” statements. During the annotation, we hire 8 experts to perform sanity checks on each of the HIT to make sure that the annotated dataset is clean and meets our requirements. 

18 

Published as a conference paper at ICLR 2020 

**==> picture [397 x 88] intentionally omitted <==**

**----- Start of picture text -----**<br>
Less(Count(Filter(incumbent== democratic )), Count(Filter(incumbent== republican )))=False Label<br>Less(Count(Filter(incumbent== republican )), Count(Filter(incumbent== democratic )))=True<br>Greater(Count(Filter(incumbent== republican )), Count(Filter(incumbent== democratic )))=False<br>Greater(Count(Filter(incumbent== democratic )),Count(Filter(incumbent== republican )))=True<br>Within((Filter(incumbent== democratic ), incumbent,  republican )=False<br>Within((Filter(incumbent==  republican ), incumbent,  democratic )=False<br>And(Same(all_rows, incumbent,  democratic ), Same(all_rows, incumbent,  republican ))=True<br>Or(Same(all_rows, incumbent,  democratic ), Same(all_rows, incumbent,  republican ))=True<br>Eq(Count(Filter(incumbent== republican )), Count(Filter(incumbent== democratic )))=False Program Encoder Statement Encoder<br>Statement: There are more  democratic  than  republican  in the election.<br>**----- End of picture text -----**<br>


Figure 15: We demonstrate the top program candidates and use the discriminator to rank them. 

19 

11/8/2019 

HIT 

**Survey Instructions** (Click to expand) 

**You are given a table with its wikipedia source, your job is to compose non-trivial statements supported by the table.** 

- **"Trivial"** : the sentence can be easily generated by looking **only a certain row** without understanding the table. 

- **"Non-trivial"** : the sentence requires reading multiple rows of the table and understanding of the table content. For example, the sentences which include **summarization, comparative, negation, relational, inclusion, superlative, aggregational, rephrase or combinations of them** are non-trivial. But non-trvial is not limited to these types, any statement involving understanding and reasoning is accepted. 

We list two examples below to help you understand, you are encouraged to open the **table wikipedia link** to understand the context of the table. (Everything in the table is lower-cased, you are free to use lower or upper case in your sentence): 

Table Wikipedia Link: Road_Rules_Challenge:_The_Island 

(https://en.wikipedia.org/wiki/Real_World/Road_Rules_Challenge:_The_Island) 

|**player**|**original season**|**gender**|**eliminated**|**placing**|
|---|---|---|---|---|
|derrick kosinski|rr : x - treme|male|winner|winner|
|evelyn smith|fresh meat|female|winner|winner|
|johnny devenanzio|rw : key west|male|winner|winner|
|kenny santucci|fresh meat|male|winner|winner|
|jenn grijalva|rw : denver|female|episode 8|runner - up|
|paula meronek|rw : key west|female|episode 8|runner - up|
|robin hibbard|rw : san diego|female|episode 8|runner - up|
|ryan kehoe|fresh meat|male|episode 8|runner - up|
|dunbar merrill|rw : sydney|male|episode 8|9th place|
|johanna botta|rw : austin|female|episode 8|10th place|
|kellyanne judd|rw : sydney|female|episode 8|11th place|
|dan walsh|rr : viewers' revenge|male|episode 8|12th place|
|colie edison|rw : denver|female|episode 7|13th place|
|cohutta grindstaff|rw : sydney|male|episode 6|14th place|
|tyrie ballard|rw : denver|male|episode 5|15th place|
|ashli robson|rw : sydney|female|episode 4|16th place|
|rachel robinson|rr : campus crawl|female|episode 3|17th place|
|abram boise|rr : south pacific|male|episode 2|18th place|
|dave malinosky|rw : hollywood|male|episode 2 (quit)|19th place|



## **Rejected ("Trivial") examples** : 

1. In the TV series "The Island", Derrick Kosinski is a male character. (Easy! You can simply look into first row to produce this sentence.) 

2. Derrick Kosinski has the placing of winner in the TV series. 

3. Kenny Santucci is from original season of "Fresh Meat". 

https://s3.amazonaws.com/mturk_bulk/hits/370501562/uNGk1Dz1zM48BZI6mALfxA.html 

1/4 

11/8/2019 

HIT 

4. Jenn Grijalva is Runner-Up of the challenge. 

## **Accepted ("Non-Trivial") examples** : 

(Superlative): In the TV series "The Island", Evelyn Smith is the highest ranked female. 

(Comparitive): In the TV series "The Island", Jenn Grijalva appears later than Colie Edison in the series. 

(Relational): Ashli Robson appears one episode later than Rachel Robinson in the TV series. 

(Summarization): there are three male winners in the challenge. 

(Rephrase): Evelyn Smith never eliminated in any episode in the TV series. 

(Combination): Derrick Kosinski is the winner and Jenn Grijalva is Runner-Up of the challenge. 

(Negation): jenn grijalva is not the female winning the challenge. 

(Inclusion): Evelyn smith is one of the four winner for the challenge. 

Table Wikipedia Link: AFC_Champions_League (https://en.wikipedia.org/wiki/AFC_Champions_League) 

|**rank**|**member association**|**points**|**group stage**|**play - off**|**afc cup**|
|---|---|---|---|---|---|
|1|saudi arabia|860.5|4|0|0|
|2|qatar|838.2|4|0|0|
|3|iran|813.5|3|1|0|
|4|uae|750.2|2|2|0|
|5|uzbekistan|680.8|1|0|0|
|6|india|106.4|0|0|2|
|7|jordan|128.7|0|0|2|



## **Rejected ("Trivial") examples** : 

2. ratar is in rank 2. 

3. When member association is india, the points is 106.4. 

## **Accepted ("Non-Trivial") examples** : 

(Negation): iran is one of the two countries getting into the 4th stage. (Average): uae and qatar have an average of 1 play - off during the champion league. 

(Algorithmic): saudi arabia achieves 22.3 more points than qatar. 

(Comparison): india got lower points than jordan in the league. 

(Summarization): there are two team which have won the afc cup twice. 

(Superlative): In the Champions League, saudi arabia achieves the highest points. 

(Combination): saudi arabia is the group stage 4 while iran is in group stage 3. 

Tips1: We set minimum length to 9, and sentences with more complicated grammar structures are preferred. 

Tips2: Do not limited to only one type of description like superlative or relative. 

Tips3: Copying the records from the table is encouraged, which can help avoid typos and mis-spelling as much as possible, . 

Tips4: Do not vague words like "maybe", "perhaps", "good", "excellent", "most", etc. 

https://s3.amazonaws.com/mturk_bulk/hits/370501562/uNGk1Dz1zM48BZI6mALfxA.html 

2/4 

11/8/2019 

HIT 

**non-trivial** facts for this given table: 

Table Source: athletics at the 1952 summer olympics - men 's pole vault 

(https://en.wikipedia.org/wiki/Athletics_at_the_1952_Summer_Olympics_%E2%80%93_Men%27s_pole_vault) 

|**athlete**|**nationality**|**3.60**|**3.80**|**3.95**|**result**|
|---|---|---|---|---|---|
|bob richards|united states|-|-|o|4.55 or|
|don laz|united states|-|-|o|4.50|
|ragnar lundberg|sweden|-|-|o|4.40|
|petro denysenko|soviet union|-|-|o|4.40|
|valto olenius|finland|-|-|-|4.30|
|bunkichi sawada|japan|-|o|xxo|4.20|
|volodymyr brazhnyk|soviet union|-|o|o|4.20|
|viktor knyazev|soviet union|-|o|o|4.20|
|george mattos|united states|-|-|o|4.20|
|erkki kataja|finland|-|-|o|4.10|
|tamás homonnay|sweden|-|o|o|4.10|
|lennart lind|hungary|-|o|o|4.10|
|milan milakov|yugoslavia|-|o|xo|4.10|
|rigas efstathiadis|greece|-|o|o|3.95|
|torfy bryngeirsson|iceland|-|o|o|3.95|
|erling kaas|norway|-|o|xxx|3.80|
|theodosios balafas|greece|o|o|xxx|3.80|
|jukka piironen|finland|-|xo|xx|3.80|
|zeno dragomir|romania|-|xo|xx|3.80|



Please write a non-trivial statement, minimum 9 words 

Please write a non-trivial statement, minimum 9 words 

Please write a non-trivial statement, minimum 9 words 

Please write a non-trivial statement, minimum 9 words 

Please write a non-trivial statement, minimum 9 words 

https://s3.amazonaws.com/mturk_bulk/hits/370501562/uNGk1Dz1zM48BZI6mALfxA.html 

3/4 

11/8/2019 

HIT 

## **Survey Instructions** (Click to expand) 

Please first read a table to understand its content, an example is shown below, which contains the leaderboard of a competition. 

|**Player**|**Original Season**|**Gender**|**Eliminated**|**Placing**|
|---|---|---|---|---|
|Derrick Kosinski|RR: X-Treme|Male|Winner|Winner|
|Evelyn Smith|Fresh Meat|Female|Winner|Winner|
|Johnny Devenanzio|RW: Key West|Male|Winner|Winner|
|Kenny Santucci|Fresh Meat|Male|Winner|Winner|
|Jenn Grijalva|RW: Denver|Female|Episode 8|Runner-Up|
|Paula Meronek|RW: Key West|Female|Episode 8|Runner-Up|
|Robin Hibbard|RW: San Diego|Female|Episode 8|Runner-Up|
|Ryan Kehoe|Fresh Meat|Male|Episode 8|Runner-Up|
|Dunbar Merrill|RW: Sydney|Male|Episode 8|9th Place|
|Johanna Botta|RW: Austin|Female|Episode 8|10th Place|
|KellyAnne Judd|RW: Sydney|Female|Episode 8|11th Place|
|Dan Walsh|RR: Viewers' Revenge|Male|Episode 8|12th Place|
|Colie Edison|RW: Denver|Female|Episode 7|13th Place|
|Cohutta Grindstaff|RW: Sydney|Male|Episode 6|14th Place|
|Tyrie Ballard|RW: Denver|Male|Episode 5|15th Place|
|Ashli Robson|RW: Sydney|Female|Episode 4|16th Place|
|Rachel Robinson|RR: Campus Crawl|Female|Episode 3|17th Place|
|Abram Boise|RR: South Pacific|Male|Episode 2|18th Place|
|Dave Malinosky|RW: Hollywood|Male|Episode 2 (quit)|19th Place|
|Tonya Cooley|RW: Chicago|Female|Episode 1|20th Place|



You are given a sentence to describe a fact in the table, please follow the following two cases to finish the job: 

## 

- **"fake" based on the following criteria** : 

1. Contradictory: it should still be a fluent and coherent, **but it needs be explicitly contrdictory to the facts in the table** . 

2. Do not simply add **NOT** to revert the sentence meaning. 

3. Do not write 

3. The fake statement needs to be clear, explicit and natural, do not use vague or ambiguous words like 

- "bad", "good", "many", etc. 

4. try to use diverse fake types during annotatoin. 

Example 1. Given statement: Ashli Robson was eliminated in episode 4. Good Faking: Ashli Robson survives through episode 1 to episode 5. Good Faking: Ashli Robson is not the only one eliminated in episode 4. Bad Faking (Simply add not): Ashli Robson was not eliminated on episode 4. 

https://s3.amazonaws.com/mturk_bulk/hits/391922557/v_5b2TrRmw9TnD5hSI_CnA.html 

1/3 

11/8/2019 

HIT 

Bad Faking (Ambiguous, who is Ashli?): Ashli was not eliminated on episode 4. 

Bad Faking (Irrelevant): Ashli was born in Mexico. 

Bad Faking (Too subjective, what do you mean by "early"): AshlDerrick Kosinski lost the game very early. Bad Faking (Not verifiable): AshlDerrick Kosinski was the most popular player. 

Example 2. Given statement: Tonya Cooley is in the 20th place. 

Good Faking: Tonya Cooley is not the last in placing. 

Good Faking: Tonya Cooley is eliminated in episode 1 but not the last in placing. Bad Faking: (There is nothing larger than 20th) Tonya Cooley is after the 20th place. Bad Faking: (Half Wrong/half Right) When the gneder is female, the player is Tonya Colley. Bad Faking (Introduce values outside the table): Tonya Cooley is in the 43th place. Bad Faking (Typo): Tonya Cooler is in the 20th palace. 

## *** If the given statement is erroneous (see following), please type in N/A in the input box.** 

1. critical grammar error like missing verbs, nouns, etc. Do not count small errors like tense, singular/plural, case errors. 

2. serious typo, misspelling. 

3. the described fact is contradictory to the table. 

**upper or lower case, not important** 

https://s3.amazonaws.com/mturk_bulk/hits/391922557/v_5b2TrRmw9TnD5hSI_CnA.html 

2/3 

11/8/2019 

HIT 

First Read the given tables, then rewrite the statements to make them **fake** : 

Table Source: 2003 - 04 isu junior grand prix 

(https://en.wikipedia.org/wiki/2003%E2%80%9304_ISU_Junior_Grand_Prix) 

|**rank**|**nation**|**gold**|**silver**|**bronze**|**total**|
|---|---|---|---|---|---|
|1|russia|10|14|8|32|
|2|united states|9|6|7|22|
|3|canada|4|2|10|16|
|4|japan|4|5|4|13|
|5|hungary|4|0|2|6|
|6|czech republic|2|1|1|4|
|6|ukraine|1|3|0|4|
|6|italy|0|1|3|4|
|7|sweden|1|2|0|3|
|8|israel|1|1|0|2|
|9|finland|0|0|1|1|
|9|france|0|1|0|1|
|Hightlight Mentions, Click Me!||||||



Hightlight Mentions, Click Me! 

Given Statement: russia won the most silver medals in the grand prix 

Please rewrite a sentence which is contradictory to the table 

## Hightlight Mentions, Click Me! 

Given Statement: france and finland won the least medals in the grand prix 

Please rewrite a sentence which is contradictory to the table 

## Hightlight Mentions, Click Me! 

Given Statement: hungary and finland were the only countries that idd not win any silver medals 

Please rewrite a sentence which is contradictory to the table 

## Hightlight Mentions, Click Me! 

Given Statement: the united states won more gold medals than canada 

Please rewrite a sentence which is contradictory to the table 

## Hightlight Mentions, Click Me! 

Given Statement: canada won the most bronze medals in the grand prix 

Please rewrite a sentence which is contradictory to the table 

Submit 

https://s3.amazonaws.com/mturk_bulk/hits/391922557/v_5b2TrRmw9TnD5hSI_CnA.html 

3/3 

