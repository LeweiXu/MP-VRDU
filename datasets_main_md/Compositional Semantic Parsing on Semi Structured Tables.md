## **Compositional Semantic Parsing on Semi-Structured Tables** 

**Panupong Pasupat** Computer Science Department Stanford University `ppasupat@cs.stanford.edu` 

## **Percy Liang** 

Computer Science Department Stanford University `pliang@cs.stanford.edu` 

## **Abstract** 

Two important aspects of semantic parsing for question answering are the breadth of the knowledge source and the depth of logical compositionality. While existing work trades off one aspect for another, this paper simultaneously makes progress on both fronts through a new task: answering complex questions on semi-structured tables using question-answer pairs as supervision. The central challenge arises from two compounding factors: the broader domain results in an open-ended set of relations, and the deeper compositionality results in a combinatorial explosion in the space of logical forms. We propose a logical-form driven parsing algorithm guided by strong typing constraints and show that it obtains significant improvements over natural baselines. For evaluation, we created a new dataset of 22,033 complex questions on Wikipedia tables, which is made publicly available. 

## **1 Introduction** 

In semantic parsing for question answering, natural language questions are converted into logical forms, which can be executed on a knowledge source to obtain answer denotations. Early semantic parsing systems were trained to answer highly compositional questions, but the knowledge sources were limited to small closed-domain databases (Zelle and Mooney, 1996; Wong and Mooney, 2007; Zettlemoyer and Collins, 2007; Kwiatkowski et al., 2011). More recent work sacrifices compositionality in favor of using more open-ended knowledge bases such as Freebase (Cai and Yates, 2013; Berant et al., 2013; Fader et al., 2014; Reddy et al., 2014). However, even these broader knowledge sources still define a 

|**Year**|**City**|**Country**|**Nations**|
|---|---|---|---|
|1896<br>1900<br>1904<br>. . .<br>2004<br>2008<br>2012|Athens<br>Paris<br>St. Louis<br>. . .<br>Athens<br>Beijing<br>London|Greece<br>France<br>USA<br>. . .<br>Greece<br>China<br>UK|14<br>24<br>12<br>. . .<br>201<br>204<br>204|



_x_ 1: _“Greece held its last Summer Olympics in which year?” y_ 1: _{_ 2004 _}_ 

_x_ 2: _“In which city’s the first time with at least 20 nations?” y_ 2: _{_ Paris _}_ 

_x_ 3: _“Which years have the most participating countries?” y_ 3: _{_ 2008, 2012 _}_ 

_x_ 4: _“How many events were in Athens, Greece?” y_ 4: _{_ 2 _}_ 

_x_ 5: _“How many more participants were there in 1900 than in the first year?” y_ 5: _{_ 10 _}_ 

Figure 1: Our task is to answer a highly compositional question from an HTML table. We learn a semantic parser from question-table-answer triples _{_ ( _xi, ti, yi_ ) _}_ . 

rigid schema over entities and relation types, thus restricting the scope of answerable questions. 

To simultaneously increase both the _breadth_ of the knowledge source and the _depth_ of logical compositionality, we propose a new task (with an associated dataset): answering a question using an HTML table as the knowledge source. Figure 1 shows several question-answer pairs and an accompanying table, which are typical of those in our dataset. Note that the questions are logically quite complex, involving a variety of operations such as comparison ( _x_ 2), superlatives ( _x_ 3), aggregation ( _x_ 4), and arithmetic ( _x_ 5). 

The HTML tables are semi-structured and not normalized. For example, a cell might contain multiple parts (e.g., “ _Beijing, China_ ” or “ _200 km_ ”). Additionally, we mandate that the training and test tables are disjoint, so at test time, we will see relations (column headers; e.g., “ _Nations_ ”) and entities (table cells; e.g., “ _St. Louis_ ”) 

that were not observed during training. This is in contrast to knowledge bases like Freebase, which have a global fixed relation schema with normalized entities and relations. 

Our task setting produces two main challenges. Firstly, the increased breadth in the knowledge source requires us to generate logical forms from novel tables with previously unseen relations and entities. We therefore cannot follow the typical semantic parsing strategy of constructing or learning a lexicon that maps phrases to relations ahead of time. Secondly, the increased depth in compositionality and additional logical operations exacerbate the exponential growth of the number of possible logical forms. 

We trained a semantic parser for this task from question-answer pairs based on the framework illustrated in Figure 2. First, relations and entities from the semi-structured HTML table are encoded in a graph. Then, the system parses the question into candidate logical forms with a high-coverage grammar, reranks the candidates with a log-linear model, and then executes the highest-scoring logical form to produce the answer denotation. We use beam search with pruning strategies based on type and denotation constraints to control the combinatorial explosion. 

To evaluate the system, we created a new dataset, WIKITABLEQUESTIONS, consisting of 2,108 HTML tables from Wikipedia and 22,033 question-answer pairs. When tested on unseen tables, the system achieves an accuracy of 37.1%, which is significantly higher than the information retrieval baseline of 12.7% and a simple semantic parsing baseline of 24.3%. 

## **2 Task** 

Our task is as follows: given a table _t_ and a question _x_ about the table, output a list of values _y_ that answers the question according to the table. Example inputs and outputs are shown in Figure 1. The system has access to a training set _D_ = _{_ ( _xi, ti, yi_ ) _}[N] i_ =1[of questions, tables, and an-] swers, but the tables in test data do not appear during training. 

The only restriction on the question _x_ is that a person must be able to answer it using just the table _t_ . Other than that, the question can be of any type, ranging from a simple table lookup question to a more complicated one that involves various logical operations. 

**==> picture [199 x 148] intentionally omitted <==**

**----- Start of picture text -----**<br>
xxx xxxx xxxxx xxxxx<br>t xxxxxxxxxxxx...xxxxxxxxxxxx xxxxxxxxxxxxxxxxx...xxxxxxxxxxxxxxxxxxx xxxxxxxxxxxxxxx...xxxxxxxxxxxxxx xxxxxx...xxxxxxxxx<br>Greece held the last<br>Summer Olympics in (1) Conversion<br>which year?<br>x (2) Parsing w<br>Zx<br>(3) Ranking<br>z (4) Execution y<br>λ [ Year  . . .  ] . argmax ( . . .  Greece ,  Index ) { 2004 }<br>**----- End of picture text -----**<br>


Figure 2: The prediction framework: (1) the table _t_ is deterministically converted into a knowledge graph _w_ as shown in Figure 3; (2) with information from _w_ , the question _x_ is parsed into candidate logical forms in _Zx_ ; (3) the highest-scoring candidate _z ∈Zx_ is chosen; and (4) _z_ is executed on _w_ , yielding the answer _y_ . 

**Dataset.** We created a new dataset, WIKITABLEQUESTIONS, of question-answer pairs on HTML tables as follows. We randomly selected data tables from Wikipedia with at least 8 rows and 5 columns. We then created two Amazon Mechanical Turk tasks. The first task asks workers to write trivia questions about the table. For each question, we put one of the 36 generic prompts such as _“The question should require calculation”_ or _“contains the word_ ‘first’ _or its synonym”_ to encourage more complex utterances. Next, we submit the resulting questions to the second task where the workers answer each question based on the given table. We only keep the answers that are agreed upon by at least two workers. After this filtering, approximately 69% of the questions remains. 

The final dataset contains 22,033 examples on 2,108 tables. We set aside 20% of the tables and their associated questions as the test set and develop on the remaining examples. Simple preprocessing was done on the tables: We omit all non-textual contents of the tables, and if there is a merged cell spanning many rows or columns, we unmerge it and duplicate its content into each unmerged cell. Section 7.2 analyzes various aspects of the dataset and compares it to other datasets. 

## **3 Approach** 

We now describe our semantic parsing framework for answering a given question and for training the model with question-answer pairs. 

**Prediction.** Given a table _t_ and a question _x_ , we predict an answer _y_ using the framework illustrated in Figure 2. We first convert the table _t_ into a _knowledge graph w_ (“world”) which encodes different relations in the table (Section 4). Next, we generate a set of candidate logical forms _Zx_ by parsing the question _x_ using the information from _w_ (Section 6.1). Each generated logical form _z ∈Zx_ is a graph query that can be executed on the knowledge graph _w_ to get a _denotation_ � _z_ � _w_ . We extract a feature vector _φ_ ( _x, w, z_ ) for each _z ∈Zx_ (Section 6.2) and define a loglinear distribution over the candidates: 

**==> picture [188 x 13] intentionally omitted <==**

where _θ_ is the parameter vector. Finally, we choose the logical form _z_ with the highest model probability and execute it on _w_ to get the answer denotation _y_ = � _z_ � _w_ . 

**Training.** Given training examples _D_ = _{_ ( _xi, ti, yi_ ) _}[N] i_ =1[,][we][seek][a][parameter][vector] _[θ]_ that maximizes the regularized log-likelihood of the correct denotation _yi_ marginalized over logical forms _z_ . Formally, we maximize the objective function 

**==> picture [213 x 33] intentionally omitted <==**

where _wi_ is deterministically generated from _ti_ , and 

**==> picture [205 x 27] intentionally omitted <==**

We optimize _θ_ using AdaGrad (Duchi et al., 2010), running 3 passes over the data. We use _L_ 1 regularization with _λ_ = 3 _×_ 10 _[−]_[5] obtained from cross-validation. 

The following sections explain individual system components in more detail. 

## **4 Knowledge graph** 

Inspired by the graph representation of knowledge bases, we preprocess the table _t_ by deterministically converting it into a _knowledge graph w_ as illustrated in Figure 3. In the most basic form, table rows become row nodes, strings in table cells become entity nodes,[1] and table columns become directed edges from the row nodes to the entity 

1Two occurrences of the same string constitute one node. 

**==> picture [207 x 109] intentionally omitted <==**

**----- Start of picture text -----**<br>
Index Year City Country · · ·<br>Next<br>0 1896 Athens Greece<br>Index Year City Country · · ·<br>Next<br>1 1900 Paris France<br>Number Date<br>...<br>1900.0 1900-XX-XX<br>**----- End of picture text -----**<br>


Figure 3: Part of the knowledge graph corresponding to the table in Figure 1. Circular nodes are row nodes. We augment the graph with different entity normalization nodes such as `Number` and `Date` (red) and additional row node relations `Next` and `Index` (blue). 

nodes of that column. The column headers are used as edge labels for these row-entity relations. 

The knowledge graph representation is convenient for three reasons. First, we can encode different forms of entity normalization in the graph. Some entity strings (e.g., _“1900”_ ) can be interpreted as a number, a date, or a proper name depending on the context, while some other strings (e.g., _“200 km”_ ) have multiple parts. Instead of committing to one normalization scheme, we introduce edges corresponding to different normalization methods from the entity nodes. For example, the node `1900` will have an edge called `Date` to another node _1900-XX-XX_ of type date. Apart from type checking, these normalization nodes also aid learning by providing signals on the appropriate answer type. For instance, we can define a feature that associates the phrase _“how many”_ with a logical form that says “traverse a row-entity edge, then a `Number` edge” instead of just “traverse a row-entity edge.” 

The second benefit of the graph representation is its ability to handle various logical phenomena via graph augmentation. For example, to answer questions of the form _“What is the next ...?”_ or _“Who came before ...?”_ , we augment each row node with an edge labeled `Next` pointing to the next row node, after which the questions can be answered by traversing the `Next` edge. In this work, we choose to add two special edges on each row node: the `Next` edge mentioned above and an `Index` edge pointing to the row index number ( _0, 1, 2, . . ._ ). 

Finally, with a graph representation, we can query it directly using a logical formalism for knowledge graphs, which we turn to next. 

|**Name**<br>**Example**<br>Join<br>`City`_._`Athens`<br>(row nodes with a`City`edge to`Athens`)<br>Union<br>`City`_._(`Athens`_⊔_`Beijing`)<br>Intersection<br>`City`_._`Athens`_⊓_`Year`_._`Number`_._`<`_.1990_<br>Reverse<br>**R**[`Year`]_._`City`_._`Athens`<br>(entities where a row in`City`_._`Athens`has a`Year`edge to)<br>Aggregation<br>`count`(`City`_._`Athens`)<br>(the number of rows with city`Athens`)<br>Superlative<br>`argmax`(`City`_._`Athens`_,_`Index`)<br>(the last row with city`Athens`)<br>Arithmetic<br>`sub`(_204, 201_)<br>(= 204_−_201)<br>Lambda<br>_λx_[`Year`_._`Date`_.x_]<br>(a binary: composition of two relations)|**Rule**<br>**Semantics**<br>**Example**|
|---|---|
||**_Anchored to the utterance_**<br>_TokenSpan →Entity_<br>match(_z_1)<br>`Greece`<br>(match(_s_)= entity with name_s_)<br>anchored to_“Greece”_<br>_TokenSpan →Atomic_<br>val(_z_1)<br>_2012-07-XX_<br>(val(_s_)= interpreted value)<br>anchored to_“July 2012”_|
||**_Unanchored (foating)_**<br>_∅→Relation_<br>_r_<br>`Country`<br>(_r_= row-entity relation)<br>_∅→Relation_<br>_λx_[_r.p.x_]<br>_λx_[`Year`_._`Date`_.x_]<br>(_p_= normalization relation)<br>_∅→Records_<br>`Type`_._`Row`<br>(list of all rows)<br>_∅→RecordFn_<br>`Index`<br>(row_←_row index)|
|||



Table 1: The lambda DCS operations we use. 

## **5 Logical forms** 

Table 2: Base deduction rules. Entities and atomic values (e.g., numbers, dates) are anchored to token spans, while other predicates are kept floating. ( _a ← b_ represents a binary mapping _b_ to _a_ .) 

As our language for logical forms, we use lambda dependency-based compositional semantics (Liang, 2013), or lambda DCS, which we briefly describe here. Each lambda DCS logical form is either a _unary_ (denoting a list of values) or a _binary_ (denoting a list of pairs). The most basic unaries are singletons (e.g., `China` represents an entity node, and _30_ represents a single number), while the most basic binaries are relations (e.g., `City` maps rows to city entities, `Next` maps rows to rows, and `>=` maps numbers to numbers). Logical forms can be combined into larger ones via various operations listed in Table 1. Each operation produces a unary except lambda abstraction: _λx_ [ _f_ ( _x_ )] is a binary mapping _x_ to _f_ ( _x_ ). 

## **6 Parsing and ranking** 

Given the knowledge graph _w_ , we now describe how to parse the utterance _x_ into a set of candidate logical forms _Zx_ 

## **6.1 Parsing algorithm** 

We propose a new _floating parser_ which is more flexible than a standard chart parser. Both parsers recursively build up derivations and corresponding logical forms by repeatedly applying deduction rules, but the floating parser allows logical form predicates to be generated independently from the utterance. 

**Chart parser.** We briefly review the CKY algorithm for chart parsing to introduce notation. Given an utterance with tokens _x_ 1 _, . . . , xn_ , the CKY algorithm applies deduction rules of the fol- 

lowing two kinds: 

**==> picture [207 x 45] intentionally omitted <==**

The first rule is a lexical rule that matches an utterance token span _xi · · · xj_ (e.g., _s_ = _“New York”_ ) and produces a logical form (e.g., _f_ ( _s_ ) = `NewYorkCity` ) with category _c_ (e.g., `Entity` ). The second rule takes two adjacent spans giving rise to logical forms _z_ 1 and _z_ 2 and builds a new logical form _f_ ( _z_ 1 _, z_ 2). Algorithmically, CKY stores derivations of category _c_ covering the span _xi · · · xj_ in a _cell_ ( _c, i, j_ ). CKY fills in the cells of increasing span lengths, and the logical forms in the top cell ( _ROOT,_ 1 _, n_ ) are returned. 

**Floating parser.** Chart parsing uses lexical rules (4) to generate relevant logical predicates, but in our setting of semantic parsing on tables, we do not have the luxury of starting with or inducing a full-fledged lexicon. Moreover, there is a mismatch between words in the utterance and predicates in the logical form. For instance, consider the question _“Greece held its last Summer Olympics in which year?”_ on the table in Figure 1 and the correct logical form **R** [ _λx_ [ `Year` _._ `Date` _.x_ ]] _._ `argmax` ( `Country` _._ `Greece` _,_ `Index` ). While the entity `Greece` can be anchored to the token _“Greece”_ , some logical predicates (e.g., `Country` ) cannot be clearly anchored to a token span. We could potentially learn to anchor the logical form `Country` _._ `Greece` to _“Greece”_ , but if the relation `Country` is not seen during training, such a mapping is impossible to learn from the 

**Rule Semantics Example** _**Join + Aggregate** Entity_ or _Atomic → Values z_ 1 `China` _Atomic → Values c.z_ 1 `>=` _.30_ (at least 30) ( _c ∈{_ `<` _,_ `>` _,_ `<=` _,_ `>=` _}_ ) _Relation_ + _Values → Records z_ 1 _.z_ 2 `Country` _._ `China` (events (rows) where the country is China) _Relation_ + _Records → Values_ **R** [ _z_ 1] _.z_ 2 **R** [ `Year` ] _._ `Country` _._ `China` (years of events in China) _Records → Records_ `Next` _.z_ 1 `Next` _._ `Country` _._ `China` (. . . before China) _Records → Records_ **R** [ `Next` ] _.z_ 1 **R** [ `Next` ] _._ `Country` _._ `China` (. . . after China) _Values → Atomic a_ ( _z_ 1) `count` ( `Country` _._ `China` ) (How often did China . . . ) ( _a ∈{_ `count` _,_ `max` _,_ `min` _,_ `sum` _,_ `avg` _}_ ) _Values → ROOT z_ 1 _**Superlative** Relation → RecordFn z_ 1 _λx_ [ `Nations` _._ `Number` _.x_ ] (row _←_ value in Nations column) _Records_ + _RecordFn → Records s_ ( _z_ 1 _, z_ 2) `argmax` ( `Type` _._ `Row` _, λx_ [ `Nations` _._ `Number` _.x_ ]) ( _s ∈{_ `argmax` _,_ `argmin` _}_ ) (events with the most participating nations) `argmin` ( `City` _._ `Athens` _,_ `Index` ) (first event in Athens) _Relation → ValueFn_ **R** [ _λx_ [ _a_ ( _z_ 1 _.x_ )]] **R** [ _λx_ [ `count` ( `City` _.x_ )]] (city _←_ num. of rows with that city) _Relation_ + _Relation → ValueFn λx_ [ **R** [ _z_ 1] _.z_ 2 _.x_ ] _λx_ [ **R** [ `City` ] _._ `Nations` _._ `Number` _.x_ ] (city _←_ value in Nations column) _Values_ + _ValueFn → Values s_ ( _z_ 1 _, z_ 2) `argmax` ( _. . . ,_ **R** [ _λx_ [ `count` ( `City` _.x_ )]]) (most frequent city) _**Other operations** ValueFn_ + _Values_ + _Values → Values o_ ( **R** [ _z_ 1] _.z_ 2 _,_ **R** [ _z_ 1] _.z_ 3) `sub` ( **R** [ `Number` ] _._ **R** [ `Nations` ] _._ `City` _._ `London` _, . . ._ ) ( _o ∈{_ `add` _,_ `sub` _,_ `mul` _,_ `div` _}_ ) (How many more participants were in London than . . . ) _Entity_ + _Entity → Values z_ 1 _⊔ z_ 2 `China` _⊔_ `France` (China or France) _Records_ + _Records → Records z_ 1 _⊓ z_ 2 `City` _._ `Beijing` _⊓_ `Country` _._ `China` (. . . in Beijing, China) 

Table 3: Compositional deduction rules. Each rule _c_ 1 _, . . . , ck → c_ takes logical forms _z_ 1 _, . . . , zk_ constructed over categories _c_ 1 _, . . . , ck_ , respectively, and produces a logical form based on the semantics. 

training data. Similarly, some prominent tokens (e.g., _“Olympics”_ ) are irrelevant and have no predicates anchored to them. 

Therefore, instead of anchoring each predicate in the logical form to tokens in the utterance via lexical rules, we propose parsing more freely. We replace the anchored cells ( _c, i, j_ ) with _floating cells_ ( _c, s_ ) of category _c_ and logical form size _s_ . Then we apply rules of the following three kinds: 

**==> picture [220 x 11] intentionally omitted <==**

**==> picture [140 x 12] intentionally omitted <==**

**==> picture [220 x 12] intentionally omitted <==**

**==> picture [133 x 12] intentionally omitted <==**

Note that rules (6) are similar to (4) in chart parsing except that the floating cell ( _c,_ 1) only keeps track of the category and its size 1, not the span ( _i, j_ ). Rules (7) allow us to construct predicates out of thin air. For example, we can construct a logical form representing a table relation `Country` in cell ( _Relation,_ 1) using the rule _∅→ Relation_ [ `Country` ] independent of the utterance. Rules (8) perform composition, where the induction is on the size _s_ of the logical form rather than the span length. The algorithm stops when the specified maximum size is reached, after which the logical forms in cells ( _ROOT, s_ ) for any 

( _Values,_ 8) **R** [ _λx_ [ `Year` _._ `Date` _.x_ ]] _._ `argmax` ( `Country` _._ `Greece` _,_ `Index` ) 

**==> picture [209 x 164] intentionally omitted <==**

Figure 4: A derivation for the utterance _“Greece held its last Summer Olympics in which year?”_ Only `Greece` is anchored to a phrase _“Greece”_ ; `Year` and other predicates are floating. 

_s_ are included in _Zx_ . Figure 4 shows an example derivation generated by our floating parser. 

The floating parser is very flexible: it can skip tokens and combine logical forms in any order. This flexibility might seem too unconstrained, but we can use strong typing constraints to prevent nonsensical derivations from being constructed. 

Tables 2 and 3 show the full set of deduction 

_“Greece held its last Summer Olympics in which year?” z_ = **R** [ _λx_ [ `Year` _._ `Number` _.x_ ]] _._ `argmax` ( `Type` _._ `Row` _,_ `Index` ) _y_ = _{2012}_ (type: NUM, column: YEAR) 

|**Feature Name**|**Note**|
|---|---|
|(_“last”,_predicate=`argmax`)|lex|
|phrase= predicate|unlex(∵_“year”_=`Year`)|
|missingentity|unlex(∵missing_Greece_)|
|denotation type= NUM||
|denotation column= YEAR||
|(_“which year”,_type= NUM)|lex|
|phrase=column|unlex(∵_“year”_= YEAR)|
|(_Q_=_“which”,_type= NUM)|lex|
|(_H_ =_“year”,_type= NUM)|lex|
|_H_ =column|unlex(∵_“year”_= YEAR)|



Table 4: Example features that fire for the (incorrect) logical form _z_ . All features are binary. (lex = lexicalized) 

rules we use. We assume that all named entities will explicitly appear in the question _x_ , so we anchor all entity predicates (e.g., `Greece` ) to token spans (e.g., _“Greece”_ ). We also anchor all numerical values (numbers, dates, percentages, etc.) detected by an NER system. In contrast, relations (e.g., `Country` ) and operations (e.g., `argmax` ) are kept floating since we want to learn how they are expressed in language. Connections between phrases in _x_ and the generated relations and operations in _z_ are established in the ranking model through features. 

## **6.2 Features** 

We define features _φ_ ( _x, w, z_ ) for our log-linear model to capture the relationship between the question _x_ and the candidate _z_ . Table 4 shows some example features from each feature type. Most features are of the form ( _f_ ( _x_ ) _, g_ ( _z_ )) or ( _f_ ( _x_ ) _, h_ ( _y_ )) where _y_ = � _z_ � _w_ is the denotation, and _f_ , _g_ , and _h_ extract some information (e.g., identity, POS tags) from _x_ , _z_ , or _y_ , respectively. 

**phrase-predicate:** Conjunctions between n- grams _f_ ( _x_ ) from _x_ and predicates _g_ ( _z_ ) from _z_ . We use both lexicalized features, where all possible pairs ( _f_ ( _x_ ) _, g_ ( _z_ )) form distinct features, and binary unlexicalized features indicating whether _f_ ( _x_ ) and _g_ ( _z_ ) have a string match. 

**missing-predicate:** Indicators on whether there are entities or relations mentioned in _x_ but not in _z_ . These features are unlexicalized. 

**denotation:** Size and type of the denotation _y_ = � _x_ � _w_ . The type can be either a primitive type (e.g., NUM, DATE, ENTITY) or the name of the column containing the entity in _y_ (e.g., CITY). **phrase-denotation:** Conjunctions between n- 

grams from _x_ and the types of _y_ . Similar to the phrase-predicate features, we use both lexicalized and unlexicalized features. 

**headword-denotation:** Conjunctions between the question word _Q_ (e.g., _what_ , _who_ , _how many_ ) or the headword _H_ (the first noun after the question word) with the types of _y_ . 

## **6.3 Generation and pruning** 

Due to their recursive nature, the rules allow us to generate highly compositional logical forms. However, the compositionality comes at the cost of generating exponentially many logical forms, most of which are redundant (e.g., logical forms with an `argmax` operation on a set of size 1). We employ several methods to deal with this combinatorial explosion: 

**Beam search.** We compute the model probability of each partial logical form based on available features (i.e., features that do not depend on the final denotation) and keep only the _K_ = 200 highest-scoring logical forms in each cell. 

**Pruning.** We prune partial logical forms that lead to invalid or redundant final logical forms. For example, we eliminate any logical form that does not type check (e.g., `Beijing` _⊔_ `Greece` ), executes to an empty list (e.g., `Year` _._ `Number` _.24_ ), includes an aggregate or superlative on a singleton set (e.g., `argmax` ( `Year` _._ `Number` _.2012,_ `Index` )), or joins two relations that are the reverses of each other (e.g., **R** [ `City` ] _._ `City` _._ `Beijing` ). 

## **7 Experiments** 

## **7.1 Main evaluation** 

We evaluate the system on the development sets (three random 80:20 splits of the training data) and the test data. In both settings, the tables we test on do not appear during training. 

**Evaluation metrics.** Our main metric is _accuracy_ , which is the number of examples ( _x, t, y_ ) on which the system outputs the correct answer _y_ . We also report the _oracle_ score, which counts the number of examples where at least one generated candidate _z ∈Zx_ executes to _y_ . 

**Baselines.** We compare the system to two baselines. The first baseline (IR), which simulates information retrieval, selects an answer _y_ among the entities in the table using a log-linear model over entities (table cells) rather than logical forms. The features are conjunctions between phrases in _x_ and 

||**dev**|**dev**|**test**|**test**|
|---|---|---|---|---|
||**acc**|**ora**|**acc**|**ora**|
|IR baseline|13.4|69.1|12.7|70.6|
|WQ baseline|23.6|34.4|24.3|35.6|
|Our system|37.0|76.7|37.1|76.6|



Table 5: Accuracy (acc) and oracle scores (ora) on the development sets (3 random splits of the training data) and the test data. 

|||**acc**|**ora**|
|---|---|---|---|
||**Our system**|37.0|76.7|
|(a)|**Rule Ablation**|||
||join only|10.6|15.7|
||join + count (= WQ baseline)|23.6|34.4|
||join + count + superlative|30.7|68.6|
||all_−{⊓, ⊔}_|34.8|75.1|
|(b)|**Feature Ablation**|||
||all_−_features involving predicate|11.8|74.5|
||all_−_phrase-predicate|16.9|74.5|
||all_−_lex phrase-predicate|17.6|75.9|
||all_−_unlex phrase-predicate|34.3|76.7|
||all_−_missing-predicate|35.9|76.7|
||all_−_features involving denotation|33.5|76.8|
||all_−_denotation|34.3|76.6|
||all_−_phrase-denotation|35.7|76.8|
||all_−_headword-denotation|36.0|76.7|
|(c)|**Anchor operations to trigger words**|37.1|59.4|



Table 6: Average accuracy and oracle scores on development data in various system settings. 

properties of the answers _y_ , which cover all features in our main system that do not involve the logical form. As an upper bound of this baseline, 69.1% of the development examples have the answer appearing as an entity in the table. 

In the second baseline (WQ), we only allow deduction rules that produce join and count logical forms. This rule subset has the same logical coverage as Berant and Liang (2014), which is designed to handle the WEBQUESTIONS (Berant et al., 2013) and FREE917 (Cai and Yates, 2013) datasets. 

**Results.** Table 5 shows the results compared to the baselines. Our system gets an accuracy of 37.1% on the test data, which is significantly higher than both baselines, while the oracle is 76.6%. The next subsections analyze the system components in more detail. 

## **7.2 Dataset statistics** 

In this section, we analyze the breadth and depth of the WIKITABLEQUESTIONS dataset, and how the system handles them. 

**Number of relations.** With 3,929 unique column headers (relations) among 13,396 columns, 

|**Operation**|**Amount**|
|---|---|
|join (table lookup)|13.5%|
|+ join with`Next`|+ 5.5%|
|+ aggregate (`count`,`sum`,`max`, . . . )|+ 15.0%|
|+ superlative (`argmax`,`argmin`)|+ 24.5%|
|+ arithmetic,_⊓_,_⊔_|+ 20.5%|
|+ otherphenomena|+ 21.0%|



Table 7: The logical operations required to answer the questions in 200 random examples. 

the tables in the WIKITABLEQUESTIONS dataset contain many more relations than closed-domain datasets such as Geoquery (Zelle and Mooney, 1996) and ATIS (Price, 1990). Additionally, the logical forms that execute to the correct denotations refer to a total of 2,056 unique column headers, which is greater than the number of relations in the FREE917 dataset (635 Freebase relations). 

**Knowledge coverage.** We sampled 50 examples from the dataset and tried to answer them manually using Freebase. Even though Freebase contains some information extracted from Wikipedia, we can answer only 20% of the questions, indicating that WIKITABLEQUESTIONS contains a broad set of facts beyond Freebase. 

**Logical operation coverage.** The dataset covers a wide range of question types and logical operations. Table 6(a) shows the drop in oracle scores when different subsets of rules are used to generate candidates logical forms. The _join only_ subset corresponds to simple table lookup, while _join + count_ is the WQ baseline for Freebase question answering on the WEBQUESTIONS dataset. Finally, _join + count + superlative_ roughly corresponds to the coverage of the Geoquery dataset. 

To better understand the distribution of logical operations in the WIKITABLEQUESTIONS dataset, we manually classified 200 examples based on the types of operations required to answer the question. The statistics in Table 7 shows that while a few questions only require simple operations such as table lookup, the majority of the questions demands more advanced operations. Additionally, 21% of the examples cannot be answered using any logical form generated from the current deduction rules; these examples are discussed in Section 7.4. 

**Compositionality.** From each example, we compute the logical form size (number of rules applied) of the highest-scoring candidate that executes to the correct denotation. The histogram in Figure 5 shows that a significant number of logical 

**==> picture [169 x 61] intentionally omitted <==**

**----- Start of picture text -----**<br>
2500<br>2000<br>1500<br>1000<br>500<br>0<br>2 3 4 5 6 7 8 9 10 11<br>formula size<br>frequency<br>**----- End of picture text -----**<br>


Figure 5: Sizes of the highest-scoring correct candidate logical forms in development examples. 

**==> picture [187 x 65] intentionally omitted <==**

**----- Start of picture text -----**<br>
with pruning without pruning<br>80 80<br>60 60<br>40 40<br>20 20<br>0 0<br>0 25 50 75 100 0 25 50 75 100<br>beam size beam size<br>score score<br>**----- End of picture text -----**<br>


Figure 6: Accuracy (solid red) and oracle (dashed blue) scores with different beam sizes. 

## forms are non-trivial. 

**Beam size and pruning.** Figure 6 shows the results with and without pruning on various beam sizes. Apart from saving time, pruning also prevents bad logical forms from clogging up the beam which hurts both oracle and accuracy metrics. 

## **7.3 Features** 

**Effect of features.** Table 6(b) shows the accuracy when some feature types are ablated. The most influential features are lexicalized phrasepredicate features, which capture the relationship between phrases and logical operations (e.g., relating _“last”_ to `argmax` ) as well as between phrases and relations (e.g., relating _“before”_ to `<` or `Next` , and relating _“who”_ to the relation `Name` ). 

**Anchoring with trigger words.** In our parsing algorithm, relations and logical operations are not anchored to the utterance. We consider an alternative approach where logical operations are anchored to “trigger” phrases, which are hand-coded based on co-occurrence statistics (e.g., we trigger a `count` logical form with _how_ , _many_ , and _total_ ). 

Table 6(c) shows that the trigger words do not significantly impact the accuracy, suggesting that the original system is already able to learn the relationship between phrases and operations even without a manual lexicon. As an aside, the huge drop in oracle is because fewer “semantically incorrect” logical forms are generated; we discuss this phenomenon in the next subsection. 

## **7.4 Semantically correct logical forms** 

In our setting, we face a new challenge that arises from learning with denotations: with deeper compositionality, a larger number of nonsensical logical forms can execute to the correct denotation. For example, if the target answer is a small number (say, 2), it is possible to count the number of rows with some random properties and arrive at the correct answer. However, as the system encounters more examples, it can potentially learn to disfavor them by recognizing the characteristics of semantically correct logical forms. 

**Generating semantically correct logical forms.** The system can learn the features of semantically correct logical forms only if it can generate them in the first place. To see how well the system can generate correct logical forms, looking at the oracle score is insufficient since bad logical forms can execute to the correct denotations. Instead, we randomly chose 200 examples and manually annotated them with logical forms to see if a trained system can produce the annotated logical form as a candidate. 

Out of 200 examples, we find that 79% can be manually annotated. The remaining ones include artifacts such as unhandled question types (e.g., yes-no questions, or questions with phrases _“same”_ or _“consecutive”_ ), table cells that require advanced normalization methods (e.g., cells with comma-separated lists), and incorrect annotations. 

The system generates the annotated logical form among the candidates in 53.5% of the examples. The missing examples are mostly caused by anchoring errors due to lexical mismatch (e.g., _“Italian” →_ `Italy` , or _“no zip code” →_ an empty cell in the zip code column) or the need to generate complex logical forms from a single phrase (e.g., _“May 2010” →_ `>=` _.2010-05-01⊓_ `<=` _.2010-05-31_ ). 

## **7.5 Error analysis** 

The errors on the development data can be divided into four groups. The first two groups are unhandled question types (21%) and the failure to anchor entities (25%) as described in Section 7.4. The third group is normalization and type errors (29%): although we handle some forms of entity normalization, we observe many unhandled string formats such as times (e.g., _3:45.79_ ) and city-country pairs (e.g., _Beijing, China_ ), as well as complex calculation such as computing time periods (e.g., _12pm–1am →_ 1 hour). Finally, we have 

ranking errors (25%) which mostly occur when the utterance phrase and the relation are obliquely related (e.g., _“airplane”_ and `Model` ). 

## **8 Discussion** 

Our work simultaneously increases the breadth of knowledge source and the depth of compositionality in semantic parsing. This section explores the connections in both aspects to related work. 

**Logical coverage.** Different semantic parsing systems are designed to handle different sets of logical operations and degrees of compositionality. For example, form-filling systems (Wang et al., 2011) usually cover a smaller scope of operations and compositionality, while early statistical semantic parsers for question answering (Wong and Mooney, 2007; Zettlemoyer and Collins, 2007) and high-accuracy natural language interfaces for databases (Androutsopoulos et al., 1995; Popescu et al., 2003) target more compositional utterances with a wide range of logical operations. This work aims to increase the logical coverage even further. For example, compared to the Geoquery dataset, the WIKITABLEQUESTIONS dataset includes a move diverse set of logical operations, and while it does not have extremely compositional questions like in Geoquery (e.g., _“What states border states that border states that border Florida?”_ ), our dataset contains fairly compositional questions on average. 

To parse a compositional utterance, many works rely on a lexicon that translates phrases to entities, relations, and logical operations. A lexicon can be automatically generated (Unger and Cimiano, 2011; Unger et al., 2012), learned from data (Zettlemoyer and Collins, 2007; Kwiatkowski et al., 2011), or extracted from external sources (Cai and Yates, 2013; Berant et al., 2013), but requires some techniques to generalize to unseen data. Our work takes a different approach similar to the logical form growing algorithm in Berant and Liang (2014) by not anchoring relations and operations to the utterance. 

**Knowledge domain.** Recent works on semantic parsing for question answering operate on more open and diverse data domains. In particular, large-scale knowledge bases have gained popularity in the semantic parsing community (Cai and Yates, 2013; Berant et al., 2013; Fader et al., 2014). The increasing number of relations and entities motivates new resources and techniques for 

improving the accuracy, including the use of ontology matching models (Kwiatkowski et al., 2013), paraphrase models (Fader et al., 2013; Berant and Liang, 2014), and unlabeled sentences (Krishnamurthy and Kollar, 2013; Reddy et al., 2014). 

Our work leverages open-ended data from the Web through semi-structured tables. There have been several studies on analyzing or inferring the table schemas (Cafarella et al., 2008; Venetis et al., 2011; Syed et al., 2010; Limaye et al., 2010) and answering search queries by joining tables on similar columns (Cafarella et al., 2008; Gonzalez et al., 2010; Pimplikar and Sarawagi, 2012). While the latter is similar to question answering, the queries tend to be keyword lists instead of natural language sentences. In parallel, open information extraction (Wu and Weld, 2010; Masaum et al., 2012) and knowledge base population (Ji and Grishman, 2011) extract information from web pages and compile them into structured data. The resulting knowledge base is systematically organized, but as a trade-off, some knowledge is inevitably lost during extraction and the information is forced to conform to a specific schema. To avoid these issues, we choose to work on HTML tables directly. 

In future work, we wish to draw information from other semi-structured formats such as colon-delimited pairs (Wong et al., 2009), bulleted lists (Gupta and Sarawagi, 2009), and top- _k_ lists (Zhang et al., 2013). Pasupat and Liang (2014) used a framework similar to ours to extract entities from web pages, where the “logical forms” were XPath expressions. A natural direction is to combine the logical compositionality of this work with the even broader knowledge source of general web pages. 

**Acknowledgements.** We gratefully acknowledge the support of the Google Natural Language Understanding Focused Program and the Defense Advanced Research Projects Agency (DARPA) Deep Exploration and Filtering of Text (DEFT) Program under Air Force Research Laboratory (AFRL) contract no. FA8750-13-2-0040. 

**Data and reproducibility.** The WIKITABLEQUESTIONS dataset can be downloaded at `http: //nlp.stanford.edu/software/sempre/wikitable/` . Additionally, code, data, and experiments for this paper are available on the CodaLab platform at `https://www.codalab.org/worksheets/ 0xf26cd79d4d734287868923ad1067cf4c/` . 

## **References** 

- [Androutsopoulos et al.1995] I. Androutsopoulos, G. D. Ritchie, and P. Thanisch. 1995. Natural language interfaces to databases – an introduction. _Journal of Natural Language Engineering_ , 1:29–81. 

- [Berant and Liang2014] J. Berant and P. Liang. 2014. Semantic parsing via paraphrasing. In _Association for Computational Linguistics (ACL)_ . 

- [Berant et al.2013] J. Berant, A. Chou, R. Frostig, and P. Liang. 2013. Semantic parsing on Freebase from question-answer pairs. In _Empirical Methods in Natural Language Processing (EMNLP)_ . 

- [Cafarella et al.2008] M. J. Cafarella, A. Halevy, D. Z. Wang, E. Wu, and Y. Zhang. 2008. WebTables: exploring the power of tables on the web. In _Very Large Data Bases (VLDB)_ , pages 538–549. 

- [Cai and Yates2013] Q. Cai and A. Yates. 2013. Largescale semantic parsing via schema matching and lexicon extension. In _Association for Computational Linguistics (ACL)_ . 

- [Duchi et al.2010] J. Duchi, E. Hazan, and Y. Singer. 2010. Adaptive subgradient methods for online learning and stochastic optimization. In _Conference on Learning Theory (COLT)_ . 

- [Fader et al.2013] A. Fader, L. Zettlemoyer, and O. Etzioni. 2013. Paraphrase-driven learning for open question answering. In _Association for Computational Linguistics (ACL)_ . 

- [Fader et al.2014] A. Fader, L. Zettlemoyer, and O. Etzioni. 2014. Open question answering over curated and extracted knowledge bases. In _International Conference on Knowledge Discovery and Data Mining (KDD)_ , pages 1156–1165. 

- [Gonzalez et al.2010] H. Gonzalez, A. Y. Halevy, C. S. Jensen, A. Langen, J. Madhavan, R. Shapley, W. Shen, and J. Goldberg-Kidon. 2010. Google fusion tables: web-centered data management and collaboration. In _Proceedings of the 2010 ACM SIGMOD International Conference on Management of data_ , pages 1061–1066. 

- [Gupta and Sarawagi2009] R. Gupta and S. Sarawagi. 2009. Answering table augmentation queries from unstructured lists on the web. In _Very Large Data Bases (VLDB)_ , number 1, pages 289–300. 

- [Ji and Grishman2011] H. Ji and R. Grishman. 2011. Knowledge base population: Successful approaches and challenges. In _Association for Computational Linguistics (ACL)_ , pages 1148–1158. 

- [Krishnamurthy and Kollar2013] J. Krishnamurthy and T. Kollar. 2013. Jointly learning to parse and perceive: Connecting natural language to the physical world. _Transactions of the Association for Computational Linguistics (TACL)_ , 1:193–206. 

- [Kwiatkowski et al.2011] T. Kwiatkowski, L. Zettlemoyer, S. Goldwater, and M. Steedman. 2011. Lexical generalization in CCG grammar induction for semantic parsing. In _Empirical Methods in Natural Language Processing (EMNLP)_ , pages 1512–1523. 

- [Kwiatkowski et al.2013] T. Kwiatkowski, E. Choi, Y. Artzi, and L. Zettlemoyer. 2013. Scaling semantic parsers with on-the-fly ontology matching. In _Empirical Methods in Natural Language Processing (EMNLP)_ . 

- [Liang2013] P. Liang. 2013. Lambda dependencybased compositional semantics. Technical report, arXiv. 

- [Limaye et al.2010] G. Limaye, S. Sarawagi, and S. Chakrabarti. 2010. Annotating and searching web tables using entities, types and relationships. In _Very Large Data Bases (VLDB)_ , volume 3, pages 1338–1347. 

- [Masaum et al.2012] Masaum, M. Schmitz, R. Bart, S. Soderland, and O. Etzioni. 2012. Open language learning for information extraction. In _Empirical Methods in Natural Language Processing and Computational Natural Language Learning (EMNLP/CoNLL)_ , pages 523–534. 

- [Pasupat and Liang2014] P. Pasupat and P. Liang. 2014. Zero-shot entity extraction from web pages. In _Association for Computational Linguistics (ACL)_ . 

- [Pimplikar and Sarawagi2012] R. Pimplikar and S. Sarawagi. 2012. Answering table queries on the web using column keywords. In _Very Large Data Bases (VLDB)_ , volume 5, pages 908–919. 

- [Popescu et al.2003] A. Popescu, O. Etzioni, and H. Kautz. 2003. Towards a theory of natural language interfaces to databases. In _International Conference on Intelligent User Interfaces (IUI)_ , pages 149–157. 

- [Price1990] P. Price. 1990. Evaluation of spoken language systems: The ATIS domain. In _Proceedings of the Third DARPA Speech and Natural Language Workshop_ , pages 91–95. 

- [Reddy et al.2014] S. Reddy, M. Lapata, and M. Steedman. 2014. Large-scale semantic parsing without question-answer pairs. _Transactions of the Association for Computational Linguistics (TACL)_ , 2(10):377–392. 

- [Syed et al.2010] Z. Syed, T. Finin, V. Mulwad, and A. Joshi. 2010. Exploiting a web of semantic data for interpreting tables. In _Proceedings of the Second Web Science Conference_ . 

- [Unger and Cimiano2011] C. Unger and P. Cimiano. 2011. Pythia: compositional meaning construction for ontology-based question answering on the semantic web. In _Proceedings of the 16th international conference on Natural language processing and information systems_ , pages 153–160. 

- [Unger et al.2012] C. Unger, L. B¨uhmann, J. Lehmann, A. Ngonga, D. Gerber, and P. Cimiano. 2012. Template-based question answering over RDF data. In _World Wide Web (WWW)_ , pages 639–648. 

- [Venetis et al.2011] P. Venetis, A. Halevy, J. Madhavan, M. Pas¸ca, W. Shen, F. Wu, G. Miao, and C. Wu. 2011. Recovering semantics of tables on the web. In _Very Large Data Bases (VLDB)_ , volume 4, pages 528–538. 

- [Wang et al.2011] Y. Wang, L. Deng, and A. Acero. 2011. Semantic frame-based spoken language understanding. _Spoken Language Understanding: Systems for Extracting Semantic Information from Speech_ , pages 41–91. 

- [Wong and Mooney2007] Y. W. Wong and R. J. Mooney. 2007. Learning synchronous grammars for semantic parsing with lambda calculus. In _Association for Computational Linguistics (ACL)_ , pages 960–967. 

- [Wong et al.2009] Y. W. Wong, D. Widdows, T. Lokovic, and K. Nigam. 2009. Scalable attribute-value extraction from semi-structured text. In _IEEE International Conference on Data Mining Workshops_ , pages 302–307. 

- [Wu and Weld2010] F. Wu and D. S. Weld. 2010. Open information extraction using Wikipedia. In _Association for Computational Linguistics (ACL)_ , pages 118–127. 

- [Zelle and Mooney1996] M. Zelle and R. J. Mooney. 1996. Learning to parse database queries using inductive logic programming. In _Association for the Advancement of Artificial Intelligence (AAAI)_ , pages 1050–1055. 

- [Zettlemoyer and Collins2007] L. S. Zettlemoyer and M. Collins. 2007. Online learning of relaxed CCG grammars for parsing to logical form. In _Empirical Methods in Natural Language Processing and Computational Natural Language Learning (EMNLP/CoNLL)_ , pages 678–687. 

- [Zhang et al.2013] Z. Zhang, K. Q. Zhu, H. Wang, and H. Li. 2013. Automatic extraction of top-k lists from the web. In _International Conference on Data Engineering_ . 

