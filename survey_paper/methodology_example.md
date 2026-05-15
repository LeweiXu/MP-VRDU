# **Natural Language Processing in Support of Evidence-based Medicine: A Scoping Review** 

**Zihan Xu[1,*] , Haotian Ma[1,*] , Gongbo Zhang[2] , Yihao Ding[3] , Chunhua Weng[2]** , **Yifan Peng[1]** 

1Weill Cornell Medicine, 2Columbia University, 3University of Sydney 

**Correspondence:** yip4002@med.cornell.edu 

*[Authors contributed equally] 

## **Abstract** 

Evidence-based medicine (EBM) is at the forefront of modern healthcare, emphasizing the use of the best available scientific evidence to guide clinical decisions. Due to the sheer volume and rapid growth of medical literature and the high cost of curation, there is a critical need to investigate Natural Language Processing (NLP) methods to identify, appraise, synthesize, summarize, and disseminate evidence in EBM. This survey presents an in-depth review of 129 research studies on leveraging NLP for EBM, illustrating its pivotal role in enhancing clinical decision-making processes. The paper systematically explores how NLP supports the five fundamental steps of EBM – Ask, Acquire, Appraise, Apply, and Assess. The review not only identifies current limitations within the field but also proposes directions for future research, emphasizing the potential for NLP to revolutionize EBM by refining evidence extraction, evidence synthesis, appraisal, summarization, enhancing data comprehensibility, and facilitating a more efficient clinical workflow. 

## **1 Introduction** 

Evidence-based medicine (EBM) is at the forefront of modern healthcare, emphasizing the use of the best available scientific evidence to guide clinical decisions (Sackett et al., 1996). By integrating clinical expertise, patient values, and the most up-todate research data, EBM facilitates healthcare decisions by patients and the general public, clinicians, guideline developers, administrators, and policymakers (Mehta et al., 2022; Kwaan and Melton, 2012; Van de Vliet et al., 2023). 

The foundation of EBM heavily relies on comprehensive research data from detailed textual sources such as clinical trial publications, cohort studies, and case reports (Blunt, 2022; Ratnani et al., 2023). Navigating this evidence hierarchy necessitates the use of advanced Natural Language 

Processing (NLP) techniques, which are crucial for streamlining literature searches and extracting PICO (Patient/Population, Intervention, Comparison, Outcomes) elements (Peng et al., 2023; Nye et al., 2018). From the early utilization of statistical machine learning (Arora et al., 2019) and recurrent neural networks (Guan et al., 2019), there has been a significant shift towards more advanced technologies such as transformer-based frameworks and large language models (LLMs). These modern approaches employ self-supervised pretraining and instruct-tuning (Rohanian et al., 2024) to capture domain-specific knowledge (Kalyan et al., 2022), enhancing the accuracy and scalability of medical information processing (Thirunavukarasu et al., 2023). Particularly, the recent advancements in LLMs have further propelled NLP capabilities within EBM, excelling in more complex tasks such as appraising and synthesizing evidence (Górska and Tacconelli, 2024), differentiating and ranking evidence (Datta et al., 2024), generating humanlike responses, answering complex clinical questions (Shiraishi et al., 2024), and identifying relevant clinical trials (Devi et al., 2024a). 

Despite these significant advancements, a comprehensive review summarizing NLP development and applications in EBM is still in demand. This paper seeks to fill the gap by offering a thorough review of essential NLP tasks in EBM, with a focus on evidence generation, such as evidence retrieval, extraction, synthesis, and summarization, as well as evidence adoption and evidence-based research, such as question-answering, clinical trial design and identification, and other cutting-edge studies across various clinical specialties. 

Furthermore, we outline key benchmarks to facilitate the development of future NLP models. Finally, we explore several potential avenues for future research. To better support both clinicians and researchers in making more informed clinical decisions and producing more comprehensive review 

21421 

_Findings of the Association for Computational Linguistics: ACL 2025_ , pages 21421–21443 July 27 - August 1, 2025 ©2025 Association for Computational Linguistics 

literature, we have made these resources publicly available.[1] 

## **2 Scope and Literature Selection** 

Our scoping review adheres to the Preferred Reporting Items for Systematic Reviews and MetaAnalyses (PRISMA[2] ) guidelines, as illustrated in Figure 1. 

## **2.1 Information sources** 

We searched 4 databases, including PubMed[3] , IEEE Xplore[4] , ACM Digital Library[5] , and ACL Anthology[6] . The search included studies from the past 5 years, spanning 2019 to 2024. 

## **2.2 Search strategy** 

Our search strategy was meticulously designed to capture the most relevant studies at the intersection of NLP and EBM (Supplementary File A.3). We targeted key NLP concepts and technologies by including terms such as ‘ _natural language processing_ ’, ‘ _language model_ ’, ‘ _large language model_ ’, ‘ _computational linguistics_ ’, ‘ _information extraction_ ’, ‘ _information retrieval_ ’, ‘ _clinical trial retrieval_ ’, ‘ _text summarization_ ’, ‘ _question answering_ ’, ‘ _sentence segmentation_ ’, ‘ _named entity recognition_ ’, ‘ _tokenization_ ’ and the abbreviations like ‘NLP’ and ‘LLM’. In the domain of EBM, we included terms like ‘ _Evidence-Based Medicine_ ’, ‘ _Evidence-Based Practice_ ’, ‘ _Clinical Trial_ ’ and their abbreviations like ‘EBM’ and ‘EBP’, also limited to appearances in the title or abstract. We used the Boolean operator to combine any word from the NLP domain and any work from the EBM domain in our search terms. 

## **2.3 Study selection and metadata extraction** 

The references of all eligible studies were imported into Covidence[7] , and duplicates were removed. We then screened the articles by title and abstract. Inclusion criteria were defined as (1) Studies published in English, (2) research applying NLP techniques specifically for EBM, and (3) Studies focusing on applications for humans. Exclusion criteria were defined as (1) articles unrelated to NLP for 

- 1https://github.com/bionlplab/ 

- awesome-nlp-in-ebm 

   - 2https://www.prisma-statement.org/ 

   - 3https://pubmed.ncbi.nlm.nih.gov/ 

   - 4https://ieeexplore.ieee.org/ 

   - 5https://dl.acm.org/ 

   - 6https://aclanthology.org/ 

   - 7https://www.covidence.org 

**==> picture [219 x 200] intentionally omitted <==**

**----- Start of picture text -----**<br>
Studies from<br>databases/registers<br>(n = 601)<br>References removed  (n = 8)<br>Duplicates iden�fied manually (n = 6)<br>Duplicates iden�fied by Covidence (n =<br>1)<br>Studies excluded  (n = 386)<br>Non-NLP studies (n = 19)<br>Studies screened  Not relevant research area (n=267)<br>(n = 603) Not EBM studies (n = 71)<br>Secondary Literature (n=26)<br>Not English (n=3)<br>Studies excluded  (n = 88)<br>Studies assessed for eligibility<br>Misaligned objec�ves (n = 52)<br>(n = 217)<br>     Lack of EBM tasks (n = 36)<br>Studies included in review<br>(n = 129)<br>Identification<br>Screening<br>Included<br>**----- End of picture text -----**<br>


Figure 1: PRISMA flow diagram. 

EBM, (2) non-English publications, and (3) secondary literature such as systematic reviews, retracted papers, survey papers, case studies, and descriptive papers lacking experimental results. 

After the screening, the metadata was extracted from each paper, including models, disease, tasks involved, results, and limitations. Two annotators cross-verified the study selection and metadata extraction processes and consulted a third in cases of disagreement. 

## **2.4 Study Statistics** 

From an initial pool of 601 papers retrieved from databases and 9 additional sources, we removed 8 duplicates. Subsequently, 386 papers were excluded during the initial screening based on predefined exclusion criteria, and 88 more were removed during full-text screening due to misaligned objectives or lack of relevance to EBM tasks. Ultimately, 129 studies met the inclusion criteria and form the basis of this review, with detailed metadata provided in Supplementary Table 1. 

Figure 2 illustrates the distribution of research papers across different years (2019–2024) and their corresponding NLP tasks. There has been a rapid growth of papers over the years, peaking in 2023. The most common tasks throughout the years are Entity Extraction, Classification, and Evaluation, showing their foundational role in NLP for EBM research. Emerging tasks like Question Answering and Quality Assessment have appeared more prominently in recent years, reflecting evolving research directions. 

21422 

**==> picture [220 x 220] intentionally omitted <==**

**----- Start of picture text -----**<br>
Summarization Evidence Ranking Quality Assessment<br>Clinical Trial Design Evidence Synthesis Question Answering<br>Entity Extraction and Classification Information Retrieval Relation Extraction<br>Evaluation<br>80<br>70<br>60<br>50<br>40<br>30<br>20<br>10<br>0 2019 2020 2021 2022 2023 2024<br>Number of Papers<br>**----- End of picture text -----**<br>


Figure 2: Distribution of papers in different EBM tasks over time. The color schema is the same as Supplementary Table 1. 

||EBM cycle|Description|NLP tasks|
|---|---|---|---|
||Ask|Search & select|Question|
|||studies|answering,|
||||Information|
||||retrieval|
||Acquire|Collect data|Named entity|
||||recognition and|
||||normalization,|
||||Relation|
||Appraise|Examine<br>relevance,<br>validity, and|extraction<br>Quality<br>assessment,<br>Evidence ranking|
|||results|and screening,<br>Evidence|
||||synthesis,|
||||Evidence|
||||summarization|
||Apply & Asses|Apply EBM in<br>practice and|Clinical trial<br>identifcation and|
|||research and|design, Question|
|||evaluate their<br>effectiveness|Answering,<br>Domain-specifc|
||||applications|



Table 1: Mapping of EBM cycle to corresponding NLP tasks. 

## **3 NLP Techniques for EBM** 

The entire EBM process consists of five steps, commonly referred to as the ‘5A’s: **Ask** , **Acquire** , **Appraise** , **Apply** , and **Assess** (Ratnani et al., 2023). NLP can be leveraged at each step to enhance the process (Table 1). For example, in the **Ask** step, clinicians or patients formulate precise clinical questions to address specific healthcare concerns. During the **Acquire** step, NLP can be employed to extract evidence, often leveraging the PICO framework. In the **Appraise** step, NLP tools can assist in evaluating and ranking the quality, validity, and relevance of the retrieved information to ensure its applicability to clinical decision-making. For the **Apply and Assess** steps, NLP can streamline the design and identification of relevant clinical trials and facilitate their integration into practice, enabling continuous assessment and refinement of patient care strategies. Detailed trends and advancements for each step of the EBM process are discussed in the following sections. 

## **4 Ask - Searching & Selecting Studies** 

EBM can help researchers and clinicians draft a successful systematic review. After the scope and questions have been determined, the first step is to search for studies to include in the reviews and ensure they remain up to date. 

This step is typically achieved using NLP-based information retrieval techniques, which extract rel- 

evant information from large text corpora based on user queries. Early heuristic methods involved structured, keyword-based queries to retrieve articles from repositories like MEDLINE or PubMed. These methods, while foundational, are limited by the high cost of expert annotation, maintenance, and domain sensitivity (Névéol et al., 2011). Despite these limitations, recent methods often rely on predefined rule-based strategies, e.g., SR[pt] and CQrs (Navarro-Ruan and Haynes, 2022), to filter and compare the retrieved results for systematic reviews. In addition, while statistical machine learning and context-aware models (Kamath et al., 2021; Samuel et al., 2021) have been widely adopted, they often lack scalability and struggle with less representative text embeddings. 

Recent advancements are leaning towards transformer-based deep learning frameworks (Ramprasad et al., 2023a; Jin et al., 2022) due to their scalability and the ability to integrate medical ontologies, improving domain-specific text representation through self-supervised pretraining. For example, (Lokker et al., 2023) used BioBERT’s (Lee et al., 2020) embeddings and attention mechanisms to improve query representation and biomedical literature retrieval in clinical practice. Furthermore, the integration of generative AI models has advanced literature retrieval despite challenges like hallucination. For example, Gwon et al. (2024) 

21423 

compared Microsoft Bing AI and ChatGPT in accelerating the systematic literature search for a clinical review on Peyronie disease treatment, finding both can speed up the search process. 

## **5 Acquire - Collecting Data** 

EBM is designed to identify all studies relevant to their research questions and synthesize data regarding the study design, risk of bias, and results. Therefore, the findings of EBM heavily depend on decisions about which data from these studies are presented and analyzed. The data collected should be accurate, complete, and accessible for future review, updates, and data-sharing purposes. Here we describe NLP approaches used to extract data directly from journal articles and other studies’ reports. 

## **5.1 Entity Extraction and Normalization** 

Initially, entity (e.g., PICO) extraction relied on rule-based approaches, which utilize predefined lexical, syntactic, and contextual rules for extracting entities from clinical trial data (Chen et al., 2019c; Borchert et al., 2022). These methods are simple, transparent, and customizable, making them practical for high-precision tasks in structured contexts. Although they face challenges with complex or ambiguous data, their interpretability and ease of adaptation remain valuable for PICO extraction (Dhrangadhariya and Müller, 2023). 

RNN/LSTM-based frameworks lacked longterm memory capabilities. Nevertheless, they have been used for sequential sentence classification to enhance context utilization and improve classification accuracy in unstructured or less structured medical abstracts (Jin and Szolovits, 2018). 

The current trend is towards the dominance of transformer-based frameworks due to their domainaware pertaining benefits. For instance, models such as SciBERT and PubMedBERT have been specifically developed for extracting ‘Intervention’ (‘I’ in PICO) (Tsubota et al., 2022), SrBERT (Aum and Choe, 2021) for classifying articles into “included’ or “excluded” categories based on predefined inclusion criteria. 

## **5.2 Relation Extraction** 

Following the identification of PICO elements, relation extraction approaches can be used to link these elements within studies. 

Initially, rule-based and machine-learning methods were used to extract meaningful relationships 

from medical literature (Alodadi and Janeja, 2019; Borchert et al., 2022). By 2021, transformative methodologies were developed, integrating deep learning frameworks like BERT and Augment Mining (AM). For example, srBERT built-in (Aum and Choe, 2021), identified key elements and defines interrelations from the titles of articles. Stylianou and Vlahavas (2021) classified the relationships between argumentative components within the texts, such as claims and evidence. Their relationships were labeled as ‘supporting’ or ‘opposing’. 

In the systematic review process, understanding the connections between different study results can influence the review outcomes. However, besides systematic reviews, automated relation extraction has shifted towards more structured approaches, such as schema-based relation extraction. For example, Sanchez-Graillet et al. (2022) utilized a richly annotated corpus that aligns with the C-TrO ontology. Complementing these advances, graphbased approaches offer a novel way to encode complex relationships between clinical entities. A knowledge graph is a structured representation of information where entities (e.g., symptoms, treatments, drugs) are represented as nodes and their relationships as edges. Graph-based approaches have emerged as an effective method to encode relationships. For example, a knowledge graph was used to organize and visualize relationships among clinical trial entities such as symptoms, treatments, and drug outcomes by structuring data into nodes and edges (Pan et al., 2021). 

## **6 Appraise, Synthesize, and Summarize Evidence** 

This task screens the included studies for risk of bias and appraises them for quality to ensure that healthcare decisions are informed by the most reliable and relevant evidence. Once the appraisal is complete, the next step is synthesizing evidence by combining findings from multiple studies, often using meta-analyses. Finally, these synthesized insights are summarized into concise, actionable conclusions. 

## **6.1 Quality Assessment** 

Developing tools to assess evidence is crucial in EBM, such as the fully automated tool that combines machine learning and rule-based techniques by Brassey et al. (2021). It assessed the evidence from randomized clinical trials and systematic re- 

21424 

views by sentiment analysis, indication of bias, and sample size calculation, and used them to estimate the potential effectiveness of the intervention. Besides, deep learning models such as BERT (Devlin et al., 2019) have been used to evaluate the quality of evidence by analyzing article titles and abstracts. For example, different variations like BioBERT, BlueBERT, and BERTBASE were fine-tuned to classify the articles based on their adherence to methodological quality criteria (Lokker et al., 2023). 

## **6.2 Evidence Ranking and Screening** 

After the quality assessment, the next step is to screen and rank the evidence. Several ranking methods are available, with statistical-based methods being among the earliest used. For example, Norman et al. (2019b) developed a method to rank references by their likelihood of relevance. Compared with randomized screening, their study showed that prioritization methods (with technological assistance) allow for fewer studies to be screened while still producing reliable results, which effectively reduces both the time and cost associated with the screening process. Rybinski et al. (2020a) introduced the platform A2A, which used Okapi Best Match 25 (BM25) that assigned scores to documents based on term frequency and document length and Divergence from Randomness (DFR) that quantified informativeness as the divergence of a term’s distribution from randomness for document ranking. Additionally, machine learning methods are implemented. Rybinski et al. (2020b) designed a search system with a simple query formulation strategy for initial ranking and used pretrained BERT models (SciBERT, BioBERT, and BlueBERT) for re-ranking in clinical trial searches, which improved the robustness. 

## **6.3 Evidence Synthesis** 

Evidence synthesis combines data from included studies to draw conclusions about a body of evidence. While the most common method used is meta-analysis, which statistically combines results from studies to estimate overall effect sizes, NLPbased approaches have also been applied to synthesize studies or findings. Mutinda et al. (2022b) proposed a method to reproduce meta-analysis, computing summary statistics (e.g., risk ratio) and visualizing results using forest plots by extracting and normalizing PICO elements from breast cancer randomized controlled trials. However, this method is built on a small amount of data. Górska and 

Tacconelli (2024) developed a system to continuously update summary statistics from key publications, further improving the meta-analysis process. However, only binary outcomes were supported in both methods, limiting the applicability to broader meta-analysis needs. Besides meta-analysis, EvidenceMap (Kang et al., 2023) effectively synthesized medical findings by employing a structured and hierarchical representation comprising Entities, Propositions, and Maps that enhances the interpretability and retrievability of evidence through its sophisticated semantic relational retranslation. 

## **6.4 Evidence Summarization** 

Finally, EBM must present a clear statement of findings or conclusions to help people make betterinformed decisions and increase usability. This summary should include information on all important outcomes, evidence certainty, and the intervention’s desirable and undesirable consequences. 

From an NLP technical perspective, evidence summarization uses extractive and abstractive strategies. Extractive summarization selects the most important sentences from the original text. Gulden et al. (2019) generated a new dataset from clinicaltrials.gov to test various algorithms (e.g., LexRank, TextRank, and Latent Semantic Analysis), identifying TextRank as the best performer in creating summaries directly from the source texts without altering the original wording. However, these algorithms suffered from inefficiency and high computational complexity when processing large datasets. Sarker et al. (2020) developed a lightweight system that leverages Maximal Marginal Relevance (MMR) and pre-trained word embeddings trained on PubMed and PMC texts to integrate semantic relevance and reduce redundancy. Similarly, Xie et al. (2022) proposed a knowledge infusion training framework called KeBioSum, which incorporated PICO into pre-trained language models (PLMs). It utilized lightweight knowledge adapters to reduce computational costs while improving semantic understanding and contextual representation. 

Abstractive summarization focuses on the most critical information and creates new text for the summary; usually, more advanced techniques are used. Lalitha et al. (2023) have implemented sophisticated techniques such as neural networkbased model T5 (Text-to-Text Transfer Transformer), BART (Bidirectional Auto-Regressive Transformer), and PEGASUS (Pre-training with 

21425 

Extracted Gap-sentences for Abstractive Summarization Sequence-to-sequence) to resolve the challenge of obtaining useful information from a vast amount of clinical documents. 

With the increased demand for user-interacted summarization, Ramprasad et al. (2023b) presented TrialsSummarizer, a system that helps automate summarizing the most relevant evidence in a set of randomized controlled trials by a multi-headed architecture, enabling each token in the generated summary to be explicitly linked to specific input aspects (e.g., population, intervention, or outcome). It introduces template-infilling capabilities, allowing users to correct or adjust generated summaries dynamically. Moreover, the application of LLMs has evolved to address these tasks with growing precision and depth. Hamed et al. (2023) explored ChatGPT’s capabilities in synthesizing diabetic ketoacidosis (KDA) guidelines by comparing, integrating, and abstracting content. Unlu et al. (2024) employed a Retrieval-Augmented Generation (RAG) framework with GPT-4 for generating responses to clinical trial eligibility questions based on retrieved patient data. Furthermore, TriSum (Jiang et al., 2024) stood out by using structured rationale-based abstractive summarization, where large language models generate aspect-triple rationales that are distilled into smaller models through a dual-scoring selection mechanism and curriculum learning. 

## **7 Apply and Assess: adoption, refinement, and research** 

Transitioning from the evidence generation and synthesis, the next critical step is its adoption and refinement, facilitated by an ‘Evidence-based Research’ approach. Adoption and refinement are crucial to consistently reassessing and enhancing clinical evidence, particularly when existing evidence gaps lead to unmet needs of clinicians and patients. Evidence-based research further ensures that these gaps inform future clinical studies. Here, we summarize several applications identified from our literature review that align with this topic. 

## **7.1 Specialty-specific adoption** 

In addition to general applications, we observed that NLP for EBM has been applied within specific medical specialties. Here, we summarized common specialties featured in the papers, such as oncology for conditions like Non-small cell lung 

cancer (NSCLC) and cardiovascular events such as heart failure. Other diseases are detailed in Supplementary Table 1. 

**Oncology.** Cancer is a central topic in EBM, as it demands continuous integration of new research findings to guide evidence-based decisions for accurate diagnosis, effective treatment, and long-term patient management. Saiz et al. (2021) introduced Watson Oncology Literature Insights (WOLI), an AI system, by automatically identifying, prioritizing, and extracting relevant oncology research, which facilitated the translation of evidence into clinical practice. Similarly, the Clinical Trial Matching (CTM) system (Alexander et al., 2020) was evaluated at a cancer center in Australia with an overall accuracy of 92% for screening lung cancer patients. These tools highlight how AI-driven systems are increasingly embedded in hospital workflows. 

**Cardiology.** Cardiology demands robust evidence to support the decision due to the high prevalence and the critical consequences of diagnostic errors, which can result in severe harm or loss of life. For example, the hybrid model proposed by Tun et al. (2023) exemplifies clinical practice by automating patient eligibility assessment directly within clinical workflows. In a real-world application on a dataset of 40,000 patients across several clinical care pathways, such as heart failure with reduced and preserved ejection fraction and atrial fibrillation, this model was deployed and achieved an impressive accuracy of 87.3%. 

## **7.2 Clinical trial design and identification** 

Not all medical specialties are fully addressed by current research, and even in those with significant focus, the integration of findings into real-world guidance remains insufficient. Automating clinical trial procedures is critical for instant reaction to pandemics or public health emergencies. A crucial step in advancing future clinical trials or experiments is the design phase, where NLP plays a pivotal role. Effective clinical trial design involves structuring and optimizing trials to ensure they align with patient needs and research objectives. NLP tools can enhance the efficiency of clinical trial design by facilitating the automated matching of patients to suitable trials and ensuring trials are aligned with the right patient cohorts. This capability supports a more effective and timely deployment of research resources in emergency health 

21426 

situations. 

Eligibility matching and cohort identification is a process of matching patients to clinical trials based on their eligibility and identifying groups of patients (cohorts) who meet specific criteria for inclusion in clinical trials. There are several applications. Vydiswaran et al. (2019) proposed a hybrid approach to identify patient cohorts for clinical trials, which combines pattern-based, knowledgeintensive, and feature-weighting techniques to determine if patients meet specific selection criteria. Segura-Bedmar and Raez (2019) explored the use of deep learning models for cohort selection, framing it as a multi-label classification task. By employing CNNs and RNNs to process freetext eligibility criteria, this method allows for automatic learning of representations directly from text. Building on these foundations, Liu et al. (2022) developed Criteria2Query (C2Q) to extract and transform free-text eligibility criteria into structured, queryable data for cohort identification. More recently, Murcia et al. (2024) proposed the “TrialMatcher” algorithm to match veterans for clinical trials using existing information within EHRs. It extracted attributes from patient profiles and eligibility criteria from trial profiles and compared them using the Sørensen-Dice Index (SDI). These applications show the potential of streamlining the process of recruitment and improving future clinical trial design. Now, researchers try to add LLMs to the studies. LLMs like GPT-3.5 or GPT-4 enhance clinical trial workflows by processing complex natural language data, such as patient profiles and trial eligibility criteria. The examples include AutoTrial (Wang et al., 2023b), focusing on trial design, specifically generating eligibility criteria using multi-step reasoning and hybrid prompting and TrialGPT (Jin et al., 2024), implementing a comprehensive framework for large-scale patienttrial matching, emphasizing real-world deployment and time-saving efficiency. 

## **7.3 Drug repurposing** 

Another frontier application in this field is drug repurposing, which utilizes NLP to analyze existing medical literature and uncover new therapeutic applications for established drugs. By automating the analysis of large datasets such as clinical trials and research papers, NLP speeds up the identification of potential treatments, offering a faster and more cost-effective alternative to traditional drug discovery methods. During the COVID-19 pandemic, 

there is an urgent need for drugs for treatment. To quickly meet this requirement, the CovidX Network Algorithm Gates and Hamed (2020) was developed, which utilized NLP to analyze vast COVID-19 biomedical literature. It ranked potential drug candidates for repurposing, highlighting NLP’s power in automating and accelerating evidence synthesis during critical times. Alzheimer’s disease (AD), a progressive neurodegenerative disorder, remains a major global health challenge with limited treatment options and no definitive cure. Despite significant investment in drug development, the failure rate for Alzheimer’s-specific drugs in clinical trials remains exceedingly high. To address this, Daluwatumulle et al. (2022) employed knowledge graph embeddings to predict AD drug candidates by linking textual data and generating hypotheses from unstructured information. 

## **7.4 Question Answering** 

While EBM is taught according to the five steps: ask, acquire, appraise, apply, and evaluate, a recent trend of application with the advancement in LLMs focuses on treating the entire process as a questionanswering (QA) task. Xie et al. (2023) experimented with the consultation of rhinoplasty questions to ChatGPT, which pre-learned knowledge and summarized texts to respond, testing the potential of LLMs to offer valuable feedback. Moreover, Mohammed and Fiaidhi (2024) added bootstrapping to BioBERT and BioGPT so that they could better understand PICO questions from physicians and find potential answers from publications. Expanding on this trend, Chuan and Morgan (2021) introduced Chatbot SOPHIA, which helps users understand their eligibility for clinical trials by answering questions based on trial criteria. Addressing rare cancers, Jang et al. (2022) fine-tuned SAPBERT for QA and NER tasks, ultimately summarizing potential drugs ranked by relevance, such as bevacizumab, temozolomide, lomustine, and nivolumab. 

## **8 EBM Benchmark dataset** 

Here, we summarize the benchmarks used in NLP and EBM (Supplementary Table 2). The tasks frequently involved with these benchmarks are Evidence Retrieval, Evidence Extraction, and Clinical Trial Identification. There is a notable gap in datasets specifically tailored for Evidence Synthesis and Appraisal, as well as Question Answering. 

21427 

The existing datasets are often built upon general texts rather than medical-specific content. For example, CNN-DailyMail (See et al., 2017) is used for Evidence Summarization, but it is not medicalrelated. We also noticed that the primary data sources for these benchmarks are scholarly articles from PubMed and clinical trials. 

## **9 Challenges and Future Directions** 

EBM is an important, rewarding, and dynamic field that organizes current data to improve healthcare decision-making. By integrating the best available evidence with a healthcare professional’s experience and the patient’s values, EBM aims to optimize health outcomes. Our focus here is on retrieving, extracting, appraising, synthesizing, and summarizing evidence from biomedical literature such as clinical trials, cohort studies, and case reports. However, conducting these analyses can be both demanding and time-consuming. In this study, we explore key NLP techniques that can streamline and facilitate this process. 

Our review indicates that NLP-based systems or pipelines have achieved impressive results in EBM, such as extracting entities like PICO, enhancing the information retrieval engines, automating the evidence synthesis, assessing evidence quality, ranking the evidence with the highest confidence, summarizing the information, and answering questions. At the same time, as in any other evolving area, there remain challenges ahead. For example, generative models in EBM tasks have demonstrated impressive fluency and scalability, yet their tendency to hallucinate facts, lack source attribution, and sensitivity to prompt phrasing remain significant limitations for clinical use. A core challenge is the validation and trustworthiness of generated outputs, especially in high-stakes domains like medicine. Mechanisms such as RAG offer potential mitigations but require further development and evaluation. 

From another perspective, particularly in handling diseases with limited literature or annotated data (Ge et al., 2023). Few-shot learning holds significant potential, as it enables models to generalize effectively from a small number of examples, reducing the dependency on large, annotated datasets. This data-efficient approach is crucial for EBM tasks in under-researched areas, such as rare diseases, where annotated resources are scarce. Fewshot learning can help these models adapt quickly 

to specific clinical needs, allowing for more accurate information extraction, question answering, and evidence synthesis, even with minimal training data. 

Additionally, there is a pressing need for more benchmark datasets, especially for Evidence Synthesis and Appraisal and Question Answering. Current resources often rely on general corpora rather than those specifically oriented toward medical content, limiting the development of specialized NLP applications. Researchers can consider and build more meaningful datasets. Moreover, NLP-based tools have not yet been widely applied across all medical specialties, such as Urology and Hepatology, indicating room for expansion in these areas. 

Another future direction for NLP in EBM involves incorporating real-world data from various sources, such as mobile devices, social media, and genomics. These data sources capture rich and diverse information beyond traditional clinical records, offering valuable insights into patient behaviors, lifestyle, environmental factors, and genetic predispositions. For example, data from mobile health apps and wearable devices can provide real-time health metrics. At the same time, social media posts may reveal patient self-reported outcomes or experiences that are often missed in clinical settings. Integrating genomic data adds another layer, enabling family history and personalized genomic code into disease risk and treatment response. 

Furthermore, the “black box" nature of many NLP models limits their interpretability and accountability. Biases within training data can restrict NLP’s effectiveness and fairness across diverse patient demographics. Additionally, the high computational demands and the need for domain expertise in both NLP and healthcare are resource-intensive. 

To fully realize the potential of NLP for EBM in real-world clinical workflows often involve interdisciplinary scenarios that span multiple conditions, comorbidities, and patient subpopulations. To address these complexities, NLP systems for EBM must evolve toward more holistic, adaptable frameworks capable of reasoning across diverse clinical questions and integrating heterogeneous data sources. 

Addressing these limitations is important for enabling efficiency and ultimately contributing to a safer, more equitable healthcare landscape. 

21428 

## **10 Conclusion** 

Our comprehensive review of over 600 papers resulted in the selection of 129 studies that focus on critical aspects of NLP within EBM. We first provide an overview of EBM, followed by a survey of NLP methods and techniques that address each step of the EBM process. We also explore use cases that demonstrate the application of EBM in various scenarios. Additionally, we review popular datasets and benchmarks. Finally, we present open challenges and future directions for research in this field. As NLP technologies evolve, they offer promising prospects for harnessing vast amounts of unstructured data, thus supporting clinical and research applications. 

- Mohammad S. Alodadi and Vandana P. Janeja. 2019. Linking knowledge discovery in clinical notes and massive biomedical literature repositories. In _2019 IEEE International Conference on Big Data_ . IEEE. 

- Paul Arora, Devon Boyne, Justin J. Slater, Alind Gupta, Darren R. Brenner, and Marek J. Druzdzel. 2019. Bayesian networks for risk prediction using realworld data: A tool for precision medicine. _Value in Health_ , 22(4):439–445. 

- Sungmin Aum and Seon Choe. 2021. srBERT: automatic article classification model for systematic review using BERT. _Syst. Rev._ , 10(1):285. 

- Jacob Beattie, Sarah Neufeld, Daniel Yang, Christian Chukwuma, Ahmed Gul, Neil Desai, Steve Jiang, and Michael Dohopolski. 2024. Utilizing large language models for enhanced clinical trial matching: A study on automation in patient screening. _Cureus_ , 16(5):e60044. 

## **Limitations** 

Our study primarily focuses on English-language publications, potentially overlooking important research published in other languages. The inclusion criteria may have excluded studies indirectly related to EBM and NLP that could provide valuable insights. Additionally, our analysis only covers articles published between 2019 and 2024, which may have led to the omission of significant earlier works that contributed to the foundation of this field. Furthermore, the databases and search engines used in this review are limited, and it is possible that some relevant studies on NLP for EBM during the specified period were not identified. 

## **Acknowledgments** 

This project was sponsored by the National Library of Medicine grants R01LM014344 and R01LM014573. 

## **References** 

- Md Abdullah Al Hafiz Khan, Md Shamsuzzaman, Sadid A. Hasan, Mohammad S Sorower, Joey Liu, Vivek Datla, Mladen Milosevic, Gabe Mankovich, Rob van Ommering, and Nevenka Dimitrova. 2019. Improving disease named entity recognition for clinical trial matching. In _2019 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)_ , pages 2541–2548. 

- Marliese Alexander, Benjamin Solomon, David L Ball, Mimi Sheerin, Irene Dankwa-Mullan, Anita M Preininger, Gretchen Purcell Jackson, and Dishan M Herath. 2020. Evaluation of an artificial intelligence clinical trial matching system in australian lung cancer patients. _JAMIA Open_ , 3(2):209–215. 

- J Thaddeus Beck, Melissa Rammage, Gretchen P Jackson, Anita M Preininger, Irene Dankwa-Mullan, M Christopher Roebuck, Adam Torres, Helen Holtzen, Sadie E Coverdill, M Paul Williamson, Quincy Chau, Kyu Rhee, and Michael Vinegra. 2020. Artificial intelligence tool for optimizing eligibility screening for clinical trials in a large community cancer center. _JCO Clin. Cancer Inform._ , 4(4):50–59. 

- Chris Blunt. 2022. The pyramid schema: The origins and impact of evidence pyramids. _SSRN Electron. J._ 

- Florian Borchert, Christina Lohr, Luise Modersohn, Thomas Langer, Markus Follmann, Jan Philipp Sachs, Udo Hahn, and Matthieu-P. Schapranow. 2020. GGPONC: A corpus of German medical text with rich metadata based on clinical practice guidelines. In _Proceedings of the 11th International Workshop on Health Text Mining and Information Analysis_ , pages 38–48, Online. Association for Computational Linguistics. 

- Florian Borchert, Laura Meister, Thomas Langer, Markus Follmann, Bert Arnrich, and Matthieu-P Schapranow. 2022. Controversial trials first: Identifying disagreement between clinical guidelines and new evidence. In _AMIA Annual Symposium Proceedings_ , pages 237–246. 

- Jon Brassey, Christopher Price, Jonny Edwards, Markus Zlabinger, Alexandros Bampoulidis, and Allan Hanbury. 2021. Developing a fully automated evidence synthesis tool for identifying, assessing and collating the evidence. _BMJ Evid. Based Med._ , 26(1):24–27. 

- Austin J. Brockmeier, Meizhi Ju, Piotr Przybyła, and Sophia Ananiadou. 2019. Improving reference prioritisation with pico recognition. _BMC Medical Informatics and Decision Making_ , 19:256. 

- Tianrun Cai, Fiona Cai, Kumar P. Dahal, Gabrielle Cremone, Ethan Lam, Charlotte Golnik, Thany Seyok, Chuan Hong, and Katherine P. Liao. 2021. Improving the efficiency of clinical trial recruitment using an 

21429 

ensemble machine learning to assist with eligibility screening. _ACR Open Rheumatology_ , 3:593–600. 

- Leonardo Campillos-Llanos, Ana Valverde-Mateos, Adrián Capllonch-Carrión, and Antonio MorenoSandoval. 2021. A clinical trials corpus annotated with umls entities to enhance the access to evidencebased medicine. _BMC Medical Informatics and Decision Making_ , 21(69). This article has been corrected. See BMC Med Inform Decis Mak. 2021 Apr 7;21:118. 

- Boyu Chen, Hao Jin, Zhiwen Yang, Yingying Qu, Heng Weng, and Tianyong Hao. 2019a. An approach for transgender population information extraction and summarization from clinical trial text. _BMC Med. Inform. Decis. Mak._ , 19(Suppl 2):62. 

- Chi-Jen Chen, Neha Warikoo, Yun Chun Chang, and Chen. 2019b. Medical knowledge infused convolutional neural networks for cohort selection in clinical trials. _Journal of the american medical informatics association_ , 26(11):1227–1236. 

- Long Chen, Yu Gu, Xin Ji, Chao Lou, Zhiyong Sun, Haodan Li, Yuan Gao, and Yang Huang. 2019c. Clinical trial cohort selection based on multi-level rulebased natural language processing system. _Journal of the American Medical Informatics Association_ , 26(11):1218–1226. 

- Ching-Hua Chuan and Susan Morgan. 2021. Creating and evaluating chatbots as eligibility assistants for clinical trials: An active deep learning approach towards user-centered classification. _ACM Trans. Comput. Healthc._ , 2(1):1–19. 

- Jonathan W Cunningham, Pulkit Singh, Christopher Reeder, Brian Claggett, Pablo M Marti-Castellote, Emily S Lau, Shaan Khurshid, Puneet Batra, Steven A Lubitz, Mahnaz Maddah, Anthony Philippakis, Akshay S Desai, Patrick T Ellinor, Orly Vardeny, Scott D Solomon, and Jennifer E Ho. 2024. Natural language processing for adjudication of heart failure in a multicenter clinical trial: A secondary analysis of a randomized clinical trial. _JAMA Cardiol._ , 9(2):174–181. 

- Geesa Daluwatumulle, Rupika Wijesinghe, and Ruvan Weerasinghe. 2022. In silico drug repurposing using knowledge graph embeddings for alzheimer’s disease. In _Proceedings of the 9th International Conference on Bioinformatics Research and Applications_ , pages 61–66, New York, NY, USA. ACM. 

- Surabhi Datta, Kyeryoung Lee, Hunki Paek, Frank J. Manion, Nneka Ofoegbu, Jingcheng Du, Ying Li, Liang-Chin Huang, Jingqi Wang, Bin Lin, Hua Xu, and Xiaoyan Wang. 2024. Autocriteria: a generalizable clinical trial eligibility criteria extraction system powered by large language models. _Journal of the American Medical Informatics Association_ , 31(2):375–385. Published: 11 November 2023. 

- Yang Deng, Yaliang Li, Ying Shen, Nan Du, Wei Fan, Min Yang, and Kai Lei. 2019. Medtruth: A semisupervised approach to discovering knowledge condition information from multi-source medical data. In _Proceedings of the 28th ACM International Conference on Information and Knowledge Management_ , CIKM ’19, page 719–728, New York, NY, USA. Association for Computing Machinery. 

- Arti Devi, Shashank Uttrani, Aryansh Singla, Sarthak Jha, Nataraj Dasgupta, Sayee Natarajan, Rajeshwari S Punekar, Larry A Pickett, and Varun Dutt. 2024a. Quantitative analysis of GPT-4 model: Optimizing patient eligibility classification for clinical trials and reducing expert judgment dependency. In _Proceedings of the 2024 8th International Conference on Medical and Health Informatics_ , pages 230–237, New York, NY, USA. ACM. 

- Arti Devi, Shashank Uttrani, Aryansh Singla, Sarthak Jha, Nataraj Dasgupta, Sayee Natarajan, Rajeshwari S. Punekar, Larry A. Pickett, and Varun Dutt. 2024b. Automating clinical trial eligibility screening: Quantitative analysis of GPT models versus human expertise. In _Proceedings of the 17th International Conference on PErvasive Technologies Related to Assistive Environments_ , New York, NY, USA. ACM. 

- Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In _Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)_ , pages 4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics. 

- Jay DeYoung, Iz Beltagy, Madeleine van Zuylen, Bailey Kuehl, and Lucy Wang. 2021. MS^2: Multidocument summarization of medical studies. In _Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing_ , pages 7494– 7513, Stroudsburg, PA, USA. Association for Computational Linguistics. 

- Houssein Dhayne, Rima Kilany, Rafiqul Haque, and Yehia Taher. 2021. Emr2vec: Bridging the gap between patient data and clinical trial. _Computers & Industrial Engineering_ , 156:107236. 

- Anjani Dhrangadhariya, Roger Hilfiker, Roger Schaer, and Henning Müller. 2020. Machine learning assisted citation screening for systematic reviews. _Digital Personalized Health and Medicine_ . 

- Anjani Dhrangadhariya, Gaetano Manzo, and Henning Müller. 2024. Pico to picos: Weak supervision to extend datasets with new labels. _Digital Health and Informatics Innovations for Sustainable Health Care Systems_ , 316. 

- Anjani Dhrangadhariya and Henning Müller. 2023. Not so weak pico: leveraging weak supervision for participants, interventions, and outcomes recognition for systematic review automation. _JAMIA Open_ , 6(1). 

21430 

- Nhan V Do, Danne C Elbers, Nathanael R Fillmore, Samuel Ajjarapu, Steven J Bergstrom, John Bihn, June K Corrigan, Rupali Dhond, Svitlana Dipietro, Arkadiy Dolgin, Theodore C Feldman, Sergey D Goryachev, Linden B Huhmann, Jennifer La, Paul A Marcantonio, Kyle M McGrath, Stephen J Miller, Vinh Q Nguyen, George R Schneeloch, Feng-Chi Sung, Kaitlin N Swinnerton, Amelia H Tarren, Hannah M Tosi, Danielle Valley, Austin D Vo, Cenk Yildirim, Chunlei Zheng, Robert Zwolinski, Gisele A Sarosy, David Loose, Colleen Shannon, and Mary T Brophy. 2024. Matching patients to accelerate clinical trials (MPACT): Enabling technology for oncology clinical trial workflow. _Stud. Health Technol. Inform._ , 310:1086–1090. 

- Nicholas J. Dobbins, Bin Han, Weipeng Zhou, Kristine F. Lan, H. Nina Kim, Robert Harrington, Özlem Uzuner, and Meliha Yetisgen. 2023. LeafAI: query generator for clinical cohort discovery rivaling a human programmer. _Journal of the American Medical Informatics Association_ , 30(12):1954–1964. 

- Nicholas J Dobbins, Tony Mullen, Özlem Uzuner, and Meliha Yetisgen. 2022. The leaf clinical trials corpus: a new resource for query generation from clinical trial eligibility criteria. _Sci. Data_ , 9(1):490. 

- Jingcheng Du, Qing Wang, Jingqi Wang, Prerana Ramesh, Yang Xiang, Xiaoqian Jiang, and Cui Tao. 2021. COVID-19 trial graph: a linked graph for COVID-19 clinical trials. _J. Am. Med. Inform. Assoc._ , 28(9):1964–1969. 

- Abdelazeem Eldawlatly, Hussain Alshehri, Abdullah Alqahtani, Abdulaziz Ahmad, Fatma Al-Dammas, and Amir Marzouk. 2018. Appearance of population, intervention, comparison, and outcome as research question in the title of articles of three different anesthesia journals: A pilot study. _Saudi Journal of Anaesthesia_ , 12(2):283–286. 

- Yilu Fang, Jae Hyun Kim, Betina Ross Idnay, Rebeca Aragon Garcia, Carmen E. Castillo, Yingcheng Sun, Hao Liu, Cong Liu, Chi Yuan, and Chunhua Weng. 2021. Participatory design of a clinical trial eligibility criteria simplification method. _Studies in Health Technology and Informatics_ , 281:984–988. 

- Lyndsey Elaine Gates and Ahmed Abdeen Hamed. 2020. The anatomy of the SARS-CoV-2 biomedical literature: Introducing the CovidX network algorithm for drug repurposing recommendation. _J. Med. Internet Res._ , 22(8):e21169. 

- Yao Ge, Yuting Guo, Sudeshna Das, Mohammed Ali Al-Garadi, and Abeed Sarker. 2023. Few-shot learning for medical text: A review of advances, trends, and opportunities. _J. Biomed. Inform._ , 144(104458):104458. 

- Madhusudan Ghosh, Shrimon Mukherjee, Asmit Ganguly, Partha Basuchowdhuri, Sudip Kumar Naskar, and Debasis Ganguly. 2024a. AlpaPICO: Extraction of PICO frames from clinical trial documents using LLMs. _Methods_ , 226:78–88. 

- Madhusudan Ghosh, Shrimon Mukherjee, Payel Santra, Girish Na, and Partha Basuchowdhuri. 2024b. BLINKtextsubscriptLSTM: BioLinkBERT and LSTM based approach for extraction of PICO frame from clinical trial text. In _Proceedings of the 7th Joint International Conference on Data Science & Management of Data (11th ACM IKDD CODS and 29th COMAD)_ , New York, NY, USA. ACM. 

- Meijian Guan, Samuel Cho, Robin Petro, Wei Zhang, Boris Pasche, and Umit Topaloglu. 2019. Natural language processing and recurrent network models for identifying genomic mutation-associated cancer treatment change from patient progress notes. _JAMIA Open_ , 2(1):139–149. 

- Christian Gulden, Melanie Kirchner, Christina Schüttler, Marc Hinderer, Marvin Kampf, Hans-Ulrich Prokosch, and Dennis Toddenroth. 2019. Extractive summarization of clinical trial descriptions. _Int. J. Med. Inform._ , 129:114–121. 

- YN Gwon, JH Kim, HS Chung, EJ Jung, J Chun, S Lee, and SR Shim. 2024. The use of generative ai for scientific literature searches for systematic reviews: Chatgpt and microsoft bing ai performance evaluation. _JMIR Medical Informatics_ , 12:e51187. 

- Anna Górska and Evelina Tacconelli. 2024. Towards autonomous living meta-analyses: A framework for automation of systematic review and meta-analyses. _Stud. Health Technol. Inform._ , 316:378–382. 

- Ehab Hamed, Ahmad Eid, and Medhat Alberry. 2023. Exploring ChatGPT’s potential in facilitating adaptation of clinical guidelines: A case study of diabetic ketoacidosis guidelines. _Cureus_ , 15(5):e38784. 

- Hamed Hassanzadeh, Sarvnaz Karimi, and Anthony Nguyen. 2020. Matching patients to clinical trials using semantically enriched document representation. _Journal of Biomedical Informatics_ , 105:103406. 

- Hendrik Ter Horst, Nicole Brazda, Jessica SchiraHeinen, Julia Krebbers, Hans-Werner Müller, and Philipp Cimiano. 2023. Automatic knowledge graph population with model-complete text comprehension for pre-clinical outcomes in the field of spinal cord injury. _Artificial Intelligence in Medicine_ , 137:102491. 

- Yan Hu, Vipina K Keloth, Kalpana Raja, Yong Chen, and Hua Xu. 2023. Towards precise PICO extraction from abstracts of randomized controlled trials using a section-specific learning approach. _Bioinformatics_ , 39(9):btad542. 

- Andy S. Huang, Kyle Hirabayashi, Laura Barna, Deep Parikh, and Louis R. Pasquale. 2024. Assessment of a large language model’s responses to questions and cases about glaucoma and retina management. _JAMA Ophthalmology_ , 142(4):371–375. 

- Bum-Sup Jang, Andrew J Park, and In Ah Kim. 2022. Exploration of biomedical knowledge for recurrent glioblastoma using natural language processing deep learning models. _BMC Med. Inform. Decis. Mak._ , 22(1):267. 

21431 

- Pengcheng Jiang, Cao Xiao, Zifeng Wang, Parminder Bhatia, Jimeng Sun, and Jiawei Han. 2024. TriSum: Learning summarization ability from large language models with structured rationale. In _Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)_ , pages 2805–2819, Mexico City, Mexico. Association for Computational Linguistics. 

- Di Jin and Peter Szolovits. 2018. PICO element detection in medical text via long short-term memory neural networks. In _Proceedings of the BioNLP 2018 workshop_ , Stroudsburg, PA, USA. Association for Computational Linguistics. 

- Qiao Jin, Chuanqi Tan, Mosha Chen, Ming Yan, and Xiaozhong Liu Ningyu Zhang, Songfang Huang. 2022. State-of-the-art evidence retriever for precision medicine: Algorithm development and validation. _JMIR Medical Informatics_ , 10(12):e40743. 

- Qiao Jin, Zifeng Wang, Charalampos S Floudas, Fangyuan Chen, Changlin Gong, Dara BrackenClarke, Elisabetta Xue, Yifan Yang, Jimeng Sun, and Zhiyong Lu. 2024. Matching patients to clinical trials with large language models. _Nat. Commun._ , 15(1):9074. 

- Tom H Johnston, Alix M B Lacoste, Paula Ravenscroft, Jin Su, Sahar Tamadon, Mahtab Seifi, Anthony E Lang, Susan H Fox, Jonathan M Brotchie, and Naomi P Visanji. 2024. Using artificial intelligence to identify drugs for repurposing to treat l-DOPA-induced dyskinesia. _Neuropharmacology_ , 248(109880):109880. 

- Katikapalli Subramanyam Kalyan, Ajit Rajasekharan, and Sivanesan Sangeetha. 2022. Ammu: A survey of transformer-based biomedical pretrained language models. _Journal of Biomedical Informatics_ , 126:103982. 

- Sowmya Kamath, Veena Mayya, and Priyadarshini. 2021. A probabilistic precision information retrieval model for personalized clinical trial recommendation based on heterogeneous data. In _2021 12th International Conference on Computing Communication and Networking Technologies (ICCCNT)_ , pages 1–5. IEEE. 

- Lara J Kanbar, Benjamin Wissel, Yizhao Ni, Nathan Pajor, Tracy Glauser, John Pestian, and Judith W Dexheimer. 2022. Implementation of machine learning pipelines for clinical practice: Development and validation study. _JMIR Medical Informatics_ , 10(12):e37833. 

- Tian Kang, Adler Perotte, Youlan Tang, Casey Ta, and Chunhua Weng. 2021. Umls-based data augmentation for natural language processing of clinical research literature. _Journal of the American Medical Informatics Association_ , 28(4):812–823. 

- Tian Kang, Yingcheng Sun, Jae Hyun Kim, Casey Ta, Adler Perotte, Kayla Schiffer, Mutong Wu, Yang 

Zhao, Nour Moustafa-Fahmy, Yifan Peng, and Chunhua Weng. 2023. EvidenceMap: a three-level knowledge representation for medical evidence computation and comprehension. _J. Am. Med. Inform. Assoc._ , 30(6):1022–1031. 

- Tian Kang, Shirui Zou, and Chunhua Weng. 2019. Pretraining to recognize PICO elements from randomized controlled trial literature. _Stud. Health Technol. Inform._ , 264:188–192. 

- Samuel Kaskovich, Kirk D. Wyatt, Tomasz Oliwa, Luca Graglia, Brian Furner, Jooho Lee, Anoop Mayampurath, and Samuel L. Volchenboum. 2023. Automated matching of patients to clinical trials: A patient-centric natural language processing approach for pediatric leukemia. _JCO Clinical Cancer Informatics_ , 7. 

- Jenna Kefeli and Nicholas Tatonetti. 2024. Tcgareports: A machine-readable pathology report resource for benchmarking text-based ai models. _Patterns_ , 5(3):100933. Published online February 21, 2024. 

- AH Khan, A Abbe, B Falissard, P Carita, C Bachert, J Mullol, M Reaney, J Chao, LP Mannent, N Amin, P Mahajan, G Pirozzi, and L Eckert. 2021. Data mining of free-text responses: An innovative approach to analyzing patient perspectives on chronic rhinosinusitis with nasal polyps in a phase iia proof-ofconcept study for dupilumab. _Dove Medical Press_ , 2021(15):2577–2586. 

- Jeongeun Kim, Mitchell Izower, and Yuri Quintana. 2023a. Parsable clinical trial eligibility criteria representation using natural language processing. In _AMIA Annual Symposium Proceedings_ , pages 616– 624. American Medical Informatics Association. 

- Jeongeun Kim, Mitchell Izower, and Yuri Quintana. 2023b. Parsable clinical trial eligibility criteria representation using natural language processing. In _AMIA Annual Symposium Proceedings_ , pages 616– 624. American Medical Informatics Association. 

- Su Nam Kim, David Martinez, Lawrence Cavedon, and Lars Yencken. 2024. Nicta-piboso dataset. https: //doi.org/10.57702/ne4r48m1. Dataset consists of 1,000 medical abstracts manually annotated with semantic tags based on the PICO criteria to support the automatic classification of sentences. 

- Bevan Koopman, Tracey Wright, Natacha Omer, Veronica McCabe, and Guido Zuccon. 2021. Precision medicine search for paediatric oncology. In _Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval_ , SIGIR ’21, page 2536–2540, New York, NY, USA. Association for Computing Machinery. 

- Bevan Koopman and Guido Zuccon. 2022. Cohortbased clinical trial retrieval. In _Proceedings of the 25th Australasian Document Computing Symposium_ , ADCS ’21, New York, NY, USA. Association for Computing Machinery. 

21432 

- Fabrício Kury, Alex Butler, Chi Yuan, Li-Heng Fu, Yingcheng Sun, Hao Liu, Ida Sim, Simona Carini, and Chunhua Weng. 2020. Chia, a large annotated corpus of clinical trial eligibility criteria. _Sci. Data_ , 7(1):281. 

- Mary R Kwaan and Genevieve B Melton. 2012. Evidence-based medicine in surgical education. _Clin. Colon Rectal Surg._ , 25(3):151–155. 

- Evani Lalitha, Kasarapu Ramani, Dudekula Shahida, Esikela Venkata Sai Deepak, M Hima Bindu, and Diguri Shaikshavali. 2023. Text summarization of medical documents using abstractive techniques. In _2023 2nd International Conference on Applied Artificial Intelligence and Computing (ICAAIC)_ , pages 939–943. IEEE. 

- Mengfei Lan, Mandy Cheng, Linh Hoang, Gerben Ter Riet, and Halil Kilicoglu. 2024. Automatic categorization of self-acknowledged limitations in randomized controlled trial publications. _J. Biomed. Inform._ , 152:104628. 

- Jinhyuk Lee, Wonjin Yoon, Sungdong Kim, Donghyeon Kim, Sunkyu Kim, Chan Ho So, and Jaewoo Kang. 2020. BioBERT: a pre-trained biomedical language representation model for biomedical text mining. _Bioinformatics_ , 36(4):1234–1240. 

- Kyeryoung Lee, Zongzhi Liu, Yun Mai, Tomi Jun, Meng Ma, Tongyu Wang, Lei Ai, Ediz Calay, William Oh, Gustavo Stolovitzky, Eric Schadt, and Xiaoyan Wang. 2024. Optimizing clinical trial eligibility design using natural language processing models and realworld data: Algorithm development and validation. _JMIR AI_ , 3:e50800. 

- Chao Li, Harsha Gurulingappa, Prathamesh Karmalkar, Jana Raab, Aastha Vij, Gerard Megaro, and Christian Henke. 2021a. Automate clinical evidence synthesis by linking trials to publications with text analytics. In _2021 International Symposium on Electrical, Electronics and Information Engineering_ , New York, NY, USA. ACM. 

- Jianfu Li, Qiang Wei, Omid Ghiasvand, Miao Chen, Victor Lobanov, Chunhua Weng, and Hua Xu. 2022. A comparative study of pre-trained language models for named entity recognition in clinical trial eligibility criteria from multiple corpora. _BMC Med. Inform. Decis. Mak._ , 22(Suppl 3):235. 

- Xinhang Li, Hao Liu, Fabrício Kury, Chi Yuan, Alex Butler, Yingcheng Sun, Anna Ostropolets, Hua Xu, and Chunhua Weng. 2021b. A comparison between human and nlp-based annotation of clinical trial eligibility criteria text using the omop common data model. In _AMIA Joint Summits on Translational Science Proceedings_ , pages 394–403. AMIA. 

- Yizhen Li, Zhongzhi Luan, Yixing Liu, Heyuan Liu, Jiaxing Qi, and Dongran Han. 2024. Automated information extraction model enhancing traditional chinese medicine rct evidence extraction (evi-bert): algorithm development and validation. _frontiers artificial intelligence_ , 7(1454945):not listed. 

- Cong Liu, Hao Liu, Casey Ta, James Roger, Alex Butler, Junghwan Lee, Jaehyun Kim, Ning Shang, and Chunhua Weng. 2022. Evaluation of Criteria2Query: Towards augmented intelligence for cohort identification. _Stud. Health Technol. Inform._ , 290:297–300. 

- Cong Liu, Chi Yuan, Alex M Butler, Richard D Carvajal, Ziran Ryan Li, Casey N Ta, and Chunhua Weng. 2019. Dquest: dynamic questionnaire for search of clinical trials. _Journal of the American Medical Informatics Association_ , 26(11):1333–1343. 

- Hao Liu, Yuan Chi, Alex Butler, Yingcheng Sun, and Chunhua Weng. 2021. A knowledge base of clinical trial eligibility criteria. _Journal of Biomedical Informatics_ , 117:103771. 

- Cynthia Lokker, Elham Bagheri, Wael Abdelkader, Rick Parrish, Muhammad Afzal, Tamara Navarro, Chris Cotoi, Federico Germini, Lori Linkins, R Brian Haynes, Lingyang Chu, and Alfonso Iorio. 2023. Deep learning to refine the identification of highquality clinical research articles from the biomedical literature: Performance evaluation. _J. Biomed. Inform._ , 142(104384):104384. 

- Khalid Mahmood Malik, Madan Krishnamurthy, Pawel Marcinek, and Ghaus M Malik. 2020. Impact of size, location, symptomatic-nature and gender on the rupture of saccular intracranial aneurysms. In _Proceedings of the 2018 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining_ , ASONAM ’18, page 995–1001. IEEE Press. 

- Iain J Marshall, Benjamin Nye, Joël Kuiper, Anna Noel-Storr, Rachel Marshall, Rory Maclean, Frank Soboczenski, Ani Nenkova, James Thomas, and Byron C Wallace. 2020. Trialstreamer: A living, automatically updated database of clinical trial reports. _J. Am. Med. Inform. Assoc._ , 27(12):1903–1912. 

- Iain J Marshall, Thomas A Trikalinos, Frank Soboczenski, Hye Sun Yun, Gregory Kell, Rachel Marshall, and Byron C Wallace. 2023. In a pilot study, automated real-time systematic review updates were feasible, accurate, and work-saving. _J. Clin. Epidemiol._ , 153:26–33. 

- Tobias Mayer, Santiago Marro, Elena Cabrio, and Serena Villata. 2021. Enhancing evidence-based medicine with natural language argumentative analysis of clinical trials. _Artificial Intelligence in Medicine_ , 118:102098. 

- Chirag Mehta, David Cohen, Priya Jaisinghani, and Payal Parikh. 2022. Internal medicine resident adherence to evidence-based practices in management of diabetes mellitus. _J. Med. Educ. Curric. Dev._ , 9:23821205221076659. 

- Stéphane M. Meystre, Paul M. Heider, Andrew Cates, Grace Bastian, Tara Pittman, Stephanie Gentilin, and Teresa J. Kelechi. 2023. Piloting an automated clinical trial eligibility surveillance and provider alert system based on artificial intelligence and standard 

21433 

data models. _BMC Medical Research Methodology_ , 23(88). 

- Rashmi Mishra, Andrea Burke, Bonnie Gitman, Payal Verma, Mark Engelstad, Ilias Alevizos, William A. Gahl, Michael T. Collins, Janice S. Lee, and Murat Sincan. 2019. Data-driven method to enhance craniofacial and oral phenotype vocabularies. _The Journal of the American Dental Association_ , 150(11):933– 939.e2. 

- Sabah Mohammed and Jinan Fiaidhi. 2023. Investigation into scaling-up the soap problem-oriented medical record into a clinical case study. In _2023 IEEE 11th International Conference_ . IEEE. 

- Sabah Mohammed and Jinan Fiaidhi. 2024. Generative AI for evidence-based medicine: A PICO GenAI for synthesizing clinical case reports. In _ICC 2024 - IEEE International Conference on Communications_ , volume 3, pages 1503–1508. IEEE. 

- Sabah Mohammed, Jinan Fiaidhi, and Rahul Kudadiya. 2023. Integrating a PICO clinical questioning to the QL4POMR framework for building evidence-based clinical case reports. In _2023 IEEE International Conference on Big Data (BigData)_ , volume 4, pages 4940–4947. IEEE. 

- Victor M Murcia, Vinod Aggarwal, Nikhil Pesaladinne, Ram Thammineni, Nhan Do, Gil Alterovitz, and Rafael B Fricks. 2024. Automating clinical trial matches via natural language processing of synthetic electronic health records and clinical trial eligibility criteria. _AMIA Summits Transl. Sci. Proc._ , 2024:125– 134. 

- Faith Mutinda, Kongmeng Liew, Shuntaro Yada, Shoko Wakamiya, and Eiji Aramaki. 2022a. PICO corpus: A publicly available corpus to support automatic data extraction from biomedical literature. In _Proceedings of the first Workshop on Information Extraction from Scientific Publications_ , pages 26–31, Online. Association for Computational Linguistics. 

- Faith Wavinya Mutinda, Kongmeng Liew, Shuntaro Yada, Shoko Wakamiya, and Eiji Aramaki. 2022b. Automatic data extraction to support meta-analysis statistical analysis: a case study on breast cancer. _BMC Med. Inform. Decis. Mak._ , 22(1):158. 

- Joshua J. Myszewski, Emily Klossowski, Patrick Meyer, Kristin Bevil, Lisa Klesius, and Kristopher M. Schroeder. 2022. Validating gan-biobert: A methodology for assessing reporting trends in clinical trials. _Frontiers in Digital Health_ , 4. 

- Tamara Navarro-Ruan and R. Brian Haynes. 2022. Preliminary comparison of the performance of the national library of medicine’s systematic review publication type and the sensitive clinical queries filter for systematic reviews in pubmed. _Journal of the Medical Library Association_ , 110(1). 

- Aurélie Névéol, Rezarta Islamaj Do˘gan, and Zhiyong Lu. 2011. Semi-automatic semantic annotation of pubmed queries: a study on quality, efficiency, satisfaction. _Journal of biomedical informatics_ , 44(2):310–318. 

- Abigail Newbury, Hao Liu, Betina Idnay, and Chunhua Weng. 2023. The suitability of UMLS and SNOMED-CT for encoding outcome concepts. _J. Am. Med. Inform. Assoc._ , 30(12):1895–1903. 

- Vincent Nguyen, Sarvnaz Karimi, and Brian Jin. 2019. An experimentation platform for precision medicine. In _Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval_ , SIGIR’19, page 1357–1360, New York, NY, USA. Association for Computing Machinery. 

- Yizhao Ni, Monica Bermudez, Stephanie Kennebeck, Stacey Liddy-Hicks, and Judith Dexheimer. 2019. A real-time automated patient screening system for clinical trials eligibility in an emergency department: Design and evaluation. _JMIR Medical Informatics_ , 7(3):e14185. 

- Mauro Nievas, Aditya Basu, Yanshan Wang, and Hrituraj Singh. 2024. Distilling large language models for matching patients to clinical trials. _Journal of the American Medical Informatics Association_ , 31(9):1953–1963. 

- Christopher Norman, Mariska Leeflang, René Spijker, Evangelos Kanoulas, and Aurélie Névéol. 2019a. A distantly supervised dataset for automated data extraction from diagnostic studies. In _Proceedings of the 18th BioNLP Workshop and Shared Task_ , pages 105–114, Florence, Italy. Association for Computational Linguistics. 

- Christopher R Norman, Mariska M G Leeflang, Raphaël Porcher, and Aurélie Névéol. 2019b. Measuring the impact of screening automation on meta-analyses of diagnostic test accuracy. _Syst. Rev._ , 8(1):243. 

- Elvira Nurmambetova, Jie Pan, Zilong Zhang, Seungwon Lee, Danielle A Southern, Elliot A Martin, Guosong Wu, Chester Ho, and Cathy A Eastwood. 2023. Developing an inpatient electronic medical record phenotype for hospital-acquired pressure injuries: Case study using natural language processing models. _JMIR AI_ , 2(2023):e41264. 

- Benjamin Nye, Junyi Jessy Li, Roma Patel, Yinfei Yang, Iain Marshall, Ani Nenkova, and Byron Wallace. 2018. A corpus with multi-level annotations of patients, interventions and outcomes to support language processing for medical literature. In _Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 197–207, Melbourne, Australia. Association for Computational Linguistics. 

- Zhenhe Pan, Shuang Jiang, Juntao Su, Muzhe Guo, and Yuanlin Zhang. 2021. Knowledge graph based platform of COVID-19 drugs and symptoms. In _Proceedings of the 2021 IEEE/ACM International Conference_ 

21434 

_on Advances in Social Networks Analysis and Mining_ , New York, NY, USA. ACM. 

- Yifan Peng, Justin F Rousseau, Edward H Shortliffe, and Chunhua Weng. 2023. AI-generated text may have a role in evidence-based medicine. _Nat. Med._ , 29(7):1593–1594. 

- Sanjana Ramprasad, Iain J. Marshall, Denis Jered McInerney, and Byron C. Wallace. 2023a. Automatically summarizing evidence from clinical trials: A prototype highlighting current challenges. In _Proceedings of the Conference of the Association for Computational Linguistics Meeting_ , pages 236–247. 

- Sanjana Ramprasad, Jered Mcinerney, Iain Marshall, and Byron Wallace. 2023b. Automatically summarizing evidence from clinical trials: A prototype highlighting current challenges. In _Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics: System Demonstrations_ , pages 236–247, Stroudsburg, PA, USA. Association for Computational Linguistics. 

- Iqbal Ratnani, Sahar Fatima, Muhammad Mohsin Abid, Zehra Surani, and Salim Surani. 2023. Evidencebased medicine: History, review, criticisms, and pitfalls. _Cureus_ , 15(2):e35266. 

- Omid Rohanian, Mohammadmahdi Nouriborji, Samaneh Kouchaki, Farhad Nooralahzadeh, Lei Clifton, and David A. Clifton. 2024. Exploring the effectiveness of instruction tuning in biomedical language processing. _Artificial Intelligence in Medicine_ , 158:103007. 

- Mohammad Abu Tareq Rony, Mohammad Shariful Islam, Tipu Sultan, Samah Alshathri, and Walid ElShafai. 2023. Medigpt: Exploring potentials of conventional and large language models on medical data. _IEEE Access_ , 12. 

- Maciej Rybinski, Sarvnaz Karimi, Vincent Nguyen, and Cecile Paris. 2020a. A2A: a platform for research in biomedical literature search. _BMC Bioinformatics_ , 21(Suppl 19):572. 

- Maciej Rybinski, Jerry Xu, and Sarvnaz Karimi. 2020b. Clinical trial search: Using biomedical language understanding models for re-ranking. _J. Biomed. Inform._ , 109(103530):103530. 

- David L. Sackett, William M. C. Rosenberg, J. A. Muir Gray, R. Brian Haynes, and W. Scott Richardson. 1996. Evidence based medicine: what it is and what it isn’t. _BMJ_ , 312(7023):71–72. 

- Jawad Sadek, Alex Inskip, James Woltmann, Georgina Wilkins, Christopher Marshall, Maria Pokora, Amey Vedpathak, Anastasija Jadrevska, Dawn Craig, and Michael Trenell. 2023. Scanmedicine: An online search system for medical innovation. _Contemporary Clinical Trials_ , 125:107042. 

- Fernando Suarez Saiz, Corey Sanders, Rick Stevens, Robert Nielsen, Michael Britt, Leemor Yuravlivker, Anita M Preininger, and Gretchen P Jackson. 2021. Artificial intelligence clinical evidence engine for automatic identification, prioritization, and extraction of relevant clinical oncology research. _JCO Clin. Cancer Inform._ , 5(5):102–111. 

- Hamman Samuel, Osmar Zaiane, and Francois Bolduc. 2021. Evaluation of applied machine learning for health misinformation detection via survey of medical professionals on controversial topics in pediatrics. In _Proceedings of the 5th International Conference on Medical and Health Informatics_ , pages 1–6. ACM. 

- Olivia Sanchez-Graillet, Christian Witte, Frank Grimm, and Philipp Cimiano. 2022. An annotated corpus of clinical trial publications supporting schema-based relational information extraction. _J. Biomed. Semantics_ , 13(1):14. 

- Abeed Sarker, Yuan-Chi Yang, Mohammed Ali AlGaradi, and Aamir Abbas. 2020. A light-weight text summarization system for fast access to medical evidence. _Front. Digit. Health_ , 2:585559. 

- Abigail See, Peter J. Liu, and Christopher D. Manning. 2017. Get to the point: Summarization with pointer-generator networks. _arXiv preprint arXiv:1704.04368_ . 

- Isabel Segura-Bedmar and Pablo Raez. 2019. Cohort selection for clinical trials using deep learning models. _J. Am. Med. Inform. Assoc._ , 26(11):1181–1188. 

- Makoto Shiraishi, Yoko Tomioka, Ami Miyakuni, Saaya Ishii, Asei Hori, Hwayoung Park, Jun Ohba, and Mutsumi Okazaki. 2024. Performance of ChatGPT in answering clinical questions on the practical guideline of blepharoptosis. _Aesthetic Plast. Surg._ , 48(13):2389–2398. 

- Irena Spasic, David Krzeminski, Paul Corcoran, and Alexander Balinsky. 2019. Cohort selection for clinical trials from longitudinal patient records: Text mining approach. _JMIR Medical Informatics_ , 7(4):e15980. 

- Nikolaos Stylianou, Gerasimos Razis, Dimitrios G. Goulis, and Ioannis Vlahavas. 2020. Ebm+: Advancing evidence-based medicine via two level automatic identification of populations, interventions, outcomes in medical literature. _Artificial Intelligence in Medicine_ , 108:101949. 

- Nikolaos Stylianou and Ioannis Vlahavas. 2021. TransforMED: End-to-end transformers for evidencebased medicine and argument mining in medical literature. _J. Biomed. Inform._ , 117(103767):103767. 

- Davide Testa, Emmanuele Chersoni, and Alessandro Lenci. 2023. We understand elliptical sentences, and language models should too: A new dataset for studying ellipsis and its interaction with thematic fit. In 

21435 

_Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 3340–3353, Toronto, Canada. Association for Computational Linguistics. 

- Arun James Thirunavukarasu, Darren Shu Jeng Ting, Kabilan. Elangovan, et al. 2023. Large language models in medicine. _Nature Medicine_ , 29:1930– 1940. 

- Shubo Tian, Arslan Erdengasileng, Xi Yang, Yi Guo, Yonghui Wu, Jinfeng Zhang, Jiang Bian, and Zhe He. 2021. Transformer-based named entity recognition for parsing clinical trial eligibility criteria. _ACM BCB_ , 2021. 

- Shubo Tian, Pengfei Yin, Hansi Zhang, Arslan Erdengasileng, Jiang Bian, and Zhe He. 2023. Parsing clinical trial eligibility criteria for cohort query by a multi-input multi-output sequence labeling model. In _2023 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)_ , pages 4426–4430. 

- Hegler C. Tissot, Anoop D. Shah, David Brealey, Steve Harris, Ruth Agbakoba, and Amos Folarin. 2020. Natural language processing for mimicking clinical trial recruitment in critical care: A semi-automated simulation based on the leopards trial. _IEEE Journal of Biomedical and Health Informatics_ , 24(10):2950– 2959. 

- Tadashi Tsubota, Danushka Bollegala, Yang Zhao, Yingzi Jin, and Tomotake Kozu. 2022. Improvement of intervention information detection for automated clinical literature screening during systematic review. _J. Biomed. Inform._ , 134(104185):104185. 

- Pyae Phyo Tun, Jiawen Luo, Jiecheng Xie, Sandi Wibowo, and Chen Hao. 2023. Automatic assessment of patient eligibility by utilizing nlp and rule-based analysis. In _2023 45th Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC)_ , page 10340494, Sydney, Australia. IEEE. 

- Ali Turfaha, Hao Liu, Latoya A Stewart, Tian Kang, and Chunhua Weng. 2022. Extending pico with observation normalization for evidence computing. In _MEDINFO 2021: One World, One Health – Global Partnership for Digital Innovation_ , pages 268–272, New York, New York, USA. International Medical Informatics Association and IOS Press, IOS Press. 

- Ozan Unlu, Jiyeon Shin, Charlotte J Mailly, Michael F Oates, Michela R Tucci, Matthew Varugheese, Kavishwar Wagholikar, Fei Wang, Benjamin M Scirica, Alexander J Blood, and Samuel J Aronson. 2024. Retrieval augmented generation enabled generative pre-trained transformer 4 (GPT-4) performance for clinical trial screening. _medRxiv_ . 

- Peter Van de Vliet, Tobias Sprenger, Linde F C Kampers, Jennifer Makalowski, Volker Schirrmacher, Wilfried Stücker, and Stefaan W Van Gool. 2023. The application of evidence-based medicine in individualized medicine. _Biomedicines_ , 11(7). 

- Bianca Vora, Denison Kuruvilla, Chloe Kim, Michael Wu, Colby S. Shemesh, and Gillie A. Roth. 2023. Applying natural language processing to clinicaltrials.gov: mrna cancer vaccine case study. _Clinical and Translational Science_ , 16:2417–2420. 

- V G Vinod Vydiswaran, Asher Strayhorn, Xinyan Zhao, Phil Robinson, Mahesh Agarwal, Erin Bagazinski, Madia Essiet, Bradley E Iott, Hyeon Joo, Pingjui Ko, Dahee Lee, Jin Xiu Lu, Jinghui Liu, Adharsh Murali, Koki Sasagawa, Tianshi Wang, and Nalingna Yuan. 2019. Hybrid bag of approaches to characterize selection criteria for cohort identification. _J. Am. Med. Inform. Assoc._ , 26(11):1172–1180. 

- Kunyuan Wang, Hao Cui, Yun Zhu, Xiaoyun Hu, Chang Hong, Yabing Guo, Lingyao An, Qi Zhang, and Li Liu. 2024. Evaluation of an artificial intelligencebased clinical trial matching system in chinese patients with hepatocellular carcinoma: a retrospective study. _BMC Cancer_ , 24(1):246. 

- Yu Wang, Yuan Wang, Zhenwan Peng, Feifan Zhang, Luyao Zhou, and Fei Yang. 2023a. Medical text classification based on the discriminative pretraining model and prompt-tuning. _Digit. Health_ , 9:20552076231193213. 

- Zifeng Wang and Jimeng Sun. 2022. Trial2Vec: Zeroshot clinical trial document similarity search using self-supervision. In _Findings of the Association for Computational Linguistics: EMNLP 2022_ , pages 6377–6390, Stroudsburg, PA, USA. Association for Computational Linguistics. 

- Zifeng Wang, Cao Xiao, and Jimeng Sun. 2023b. AutoTrial: Prompting language models for clinical trial design. In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_ , pages 12461–12472, Singapore. Association for Computational Linguistics. 

- Christian Witte, David M Schmidt, and Philipp Cimiano. 2024. Comparing generative and extractive approaches to information extraction from abstracts describing randomized clinical trials. _J. Biomed. Semantics_ , 15(1):3. 

- Qianqian Xie, Jennifer Amy Bishop, Prayag Tiwari, and Sophia Ananiadou. 2022. Pre-trained language models with domain knowledge for biomedical extractive summarization. _Knowl. Based Syst._ , 252(109460):109460. 

- Shiyao Xie, Wenjing Zhao, Guanghui Deng, Guohua He, Na He, Zhenhua Lu, Weihua Hu, Mingming Zhao, and Jian Du. 2024. Utilizing chatgpt as a scientific reasoning engine to differentiate conflicting evidence and summarize challenges in controversial clinical questions. _Journal of the American Medical Informatics Association_ , 31(7):1551–1560. Published online: 17 May 2024. 

- Yi Xie, Ishith Seth, David J Hunter-Smith, Warren M Rozen, Richard Ross, and Matthew Lee. 2023. Aesthetic surgery advice and counseling from artificial 

21436 

intelligence: A rhinoplasty consultation with ChatGPT. _Aesthetic Plast. Surg._ , 47(5):1985–1993. 

- Quan Xu, Yueyue Liu, Dawei Sun, Xiaoqian Huang, Feihong Li, Jincheng Zhai, Yang Li, Qiming Zhou, Niansong Qian, and Beifang Niu. 2023. OncoCTMiner: streamlining precision oncology trial matching via molecular profile analysis. _Database (Oxford)_ , 2023:baad077. 

- Ce Zheng, Hongfei Ye, Jinming Guo, Junrui Yang, Ping Fei, Yuanzhi Yuan, Danqing Huang, Yuqiang Huang, Jie Peng, Xiaoling Xie, Meng Xie, Peiquan Zhao, Li Chen, and Mingzhi Zhang. 2024. Development and evaluation of a large language model of ophthalmology in chinese. _Br. J. Ophthalmol._ , 108(10):1390– 1397. 

## **A Appendix** 

- Yumeng Yang, Soumya Jayaraj, Ethan Ludmir, and Kirk Roberts. 2023. Text classification of cancer clinical trial eligibility criteria. _AMIA Annu. Symp. Proc._ , 2023:1304–1313. 

- Xiaoxi Yao, Zachi I. Attia, Emma M. Behnken, Kelli Walvatne, Rachel E. Giblon, Sijia Liu, Konstantinos C. Siontis, Bernard J. Gersh, Jonathan GraffRadford, Alejandro A. Rabinstein, Paul A. Friedman, and Peter A. Noseworthy. 2021. Batch enrollment for an artificial intelligence-guided intervention to lower neurologic events in patients with undiagnosed atrial fibrillation: rationale and design of a digital clinical trial. _American Heart Journal_ , 239:73–79. 

- Fatin Syafiqah Yazi, Wan-Tze Vong, Valliappan Raman, Patrick Hang Hui Then, and Mukulraj J Lunia. 2021. Towards automated detection of contradictory research claims in medical literature using deep learning approach. In _2021 Fifth International Conference on Information Retrieval and Knowledge Management (CAMP)_ , pages 116–121. 

- Jiayi Yuan, Ruixiang Tang, Xiaoqian Jiang, and Xia Hu. 2024. Large language models for healthcare data augmentation: An example on patient-trial matching. In _AMIA Annual Symposium Proceedings_ , volume 2024, pages 1324–1333. AMIA. 

- Kun Zeng, Zhiwei Pan, Yibin Xu, and Yingying Qu. 2020. An ensemble learning strategy for eligibility criteria text classification for clinical trial recruitment: Algorithm development and validation. _JMIR Medical Informatics_ , 8(7):e17832. 

- Gongbo Zhang, Qiao Jin, Yiliang Zhou, Song Wang, Betina Idnay, Yiming Luo, Elizabeth Park, Jordan G. Nestor, Matthew E. Spotnitz, Ali Soroush, Thomas R. Campion Jr, Zhiyong Lu, Chunhua Weng, and Yifan Peng. 2024a. Closing the gap between open source and commercial large language models for medical evidence summarization. _NPJ Digital Medicine_ , 7(1):239. 

- Gongbo Zhang, Yiliang Zhou, Yan Hu, Hua Xu, Chunhua Weng, and Yifan Peng. 2024b. A span-based model for extracting overlapping PICO entities from RCT publications. _J. Am. Med. Inform. Assoc._ 

- Yixuan Zhang, Junzhen Liu, and Wei Lu. 2023. Medictgp: An accurate entity recognition model combining medical domain knowledge and globalization ideas. In _Proceedings of the 2023 9th International Conference on Computing and Artificial Intelligence_ , ICCAI ’23, page 477–483, New York, NY, USA. Association for Computing Machinery. 

21437 

## **A.1 Included Studies** 

Supplementary Table 1: Overview of Included Studies. P - Precision, R - Recall, Acc - Accuracy, NDCG - Normalized Discounted Cumulative Gain. 

Sy) S - Abstractive Summarization, @ T - Clinical Trial Design, @ N - Entity Extraction and Classification, ® E - Evaluation of Performance, L - Evidence Ranking and Screening, S - Extractive Summarization, Y - Evidence Synthesis, @ I - Information Retrieval, Sy) @ A - Quality Assessment, @ Q O - Question Answering, @ R - Relation Extraction. O 

|Study|Model|Disease|Task|
|---|---|---|---|
|Al Hafiz Khan et al.(2019)|RNN|–|T N<br>E<br>@0e|
|Alexander et al.(2020)|Statistical|Lung Cancer|T N<br>E<br>@0e|
|Alodadi and Janeja(2019)|Rules|–|N<br>E<br>I<br>R<br>@000|
|Aum and Choe(2021)|Transformer|Cognitive Impairment|N<br>E<br>L R<br>@O00|
|Beattie et al.(2024)|LLM|–|T N<br>E<br>C@e@|
|Beck et al.(2020)|Rules|Breast Cancer|T N<br>E<br>@@e@|
|Borchert et al.(2022)|Rules|Cancer|N R<br>eo|
|Brassey et al.(2021)|Rules|–|N Y A<br>@0®8|
|Brockmeier et al.(2019)|Transformer|–|N<br>L<br>ee|
|Cai et al.(2021)|Random Forest, Logistic LASSO|Rheumatoid Arthritis|T<br>E<br>@®@|
|Campillos-Llanos et al.|Transformer|–|N<br>E<br>ee|
|(2021)||||
|Chen et al.(2019a)|Rules|–|S<br>T N<br>E R<br>@O000|
|Chen et al.(2019b)|CNN|Myocardial infarction|T N<br>E<br>Cee|
|Chen et al.(2019c)|Rules|–|T N<br>E<br>@0e|
|Chuan and Morgan(2021)<br>Cunningham et al.(2024)|CNN<br>Transformer|Cancer<br>Heart Failure|T N<br>E<br>Q<br>T N<br>E<br>C00e@<br>@0e|
|Daluwatumulle et al.(2022)|Graph|Alzheimer’s Disease (AD)|N<br>E<br>I<br>@e2|
|Datta et al.(2024)|LLM|–|N<br>E<br>L<br>@ee@|
|Devi et al.(2024b)|LLM|Non-Small Cell Lung Cancer|T N<br>E<br>Cee|
|||(NSCLC)||
|DeYoung et al.(2021)|Transformer|–|S N<br>E R<br>@0e0|
|Deng et al.(2019)|Graph, Statistical|Coronary heart disease, Chest<br>pain, Bronchitis|E<br>Q<br>ee|
|Do et al.(2024)|Rules, Statistical|Cancer|T N<br>@@|
|Dobbins et al.(2022)|Transformer|–|T N<br>E R<br>aT I®|
|Dobbins et al.(2023)|Transformer, Rules|–|T N<br>E<br>I<br>R<br>@0000|
|Dhrangadhariya et al.|Statistical, Learning to Rank|–|N<br>E Y A<br>@0@00|
|(2020)||||
|Dhrangadhariya and Müller|Transformer|–|N<br>E<br>ee|
|(2023)||||
|Dhrangadhariya et al.|Rules, Statistical|–|N<br>E<br>ee|
|(2024)||||
|Dhayne et al.(2021)|SVM, CNN, RNN, Transformer|Viral Infection|T N<br>E<br>@0e|
|Du et al.(2021)|Graph|COVID-19|E<br>I<br>@O|
|Fang et al.(2021)|Rules|–|T N<br>@@|
|Gates and Hamed(2020)|Graph, Statistical|SARS|N<br>L<br>@e@|
|Ghosh et al.(2024a)|LLM|–|N<br>E<br>@e@|
|Ghosh et al.(2024b)|Transformer, RNN|–|N<br>E<br>@e@|
|Górska and Tacconelli|Transformer, LLM|–|N<br>L Y<br>@O0O|
|(2024)||||
|Gulden et al.(2019)|Graph|–|N<br>E<br>S<br>@@0|
|Gwon et al.(2024)|LLM|Peyronie Disease|N<br>E<br>I<br>A<br>0000|
|Hamed et al.(2023)<br>Hassanzadeh et al.(2020)|LLM<br>Support Vector Machine (SVM),|Diabetic Ketoacidosis<br>–|S<br>E<br>Q<br>T N<br>E<br>I<br>TT}<br>@000|
||Random Forest (RF), Logistic|||
||Regression (LR), and Stochastic|||
||Gradient Descent (SGD), RNN|||
|Horst et al.(2023)|Graph, CRF, Statistical|Spinal Cord Injury|T N<br>E Y<br>@@@0|
|Hu et al.(2023)<br>Huang et al.(2024)<br>Jang et al.(2022)<br>Jiang et al.(2024)|Transformer<br>LLM<br>Transformer<br>LLM|COVID-19, AD<br>Glaucoma<br>Recurrent Glioblastoma<br>–|N<br>E<br>E<br>Q<br>N<br>E<br>Q<br>S<br>E<br>Q<br>$0<br>@S98|
||||Continued on next page|



21438 

**Supplementary Table 1 – continued from previous page** 

|**Supplementary Table 1 – continued frompreviouspage**|**Supplementary Table 1 – continued frompreviouspage**|**Supplementary Table 1 – continued frompreviouspage**|**Supplementary Table 1 – continued frompreviouspage**|
|---|---|---|---|
|Study<br>Model<br>Disease<br>Task||||
|Jin et al.(2022)<br>Transformer<br>–<br>E<br>I<br>A||||
||E<br>I<br>A|||
|Jin et al.(2024)<br>Transformer, LLM<br>–|T<br>E<br>L<br>I|||
|Johnston et al.(2024)<br>Vector Space Model, Statistical<br>l-DOPA-induced dyskinesia in<br>Parkinson<br>N<br>L|N<br>L|||
|Kamath et al.(2021)<br>Statistical<br>–|T<br>L<br>I|||
|Kanbar et al.(2022)<br>SVM, Naive Bayes, Random<br>Forest<br>Epilepsy<br>T N<br>E|T N<br>E|||
|Kang et al.(2019)<br>RNN<br>–|N<br>E|||
|Kang et al.(2021)<br>Transformer, RNN<br>–|N<br>E|||
|Kang et al.(2023)<br>Transformer<br>COVID-19|S N<br>E Y|||
|Kaskovich et al.(2023)<br>SVM<br>Pediatric Leukemia|T N<br>E<br>L|||
|Kefeli and Tatonetti(2024)<br>BERT<br>Cancer|N<br>E|||
|Khan et al.(2021)<br>Statistical<br>Chronic Rhinosinusitis with Nasal<br>Polyps<br>N|N|||
|Kim et al.(2023a)<br>LR, NB, kNN, SVM, CNN, RNN,<br>FastText, Transformer, ERNIE<br>Hepatocellular Carcinoma<br>T N<br>E||||
|Kim et al.(2023b)<br>Transformer<br>–|T N<br>E|||
|Koopman et al.(2021)<br>Graph<br>Cancer|N<br>I|||
|Koopman and Zuccon<br>(2022)<br>MMR<br>Cancer<br>T<br>L<br>I|T<br>L<br>I|||
|Kury et al.(2020)<br>Rules<br>–<br>T<br>E||||
|Lalitha et al.(2023)<br>T5(Text-to-Text Transfer<br>Transformer), BART<br>(Bidirectional Auto-Regressive<br>Transformer) and PEGASUS<br>(Pre-training with Extracted<br>Gap-sentences for Abstractive<br>Summarization<br>Sequence-to-sequence)<br>–<br>S<br>E||||
|Lan et al.(2024)<br>Transformer<br>–|N<br>E A|||
|Lee et al.(2024)<br>RNN<br>Cancer|T N<br>E|||
|Li et al.(2021a)<br>Vector Space model<br>–|L<br>I|||
|Li et al.(2021b)<br>Rules<br>–|N|||
|Li et al.(2022)<br>Transformer<br>–|T N<br>E|||
|Li et al.(2024)<br>Transformer<br>Stroke, Colorectal Cancer,<br>Coronary Heart Disease, Heart<br>Failure, Chronic Obstructive<br>Pulmonary Disease, Diabetes,<br>Diabetic Nephropathy,<br>Osteoarthritis, Obesity,<br>Rheumatoid Arthritis, and<br>Diarrhea<br>N<br>E|N<br>E|||
|Liu et al.(2019)<br>RNN<br>–|N<br>E<br>I|||
|Liu et al.(2021)<br>Rules<br>–|T N<br>E R|||
|Liu et al.(2022)<br>Rules, RNN<br>–|T N R|||
|Lokker et al.(2023)<br>Transformer<br>–|E<br>I<br>A|||
|Malik et al.(2020)<br>Statistical<br>–|N|||
|Marshall et al.(2023)<br>Transformer<br>COVID-19|N Y<br>I|||
|Mayer et al.(2021)<br>GRU, CRF, RNN, Transformer<br>–|N<br>E R|||
|Meystre et al.(2023)<br>Rules<br>–|T N<br>L|||
|Mishra et al.(2019)<br>Statistical<br>Craniofacial Abnormalities|N<br>E|||
|Mohammed and Fiaidhi<br>(2023)<br>Graph<br>–<br>N<br>I|N<br>I|||
|Mohammed et al.(2023)<br>Graph<br>–|N<br>E<br>S|||
|Mohammed and Fiaidhi<br>(2024)<br>LLM<br>–<br>N<br>E Y<br>Q|N<br>E Y<br>Q|||
|Murcia et al.(2024)<br>Transformer<br>–|T N|||
|Mutinda et al.(2022b)<br>Rules, Statistical, Transformer<br>Breast cancer|N<br>E Y|||
|Myszewski et al.(2022)<br>Transformer<br>-|N<br>E A|||
|Navarro-Ruan and Haynes<br>(2022)<br>Rules, Statistical<br>–<br>L<br>I|L<br>I|||
|Newbury et al.(2023)<br>Rules<br>–<br>N<br>E||||



Continued on next page 

21439 

**Supplementary Table 1 – continued from previous page** 

|**Supplementary Table 1 – continued frompreviouspage**|**Supplementary Table 1 – continued frompreviouspage**|**Supplementary Table 1 – continued frompreviouspage**|**Supplementary Table 1 – continued frompreviouspage**|**Supplementary Table 1 – continued frompreviouspage**|
|---|---|---|---|---|
|Study<br>Model<br>Disease<br>Task|||||
|Nguyen et al.(2019)<br>Transformer, RNN, SVM<br>–<br>L<br>I|||||
|Ni et al.(2019)<br>Rules<br>Respiratory Tract Infection,<br>Traumatic Brain Injury, and<br>Serious Bacterial Infections<br>T N<br>E|||||
|Nievas et al.(2024)<br>LLM<br>–|T<br>E||||
|Norman et al.(2019b)<br>Statistical<br>–|L||||
|Nurmambetova et al.(2023)<br>Random Forest, XGBoost<br>Acquired Pressure Injuries|N<br>E||||
|Pan et al.(2021)<br>Transformer, Graph<br>COVID-19|N<br>E<br>S<br>Q R||||
|Ramprasad et al.(2023a)<br>Transformer, Longformer<br>–|S<br>I||||
|Rony et al.(2023)<br>LLM<br>–|E<br>Q||||
|Rybinski et al.(2020a)<br>Rules<br>–|I||||
|Rybinski et al.(2020b)<br>Transformer<br>–|N<br>E<br>L<br>I||||
|Sadek et al.(2023)<br>Knowledge<br>–|N<br>I||||
|Saiz et al.(2021)<br>Gradient-boosted Trees<br>Cancer|T N<br>E<br>L||||
|Samuel et al.(2021)<br>Statistical<br>Autism|E<br>I||||
|Sanchez-Graillet et al.<br>(2022)<br>Transformer, Schema<br>Glaucoma, Type 2 diabetes<br>mellitus<br>N<br>E|N<br>E||||
|Sarker et al.(2020)<br>Statistical<br>COVID-19|E<br>S||||
|Segura-Bedmar and Raez<br>(2019)<br>CNN, RNN<br>–<br>T<br>E|T<br>E||||
|Shiraishi et al.(2024)<br>LLM<br>Blepharoptosis|E<br>Q||||
|Spasic et al.(2019)<br>SVM, Logistic Regression, Naive<br>Bayes, Gradient Tree Boosting,<br>Rules, Decision Trees, Random<br>Forests<br>–<br>T N<br>E|T N<br>E||||
|Stylianou et al.(2020)<br>Transformer, RNN<br>–|N<br>E||||
|Stylianou and Vlahavas<br>(2021)<br>Transformer<br>–<br>N<br>E|N<br>E||||
|Tian et al.(2021)<br>Transformer<br>–<br>T N A|||||
|Tian et al.(2023)<br>Statistical, Transformer<br>AD<br>T N<br>E|||||
|Tissot et al.(2020)<br>Rules<br>Organ dysfunction in septic shock<br>T N<br>E<br>I|||||
|Tsubota et al.(2022)<br>Transformer<br>–<br>N<br>E<br>I|||||
|Tun et al.(2023)<br>Rules<br>Cardiovascular Events<br>T<br>E|||||
|Turfaha et al.(2022)<br>Rules<br>–<br>N<br>E|||||
|Unlu et al.(2024)<br>LLM<br>Heart Failure<br>E<br>Q|||||
|Vora et al.(2023)<br>Rules<br>mRNA Cancer<br>N<br>I|||||
|Vydiswaran et al.(2019)<br>Rules<br>–<br>T N<br>E|||||
|Wang and Sun(2022)<br>Transformer<br>–<br>N<br>E<br>I|||||
|Wang et al.(2023a)<br>Statistical, Transformer<br>–<br>N<br>E|||||
|Wang et al.(2023b)<br>LLM<br>–<br>T<br>E Y R|||||
|Wang et al.(2024)<br>CNN, Graph, Rules<br>Hepatocellular Carcinoma<br>N<br>E|||||
|Witte et al.(2024)<br>Transformer<br>Glaucoma, Type II Diabetes<br>N<br>E|||||
|Xie et al.(2022)<br>Transformer<br>–<br>N<br>S|||||
|Xie et al.(2023)<br>LLM<br>Rhinoplasty<br>Q|||||
|Xie et al.(2024)<br>LLM<br>–<br>S<br>E Y A|||||
|Xu et al.(2023)<br>Graph<br>Cancer<br>N<br>L A|||||
|Yang et al.(2023)<br>Transformer<br>Cancer<br>T N|||||
|Yao et al.(2021)<br>Not Specify<br>Atrial Fibrillation<br>T N<br>E|||||
|Yazi et al.(2021)<br>Transformer<br>–<br>E A|||||
|Yuan et al.(2024)<br>LLM<br>–<br>T N<br>E|||||
|Zeng et al.(2020)<br>Transformer<br>–<br>N A|||||
|Zhang et al.(2023)<br>Transformer, RNN<br>–<br>N<br>E|||||
|Zhang et al.(2024b)<br>Transformer<br>COVID-19, AD<br>N<br>E<br>I|||||
|Zheng et al.(2024)<br>LLM<br>Glaucoma<br>E<br>Q|||||



21440 

## **A.2 Benchmark Dataset** 

Supplementary Table 2: Overview of recent benchmark datasets. P - Population. I - Intervention. C - Comparison. O - Outcome (Eldawlatly et al., 2018). RCT - Randomized controlled trial. ETC - Anything that doesn’t fit into the categories above. GPG - GNU Privacy Guard. CMS - Content Management System. 

|Dataset|Avail.|Label|Annotation|Description|
|---|---|---|---|---|
|Alzheimer’s disease|Public|P, I, C, O|Manual|150 Alzheimer’s disease RCT|
|RCT (Hu et al.,2023)||||abstracts|
|Chia (Kury et al.,|Public|Non-query-able, Post-eligibility,|Manual|Eligibility statements from 1000|
|2020)||Informed consent, Pregnancy||clinical trials and the dataset|
|||considerations, Parsing error,||includes 12,409 annotated|
|||Non-representable, Competing||eligibility criteria|
|||trial, Context error, Subjective|||
|||judgment, Not a criteria,<br>Undefned semantics, Intoxication|||
|||considerations|||
|Clinical trials on|Private|Eligible, Not eligible|Manual|6M eligibility statements in|
|cancer (Rony et al.,||||clinical trials|
|2023)|||||
|COVID-19 corpus (Hu|Public|P, I, C, O|Manual|150 COVID-19 RCT abstracts|
|et al.,2023)|||||
|CT-EBM-SP (Campillos-|Public|Anatomy, pharmacological and|Manual|1,200 texts about clinical trials|
|Llanos et al.,2021)||chemical substances, pathologies,||with entities|
|||and lab tests, diagnostic or|||
|||therapeutic procedures|||
|EBM-COMET (Ghosh|Public|Physiological or clinical, Death,|Manual|300 RCT abstracts|
|et al.,2024b)||Life impact, Resource use,|||
|||Adverse events|||
|EBM-NLP (Nye et al.,|Public|P, I, O|Manual|4,993 medical abstracts from|
|2018)||||literatures on PubMed|
|EliIE (Testa et al.,|Public|Condition, observation,|Manual|230 Alzheimer’s disease RCT|
|2023)||drug/substance, and procedure or||documents|
|||device|||
|GGPONC (Borchert|Public|Recommendation creation date,|Automatic|25 GPGs with 8,414 text segments|
|et al.,2020)||Type of recommendation,||from the CMS|
|||Recommendation grade, Strength|||
|||of consensus, Total vote in|||
|||percentage, Literature references,|||
|||Expert opinion, Level of evidence,|||
|||Edit State|||
|LCT (Dobbins et al.,<br>2022)|Public|Clinical, Demographic, Logical,<br>Qualifers, Temporal and|Manual|1,000+ eligibility statements|
|||Comparative, Other|||
|Limsi-Cochrane|Public|Systematic reviews, Included|Manual|1,939 meta-analyses from 63|
|dataset (Norman et al.,||studies, Data forms, Text entries,||systematic reviews of diagnostic|
|2019a)||Excluded studies, Diagnostic tests,||test accuracy from the Cochrane|
|||Test results, Study IDs, Numerical,||Library|
|||Summary scores|||
|MedReview (Zhang|Private|medical related topics (e.g.|Manual|Meta-analysis results and narrative|
|et al.,2024a)||Wounds, Urology)||summaries from the Cochrane|
|||||Library|
|MS_∧_2 (DeYoung<br>et al.,2021)|Public|Background, Goal, Methods,<br>Detailed fndings, Further study,|Manual|470k documents and 20K<br>summaries from the scientifc|
|||Recommendation, Evidence||literature|
|||quality, effect, ETC|||
|NICTA-PIBOSO (Kim|Public|P, I, O, Background, Study Design,|Manual|1,000 biomedical abstracts|
|et al.,2024)||Other|||
|PICO-Corpus (Mutinda|Public|P, I, C, O|Manual|1,011 breast cancer RCT abstracts|
|et al.,2022a)|||||
|RedHOT (Ghosh et al.,|Public|P, I, O|Manual|22,000 social media posts from|
|2024b)||||Reddit spanning 24 health|
|||||conditions|
|Illness dataset (Rony|Private|Alzheimer’s, Parkinson’s, Cancer,|Manual|22,660 tweets|
|et al.,2023)||and Diabetes domains|||
|||||Continued on next page|



21441 

## **Supplementary Table 2 – continued from previous page** 

|Dataset|Avail.|Label|Annotation|Description|
|---|---|---|---|---|
|Symptom2Disease<br>dataset (Rony et al.,|Private|24 diseases, each described by 50<br>symptom profles|Manual|1,200 data points|
|2023)|||||
|Trialstreamer (Mar-|Public|P, I, O, RCT classifers|Manual|191 RCT publications|
|shall et al.,2020)|||||



21442 

## **A.3 Queries** 

{nlp_keywords} = natural language processing OR nlp OR language model OR large language model OR llm OR computational linguistics OR information extraction OR information retrieval OR clinical trial retrieval OR text summarization OR question answering OR sentence segmentation OR ner OR named entity recognition OR tokenization) 

{ebm_keywords) = evidence-based medicine OR ebm OR evidence-based practice OR ebp OR clinical trial 

## **PubMed** 

(({nlp_keywords}[Title/Abstract]) AND ({ebm_keywords}[Title/Abstract])) 

## **IEEE Xplore** 

(("Abstract":{nlp_keywords}) AND ("Abstract":{ebm_keywords})) OR (("Title":{nlp_keywords}) AND (("Title": {ebm_keywords}))) 

## **ACM** 

(Abstract:{nlp_keywords} AND Abstract:{ebm_keywords}) OR (Title:{nlp_keywords} AND Title:{ebm_keywords}) 

## **ACL** 

(Abstract:{nlp_keywords} AND Abstract:{ebm_keywords}) OR (Title:{nlp_keywords} AND Title:{ebm_keywords}) 

21443 

