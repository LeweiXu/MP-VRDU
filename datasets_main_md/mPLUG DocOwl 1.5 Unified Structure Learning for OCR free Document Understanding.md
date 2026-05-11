# **mPLUG-DocOwl 1.5: Unified Structure Learning for OCR-free Document Understanding** 

**Anwen Hu[1] , Haiyang Xu[1] , Jiabo Ye** _[∗]_ **[1] , Ming Yan[1]** _[∗]_ **Liang Zhang[2] , Bo Zhang[1] , Chen Li[1] , Ji Zhang[1] , Qin Jin[2] , Fei Huang[1] , Jingren Zhou[1]** 1Alibaba Group 

2Renmin University of China `{huanwen.haw,shuofeng.xhy,ym119608}@alibaba-inc.com` 

Figure 1: Compared with similar-size generalists, our DocOwl 1.5 achieves state-of-the-art OCR-free performance on 10 Visual Document Understanding benchmarks. 

## **Abstract** 

Structure information is critical for understanding the semantics of text-rich images, such as documents, tables, and charts. Existing Multimodal Large Language Models (MLLMs) for Visual Document Understanding are equipped with text recognition ability but lack general structure understanding abilities for text-rich document images. In this work, we emphasize the importance of structure information in Visual Document Understanding and propose the Unified Structure Learning to boost the performance of MLLMs. Our Unified Structure Learning comprises structureaware parsing tasks and multi-grained text localization tasks across 5 domains: document, webpage, table, chart, and natural image. To better encode structure information, we design a simple and effective vision-to-text module H-Reducer, which can not only maintain the layout information but also reduce the length of visual features by merging horizontal adjacent patches through convolution, enabling the LLM to understand high-resolution images more efficiently. Furthermore, by constructing structure-aware text sequences and multi-grained pairs of texts and bounding boxes for publicly available text-rich images, we build a comprehensive 

> _∗_ Corresponding authors 

Preprint. Under review. 

**==> picture [389 x 297] intentionally omitted <==**

**----- Start of picture text -----**<br>
“tue Sek revew Atednce Rol B= Cate Tes (b) which edition has unlimited remote desktop services and<br>Dect:Proved REMSED- Recollof (CHANGED oncerSUPPLIER)atoning a ak oven afte at mayoat be virtulization rights? Datacenter<br>saeco=tte = Two-thirdsi special of Afghanimmi and<br>inte reps: 10808% a Iraqi special immigrant<br>Datnton Dopp =100208 pt ate VISE ‘pepeSc c " soa visa= recipientssoe  were<br>q ‘ Bevtcdsciene a= < | |dependents of applicants<br>soranbeesysen pal aR document on . on jou at Stance:Balance ControlA State-of-theory  in Obesecau? ©)Art Subjects during Quietmar ca no? a]a Number¥  of Afghan7. and Iraqi acitizensTt<br>= - eae «| |admittedto the US. under special<br>: suite = ArticleAccs overviewMenu +sepSonsjossnpacinunacononeaebeci2,19, Warp SmaIhg nr Rn ep 0 2 mo | limmigrantvisa programs, 2007-17<br>mews § (oa baat re tees,eeeee Principal applicants<br>——— Abstract<br>(a) What is the assigned  (c) What is the title of the paper in the website?  Afghan<br>response code? W24 Balance Control in Obese Subjects during  22%<br>Quiet Stance: A state-of-the Art.<br>Sm 12% °<br>Bs72i\ = Org! [B120 h ------+2-- 2222222 22oe  enn 19%_Se -<br>Of teensHowgesKids 12-17..use Social Media fosAEEEE! § 90 ‘ Dependents-<br>95use the %Internet. 81usesocial media0(0) £i 604 specialNote:U.S. governmentinPrincipal immigrantapplicants visIr a qs worked andeligibleAfghanistan. forfor the<br>es oo os Fiscal years begin Oct. 1. Dependents<br>of50radolescents(0) loginto% Arnang21%  kidsAAOunder 1% 20° ES IIL ILS IP LS LOL GLI ILE SABA ABA theyincludeunmarrieddependentsaccount4 principalchikirenare notincludedinfor lessapplicant’syoungerthanthan 1% spouseofthis total21. Other chart,and<br>aa ram) rewortngste AAAS ae recipients. Shares maynotadd to 100%<br>Ay U horhoe26% eeeAM le habia casa eae due to rounding.Source: U.S. State Department Bureau of<br>(d) What percentage of teenagers from the  (e) What is the forecast for the increase  (f) What is the percentage of<br>age group 12-17 didn't use the Internet? 5% in customs duty revenue in 2030? 100 Iraqi dependents citizen? 0.19.<br>**----- End of picture text -----**<br>


Figure 2: Illustrations of the importance of structure information in Visual Document Understanding on documents (a), tables (b), webpages (c), infographics (d), and charts (e-f). 

training set DocStruct4M to support structure learning. Finally, we construct a small but high-quality reasoning tuning dataset DocReason25K to trigger the detailed explanation ability in the document domain. Our model DocOwl 1.5 achieves state-of-the-art performance on 10 visual document understanding benchmarks, improving the SOTA performance of MLLMs with a 7B LLM by more than 10 points in 5/10 benchmarks. Our codes, models, and datasets are publicly available at `https://github.com/X-PLUG/mPLUG-DocOwl/tree/main/DocOwl1.5` . 

## **1 Introduction** 

Leveraging the strong language understanding and generation ability of Large Language Models (LLM) [5, 46, 48, 62], some recent works [57, 58, 27, 26, 64, 24] have developed Multimodal Large Language Models (MLLMs) for general vision-and-language understanding. By aligning a pre-trained visual encoder (e.g. the ViT/L-14 [12] from CLIP [36]) and the LLM with a Vision-toText (V2T) module, these models present promising performance on understanding general images. However, they still face great challenges with images with rich text information, such as documents, webpages, tables, and charts [28]. This is mainly because the visual encoder and V2T module are trained on general image-text pairs and not specifically optimized to represent the textual and structural information in text-rich images. 

Textual information in images manifests with a multitude of visual structures, spanning the simplicity of plain text to the systematic grid layouts of tables and incorporating a spectrum of graphical 

2 

representations such as pie, line, and bar charts. These elements may appear in isolation or be intricately interwoven within the framework of documents and webpages, reflecting a rich diversity of informational architecture across posters, invoices, infographics, scientific reports, academic and news websites, etc. As shown in Fig. 2, besides the basic textual content, structure information also plays a big role in Visual Document Understanding [53, 18, 45, 23]. With basic abilities to understand general images and comprehend structured texts through the LLM decoder, MLLM has the potential to achieve unified structure learning on text-rich images. For better Visual Document Understanding with MLLMs, some works [55, 56, 3, 13] attempt to design text-reading tasks to strengthen the text recognition ability, but either ignore the structure comprehension or only cover limited domains of text-rich images, such as just webpages [23] or documents [13]. In this work, we first propose to perform unified structure learning on text-rich images for MLLMs across 5 domains: document, webpage, table, chart, and natural image. 

For better structural understanding, we first design a simple and effective vision-to-text module, namely H-Reducer. Unlike the Resampler [1] or Q-former [24] which fuses visual features with learnable queries but affects spatial information, the H-Reducer accumulates neighborhood visual features through convolution to keep the relative positional relationships. Compared with V2T modules with only linear layers [27, 26], it produces much fewer visual features, which is more efficient for LLM to understand high-resolution document images. Considering texts in document images are most organized from left to right, H-Reducer merges visual features at the horizontal level. Our Unified Structure Learning comprises structure-aware parsing tasks and multi-grained text localization tasks. To learn the organization of text contents, the former mainly teaches the model to parse the texts in the image in a structure-aware style, such as using line feeds and spaces to represent the structure of documents or webpages, and using extended Markdown syntax to represent the structure of tables and charts. Multi-grained text localization tasks further enhance the ability to correlate visually situated texts and concrete positions in the image. To support unified structure learning, based on publicly available datasets, we carefully build a comprehensive training set DocStruct4M by constructing structure-aware sequences and multi-grained pairs of text and bounding boxes. The DocOwl 1.5 is trained in a two-stage framework, starting with the Unified Structure Learning and then followed by the Multi-task Tuning among downstream tasks. Finally, to trigger the reasoning ability of MLLM in Visual Document Understanding, we construct a high-quality instruction tuning dataset DocReason25K. By performing joint training on DocReason25K and downstream datasets, DocOwl 1.5-Chat well balance giving a simple answer or detailed explanations. 

Our contributions in this work are four-fold: 

- We first propose Unified Structure Learning on text-rich images for MLLMs and design both structure-aware parsing tasks and multi-grained text localization tasks across 5 domains. A comprehensive dataset DocStruct4M is carefully built to support Unified Structure Learning. 

- We design a simple and effective vision-to-text module for structure learning and perform extensive experiments to validate its effectiveness. 

- We construct a high-quality instruction tuning set to trigger the reasoning ability of MLLMs on Visual Document Understanding. 

- DocOwl 1.5 and DocOwl 1.5-Chat achieves state-of-the-art OCR-free performance on 10 Visual Document Understanding tasks, achieving improvement of more than 10 points on 5/10 tasks among similar-sized models. 

## **2 Related Work** 

**Visual Document Understanding** (VDU), also known as Visually-situated Language Understanding [23, 56], aims to comprehend images with rich text information. Such images range from documents [30, 31, 42, 41, 60], tables [34, 8, 63], charts [29, 19, 32, 21, 44, 17], natural images [39, 40, 16] to webpage screenshots [43, 9], where diverse composition of text and visual objects contains a wealth of information. To evaluate the multimodal document understanding performance, the task formats include low-level recognition, e.g. information extraction [42, 41], and high-level semantic understanding, such as visual question answering [30, 31, 34, 29, 43, 40], image captioning [39, 21, 44], and natural language inference [8]. According to whether relying on an off-the-shelf OCR system to recognize texts in the image, models for Visual Document Understanding can be categorized into OCR-dependent models [45, 53, 18, 54] and OCR-free ones [22, 23]. To leverage recognized texts 

3 

from an OCR system, OCR-dependent models are always trained to align textual and visual inputs. For example, UDOP [45] is pre-trained to recover masked text and layout information given image and retained text as inputs. As for OCR-free methods, training with tasks about text recognition is indispensable. Dount [22] design the text reading task to output continuous text sequences that ignore structure information. To leverage structure information, Pix2Struct [23] designs a Screenshot Parsing Task to generate the HTML DOM tree for webpage screenshots but is hard to apply to other types of images. In this work, we first propose Unified Structure Learning for all image types and carefully build a comprehensive dataset to support layout learning. 

**Multimodal Large Language Models** (MLLM) have shown strong vision understanding and openended conversation abilities [57, 58, 64, 10, 3, 15, 59] for natural images. They follow the architecture paradigm of connecting a vision encoder,e.g. ViT [12, 36], with a Large Language Model(LLM) [46, 48, 2] by a vision-to-text module, such as simple linear layers [27, 26] or a Q-Former [24]/Resampler [1]/Abstractor [57, 58] with learnable queries. To enable MLLMs to comprehend images with rich texts, there are major two challenges: how to encode high-resolution images and how to understand visually-situated texts. To tackle high-resolution images, most works choose to further train [3, 13] or extraly add a high-resolution vision encoder [15]. UReader [56] first proposes to keep the low-resolution vision encoder and use a shape-adaptive cropping module to crop raw images into multiple sub-images with low resolution. To enhance the visually-situated text understanding, some work design tasks of reading texts from top-left to bottom-right without taking into account the importance of structure [56, 3]. CogAgent [15] and DocPedia [13] further try strengthening the layout understanding for documents, webpages, and natural images with text grounding tasks. However, the comprehension of the overall structure is ignored, and tables and charts are not covered. In this work, we follow UReader to process high-resolution images. To strengthen structure understanding, we design structure-aware praising and multi-grained text localization tasks for all types of images, covering documents, tables, charts, webpages, and natural images. We propose a vision-to-text architecture to better maintain spatial information of visual features by convolution. Finally, to support unified structure learning, we build a comprehensive training dataset DocStruct4M and greatly improve the visual document understanding performance. 

## **3 DocOwl 1.5** 

DocOwl 1.5 follows the typical architecture of Multimodal Large Language Models, which consists of a visual encoder, a vision-to-text module, and a large language model as the decoder. To better keep the textual and layout information in text-rich images of high resolution, we design an H-Reducer as the vision-to-text module to ensemble horizontal visual features. As shown in Fig. 3(a), to enhance the text recognition and structure understanding abilities, we first perform Unified Structure Learning with structure-aware parsing and multi-grained text localization tasks for all types of images. Then, the model is jointly tuned on multiple downstream tasks of Visual Document understanding. 

## **3.1 Model Architecture** 

**High-resolution Image Encoding.** As proved by previous works [22, 23, 56], the ability to encode high-resolution images is critical to ensuring that the decoder can use rich text information from document images. As shown in Fig. 3(b), following UReader [56] , we utilize a parameter-free Shape-adaptive Cropping Module to crop a shape-variable high-resolution image _I_ into multiple fixed-size sub-images ( _I_ 1 _, I_ 2 _, ..., IC_ ), where _C_ is the number of crops. To keep the overall layout information, the raw image is also resized to a low-resolution one as the global image _I_ 0. Then, each image _Ii_ in ( _I_ 0 _, I_ 1 _, ..., IC_ ) is independently encoded to a sequence of visual features _Vi_ = ( _vi_[1] _[, v] i_[2] _[, ..., v] i[L]_[)] _[,]_[ 0] _[≤][i][≤][C]_[by][a][transformer-based][Visual][Encoder,][where] _[v] i[j][,]_[ 1] _[≤][j][≤][L]_[is][a] _D_ -dimension vector, _L_ is the length of visual features for each image. 

**Spatial-aware Vision-to-Text Module: H-Reducer.** There are two kinds of popular vision-to-text modules for Multimodal Large Language Models: a MLP [27, 26, 64] or a cross-attention module with learnable queries [57, 3, 1, 24]. Both two are not quite suitable for representing high-resolution text-rich images. The former projects complete visual features into the language embedding space. It maintains all spatial information in the document image but keeps the sequence length of raw visual features, which is too long when processing high-resolution images. For example, encoding a 1,344x1,344 image with the ViT/L-14 results in 9,216 visual tokens. The cross-attention module 

4 

**==> picture [383 x 297] intentionally omitted <==**

**----- Start of picture text -----**<br>
Document Parsing<br>VQA<br>oe FS iG jessccccesceny<br>Table Parsing LLM LLM<br>Information Extraction<br>Chart Parsing H-Reducer H-Reducer<br>Natural Image Parsing ViT Image Captioning ViT<br>Multi-grained Text Localization i = - Natural Language Inference a —<br>Stage 1: Unified Structure Learning Stage 2: Multi-task Tuning<br>(a)<br>Twitter. According to the image, there are 560 million and 70 million<br>active users for Twitter and Pinterest. Thus, Twitter has more active users.<br>Large Language Model MAM<br>————<br>𝑇! 𝑇" 𝑇# 𝑇$ 𝑇% 𝑇& 𝑇' 𝑋 H-Reducer<br><row1-col1> <row1-col2> <row1-col3> <row2-col1> <row2-col2> <row2-col3><br>Py) iy ty ty ty ty ty 4 Convolution<br>#𝑉! #𝑉" #𝑉# #𝑉$ #𝑉% #𝑉& #𝑉' (1x4)<br>H-Reducer<br>|<br>𝑉! 𝑉" 𝑉# 𝑉$ 𝑉% 𝑉& 𝑉'<br>C8a Visual Encoder    RRR R an ool le<br>7 𝐼! 88 𝐼" 𝐼# 86a 𝐼$ 𝐼% 𝐼& _e 𝐼' ||<br>Shape-Adaptive  Cropping Module<br>cmap ECEERREA FA<br>𝑉( $𝑉(<br>Who has more active users, Pinterest or Twitter? #𝑉(<br>(b)<br><global-img><br>FC<br>flatten<br>**----- End of picture text -----**<br>


Figure 3: The two-stage training framework (a) and overall architecture (b) of DocOwl 1.5. The global image and cropped images are processed independently by the Visual Encoder and H-Reducer. `<rowx-coly>` is the special textual token to indicate that the position of the cropped image in the original image is the _x[th]_ row and _y[th]_ column. 

could greatly reduce the length of the visual sequence to the number of learnable queries, but may lose spatial information during semantic fusion. 

In this work, we design a more appropriate vision-to-text module for Visual Document Understanding, namely H-Reducer, which not only reduces visual sequence length but also keeps the spatial information. As shown in Fig. 3(b), the H-Reducer is comprised of a convolution layer to reduce sequence length and a fully-connected layer to project visual features to language embedding space. Since most textual information in document images is arranged from left to right, the horizontal text information is usually semantically coherent. Thus, the kernel size and stride size in the convolution layer are set as 1x4 to ensemble horizontal 4 visual features. The output channel is set equal to the input channel _D_ . The convolution calculation is as follows: 

**==> picture [296 x 47] intentionally omitted <==**

where _f_ represents the dot product with kernel weights on multiple channels. After the convolution layer, the visual features of image _Ii_ are converted to the _V i_ , the feature length of which is _L/_ 4. 

Then, with a fully connected layer to align visual features to the language embedding space, the _V i_ are transferred to _V_[ˆ] _i_ = (ˆ _vi_[1] _[,]_[ ˆ] _[v] i_[2] _[, ...,]_[ ˆ] _[v] i[L/]_[4] ). 

**Multimodal Modeling with LLM.** As the decoder of MLLM, large language models should understand both the visual features of images and the textual features of language instructions. Following 

5 

**==> picture [374 x 240] intentionally omitted <==**

**----- Start of picture text -----**<br>
Document Parsing Natural Image Parsing<br>emotional players celebrate at the  <doc> Universities and<br>=a final whistle of their game against  departments \n<br>football team <ocr> QATAR  Students are members of the \n<br>2 ae AIRWAYS QATAR 15 CAP  University, a department and a \n<br></ocr> College.\n<br> •Course content \n<br>Read text in the image. Describe the content and text   •Lectures,…</doc><br>within the image.<br>Table Parsing <md> | | <COLSPAN=2> Factors levels in actual values | | | |<br>\n<br>| Formula | Pluronic to drug ratio | P123 percentage (%) |<br>EE% + SD (%)a | PS + SD (nm)a | PDI + SDa | \n<br>LLM MAM  | --- | --- | --- | --- | --- | --- | \n<br>| M1 | 10 | 10 | 7.27 ± 0.47 | 58.73 ± 1.56 | 0.23 ± 0.01 | \n<br>Parse the table in Markdown style.Chart Parsing H-Reducer …| M11 | 30 | 50 | 99.43 </md> ± 0.73 | 23.74 ± 0.95 | 0.13 ± 0.03 |<br>Viusal Encoder<br>Convert the<br>chart into Markdown  DocOwl 1.5 <md> | --- | --- | --- | --- | --- | | Country | 1960 | 1961 | 1962 | 1963 | \n \n<br>format. | Ecuador | 45 | 44.1 | 43.4 | 43.4 | \n<br>…<br>Text Recognition  | Iran | 45.2 | 44.3 | 43.7 | 43.8 | </md><br>Text Grounding<br><ocr> September 15, 1972 \n<br>Identify the text within the  Predict the bounding box of the text <ocr> 17. Loans Given 75.64 55.24 \n DATE BIOGRAPHICAL DATA NAME Mr. Milovan Bosnjak \n \n<br>bounding box  …<br><bbox>212, 52,896, 418</bbox> 18. Receipt towards Loan Repayment 64.11 4.64 0.13 0.11 \n PROFESSIONAL EXPERIENCE (In decending chronological order; position </ocr><br>19. Advances Given 26.27<br>0.88 6.50 </ocr> <bbox>120, 538, 753, 584 </bbox><br>**----- End of picture text -----**<br>


Figure 4: The illustration of Unified Structure Learning of DocOwl 1.5. 

mPLUG-Owl2 [58], we apply the Modality-adaptive Module(MAM) in LLM to better distinguish visual and textual inputs. During self-attention, MAM utilizes two sets of linear projection layers to separately perform the key/value projection for visual features and textual features. To help the LLM correlate multiple cropped sub-images, UReader [56] designs learnable crop position embeddings to denote the row and column position in the raw image. In this work, we simply add special textual tokens `‘<row` _x_ `_col` _y_ `>’` before the visual features of each cropped image, where _x_ and _y_ refer to the row and column index respectively. For the global image, the textual indicator token is `‘<global_img>’` . This design eliminates the need to introduce additional parameters and is more friendly to the LLM decoder. Our experiments validate that it achieves comparable effects as the crop position embedding. Overall, the decoding of the LLM is as follows: 

**==> picture [288 x 13] intentionally omitted <==**

where [; ] means the concatenation operation, _C_ is the crop number of the image, _Tj,_ 0 _≤ j ≤ C_ is the textual embeddings of the special textual indicator for the global image or positions of cropped images, _V_[ˆ] _j_ is the visual features of a global or cropped image, _X_ is the textual embeddings of the instruction, _Y_ is the predicted answer. 

## **3.2 Unified Structure Learning** 

Most Multimodal Large Language Models [27, 58, 50] are trained with image-text pairs of natural images to align the visual encoder with the LLM, such as Conceptual Captions [7], LAION [37] and COYO [6]. Initializing from such models could inherit the shallow text recognition ability, but is far from understanding complex textual and structural information in various text-rich images. In this work, to empower the comprehensive document understanding abilities of MLLM, we design a Unified Structure Learning across 5 domains, including natural images, documents, tables, charts, and webpages. It involves both structure-aware parsing tasks and multi-grained text localization tasks, as shown in Fig. 4. 

**Document Parsing.** For representing the structure information, Pix2Struct [23] parses webpage screenshots with condensed HTML DOM trees, which are built based on the HTML source codes and are not available for other formats of documents or webpage screenshots, e.g. PDF. In documents or webpages, horizontal and vertical distances between texts form the main layout information. Therefore, to make the structure-aware parsing task applicable to most documents and webpage 

6 

screenshots, we choose to add extra line feeds( `‘` _\n_ `’` ) and spaces into the text sequence to denote different lines and horizontal distances. The greater the horizontal distance, the more space characters. 

We choose CCpdf [47], RVL-CDIP [14], VisualMRC [43] and datasets encapsulated in DUEBenchmark [4] (DocVQA [30], InfoVQA [31], DeepForm [42], KLC [41], WTQ [34], TabFact [8]) to support the Document Parsing task. CCpdf [47] is a multi-lingual PDF dataset built upon webpages from Common Cramwl[2] , covering diverse domains of documents, such as industry, academic, and medical. In this work, we mainly focus on English Document Understanding and drop PDFs detected as other languages. RVL-CDIP contains 16 categories of industry documents, such as ‘letter’, ‘email’, and ‘scientific reports’. We further remove some categories with flipping and blurring texts, such as ‘handwritten’ and ‘form’. DUE-Benchmark is a collection of available and reformulated datasets over various document domains and layouts featuring tables, graphs, lists, and infographics. VisualMRC is a webpage screenshot dataset across 35 websites. OCR annotations in VisualMRC are aligned with local regions, thus, we follow them to utilize crops of a screenshot as input for this parsing task. For CCpdf and DUE-Benchmark, a PDF-parsing tool pdfplumber[3] can be directly used to generate structure-aware text sequence with a PDF page as the input. For RVL-CDIP and VisualMRC, there are no PDF files, just annotations of bounding boxes of texts. As an alternative, akin to the LATIN-Prompt [51], we insert the line feeds and spaces by calculating and comparing the horizontal and vertical distances of bounding boxes. To avoid too many space characters resulting in sparse texts, we further limit the maximum number of consecutive spaces to 4. This strategy allows us to construct structure-aware text sequences in the same style as pdfplumber. 

**Table Parsing.** Different from documents or webpages, tables are structured in a more standardized way, where row and column correspondences represent key-value pairs. HTML and Markdown codes are mainly two kinds of text sequences used to represent a table. HTML codes can represent all kinds of tables, with or without cells spanning multiple rows and grids, but they contain too many paired labels (e.g. `‘<tr></tr>’` and `‘<td></td>’` ), causing text sequences to be too long. Markdown codes can represent a table with concise text sequence, but they cannot represent cells spanning multiple rows and columns. To represent all tables with concise text sequence, we follow the main grammar of Markdown to represent table structure with `‘|’` and line feeds( `‘` _\n_ `’` ). To represent cells spanning multiple rows and columns, we add special text tokens `‘<COLSPAN=x>’` and `‘<ROWSPAN=y>’` before the value, as shown in Fig. 4. 

We choose TURL [11] and PubTabNet [63] to do the structure-aware table parsing task, where tables are collected from Wikipedia pages and scientific articles, respectively. Without cells across rows and columns, tables in TURL can be directly represented with Markdown codes. Due to lacking table images in TURL, we transfer tables into HTML codes and render table images with variations in background color and font size. PubTabNet contains pairs of table images and HTML codes. We convert HTML codes into Markdown style and add `‘<ROWSPAN=x>’` or `‘<COLSPAN=y>’` before the value when attributes `‘rowspan=x’` or `‘colspan=y’` are set in the `‘<td>’` label. 

**Chart Parsing.** Unlike documents and tables, organizing texts in reading order cannot represent the structure of charts. Considering that the chart is a visualization form of the table, parsing charts to tables could best maintain the mathematical characteristics of the chart. This requires the model to understand the structure of the chart and the alignment of the x/y axis. Besides, to keep consistent with the Table Parsing task, we also use Markdown codes to represent the data tables of charts, as shown in Fig. 4. 

We adopt PlotQA [32], FigureQA [20], DVQA [19], and ChartQA [29] to support the structureaware chart parsing task. These datasets cover charts on both synthetic [20, 19] data and data from real-world sources [32, 29]. Chart types include vertical bar, horizontal bar, line, dot line, and pie chart. Source data of the chart is provided in the JSON [32, 20, 32] or CSV format [29], both can be conveniently converted to Markdown codes. However, some raw values are not suitable as standard answers for parsing because there are too many significant digits to be represented on the chart. Therefore, to reduce the difficulty of estimating values and make the model focus more on structural understanding, we keep 4 significant digits for all values. 

**Natural Image Parsing.** Quite different from text-dominant images mentioned above, the semantics of natural images is a combination of natural objects and scene texts. Thus, parsing natural images is 

> 2 `https://commoncrawl.org` 

> 3 `https://github.com/jsvine/pdfplumber` 

7 

Figure 5: Detailed statistics of DocStruct4M. 

necessary to organize scene texts and mention the main image content. Manually annotating captions to describe the relationship between objects and scene texts is labour- and financial-intensive. Like TAP [54], we concatenate the general caption with OCR texts to form the target parsing sequence. 

We utilize OCR-CC [54] to support the Natural Image Parsing task. OCR-CC is a subset of Conceptual Caption [38], which contains images with scene texts detected by the Microsoft Azure OCR system. 

**Multi-grained Text Localization.** As proved in previous works [52, 49, 35] on general image understanding, semantic comprehension and object grounding tasks can be well unified in a single model. For Visual Document Understanding, structure-aware parsing tasks mainly focus on organizing texts according to the overall structure, while neglecting the correspondence between specific texts and local positions. Correlating texts with the concrete position in images is another basic structure understanding ability for visual documents. To support text position learning, we design two symmetrical tasks, namely Multi-grained Text Grounding and Multi-grained Text Recognition. The former aims to predict the bounding box given the visually-situated texts, while the latter does the opposite. We set four granularities of texts for these two tasks: word, phrase, line, and block. The ‘word’ is the smallest granularity of the bounding box, referring to only 1 word. To ensure that the word is visible and the answer is unique, words that are too small (normalized area < 0.001) and words that appear multiple times in the same image are excluded from candidates. The ‘line’ consists of texts that are judged to be horizontally parallel by vertical distance, and the ‘phrase’ is comprised of multiple adjacent words within the same line. The ‘block’ is a combination of multiple successive lines, ranging from 2 to half of the total lines. The text sequences of word-level and phrase-level question answering are much shorter than the other two. Therefore, in order to learn localization more efficiently, each word-level or phrase-level sample consists of up to 5 question-answer pairs for the same image. As for the representation of bounding boxes, we transfer each continuous value in the normalized bounding box into a discrete position token, ranging from 0 to 999. 

The bounding box annotation is necessary for constructing samples for Multi-grained Text Localization tasks. Therefore, we take DocVQA, InfoVQA, WTQ, TabFact, DeepForm, KLC, ChartQA, VisualMRC, and TextVQA [40] for this task, across domains of the document, table, chart, webpage, and natural image. 

Overall, to support the unified structure learning for text-rich images, we build a DocStruct4M dataset by ensembling multiple training sets of publicly available datasets and constructing structure-aware text sequences or text-position pairs as the targets. The form of instructions for each task is very diverse for developing the general instruction-following ability of the model. Fig. 5 shows the detailed statistics of DocStruct4M. 

## **3.3 Multi-task Fine-tuning** 

Through Unified Structure Learning, models could well understand the structure of diverse document images but cannot follow users’ instructions to do different types of tasks, such as information 

8 

Table 1: The detailed statistics of DocReason25K. The ‘Avg Length’ refers to the average token length of the answer. 

||DocVQA<br>InfoVQA<br>WTQ<br>VisualMRC<br>ChartQA<br>TextVQA|ALL|
|---|---|---|
||||
|Image<br>Sample<br>Avg Length|1,491<br>1,614<br>850<br>1,927<br>1,252<br>1,612<br>5,119<br>5,421<br>5,994<br>5,263<br>1,827<br>2,253<br>79.2<br>95.4<br>77.7<br>103.4<br>106.9<br>88.0|8,746<br>25,877<br>89.9|



extraction or image captioning. So, we further perform multi-task fine-tuning to train a generalist of visual document understanding as UReader [56]. 

## **3.4 Training Paradigm** 

As shown in Fig. 3(a), DocOwl 1.5 is trained in a two-stage framework. Considering the LLM has strong comprehension abilities for structured text [51, 61], we argue that the main limitation of MLLM in visual document understanding is the representation ability of the Visual Encoder and Vision-to-Text module for visually-situated text and structure information. Thus, during the Unified Structure Learning, we freeze the LLM parameters and tune the Visual Encoder and H-Reducer. The MAM is also optimized to help the LLM better distinguish visual features and texts parsed from the image. During the stage of Multi-task Fine-tuning, the model mainly learns how to follow the user’s instructions to give answers based on visually-situated text and structure understanding capabilities acquired in the first stage. Therefore, the Visual Encoder is frozen and other modules are tuned. 

## **4 DocOwl 1.5-Chat** 

Existing benchmarks mainly evaluate the document understanding ability by answering the question with simple phrases and neglect detailed explanations. In this work, to better leverage the strong language reasoning ability of Large Language Models on Visual Document Understanding, we build a small instruction-tuning set with detailed explanations on text-rich image understanding, namely DocReason25K. Based on raw questions from DocVQA [30], InfoVQA [31], WTQ [34], VisualMRC [43], ChartQA [29] and TextVQA [40], we collect detailed explanations with ChatGPT[4] . Text contents are dominant information on documents, tables or webpage screenshots. Therefore, for DocVQA, InfoVQA, WTQ, and VisualMRC, we take the structure-aware text sequence of the image as the input to `gpt-3.5-turbo-0301` and prompt it to answer the question with simple answers and detailed explanations. As for ChartQA and TextVQA, we take the image as the input and utilize the `gpt-4-vision-preview` to answer the question with detailed explanations. In order to filter out samples where ChartGPT answers incorrectly, we further prompt `gpt-3.5-turbo-0301` to judge whether the answer given by ChartGPT is consistent with the concise human-annotated ground-truth answer. Compared with raw questions in benchmark datasets, questions in DocReason25K are added with a prompt `‘Answer the question with detailed explanation’` . Detailed statistics of DocReason25K are presented in Table 1. DocOwl 1.5-Chat is trained by combining downstream datasets with DocReason25K and performing multi-task tuning after Unified Structure Learning. 

## **5 Experiments** 

## **5.1 Implementation Details** 

DocOwl 1.5 is initialized from mPLUG-Owl2 [58], which utilizes the ViT/L-14 [12] as the Visual Encoder and a 7B Large Langauge Model with the Modality Adaptive Module as the language decoder. According to the aspect ratio and resolution, each image is cropped into up to 9 sub-images with a fixed resolution of 448x448. Each sub-image is encoded to 1,024 features by the ViT/L-14 and then reduced to 256 features by the H-Reducer. The model is trained with 12,000 iterations on DocStruct4M, with the learning rate and batch size set as 1e-4 and 1,024. It costs about 128 A100 days. During the Multi-task finetuning, the model is trained for 6,500 iterations with the batch size set as 256 and the learning rate set as 2e-5. This further costs about 24 A100 days. 

> 4 `https://openai.com/chatgpt` 

9 

Table 2: Different settings of OCR-free Visual Document Understanding models. ‘Open’ refers to whether all OCR learning data is open-source. 

|**Model**|**Init**<br>**Resolution**|**OCR Learning**<br>**Text**<br>**Bbox**<br>**Size**<br>**Domain**<br>**Open**|
|---|---|---|
|Donut [22]<br>Pix2Struct [23]<br>QwenVL [3]<br>Monkey [25]<br>UReader [56]<br>DocPedia [13]<br>CogAgent [15]|-<br>2560x1920<br>-<br>219(shape variable)<br>-<br>448x448<br>QwenVL [3]<br>896x896<br>Owl [57]<br>224x224(x20 crops)<br>-<br>2560×2560<br>CogVLM [50]<br>1120×1120|✓<br>_×_<br>13M<br>Synthetic, Doc<br>✓<br>✓<br>_×_<br>80M<br>Web<br>_×_<br>✓<br>_×_<br>24.8M<br>Synthetic, Doc, Web<br>_×_<br>_×_<br>_×_<br>-<br>-<br>-<br>✓<br>_×_<br>0.1M<br>Doc, Table, Chart, Web, Natural<br>✓<br>✓<br>✓<br>0.9M<br>Doc<br>_×_<br>✓<br>✓<br>107M<br>Synthetic, Nature, Doc, Web<br>_×_|
|DocOwl 1.5|Owl2 [58]<br>448x448(x9 crops)|✓<br>✓<br>4M<br>Doc, Table, Chart, Web, Natural<br>✓|



Table 3: Comparison with OCR-free methods on various types of text-rich image understanding tasks. The superscript ‘ _∗_ ’ refers to models separately fine-tuned on each downstream task, rather than generalists. The _underline_ means the best performance among models with <10B parameters. 

|generalists. The _underline_<br>means the bestperformance amongmodels with <10Bparameters.|generalists. The _underline_<br>means the bestperformance amongmodels with <10Bparameters.|generalists. The _underline_<br>means the bestperformance amongmodels with <10Bparameters.|generalists. The _underline_<br>means the bestperformance amongmodels with <10Bparameters.|generalists. The _underline_<br>means the bestperformance amongmodels with <10Bparameters.|generalists. The _underline_<br>means the bestperformance amongmodels with <10Bparameters.|
|---|---|---|---|---|---|
|**Model**<br>**Size**<br>**Doc**<br>**Info**<br>**Deep**<br>**KLC**<br>**WTQ**<br>**Tab**<br>**Chart**<br>**Text**<br>**Text**<br>**Visual**<br>**VQA**<br>**VQA**<br>**Form**<br>**Fact**<br>**QA**<br>**VQA**<br>**Caps**<br>**MRC**||||||
|Dessurt_∗_<br><1B<br>Donut_∗_<br><1B<br>Pix2Struct_∗_<br>_base_<br><1B<br>Pix2Struct_∗_<br>_large_<br>1.3B|63.2<br>-<br>-<br>-<br>67.5<br>11.6<br>61.6<br>30.0<br>72.1<br>38.2<br>-<br>-<br>76.6<br>40.0<br>-<br>-|-<br>-<br>18.8<br>54.6<br>-<br>-<br>-<br>-|-<br>41.8<br>56.0<br>58.6|-<br>-<br>43.5<br>74.4<br>-<br>88.0<br>-<br>95.5|-<br>93.91<br>-<br>-|
|DocPeida<br>7.0B<br>DocOwl<br>7.1B<br>QwenVL<br>9.6B<br>UReader<br>7.1B<br>Monkey<br>9.8B<br>CogAgent<br>17.3B|47.1<br>15.2<br>-<br>-<br>62.2<br>38.2<br>42.6<br>30.3<br>65.1<br>35.4<br>-<br>-<br>65.4<br>42.2<br>49.5<br>32.8<br>66.5<br>36.1<br>40.6<br>32.8<br>81.6<br>44.5<br>-<br>-|-<br>-<br>26.9<br>60.2<br>-<br>-<br>29.4<br>67.6<br>25.3<br>-<br>-<br>-|46.9<br>57.4<br>65.7<br>59.3<br>-<br>68.4|60.2<br>-<br>52.6<br>111.9<br>63.8<br>-<br>57.6<br>118.4<br>67.6<br>93.2<br>**76.1**<br>-|-<br>188.8<br>-<br>221.7<br>-<br>-|
|DocOwl-1.5<br>8.1B<br>DocOwl-1.5-Chat<br>8.1B|81.6<br>50.4<br>68.8<br>37.9<br>**82.2**<br>**50.7**<br>**68.8**<br>**38.7**|39.8<br>**80.4**<br>**40.6**<br>80.2|**70.5**<br>70.2|68.8<br>**132.0**<br>68.6<br>131.6|239.5<br>**246.4**|



## **5.2 Main Results** 

We evaluate the Visual Document Understanding performance on 10 text-rich image benchmarks, covering documents (DocVQA [30], InfoVQA [31], DeepForm [42], KLC [41]), tables (WTQ [34], TabFact [8]), charts (ChartQA [29]), natural images (TextVQA [40], TextCaps [39]), and webpage screenshots (VisualMRC [43]). We compare DocOwl 1.5 with state-of-the-art OCR-free models, including both Multimodal Large Language Models adapted for recognizing texts and much smaller models trained only for document understanding. The detailed comparison of model settings can be found in Table 2. As shown in Table 3, previous MLLMs with more than 7B parameters underperform domain-specific models with less than 1B parameters, showing that the document understanding is still a shortcoming for existing MLLMs. Our DocOwl 1.5 outperforms both domain-specific models and MLLMs with similar sizes on all 10 benchmarks. This validates that DocOwl 1.5 is much stronger on visual document understanding across 5 domains, covering visual question answering, information retrieval, natural language inference, and image captioning tasks. Besides, with much fewer unnatural data (3M vs 9M) and parameters (8.1B vs 17.3B), DocOwl 1.5 outperforms CogAgent [15] on InfoVQA and ChartQA, and achieves comparable performance on DocVQA. This suggests that our unified structure learning with DocStruct4M is more efficient in learning printed text recognition and how to analyze documents. However, our model still underperforms CogAgent on TextVQA, which requires the ability of scene text recognition and general knowledge about natural objects. The primary reason is that scene texts are more diverse in shapes than printed texts and CogAgent is trained on 98M samples of scene text recognition from LAION-2B [37] and COYO-700M [6], much more than the natural images (1M) in DocStruct4M. In this work, we mainly focus on improving the unified structure comprehension of visual documents and leave further scaling up data on natural scenes as future work. Finally, DocOwl 1.5-Chat can also be evaluated on these concise-answer benchmarks by removing the prompt of detailed explanation. It achieves comparable or slightly better performance than DocOwl 1.5, showing that a small amount of detailed explanatory data may better help the model understand the semantics of text-rich images. 

10 

Table 4: Ablation study of model setting. ‘Crop’ refers to the maximum number of cropped images. ‘CropPos’ means using learnable embeddings (‘Emb’) or textual tokens (‘Text’) to represent the position of cropped images. ‘Parsing’ and ‘MTL’ refer to structure-aware parsing tasks and the Multigrained Text Location task, respectively. ‘Owl(224)’ and ‘Owl2(448)’ refer to mPLUG-Owl [57] with 224 resolution and mPLUG-Owl2 [58] with 448 resolution, respectively. 

||**Model Architecture**<br>**Init**<br>**V2T**<br>**Crop**<br>**CropPos**|**Structure**<br>**Learning**|**Multi-task Tuning**<br>**ViT**<br>**LLM**|**DocVQA**<br>**TabFact**<br>**ChartQA**|
|---|---|---|---|---|
|r1<br>r2<br>r3<br>r4|Owl(224)<br>Abstractor<br>20<br>Emb<br>Owl2(448)<br>Abstractor<br>20<br>Emb|_×_<br>_×_|_×_<br>_×_<br>_×_<br>_×_|65.4<br>67.6<br>59.3<br>66.3<br>69.8<br>60.6|
||Owl2(448)<br>Abstractor<br>20<br>Emb<br><br><br>|_×_|✓<br>_×_<br>|71.4<br>70.3<br>64.2<br><br>|
||Owl2(448)<br>Abstractor<br>9<br>Emb|_×_|✓<br>_×_|68.0<br>70.0<br>64.2|
|r5<br>r6<br>r7<br>r8<br>r9|Owl2(448)<br>H-Reducer(1x4)<br>9<br>Emb<br><br><br>|_×_|✓<br>_×_<br>|72.8<br>72.9<br>65.0<br>|
||Owl2(448)<br>H-Reducer(2x2)<br>9<br>Emb<br>Owl2(448)<br>H-Reducer(2x4)<br>9<br>Emb<br>Owl2(448)<br>H-Reducer(1x8)<br>9<br>Emb<br>Owl2(448)<br>H-Reducer(2x8)<br>9<br>Emb|_×_<br>_×_<br>_×_<br>_×_|✓<br>_×_<br>✓<br>_×_<br>✓<br>_×_<br>✓<br>_×_|71.8<br>72.1<br>65.2<br>71.4<br>71.1<br>66.0<br>69.9<br>71.2<br>64.4<br>69.2<br>70.2<br>65.6|
|r10<br>r11<br>r12<br>r13|Owl2(448)<br>H-Reducer(1x4)<br>9<br>Emb<br>Owl2(448)<br>H-Reducer(1x4)<br>9<br>Emb<br>Owl2(448)<br>H-Reducer(1x4)<br>9<br>Text|Parsing<br>Parsing<br>Parsing|_×_<br>_×_<br>_×_<br>✓<br>_×_<br>✓|77.7<br>76.5<br>67.5<br>78.9<br>78.1<br>68.1<br>79.8<br>77.7<br>69.1|
||Owl2(448)<br>H-Reducer(1x4)<br>9<br>Text|Parsing+MTL|_×_<br>✓|81.6<br>80.4<br>70.5|



## **5.3 Ablation Study** 

As shown in Table 4, we further perform a comprehensive ablation study to validate the effectiveness of our H-Reducer and Unified Structure Learning. 

Firstly, initializing from a stronger general MLLMs brings better performance on text-rich images (r2 vs r1), showing general vision-and-language knowledge benefits visual document understanding. Tuning the visual encoder during multi-task fine-tuning significantly improves the document understanding performance (r3 vs r2). This suggests that the visual representation of document images may be the main shortcoming of MLLMs and inspires us to design Unified Structure Learning to enhance the representation ability of the visual encoder for visually situated texts and structure. 

**Effectiveness of H-Reducer.** When using the Shape-adaptive Cropping Module, the image resolution supported by the MLLM is the product of the cropping number and basic resolution of each crop. With the Abstractor as the vision-to-text module, reducing the cropping number causes an obvious performance decrease (r4 vs r3) on documents. However, with a smaller cropping number, the H-Reducer achieves better performance than the Abstractor (r5 vs r3), showing that 448[2] _×_ 9 _≈_ 2[21] is an acceptable resolution for existing benchmarks and the H-Reducer is stronger on maintaining rich text information during vision-and-language feature alignment. Besides, we further compare different settings of the merging shape in the convolution layer. With the same number of merged tokens, the model with the 1x4 merging shape achieves better performance than the one with the 2x2 merging shape on document and table datasets but slightly worse performance on chart understanding (r6 vs r5). This is consistent with the common sense that documents and tables mainly organize texts in the left-to-right order while the semantic structures of charts are much more flexible. A square merging shape is more suited to encode visual features in the form of bars, lines, or pies while the 1x4 merging shape is more appropriate for general document understanding. As shown in r7-r9, further extending the 1x4 merging shape horizontally and vertically decreases the length of visual features but at the cost of performance degradation. Considering the overall performance on all text-rich images, we finally choose the 1x4 as the merging shape in H-Reducer. 

**Effectiveness of Unified Structure Learning.** After determining the vision-to-text module, we perform two-stage training with Unified Structure Learning. With only the structure-aware parsing tasks, there is significant improvement across different domains (r10 vs r5). This validates that finetuning the visual encoder and H-Reducer with structure-aware parsing tasks greatly helps MLLMs understand text-rich images. Further tuning the parameters of LLM brings slight improvement (r11 vs r10), suggesting that general language knowledge is not the main obstacle to visual document understanding. By replacing the learnable crop position embeddings with special textual tokens, the model achieves better performance (r12 vs r11), showing that the LLM can well understand the relative positions of multiple cropped images with just simple textual indicators. Finally, by introducing Multigrained Text Localization tasks, DocOwl 1.5 achieves the best performance, validating that correlating visually situated texts with concrete positions helps comprehend documents more accurately. 

11 

Table 5: The comparison of two-stage training and one-stage joint training with increasing samples from DocStruct4M. For a fair comparison, the LLM is frozen for both two-stage and one-stage training. The bath size of one-stage training is always set as 256, the same as the Multi-task Tuning in two-stage training. 

||**One-Stage**|**Two-Stage**|
|---|---|---|
||||
|DocStruct4M samples<br>Benchmark samples<br>Epoch/iteration<br>Cost (A100 days)|0.0M<br>0.5M<br>1.0M<br>2.0M<br>4.0M<br>0.6M<br>0.6M<br>0.6M<br>0.6M<br>0.6M<br>7/18k<br>6/25k<br>6/37k<br>4/40k<br>3/54k|4.0M<br>0.6M<br>3/12k + 3/6.5k|
||60.0<br>83.3<br>123.3<br>133.3<br>180.0|144.8|
|DocVQA|72.8<br>75.5<br>78.6<br>78.8<br>78.9|79.9|



Table 6: The detailed statistic of DocLocal4K. 

|**Task**|**Text Granularity**<br>**Word**<br>**Phrase**<br>**Line**<br>**Block**|**Image Domain**<br>**Doc**<br>**Table**<br>**Chart**<br>**Web**<br>**Natural**|
|---|---|---|
||||
|Text Recognition<br>Text Grounding|622<br>499<br>522<br>482<br>595<br>542<br>503<br>485|1,004<br>491<br>229<br>267<br>134<br>1,011<br>524<br>240<br>242<br>108|



**Effectiveness of the Two-stage Training.** As shown in Table 5, instead of two-stage training, we also try one-stage joint training of the structure learning and downstream tasks and gradually increase the samples from DocStruct4M. The epoch is gradually reduced because we didn’t observe performance improvements with more iterations. For joint training, the model improves significantly on DocVQA as the samples of Unified Structure Learning increase when it is below 1M. However, as the Unified Structure Learning samples are further increased, the improvement of the model becomes subtle and its performance is not as good as the one using two-stage training. This shows that the two-stage training could better enhance basic text recognition and structure parsing abilities and is more beneficial and efficient for downstream document understanding. 

## **5.4 Text Localization Evaluation** 

Besides proving the effectiveness of H-Reducer through downstream text-rich image understanding performance in Table 4, we further directly compare the text localization performance after the Unified Structure Learning to validate its superiority in preserving spatial features. We build a text localization evaluation set DocLocal4K with 4,250 samples balanced on 4 granularities and covering both text recognition and text grounding tasks. The detailed statistics of DocLocal4K are shown in Table 6. Considering that document images are much more diverse and complex than other images, there are more samples in this domain than others. The IOU@0.5 is used to evaluate the text grounding performance. As for text recognition, the word, phrase, line, and block granularity is evaluated with BLEU1, BLEU2, BLEU3, and BLEU4 [33], respectively. As shown in Table 7, when trained with the same iterations, the H-Reducer achieves much better performance on both Text Recognition and Text Grounding tasks, showing that H-Reducer with the 1x4 merging shape helps the LLM better understand concrete positions in images. 

## **5.5 Qualitative Results** 

**Question Answering with Simple Phrases.** Besides quantitative results, we further present some qualitative results of visual document understanding on different domains of images. As shown in 

Table 7: Multi-grained text localization performance of models with different vision-to-text modules. 

|**Module**<br>**Iter**|**Text Grounding**<br>**Word**<br>**Phrase**<br>**Line**<br>**Block**<br>**ALL**|**Text Recognition**<br>**Word**<br>**Phrase**<br>**Line**<br>**Block**<br>**ALL**|
|---|---|---|
|Abstractor<br>1,800<br>H-Reducer(2x2)<br>1,800<br>H-Reducer(1x4)<br>1,800|10.92<br>25.83<br>34.59<br>87.01<br>37.69<br>14.19<br>34.87<br>43.94<br>89.07<br>43.94<br>**17.82**<br>**39.30**<br>**53.28**<br>**90.52**<br>**48.28**|30.68<br>28.58<br>40.12<br>32.73<br>33.03<br>37.20<br>38.33<br>48.68<br>41.99<br>41.55<br>**39.60**<br>**41.84**<br>**55.37**<br>**49.84**<br>**46.66**|
|H-Reducer(1x4)<br>12,000|70.42<br>76.38<br>85.88<br>91.34<br>80.38|70.10<br>67.86<br>73.88<br>70.70<br>70.63|



12 

**==> picture [347 x 137] intentionally omitted <==**

**----- Start of picture text -----**<br>
Human:  What is the<br>LRT4 Human:  Dept.No?   osZlSSRIeeDenSRmoe What is the  NowiHY !1i|ii|||1 {||Clinical[Marketing/Ladecketing—ManufacturingQuatity=R&D Human:  - Jeff BrianResearch Ondrla- Stove- Motter Steve- Who did clinical research? Phil Hans/  JenksDick/Tarr-AreHaas/ Sam/ TimSteveO’Noa!7 /PruittFrybackArtPeoples/ Don O’NealmeeeMcNulty“~~.ee HHlI|ry1}\\s1! [eeneaXe]rn 33ieegmiter§ 10060 LITttHT ty forecast for the increase in customs duty revenue in 2030? UReader: DocOwl 1.5:  90.5100  (×()√)<br>UReader:  76/77-142 (×) 1 UReader:  Steve Haas / Art O‘Neal (×) 1 \ q<br>（a） DocOwl 1.5:  218-12 (√) f!'1\ DocOwl 1.5:  Steve peoples（b）  (√) !1 ‘| 2 4 Sof BoB （  of c）<br>Datacenter<br>aonFoundationomcota3268 SSSTamEssentialsaecs sksrecumpeecns[_stenaerda ce “STvaeemepanreoereoump [oatecenter] ate sca [[-] Human:  remote desktop services and virtulization rights?which edition has unlimited<br>SORRAS connectionsand 101S cnmactone 250 RRAS connection. 0 AS connections, nd 21S<br>=a a ‘Ether n VM oF 1 physica server, but not bathat once f w UReader:  Enterprise edition (×)<br>DocOwl 1.5:  Standard (×)<br>Ground Truth : Datacenter<br>[Services] [limits] （d）<br>**----- End of picture text -----**<br>


Figure 6: Qualitative results of DocOwl 1.5 and UReader on different domains of images. 

Fig. 6(a) and (b), both models answer the question with texts in the image. DocOwl 1.5 can better understand the structure of two documents and give correct answers. In Fig. 6(c), due to the learning of parsing chart with Markdown codes, DocOwl 1.5 can better understand the chart and successfully correlate the x/y axis. Fig. 6(d) shows that although inconsistent with the ground truth, DocOwl 1.5 gives another correct answer with the help of stronger structure understanding on tables. 

**Question Answering with Detailed Explanations.** Fig. 7 and Fig. 8 present qualitative results of detailed explanations. Through a small amount of reasoning training, DocOwl 1.5-Chat can well inherit the reasoning ability of LLM and provide detailed explanations about the answer. However, as presented in Fig. 8(c), like most general Multimoal large Language Models [57, 58, 3], DocOwl 1.5-Chat may also suffer from the hallucination problem in Visual Document Understanding. In this work, we mainly focus on enhancing the unified structure understanding ability of MLLMs and leave how to resolve the hallucination problem in OCR-free document understanding as future work. 

**Structure-aware Parsing.** As shown in Fig. 9, DocOwl 1.5 could parse a document image by using line feeds and spaces to represent the structure of text contents. Besides parsing the whole document, as shown in Fig. 10, it could also parse texts from the middle of the image according to human instruction. Fig. 11 presents qualitative results of structure-aware table parsing through extended Markdown syntax on tables with cells spanning multiple columns or not. Furthermore, Fig. 12 shows some cases of parsing different types of charts into Markdown codes, including vertical bar, horizontal bar, pie, and line charts. When all data points are presented in the chart, DocOwl 1.5 can accurately align statistic objects with corresponding numbers. It makes some mistakes in Fig. 12(d) because estimating the concrete numbers is quite challenging when no data points are provided. Finally, as shown in Fig. 13, DocOwl 1.5 can both describe the content of natural images and read scene texts. 

**Multi-grained Text Localization.** Fig. 14 and Fig. 15 show qualitative results of text grounding and text recognition at granularities of word, phrase, line and block. The image domains range from documents, webpages, charts, and tables to natural images. 

## **6 Conclusion** 

To enhance the Visual Document Understanding performance of Multimodal Large Language Models, we first propose Unified Structure Learning across 5 domains of text-rich images, including both structure-aware parsing tasks and multi-grained text localization tasks. To better maintain structure and spatial information during vision-and-language feature alignment, we design a simple and effective vision-to-text module, named H-Reducer. It mainly utilizes a convolution layer to aggregate horizontally neighboring visual features. To support the Unified Structure Learning, we build a training dataset DocStruct4M by collecting publicly available images and carefully constructing structure-aware text sequences and multi-grained pairs of texts and bounding boxes. With Unified Structure Learning, our model DocOwl 1.5 achieves state-of-the-art OCR-free performance on 10 visual document understanding benchmarks. 

13 

**==> picture [370 x 502] intentionally omitted <==**

**----- Start of picture text -----**<br>
What is the SIS # of L.<br> errs") MacWilliams? Answer the<br>Howeecayatvsees)aon ua) question with detailed<br>[nn lacWilliams<br> ca as nn es [sis# Direct Account Name Information Required explanation.<br> mene, 158120 Discount Drug on Beau .<br>rtive 8/28/98 (Accu-Bag) 124 PO#<br>lA To arrive(6M) Cases 8/18/95 (6 each$  style (4 styles) DocOwl 1.5-Chat:<br>2 Displays per store will be shipped directly to stores to arrive 8/21/95 z<br>lesocenacnnnennma<br> eins ° SIS # 158120.<br>SES EES SESE] The SIS # of L. MacWilliams is 158120, as stated in<br>fswcemagegremenmn TL the text under the section "L. MacWilliams SIS #<br>Unraeeeeeeesr we sips dnoeesreeeeeooooelan M8 ! Direct Account Name Information Required PO #".<br>J.Campi :<br>(a)<br>b.<br>95% 81% {=) t ) What percentage of teenagers<br>from the age group 12-17<br>CT Al Of teens ages 12-17... didn't use the Internet?<br>Answer the question with<br>lg el use the Internet use social media oe ; detailed explanation.<br>——————————es ss oe] DocOwl 1.5-Chat:<br>ie—— as<br>According to the text, 95% of teenagers<br>ages 12-17 use the Internet. Therefore,<br>the percentage of teenagers from the<br>a age group 12-17 who didn't use the<br>Internet would be 100% - 95% = 5%.<br>(b)<br>Date Time Opponent* Rank* Site TV Result | Attendance<br>September 1 2:30 PM = #9 (FCS) Northern lowa*| #12 | Camp Randall Stadium * Madison, WI BTN W 26-21 79,568<br>September 8 3:00 PM = at Oregon State* #13 | Reser Stadium * Corvallis, OR FX L7-10 42,189<br>September 15 7:00PM _ Utah State* #22 Camp Randall Stadium + Madison, WI BTNI2] W 16-14 79,332<br>September 22 | 11:00 AM_ UTEP* #24 Camp Randall Stadium + Madison, WI ESPN2 W 37-26 79,806<br>September 29 7:00 PM_~— at #20 Nebraska #23 | Memorial Stadium « Lincoln, NE ABC L 27-30 85,962<br>October 6 2:30 PM illinois Camp Randall Stadium * Madison, WI ABC/ESPN2 W 31-14 80,096<br>October 13 Ross-Ade Stadium * West Lafayette, IN BTN W 38-14 46,007<br>October 20 11:00 AM_ Minnesota Camp Randall Stadium + Madison, WI (Paul Bunyan's Axe) ESPNU W 38-13 80,587<br>October 27 2:30 PM Michigan Statet Camp Randall Stadium * Madison, WI ABC/ESPN2 L13-16°T | 80,538<br>November 10 Memorial Stadium * Bloomington, IN ESPN2 W 62-14 43,240<br>November 17 2:30 PM = Ohio State Camp Randall Stadium * Madison, WI ABC/ESPN2 L 14-21 OT 80,112<br>November 24 2:30 PM at Penn State Beaver Stadium * University Park, PA ESPN2 L21-24 OT 93,505<br>December 1 7:00 PM vs. #14 Nebraska Lucas Oil Stadium ¢ Indianapolis, IN (Big Ten Championship Game) FOX W 70-31 41,260<br>January 1, 2013 4:10PM vs. #8 Stanford #23 | Rose Bowl « Pasadena, CA (Rose Bowl) ESPN L 14-20 93,259<br>*Non-conference game. tHomecoming. *Rankings from Coaches Poll released priorto game. All times are in Central Time<br>what is the last date and time on the table? Answer the question with detailed explanation.<br>DocOwl 1.5-Chat:<br>The last date and time on the chart is January 1, 2013 at 4:10 PM. The chart lists all the games played by a certain team, along with<br>the date, time, opponent, rank, site, and result. The last game listed on the chart is against Stanford, which took place on January 1,<br>2013 at 4:10 PM. Therefore, the last date and time on the chart is January 1, 2013 at 4:10 PM.<br>(c)<br>**----- End of picture text -----**<br>


Figure 7: Qualitative results of question answering with detailed explanations. Some regions are enlarged for better visualization. 

14 

**==> picture [337 x 482] intentionally omitted <==**

**----- Start of picture text -----**<br>
76 What's the value of the Very well bar<br>TNNA2) a2 | in the 65+ age group? Answer the<br>S SIN2601 929 || ‘ question with detailed explanation.<br>SETTAOUT 20s 7 y DocOwl 1.5-Chat:<br>TTS6T 87<br>TST 89<br>98 68%.<br>75 In the image, which appears to be a chart from the Pew Research Center, the bar<br>T NARS 7 93 representing "Very well" for the age group 65 and older is marked at 68%. This indicates that 68% of Republicans and Republican leaners in this age group<br>Feb, 415, 2020 believe that Trump is fighting for their beliefs "Very well."<br>(a)<br>what is written on the blue<br>shirt on the left? Answer<br>the question with detailed<br>SS any Y y explanation.<br>x — — 7<br>i<br>DocOwl 1.5-Chat:<br>1s .<br>Fly Emirates.<br> The blue shirt on the left has the word "Fly Emirates"<br>written across the chest. The logo is in white lettering<br>against a blue background, which is the color scheme<br>typically used for the airline's branding.<br>(b)<br>“PE:Now LLP OP MAGAZINE SOUTH AFRICA'S.  NostaeisHOP MAGAZINE<br>I a eee oe<br>ad a<br>i) Li ia PFA»<br>bps<br>eS Seo - —_<br> Boeote , <= ee RAPLECONOMIX’<br>—_—ooo, SFa [~ ,<br> — : 4 S|\DEF;<br>f= SLUM VILEAGE [NIS™= Pens-once Tome to Non rsme ona<br>what nationality of rappers are lazy? Answer the question with detailed explanation.<br>DocOwl 1.5-Chat:<br>South African rappers are lazy. The text in the image states that "South African rappers are lazy." This is a reference to a quote from<br>an article titled "Rap Economix" in the magazine Hype, which is South Africa's No. 1 hip hop magazine. The article discusses the<br>that they rely on the success of Kool & the Gang's "Stayin' Alive" instead of<br>creating their own music. The article also mentions that this perception is changing, with new artists emerging and breaking away<br>(c)<br>**----- End of picture text -----**<br>


**==> picture [360 x 40] intentionally omitted <==**

**----- Start of picture text -----**<br>
South African rappers are lazy. The text in the image states that "South African rappers are lazy." This is a reference to a quote from<br>an article titled "Rap Economix" in the magazine Hype, which is South Africa's No. 1 hip hop magazine. The article discusses the<br>perception that South African rappers are lazy and that they rely on the success of Kool & the Gang's "Stayin' Alive" instead of<br>creating their own music. The article also mentions that this perception is changing, with new artists emerging and breaking away<br>from the stereotype.<br>**----- End of picture text -----**<br>


Figure 8: Qualitative results of question answering with detailed explanations. Hallucination in answers are marked in red. 

15 

**==> picture [201 x 211] intentionally omitted <==**

**----- Start of picture text -----**<br>
Extract words from the graphic.<br>DocOwl 1.5:<br>&<br><doc> Tropical Birding Trip Report NAMIBIA NOV DEC 2021<br>Tinkling Cisticola    Cisticola rufilatus<br>Seen near Kayova.<br>Chirping Cisticola    Cisticola pipiens<br>Seen at Kayova.<br>Piping Cisticola     Cisticola fulvicapilla<br>Seen near Kayova.<br>Zitting Cisticola    Cisticola juncidis<br>Seen at Etosha & Windhoek.<br>Desert Cisticola     Cistcola aridulus<br>Seen at Etosha.<br>Reed Warblers and Allies (Acrocephalidae)<br>Icterine Warbler     Hippolais icterina<br>Seen at Etosha & Windhoek.<br>Sedge Warbler      Acrocephalus schoenobaenus<br>Seen at Kayova.<br>African Reed Warbler   Acrocephalus baeticus<br>Seen at Walvis Bay & Windhoek.<br>Lesser Swamp Warbler   Acrocephalus gracilirostris<br>Seen at Kayova & Windhoek.<br>Swallows (Hirundinidae)<br>Plain Martin       Riparia paludicola<br>Seen at Kayova.<br>Bank Swallow       Riparia riparia<br>Seen at Caprivi.<br>www.tropicalbirding.com    +1-409-515-9110     info@tropicalbirding.com </doc><br>**----- End of picture text -----**<br>


Figure 9: A qualitative result of structure-aware document parsing. 

**==> picture [195 x 203] intentionally omitted <==**

**----- Start of picture text -----**<br>
Continue reading the text:  <doc> 5.4. Quality Criteria for a Blended Learning Platform<br>5.4.1. Introduction<br>Our objective is to create a list of absolutely necessary features for the eLearning platform (LMS) used<br>in a blended Learning course. These features are focused on the technical and pedagogical background<br>as well as on the learner’s situation. In addition, recommends for useful features and tools are defined<br>that used either inside the LCMS or as additional tools.<br>The target of the blended learning platform is the provision of high quality learning that will lead to the<br>development of the 21st century skills. These are characterized, according to Kong et al.(2014) by three<br>emphases:<br>First emphasis is on skills development in both formal and informal learning contexts (Cox, 2013; Huang,<br>Kinshuk. & Spector. 2013). The learners will be engaged in a seamless learning environment to<br>coherently apply various generic skills for in-school teacher-led learning process initiated in digital<br>classrooms and after-school learner-initiated learning process in social learning platforms/tools<br>according to individual needs (Milrad, Wong, Sharpies, Hwang, Looi, & Ogata. 2013; Otero, Milrad,<br>Rogers, Santos, Verissimo, & Tones. 2011; Wong & Looi. 2011).<br>Second emphasis is on skills development through both individualized and collaborative learning<br>approaches. On their own or with peers, learners take responsibilities to apply various generic skills to<br>plan goals, implement tasks, monitor progress and evaluate outcomes in their learning process (Kicken,<br>Brand-Gruwel, Merrienboer, & Slot. 2009; Norris & Soloway. 2009). The feedback for learners in a<br>minimal but sufficient amount identifies individual needs and directions for future improvement<br>(Caballero, van Riesen, Alvarez, Nussbaum, De Jong. 2014; Sims. 2003: Van Merrienboer. &<br>P<br>DocOwl 1.5:<br>Sluisjmans.<br>2009).<br>Figure 5-24: Learning with Tech trends31<br>Third emphasis is on skills development supported by evidence of improvement and awareness of<br>progress. The learning process in the e-learning environment can be designed in a range of activities in<br>authentic learning contexts. Rich evidence of improvement and productive failure is collected by<br>learners performance during the learning process; indications on applying 21st century skills for<br>processing real-life information, reflecting on problem-solving ways, articulating tacit knowledge and<br>31 Adapted from Olenka Villavicencio (2013): Learning with Tech trends<br>(Retrieved from http://olevilla.blogspot.gr/2013_07_01_archive.html) and Planning for<br>Personalization,(retrieved from http://education.vermont.gov/plo-working-group/main).<br>A guidance to Blended Learning                 79 </doc><br>**----- End of picture text -----**<br>


Figure 10: A qualitative result of structure-aware document parsing from the middle of the image. The red dotted box is only used to mark the location of the answers for better visualization and is not included in the input image. 

16 

**==> picture [110 x 6] intentionally omitted <==**

**----- Start of picture text -----**<br>
Convert the figure into Markdown language.<br>**----- End of picture text -----**<br>


**==> picture [33 x 5] intentionally omitted <==**

**----- Start of picture text -----**<br>
DocOwl 1.5:<br>**----- End of picture text -----**<br>


- <md> |  | <COLSPAN=2> N (%)** | <COLSPAN=2> Crude results (n = 61,129) | <COLSPAN=2> Adjusted results** (n = 47,817) | | LAN exposure | Cases | Referents | Odds ratios | 95% CI | Odds ratios | 95% CI | | --- | --- | --- | --- | --- | --- | --- | 

- | Low | 27,121 (63) | 10,970 (60) |  | Ref. |  | Ref. | 

- | Medium | 5,974 (14) | 2,623 (14) | 1.13 | (1.07-1.19) | 1.06 | (0.97-1.16) | | High | 9,659 (23) | 4,782 (26) | 1.22 | (1.18-1.28) | 1.12 | (1.04-1.20)† | </md> 

**==> picture [9 x 8] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a)<br>**----- End of picture text -----**<br>


**==> picture [294 x 156] intentionally omitted <==**

**----- Start of picture text -----**<br>
Parse the picture content behind with markdown code : <md> | region | population centers | notes |<br>| --- | --- | --- |<br>| Bačka | Novi Sad , Subotica | part of Vojvodina ; partially in Hungary (Bácska) |<br>| Gornji Breg | Gornji Breg | sub-region of Bačka |<br>| Potisje | Bečej , Senta | sub-region of both Bačka and Banat |<br>| Šajkaška | Temerin , Žabalj | sub-region of Bačka |<br>| Telečka | Vrbas , Kula | sub-region of Bačka |<br>| Banat | Zrenjanin , Pančevo | part of Vojvodina ; partially in Romania , Hungary (Bánság) and Central Serbia |<br>| Gornje Livade |  | sub-region of Banat |<br>| Ilandžanski Rit | Ilandža , Lokve<br>Oe<br>DocOwl 1.5:<br>| sub-region of Banat |<br>| Pančevački Rit | Borča , Knjača | sub-region of Banat; part of Central Serbia |<br>| Pomorišje | Novi Kneževac , Srpski Krstur | today considered sub-region of Banat; partially in Hungary and Romania |<br>| Repište | Knjačanin | sub-region of Banat |<br>**----- End of picture text -----**<br>


- | Veliki Rit | north-west of Vršac | sub-region of Banat | 

- | Srem | Belgrade , Sremska Mitrovica | part of Vojvodina; partially in Central Serbia and Croatia (Srijem) | | Crmi Lug | Progar , Boljevci | sub-region of Podlužje | | Kupinovski Kut | Kupinovo | sub-region of Podlužje | | Podlužje | Jakovo , Hrtkovci | sub-region of Srem; partially in Central Serbia | </md> ~~(b)~~ 

Figure 11: Qualitative results of structure-aware table parsing on the table with cells spanning multiple columns (a) and structure-aware table parsing from the middle of the image (b). The red dotted box is only used to mark the location of the answers for better visualization and is not included in the input image. 

17 

**==> picture [217 x 517] intentionally omitted <==**

**----- Start of picture text -----**<br>
Translate the chart into Markdown format.<br>DocOwl 1.5:<br><md> | Age group | November 2010 | mid-December 2011 | mid-January 2012 |<br>| --- | --- | --- | --- |<br>| 18-29 | 6% | 7% | 18% |<br>| 30-49 | 5% | 12% | 24% |<br>| 50-64 | 9% | 11% | 19% |<br>| 65 and older | 4% | 8% | 12% | </md><br>(a)<br>@<br>@ Render the figure in Markdown format.<br>.<br>DocOwl 1.5:<br><md> | Entity | No confidence | Confidence |<br>| --- | --- | --- |<br>| U.S | 63 | 20 |<br>| Lebanon | 64 | 23 |<br>| Turkey | 68 | 14 |<br>| Tunisia | 69 | 18 |<br>| Israel | 83 | 6 | </md><br>(b)<br>t )<br>@ Convert the image into Markdown language.<br>DocOwl 1.5:<br><md> | Teal | Yellow Green | Light Salmon | Chartreuse | Chocolate |<br>| --- | --- | --- | --- | --- |<br>| 19.5% | 31.1% | 4.17% | 24.2% | 21.9% | </md><br>(c)<br>|<br>Parse the figure content behind with markdown code:<br><md> | Years | Dominican Republic | Kenya | Zambia |<br>| --- | --- | --- | --- |<br>DocOwl 1.5:<br>| 2007 | 1 | 10 | 9.9 |<br>| 2010 | 3.47 | 4 | 6 |<br>| 2012 | 2 | 4 | 7 |<br>| 2014 | 3 | 4 | 4 | </md><br>(d)<br>**----- End of picture text -----**<br>


Figure 12: Qualitative results of structure-aware chart parsing on the charts of the vertical bar (a), horizontal bar (b), pie (c), and line (d). Incorrect words in the answer are marked in red. 

18 

**==> picture [390 x 309] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a a a)<br>RO TOnw<br>( NIMA RESTAURNT & BAR PARO TOWN<br>Specialist in SHUTNIS DI SM PSTAIRS )<br>=p. Ys <i iw<br>s Sa ee nee ue . ‘) _<br>cemeron eeof Maurer,rere ee Gee ae | 4 i F-.<br>A. ur« e, i‘BY t |i shat uf.as asks . PA WO LinaSEERA\ SN?oy f i<br>i] x e : a a " 8 7 oe ote a<br>©) ee =e Oe Comte! § Gime nr i _. r<br>4 ce Ge" a a a<br>EJ] alamy stock photo ree<br>(PEMA  DORA GENERAL SHOP)<br>E} alamy stock photo rei!<br>(a alamy stock photo E7BDR1  www.alamy.com)<br>A Detail the information and textual components depicted in the image.<br>DocOwl 1.5:<br>& [:]<br>the main street in the town <ocr> a a a alamy NIMA RESTAURANT & BAR Special TOURS & DEL PSTAIRS PERIA DORA GENERAL SHOP<br>SONAR ENTERPRISE a a a a a a alamy photo stock E7BDR1 a www.alamy.com </ocr><br>**----- End of picture text -----**<br>


**==> picture [311 x 200] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a)<br>(PORTERHIRE)<br>Detail the information and textual components<br>depicted in the image.<br>DocOwl 1.5:<br>& [as]<br>a digger works to clear trees from the path of the railway line . <ocr> PORTERHIRE </ocr><br>(b)<br>**----- End of picture text -----**<br>


Figure 13: Qualitative results of natural image parsing. For better visualization, some regions are enlarged and labeled with corresponding scene texts. Incorrect words in the answer are marked in red. 

19 

**==> picture [277 x 596] intentionally omitted <==**

**----- Start of picture text -----**<br>
 data @ Predict the bounding<br>box of the text <ocr><br>Disapprove 54 </ocr><br>hoo DocOwl 1.5:<br><bbox>329,341,371,381</bbox><br>(a)<br>final<br>Give the bounding<br>It Means Losing You" | [57 sth si @ box of the text <ocr><br>Losing You" </ocr><br>Sin ("That's That")"<br>On Me" DocOwl 1.5:<br>The People In The World" (105 | 2nd |<br><bbox>564,214,696,272</bbox><br>edadSomewhere In Europe" Ld<br>(b)<br>@<br>Predict the bounding<br>box of the text <ocr><br>networks in 2012 to<br>find talent and<br>offy networksofOMI CLS inompanicsTO 2012TOT tohavefind talentdsocigl andpotentialmedia potential </ocr>"<br>DocOwl 1.5:<br><bbox>569,172,949,182</bbox><br>(c)<br>Predict the bounding box of the text<br><ocr> particular to:<br>Nicholas Capaldi<br>• observe the Accounts Direction issued by the Accounting Officer<br>Secretary of State for Culture, Media and<br>Sport, including the relevant accounting and 7 July 2017<br>disclosure requirements, and apply suitable<br>accounting policies on a consistent basis; Endorsed on behalf of Council:<br>• make judgements and estimates on a<br>reasonable basis;<br>• state whether applicable accounting standards </ocr><br>DocOwl 1.5:<br><bbox>71,437,761,658</bbox><br>accounts, the Accounting Officer<br> Financialcomply with Reportingthe requirementsManual andofinthe Nichsta. Af ‘ .<br>Nicholas Capaldi<br>Accounts Direction issued by the Accounting Officer<br>State for Culture, Media and<br>the relevant accounting and 7 July 2017<br>requirements, and apply suitable<br>policies on a consistent basis; Endorsed on behalf of Council<br>and estimates on a<br> basis; applicable accounting standard ()| fry5 £1"<br>in the Government Financial U/<br>Manual have been followed and Phil eorge.<br>(d)<br>**----- End of picture text -----**<br>


Figure 14: Qualitative results of Multi-grained Text Grounding. Some regions are enlarged for better visualization. Bounding boxes predicted by DocOwl 1.5 are drawn in images as solid red boxes. 

20 

**==> picture [240 x 322] intentionally omitted <==**

**----- Start of picture text -----**<br>
Detect the text in the<br>@<br>bounding box<br>[Emirates_| <bbox>859,2,962,36<br></bbox><br>- DocOwl 1.5:<br><ocr> Emirates </ocr><br>(a)<br>Detect the text in the bounding box<br><bbox>128,430,256,455</bbox><br><<br>DocOwl 1.5:<br>$303.55 trillion<br><ocr> Produced capital </ocr><br>(b)<br>Detect the text in the bounding box<br><bbox>76,674,833,682</bbox><br>DocOwl 1.5:<br><ocr> SPOT 1:30:00AM- 2:35:00APMRPer week (M),Tu,We 02/14/20<br>1:45AM (Mo) 01:00 DATELINE 1:35A MRBPBK6033EH $150.00 </ocr><br>[.<br>lo2/1q/20. 2:16AM (Ma) 1:00 | DATELINE 1:354, MRBPBK6033EH $150.00<br>lo2/12/20 1:45AM (We) 1:00 | DATELINE 1:358, MRBPBK6033EH $150.00 worsens<br>”” Yonfa3y20 2to4am (wie) ~~)” a:00)pateune i358 Pmeekeos3en ~~ ysoaa PO<br>(c)<br>**----- End of picture text -----**<br>


**==> picture [364 x 118] intentionally omitted <==**

**----- Start of picture text -----**<br>
Detect the text in the bounding box <bbox>14,532,981,850</bbox><br>DocOwl 1.5:<br><ocr> Ludwig   Wittgenstein   [edit] \n<br>Born 1889, died 1951. An Austrian born Cambridge scholar, Wittgenstein was involved in his early years with the Logical<br>Positivists of the Vienna Circle. \n<br>Wittgenstein left Philosophy for a while but was persuaded to return in the 1920s. Wittgenstein's return prompted a different way<br>of looking at religious \n<br>language, he no longer sought to prove "whether" language refers, but looked at the way in which language refers to things. \n<br>He decided that the best way to determine a word's meaning was to look at how it is used, and not to come up with a theoretical<br>definition. Wittgenstein \n<br>said this was about: "Getting back to the rough ground". </ocr><br>(d)<br>**----- End of picture text -----**<br>


Figure 15: Qualitative results of Multi-grained Text Recognition. Some regions are enlarged for better visualization. Input bounding boxes are drawn in images as solid blue boxes. Incorrect words in answers are marked in red. 

21 

## **References** 

- [1] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katie Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob Menick, Sebastian Borgeaud, Andy Brock, Aida Nematzadeh, Sahand Sharifzadeh, Mikolaj Binkowski, Ricardo Barreira, Oriol Vinyals, Andrew Zisserman, and Karen Simonyan. Flamingo: a visual language model for few-shot learning. _ArXiv_ , abs/2204.14198, 2022. 

- [2] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, Keming Lu, Jianxin Ma, Rui Men, Xingzhang Ren, Xuancheng Ren, Chuanqi Tan, Sinan Tan, Jianhong Tu, Peng Wang, Shijie Wang, Wei Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang, Hao Yang, Jian Yang, Shusheng Yang, Yang Yao, Bowen Yu, Hongyi Yuan, Zheng Yuan, Jianwei Zhang, Xingxuan Zhang, Yichang Zhang, Zhenru Zhang, Chang Zhou, Jingren Zhou, Xiaohuan Zhou, and Tianhang Zhu. Qwen technical report. _arXiv preprint arXiv:2309.16609_ , 2023. 

- [3] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. _arXiv preprint arXiv:2308.12966_ , 2023. 

- [4] Lukasz Borchmann, Michal Pietruszka, Tomasz Stanislawek, Dawid Jurkiewicz, Michal Turski, Karolina Szyndler, and Filip Gralinski. DUE: end-to-end document understanding benchmark. In _NeurIPS Datasets and Benchmarks_ , 2021. 

- [5] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. _Advances in neural information processing systems_ , 33:1877–1901, 2020. 

- [6] Minwoo Byeon, Beomhee Park, Haecheon Kim, Sungjun Lee, Woonhyuk Baek, and Saehoon Kim. Coyo-700m: Image-text pair dataset. `https://github.com/kakaobrain/ coyo-dataset` , 2022. 

- [7] Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. Conceptual 12m: Pushing web-scale image-text pre-training to recognize long-tail visual concepts. In _CVPR_ , pages 3558–3568. Computer Vision Foundation / IEEE, 2021. 

- [8] Wenhu Chen, Hongmin Wang, Jianshu Chen, Yunkai Zhang, Hong Wang, Shiyang Li, Xiyou Zhou, and William Yang Wang. Tabfact : A large-scale dataset for table-based fact verification. In _International Conference on Learning Representations (ICLR)_ , Addis Ababa, Ethiopia, April 2020. 

- [9] Xingyu Chen, Zihan Zhao, Lu Chen, JiaBao Ji, Danyang Zhang, Ao Luo, Yuxuan Xiong, and Kai Yu. Websrc: A dataset for web-based structural reading comprehension. In _Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing_ , pages 4173–4185, 2021. 

- [10] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven C. H. Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning. _CoRR_ , abs/2305.06500, 2023. 

- [11] Xiang Deng, Huan Sun, Alyssa Lees, You Wu, and Cong Yu. TURL: table understanding through representation learning. _SIGMOD Rec._ , 51(1):33–40, 2022. 

- [12] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In _ICLR_ . OpenReview.net, 2021. 

- [13] Hao Feng, Qi Liu, Hao Liu, Wengang Zhou, Houqiang Li, and Can Huang. Docpedia: Unleashing the power of large multimodal model in the frequency domain for versatile document understanding. _CoRR_ , abs/2311.11810, 2023. 

22 

- [14] Adam W. Harley, Alex Ufkes, and Konstantinos G. Derpanis. Evaluation of deep convolutional nets for document image classification and retrieval. In _ICDAR_ , pages 991–995. IEEE Computer Society, 2015. 

- [15] Wenyi Hong, Weihan Wang, Qingsong Lv, Jiazheng Xu, Wenmeng Yu, Junhui Ji, Yan Wang, Zihan Wang, Yuxuan Zhang, Juanzi Li, Bin Xu, Yuxiao Dong, Ming Ding, and Jie Tang. Cogagent: A visual language model for GUI agents. _CoRR_ , abs/2312.08914, 2023. 

- [16] Anwen Hu, Shizhe Chen, and Qin Jin. Question-controlled text-aware image captioning. In _ACM Multimedia_ , pages 3097–3105. ACM, 2021. 

- [17] Anwen Hu, Yaya Shi, Haiyang Xu, Jiabo Ye, Qinghao Ye, Ming Yan, Chenliang Li, Qi Qian, Ji Zhang, and Fei Huang. mplug-paperowl: Scientific diagram analysis with the multimodal large language model. _arXiv preprint arXiv:2311.18248_ , 2023. 

- [18] Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, and Furu Wei. Layoutlmv3: Pre-training for document AI with unified text and image masking. In _ACM Multimedia_ , pages 4083–4091. ACM, 2022. 

- [19] Kushal Kafle, Brian L. Price, Scott Cohen, and Christopher Kanan. DVQA: understanding data visualizations via question answering. In _CVPR_ , pages 5648–5656. Computer Vision Foundation / IEEE Computer Society, 2018. 

- [20] Samira Ebrahimi Kahou, Vincent Michalski, Adam Atkinson, Ákos Kádár, Adam Trischler, and Yoshua Bengio. Figureqa: An annotated figure dataset for visual reasoning. In _ICLR (Workshop)_ . OpenReview.net, 2018. 

- [21] Shankar Kantharaj, Rixie Tiffany Ko Leong, Xiang Lin, Ahmed Masry, Megh Thakkar, Enamul Hoque, and Shafiq R. Joty. Chart-to-text: A large-scale benchmark for chart summarization. In _ACL (1)_ , pages 4005–4023. Association for Computational Linguistics, 2022. 

- [22] Geewook Kim, Teakgyu Hong, Moonbin Yim, JeongYeon Nam, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, Sangdoo Yun, Dongyoon Han, and Seunghyun Park. Ocr-free document understanding transformer. In _ECCV (28)_ , volume 13688 of _Lecture Notes in Computer Science_ , pages 498–517. Springer, 2022. 

- [23] Kenton Lee, Mandar Joshi, Iulia Raluca Turc, Hexiang Hu, Fangyu Liu, Julian Martin Eisenschlos, Urvashi Khandelwal, Peter Shaw, Ming-Wei Chang, and Kristina Toutanova. Pix2struct: Screenshot parsing as pretraining for visual language understanding. In _ICML_ , volume 202 of _Proceedings of Machine Learning Research_ , pages 18893–18912. PMLR, 2023. 

- [24] Junnan Li, Dongxu Li, Silvio Savarese, and Steven C. H. Hoi. BLIP-2: bootstrapping language-image pre-training with frozen image encoders and large language models. _CoRR_ , abs/2301.12597, 2023. 

- [25] Zhang Li, Biao Yang, Qiang Liu, Zhiyin Ma, Shuo Zhang, Jingxu Yang, Yabo Sun, Yuliang Liu, and Xiang Bai. Monkey: Image resolution and text label are important things for large multi-modal models. _CoRR_ , abs/2311.06607, 2023. 

- [26] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. _CoRR_ , abs/2310.03744, 2023. 

- [27] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. _CoRR_ , abs/2304.08485, 2023. 

- [28] Yuliang Liu, Zhang Li, Hongliang Li, Wenwen Yu, Mingxin Huang, Dezhi Peng, Mingyu Liu, Mingrui Chen, Chunyuan Li, Lianwen Jin, et al. On the hidden mystery of ocr in large multimodal models. _arXiv preprint arXiv:2305.07895_ , 2023. 

- [29] Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq R. Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. In _ACL (Findings)_ , pages 2263–2279. Association for Computational Linguistics, 2022. 

23 

- [30] Minesh Mathew, Dimosthenis Karatzas, and C. V. Jawahar. Docvqa: A dataset for VQA on document images. In _WACV_ , pages 2199–2208. IEEE, 2021. 

- [31] Minesh Mathew, Viraj Bagal, Rubèn Tito, Dimosthenis Karatzas, Ernest Valveny, and C. V. Jawahar. Infographicvqa. In _WACV_ , pages 2582–2591. IEEE, 2022. 

- [32] Nitesh Methani, Pritha Ganguly, Mitesh M. Khapra, and Pratyush Kumar. Plotqa: Reasoning over scientific plots. In _WACV_ , pages 1516–1525. IEEE, 2020. 

- [33] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In _Proceedings of the 40th annual meeting of the Association for Computational Linguistics_ , pages 311–318, 2002. 

- [34] Panupong Pasupat and Percy Liang. Compositional semantic parsing on semi-structured tables. In _ACL (1)_ , pages 1470–1480. The Association for Computer Linguistics, 2015. 

- [35] Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, and Furu Wei. Kosmos-2: Grounding multimodal large language models to the world. _CoRR_ , abs/2306.14824, 2023. 

- [36] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In _ICML_ , volume 139 of _Proceedings of Machine Learning Research_ , pages 8748–8763. PMLR, 2021. 

- [37] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, Patrick Schramowski, Srivatsa Kundurthy, Katherine Crowson, Ludwig Schmidt, Robert Kaczmarczyk, and Jenia Jitsev. LAION-5B: an open large-scale dataset for training next generation image-text models. In _NeurIPS_ , 2022. 

- [38] Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In _ACL (1)_ , pages 2556–2565. Association for Computational Linguistics, 2018. 

- [39] Oleksii Sidorov, Ronghang Hu, Marcus Rohrbach, and Amanpreet Singh. Textcaps: A dataset for image captioning with reading comprehension. In _ECCV (2)_ , volume 12347 of _Lecture Notes in Computer Science_ , pages 742–758. Springer, 2020. 

- [40] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards VQA models that can read. In _CVPR_ , pages 8317–8326. Computer Vision Foundation / IEEE, 2019. 

- [41] Tomasz Stanislawek, Filip Gralinski, Anna Wróblewska, Dawid Lipinski, Agnieszka Kaliska, Paulina Rosalska, Bartosz Topolski, and Przemyslaw Biecek. Kleister: Key information extraction datasets involving long documents with complex layouts. In _ICDAR (1)_ , volume 12821 of _Lecture Notes in Computer Science_ , pages 564–579. Springer, 2021. 

- [42] S Svetlichnaya. Deepform: Understand structured documents at scale, 2020. 

- [43] Ryota Tanaka, Kyosuke Nishida, and Sen Yoshida. Visualmrc: Machine reading comprehension on document images. In _AAAI_ , pages 13878–13888. AAAI Press, 2021. 

- [44] Benny J. Tang, Angie Boggust, and Arvind Satyanarayan. Vistext: A benchmark for semantically rich chart captioning. In _ACL (1)_ , pages 7268–7298. Association for Computational Linguistics, 2023. 

- [45] Zineng Tang, Ziyi Yang, Guoxin Wang, Yuwei Fang, Yang Liu, Chenguang Zhu, Michael Zeng, Cha Zhang, and Mohit Bansal. Unifying vision, text, and layout for universal document processing. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 19254–19264, 2023. 

- [46] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. _arXiv preprint arXiv:2302.13971_ , 2023. 

24 

- [47] Michal Turski, Tomasz Stanislawek, Karol Kaczmarek, Pawel Dyda, and Filip Gralinski. Ccpdf: Building a high quality corpus for visually rich documents from web crawl data. In _ICDAR (3)_ , volume 14189 of _Lecture Notes in Computer Science_ , pages 348–365. Springer, 2023. 

- [48] Vicuna. Vicuna: An open chatbot impressing gpt-4. `https://github.com/lm-sys/ FastChat` , 2023. 

- [49] Peng Wang, An Yang, Rui Men, Junyang Lin, Shuai Bai, Zhikang Li, Jianxin Ma, Chang Zhou, Jingren Zhou, and Hongxia Yang. OFA: unifying architectures, tasks, and modalities through a simple sequence-to-sequence learning framework. In _ICML_ , volume 162 of _Proceedings of Machine Learning Research_ , pages 23318–23340. PMLR, 2022. 

- [50] Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, Jiazheng Xu, Bin Xu, Juanzi Li, Yuxiao Dong, Ming Ding, and Jie Tang. Cogvlm: Visual expert for pretrained language models. _CoRR_ , abs/2311.03079, 2023. 

- [51] Wenjin Wang, Yunhao Li, Yixin Ou, and Yin Zhang. Layout and task aware instruction prompt for zero-shot document image question answering. _CoRR_ , abs/2306.00526, 2023. 

- [52] Haiyang Xu, Ming Yan, Chenliang Li, Bin Bi, Songfang Huang, Wenming Xiao, and Fei Huang. E2E-VLP: end-to-end vision-language pre-training enhanced by visual learning. In _ACL/IJCNLP (1)_ , pages 503–513. Association for Computational Linguistics, 2021. 

- [53] Yang Xu, Yiheng Xu, Tengchao Lv, Lei Cui, Furu Wei, Guoxin Wang, Yijuan Lu, Dinei A. F. Florêncio, Cha Zhang, Wanxiang Che, Min Zhang, and Lidong Zhou. Layoutlmv2: Multi-modal pre-training for visually-rich document understanding. In _ACL/IJCNLP (1)_ , pages 2579–2591. Association for Computational Linguistics, 2021. 

- [54] Zhengyuan Yang, Yijuan Lu, Jianfeng Wang, Xi Yin, Dinei Florêncio, Lijuan Wang, Cha Zhang, Lei Zhang, and Jiebo Luo. TAP: text-aware pre-training for text-vqa and text-caption. In _CVPR_ , pages 8751–8761. Computer Vision Foundation / IEEE, 2021. 

- [55] Jiabo Ye, Anwen Hu, Haiyang Xu, Qinghao Ye, Ming Yan, Yuhao Dan, Chenlin Zhao, Guohai Xu, Chenliang Li, Junfeng Tian, Qian Qi, Ji Zhang, and Fei Huang. mplug-docowl: Modularized multimodal large language model for document understanding. _CoRR_ , abs/2307.02499, 2023. 

- [56] Jiabo Ye, Anwen Hu, Haiyang Xu, Qinghao Ye, Ming Yan, Guohai Xu, Chenliang Li, Junfeng Tian, Qi Qian, Ji Zhang, Qin Jin, Liang He, Xin Lin, and Fei Huang. Ureader: Universal ocr-free visually-situated language understanding with multimodal large language model. In _EMNLP (Findings)_ , pages 2841–2858. Association for Computational Linguistics, 2023. 

- [57] Qinghao Ye, Haiyang Xu, Guohai Xu, Jiabo Ye, Ming Yan, Yiyang Zhou, Junyang Wang, Anwen Hu, Pengcheng Shi, Yaya Shi, Chenliang Li, Yuanhong Xu, Hehong Chen, Junfeng Tian, Qian Qi, Ji Zhang, and Fei Huang. mplug-owl: Modularization empowers large language models with multimodality. _CoRR_ , abs/2304.14178, 2023. 

- [58] Qinghao Ye, Haiyang Xu, Jiabo Ye, Ming Yan, Anwen Hu, Haowei Liu, Qi Qian, Ji Zhang, Fei Huang, and Jingren Zhou. mplug-owl2: Revolutionizing multi-modal large language model with modality collaboration. _CoRR_ , abs/2311.04257, 2023. 

- [59] Duzhen Zhang, Yahan Yu, Chenxing Li, Jiahua Dong, Dan Su, Chenhui Chu, and Dong Yu. Mmllms: Recent advances in multimodal large language models. _arXiv preprint arXiv:2401.13601_ , 2024. 

- [60] Liang Zhang, Anwen Hu, Jing Zhang, Shuo Hu, and Qin Jin. MPMQA: multimodal question answering on product manuals. _CoRR_ , abs/2304.09660, 2023. 

- [61] Tianshu Zhang, Xiang Yue, Yifei Li, and Huan Sun. Tablellama: Towards open large generalist models for tables. _CoRR_ , abs/2311.09206, 2023. 

- [62] Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen Yang, Yushuo Chen, Zhipeng Chen, Jinhao Jiang, Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu, Peiyu Liu, Jian-Yun Nie, and Ji-Rong Wen. A survey of large language models. _CoRR_ , abs/2303.18223, 2023. 

25 

- [63] Xu Zhong, Elaheh ShafieiBavani, and Antonio Jimeno-Yepes. Image-based table recognition: Data, model, and evaluation. In _ECCV (21)_ , volume 12366 of _Lecture Notes in Computer Science_ , pages 564–580. Springer, 2020. 

- [64] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models, 2023. 

26 

