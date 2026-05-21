# Datasets used in MP-VRDU papers (46-paper corpus, v2)

This document catalogues every dataset referenced across the **46** papers in `models_main_md/`, grouped by the role the dataset plays in each paper: **Pretraining**, **Instruct-tuning / Fine-tuning**, and **Benchmarking**. A dataset may appear in more than one table when used in different roles by different papers (or by the same paper in multiple stages).

This v2 build extends the v1 (36-paper) tables with the 10 newly added papers — **DMAP, DREAM, MM-Doc-R1, MM-R5, SV-RAG, SLEUTH, VisDoMRAG, VRAG-RL, URaG, DocSeeker** — and incorporates one correction to the v1 DREAM/baseline mapping (DREAM also fine-tunes on DUDE's training split).

Naming conventions:
- "MP-DocVQA" subsumes "Multi-Page DocVQA" / "MPDocVQA".
- "MMLongBench-Doc" (Ma et al., NeurIPS 2024) is the 47-page-average benchmark and is distinct from "MMLongBench" (Wang et al., NeurIPS 2025), the multi-task long-context benchmark. The label "MMLongBench" in DMAP, MM-Doc-R1, VRAG-RL, SLEUTH, and SV-RAG all refer to **MMLongBench-Doc** (Ma 2024) — confirmed via grep of the cited reference. Where "MMLongBench" (2025) is meant, the table flags it explicitly.
- "InfographicVQA" subsumes "InfoVQA" / "InfographicsVQA".
- "WikiTableQuestions" subsumes "WTQ".
- Datasets introduced by a paper are marked with † beside their name.

---

## Table 1 — Pretraining Datasets

| Dataset | Papers | Usage per paper |
|---|---|---|
| DocStruct4M | mPLUG-DocOwl2; VDocRAG | mPLUG-DocOwl2: 4M samples for single-image structure-aware pretraining. VDocRAG: 500K samples used to pretrain VDocRetriever. |
| MP-DocStruct1M† | mPLUG-DocOwl2 | mPLUG-DocOwl2: 1.1M multi-image samples used for multi-page continue-pretraining. |
| LLaVA-558K | Leopard | Leopard: 558K multimodal samples used for vision–language connector pretraining. |
| Recap-COCO-30K | Leopard | Leopard: text-rich image pretraining for the Pro variant. |
| CC-3M | Leopard | Leopard: vision–language pretraining mix. |
| Donut | Leopard | Leopard: pretraining data for OCR-free document encoding. |
| Cauldron | Leopard; Texthawk2 | Leopard: pretraining-stage multimodal mix. Texthawk2: large-scale instruction/pretraining mix. |
| Arxiv Pages (4M) | Leopard | Leopard: 4M OCR-processed scientific pages for the Pro pretraining stage. |
| OCR-IDL (UCSF-IDL) | Hi-VT5 | Hi-VT5: 200K pages with hierarchical layout-aware denoising self-supervision. |
| LayoutLLM SFT data | LayTokenLLM | LayTokenLLM: layout-aware text used to pretrain the layout token mechanism. |
| Sci-Hub PDFs | Docopilot | Docopilot: raw PDFs used for next-token prediction pretraining. |
| Arxiv PDFs | Docopilot; CoR | Docopilot: next-token prediction pretraining source. CoR: Mask-AR self-supervised pretraining on figure–caption pairs. |
| OpenReview PDFs | Docopilot | Docopilot: paper PDFs for next-token prediction pretraining. |
| CCpdf | Arctic-TILT | Arctic-TILT: self-supervised pretraining for ~900K steps on PDF documents. |
| PaperPDF† | PDF-WuKong | PDF-WuKong: 1.1M bilingual PDF/QA corpus used for self-supervised + supervised pretraining. |
| LAION-400M (CapsFusion) | Texthawk2 | Texthawk2: web image–caption pairs with LVLM-rewritten captions. |
| Wanjuan1.0 | Texthawk2 | Texthawk2: bilingual interleaved image–text from Wikipedia and news. |
| GrIT-20M | Texthawk2 | Texthawk2: synthetic grounded captions with location labels. |
| UMG-41M | Texthawk2 | Texthawk2: visual grounding pretraining (CC3M, CC12M, SBU, Flickr, VG, YFCC-15M, ImageNet-21K). |
| Wukong | Texthawk2 | Texthawk2: Chinese OCR pretraining corpus. |
| IIT-CDIP | Texthawk2 | Texthawk2: English OCR pretraining corpus. |
| Common Crawl PDFs | Texthawk2 | Texthawk2: text extracted from web PDFs for OCR pretraining. |
| RenderedText | Texthawk2 | Texthawk2: synthetic English handwriting recognition data. |
| Multilingual OCR collection (ArT, COCO-Text, CTW, IC15, LSVT, MLT, MTWI, RCTW-17, ReCTS, SCUT-HCCDoc) | Texthawk2 | Texthawk2: aggregated scene/document OCR pretraining data. |
| arXiv LaTeX | Texthawk2 | Texthawk2: LaTeX → markdown pretraining data. |
| GitHub READMEs | Texthawk2 | Texthawk2: README files converted to markdown for layout pretraining. |
| Common Crawl DOCX | Texthawk2 | Texthawk2: Word documents converted to markdown for pretraining. |
| PubTables-1M | Texthawk2 | Texthawk2: table recognition pretraining. |
| MMC | Texthawk2 | Texthawk2: chart-to-table and chart-QA pretraining. |
| ChartSFT | Texthawk2 | Texthawk2: chart understanding pretraining. |

> Net: 30 distinct pretraining-stage datasets across 9 papers. None of the 10 newly added papers performs from-scratch pretraining — all inherit backbones (Qwen2.5-VL, InternVL2, Qwen3-VL, Phi-3-V, PaliGemma) and skip directly to SFT/RL or remain training-free.

---

## Table 2 — Instruct-Tuning / Fine-Tuning Datasets

| Dataset | Papers | Usage per paper |
|---|---|---|
| DocStruct4M | mPLUG-DocOwl2 | mPLUG-DocOwl2: 501,781 samples reused in continue-pretraining + SFT phase. |
| DocVQA | mPLUG-DocOwl2; VDocRAG; LayTokenLLM; Self-Attention Scoring; Docopilot; Arctic-TILT; DocR1; CREAM; PDF-WuKong; SV-RAG | mPLUG-DocOwl2: single-image SFT. VDocRAG: VDocGenerator SFT. LayTokenLLM: 50K QA pairs over 12K images for SFT. Self-Attention Scoring: Pix2Struct page encoder fine-tuned on it (Stage 1). Docopilot: 56K Qs in instruction mix. Arctic-TILT: task-specific FT. DocR1: included in EviBench training pool. CREAM: SFT for single-page VQA. PDF-WuKong: SFT for single-page document QA. SV-RAG: 39,463 DocVQA question-page pairs in the ColPali-style Col-retrieval training set. |
| InfographicVQA | mPLUG-DocOwl2; VDocRAG; RAG-DocVQA; Arctic-TILT; DocR1; CREAM; PDF-WuKong; Docopilot; SV-RAG | mPLUG-DocOwl2: SFT. VDocRAG: VDocGenerator SFT. RAG-DocVQA: embedding model + generator FT. Arctic-TILT: task-specific FT. DocR1: EviBench training. CREAM: SFT for infographic VQA. PDF-WuKong: SFT. Docopilot: 25.5K Qs in instruction mix. SV-RAG: 10,074 InfoVQA pairs in Col-retrieval training. |
| DeepForm | mPLUG-DocOwl2; Arctic-TILT; DocR1 | mPLUG-DocOwl2: form understanding SFT. Arctic-TILT: form-extraction FT. DocR1: included in EviBench training pool. |
| KleisterCharity | mPLUG-DocOwl2 (KLC); Arctic-TILT; DocR1 | mPLUG-DocOwl2: layout/IE SFT. Arctic-TILT: financial-report key-information extraction FT. DocR1: included in EviBench training pool. |
| WikiTableQuestions | mPLUG-DocOwl2 (WTQ); DocR1 | mPLUG-DocOwl2: table understanding SFT. DocR1: included in EviBench training pool. |
| TabFact | mPLUG-DocOwl2; DocR1 | mPLUG-DocOwl2: table verification SFT. DocR1: EviBench training pool. |
| ChartQA | mPLUG-DocOwl2; VDocRAG; Docopilot; DocR1; PDF-WuKong | mPLUG-DocOwl2: chart SFT. VDocRAG: VDocGenerator SFT. Docopilot: 30.2K Qs in mix. DocR1: EviBench training. PDF-WuKong: SFT. |
| TextVQA | mPLUG-DocOwl2; DocR1 | mPLUG-DocOwl2: text-in-image SFT. DocR1: EviBench training. |
| TextCaps | mPLUG-DocOwl2; DocR1 | mPLUG-DocOwl2: text captioning SFT. DocR1: EviBench training. |
| VisualMRC | mPLUG-DocOwl2; VDocRAG; DocR1; CREAM | mPLUG-DocOwl2: SFT. VDocRAG: VDocGenerator SFT. DocR1: EviBench training. CREAM: layout-rich web SFT. |
| DocReason25K | mPLUG-DocOwl2 | mPLUG-DocOwl2: document reasoning SFT. |
| MP-DocVQA | mPLUG-DocOwl2; Hi-VT5†; LayTokenLLM; RAG-DocVQA; Self-Attention Scoring; Docopilot; DFVC; CREAM; GRAM; RM-T5; Doc-V*; DocSLM; M2RAG; DocR1; PDF-WuKong; DREAM; MM-R5; URaG; DocSeeker | Hi-VT5: introduced and used as primary SFT corpus. mPLUG-DocOwl2: multi-page SFT. LayTokenLLM/RAG-DocVQA/CREAM/GRAM/Self-Attention Scoring/RM-T5: standard supervised FT for multi-page VQA. Docopilot: 51K Qs in mix. DFVC: 80% split for adapter training. Doc-V*: SFT with 9,019 trajectories. DocSLM: curriculum FT. M2RAG: VLM FT. DocR1: EviBench training. PDF-WuKong: SFT. DREAM: trains LightGBM ranker + InternVL2-4B QA module on MP-DocVQA train split. MM-R5: included in 73,843-QA SFT mix. URaG: Stage-1 retrieval pretraining + Stage-2 LoRA FT. DocSeeker: 13,986 distilled CoT samples for SFT + EviGRPO RL pool. |
| DUDE | mPLUG-DocOwl2; VDocRAG; LayTokenLLM; RAG-DocVQA; Arctic-TILT; GRAM; Doc-V*; CREAM; DocR1; PDF-WuKong; Docopilot; DocDancer; DREAM; MM-R5; URaG; DocSeeker | Most papers: standard supervised FT. Doc-V*: trajectories for SFT. DocDancer: source documents for synthesizing training trajectories. Docopilot: 27K Qs in mix. Arctic-TILT: diverse-domain FT. DREAM: DUDE train split used together with MP-DocVQA for QA-module FT. MM-R5: in 73,843-QA SFT mix. URaG: retrieval + LoRA FT. DocSeeker: SFT + EviGRPO pool. |
| SlideVQA | VDocRAG; DocR1; ViDoRAG; SV-RAG; MM-R5; VRAG-RL; URaG | VDocRAG: VDocGenerator SFT. DocR1: EviBench training. ViDoRAG: refined into ViDoSeek benchmark. SV-RAG: 10,290-sample training split for QA module fine-tuning. MM-R5: in 73,843-QA SFT mix. VRAG-RL: SFT + GRPO RL training data (Qwen-VLmax trajectory collection). URaG: retrieval + LoRA FT. |
| TAT-DQA | Arctic-TILT; SV-RAG; MM-R5 | Arctic-TILT: table-aware document QA FT. SV-RAG: 13,251 pairs in ColPali-style retrieval training set. MM-R5: in 73,843-QA SFT mix. |
| MP-DocReason51K† | mPLUG-DocOwl2 | mPLUG-DocOwl2: introduced multi-page reasoning SFT data. |
| DocGenome12K | mPLUG-DocOwl2 | mPLUG-DocOwl2: document structure SFT. |
| NewsVideoQA | mPLUG-DocOwl2 | mPLUG-DocOwl2: video-document SFT. |
| OpenWikiTable | VDocRAG | VDocRAG: VDocGenerator SFT for table QA. |
| MPMQA | VDocRAG | VDocRAG: multi-page document SFT. |
| MHDocVQA† | VDocRAG | VDocRAG: newly created multi-hop document QA SFT data. |
| Leopard-Instruct† | Leopard | Leopard: introduced 925K-instance instruction set (739K text-rich multi-image, 186K single-image). |
| Pew Research Charts | Leopard | Leopard: multi-chart QA generated with GPT-4o. |
| SlideShare | Leopard | Leopard: slide-deck QA generated with GPT-4o. |
| MultiHiertt | Leopard; DocR1 | Leopard: multi-table images for multi-image FT. DocR1: EviBench training. |
| MultiTabQA | Leopard | Leopard: multi-table FT data. |
| TableGPT | Leopard | Leopard: single-table data split into sub-tables for multi-image FT. |
| ArxivQA | Leopard; Docopilot; SV-RAG | Leopard: assembled into multi-image SFT. Docopilot: 31K Qs in instruction mix. SV-RAG: 10,000 arXivQA pairs in ColPali-style retrieval training set. |
| RICO | Leopard | Leopard: assembled into multi-image SFT. |
| FigureQA | Leopard; DocR1 | Leopard: assembled into multi-image SFT. DocR1: EviBench training. |
| MapQA | Leopard | Leopard: assembled into multi-image SFT. |
| ShareGPT4V | Leopard | Leopard: natural-image SFT to preserve general capability. |
| DocHieNet | MultiDocFusion | MultiDocFusion: hierarchical document parsing FT. |
| HRDH | MultiDocFusion | MultiDocFusion: academic-document hierarchy FT. |
| MMLongBench-Doc | MoLoRAG; DFVC; VRAG-RL | MoLoRAG: 5,500 GPT-4o-synthesised QA pairs to train logical relevance scorer. DFVC: 80% split for adapter training. VRAG-RL: SFT + GRPO RL training (trajectory collection over MMLongBench-Doc documents). |
| LongDocURL | MoLoRAG; DocDancer; MM-Doc-R1 | MoLoRAG: document snapshots used for synthetic-data generation. DocDancer: source documents for training trajectory synthesis. MM-Doc-R1: 300 samples held as validation set; remainder used as SPO/GRPO RL training data. |
| MMDocRAG | DocDancer | DocDancer: source documents for training trajectory synthesis. |
| CUAD | DocDancer; MM-R5 | DocDancer: source documents for training trajectory synthesis. MM-R5: included in 73,843-QA SFT mix. |
| HotpotQA | KGP | KGP: fine-tunes DPR/MDR encoders for next-passage prediction; instruction-tunes LLaMA-7B and T5-Large. |
| SIBR | LayTokenLLM | LayTokenLLM: real-world VIE FT with difficult-to-recognize text. |
| Doc-750K† | Docopilot | Docopilot: introduced 758K QA / 3.1M images / 251K conversations; main SFT corpus. |
| MP-Docmatix | Docopilot | Docopilot: 141K multi-page samples in SFT mix. |
| DocReason | Docopilot | Docopilot: 25.8K reasoning Qs in SFT mix. |
| MMDU | Docopilot | Docopilot: 45K multi-image general QA in SFT mix. |
| Long-context text mix (LongAlpaca, LongQLoRA, LongCite, LongAlign, LongReward) | Docopilot | Docopilot: long-context text SFT to support long-doc inputs. |
| OCR-VQA | mPLUG-DocOwl2; Docopilot; DocR1 | mPLUG-DocOwl2/Docopilot: SFT mix. DocR1: EviBench training. |
| CoR-Dataset† | CoR | CoR: introduced 26,088 high-quality QA pairs with reasoning traces for Chain-of-Reading SFT. |
| Mask-AR dataset† | CoR | CoR: self-supervised FT on figure–caption pairs from scientific documents. |
| Preference dataset (DPO) | CoR | CoR: 5,000 QA pairs used for Direct Preference Optimization. |
| InstructDoc† | InstructDoc | InstructDoc: introduced 30-source / 12-task instruction set used for instruction tuning. |
| Kleister NDA | Arctic-TILT | Arctic-TILT: legal-document key-information extraction FT. |
| CHART-Infographics | Arctic-TILT | Arctic-TILT: chart understanding FT. |
| SQuAD 2.0 | Arctic-TILT | Arctic-TILT: extractive QA FT. |
| VQA-CD | Arctic-TILT | Arctic-TILT: invoice / purchase-order FT. |
| VQAonBD | Arctic-TILT | Arctic-TILT: business-document VQA FT. |
| DVQA | DocR1 | DocR1: EviBench training pool. |
| TATDoc | DocR1 | DocR1: EviBench training pool. |
| ArxivFullQA-train | DocR1 | DocR1: scientific paper QA in EviBench training pool. |
| OpenHermes2.5 | Texthawk2 | Texthawk2: high-quality text instruction data. |
| COIG-CQIA | Texthawk2 | Texthawk2: Chinese text instruction data. |
| ShareGPT-4o | Texthawk2 | Texthawk2: instruction-following data. |
| LVIS-Instruct4V | Texthawk2 | Texthawk2: visual instruction data. |
| LAION-GPT4V | Texthawk2 | Texthawk2: vision–language instruction data. |
| LLaVAR | Texthawk2 | Texthawk2: text-rich visual SFT. |
| KVQA | Texthawk2 | Texthawk2: knowledge VQA SFT. |
| ViQuAE | Texthawk2 | Texthawk2: VQA FT. |
| Geo170K | Texthawk2 | Texthawk2: geographic visual SFT. |
| HME100K | Texthawk2 | Texthawk2: handwritten math expression FT. |
| UniMER-1M | Texthawk2 | Texthawk2: math equation recognition FT. |
| FUNSD | Texthawk2 | Texthawk2: form understanding FT. |
| XFUND | Texthawk2 | Texthawk2: multilingual form understanding FT. |
| SROIE | Texthawk2 | Texthawk2: receipt FT. |
| POIE | Texthawk2 | Texthawk2: invoice/product FT. |
| ST-VQA | Texthawk2 | Texthawk2: scene-text VQA FT. |
| ESTVQA | Texthawk2 | Texthawk2: scene-text VQA FT. |
| PaperPDF† | PDF-WuKong | PDF-WuKong: 1.1M-pair SFT corpus from scientific PDFs (introduced). |
| ScienceQA | InstructDoc | InstructDoc: held-in multimodal science instruction tuning. |
| ColPali synthetic | SV-RAG | SV-RAG: 45,940 synthetic question-page pairs (government, healthcare, AI, energy) for Col-retrieval module FT. |
| DocMatix-IR | SV-RAG | SV-RAG: ColPali-augmented IR data, GPT-4o-filtered, used for Col-retrieval training. |
| PFLDocVQA | SV-RAG | SV-RAG: federated DocVQA pairs, GPT-4o-filtered, used for Col-retrieval training. |
| SciQAG | MM-R5 | MM-R5: in 73,843-QA SFT mix from MMDocIR training set. |
| MM-R5 Reasoning Set† (7,200 CoT) | MM-R5 | MM-R5: 7,200 GPT-4o-annotated reasoning-chain samples used for SFT; ~3,000 resolution-balanced samples used for GRPO RL. |
| ViDoSeek | VRAG-RL | VRAG-RL: SFT + GRPO RL training (trajectory collection over ViDoSeek corpus). |
| Re-annotated VRAG-RL trajectories† | VRAG-RL | VRAG-RL: self-collected SFT trajectories from Qwen-VLmax over SlideVQA, ViDoSeek, MMLongBench-Doc; remains the primary SFT corpus. |
| DocSeeker ALR-CoT† (13,986) | DocSeeker | DocSeeker: 13,986 SFT samples distilled from Gemini-2.5-Flash with Analysis–Localization–Reasoning chain-of-thought structure. |

> Net: ~70 distinct fine-tuning-stage datasets across 27 papers (vs. 64 across 22 in v1). The 10 new papers introduce 5 new training-data artefacts (SV-RAG's ColPali synthetic mix, DocMatix-IR usage, PFLDocVQA usage, MM-R5's 7,200-sample reasoning set, VRAG-RL's trajectory pool, DocSeeker's ALR-CoT set) and trigger four corrections to the v1 mapping (DREAM also fine-tunes on DUDE; SlideVQA, MP-DocVQA, DUDE each gain four more FT users; CUAD and LongDocURL each gain a new FT user).

---

## Table 3 — Benchmarking Datasets

| Dataset | Papers | Usage per paper |
|---|---|---|
| MP-DocVQA | mPLUG-DocOwl2; Hi-VT5†; RAG-DocVQA; LayTokenLLM; Self-Attention Scoring; GRAM; RM-T5; Arctic-TILT; AVIR; DocR1; Doc-V*; DocSLM; M3DocRAG; PDF-WuKong; Leopard; CREAM; DFVC; DREAM; URaG; DocSeeker | Standard multi-page VQA evaluation; Hi-VT5 introduces the dataset (46K Qs, 5,928 docs). Self-Attention Scoring evaluates on the extended version (up to 793 pages). DREAM/URaG/DocSeeker: cross-page eval on the standard test split. |
| DocVQA | mPLUG-DocOwl2; Leopard; LayTokenLLM; Self-Attention Scoring; Hi-VT5; Arctic-TILT; DocR1; CREAM; PDF-WuKong; MACT; Texthawk2; CoR; DocReact; Docopilot; SV-RAG | Single-page document VQA evaluation across many papers. SV-RAG: 5,349 SP + 5,187 MP QA pairs from the validation split. |
| DUDE | Leopard; RAG-DocVQA; LayTokenLLM; GRAM; Arctic-TILT; AVIR; DocR1; Doc-V*; DocSLM; CREAM; PDF-WuKong; MultiDocFusion; MACT; M3DocRAG; VDocRAG; MLDocRAG; DREAM; SV-RAG; URaG; DocSeeker | Multi-domain multi-page document understanding evaluation. SV-RAG: 6,307 QA pairs from the validation split. URaG/DocSeeker: cross-page evaluation. |
| MMLongBench-Doc | MoLoRAG; SimpleDoc; DocReact; MDocAgent; DocAgent; MLDocRAG; MHier-RAG; MACT; DocLens; M2RAG; DocDancer; M3DocRAG; Doc-V*; DocSLM; Arctic-TILT; CoR; Docopilot; DMAP; DREAM; MM-Doc-R1; SV-RAG; SLEUTH; VRAG-RL; URaG; DocSeeker | Long-context multimodal document QA (135 docs / 1,082 questions, ~47.5 pages avg). Arctic-TILT evaluates on documents up to 400 pages. Doc-V* up to 468 pages. **25 of 46 papers** — the de-facto benchmark for the field. |
| LongDocURL | MoLoRAG; SimpleDoc; MDocAgent; MLDocRAG; MHier-RAG; DocLens; M2RAG; Doc-V*; CoR; DocDancer; DMAP; SLEUTH; URaG; DocSeeker | Comprehensive long multimodal document evaluation (396 docs / 2,325 Qs / 85.6 pages avg). |
| SlideVQA | DocReact; MACT; Arctic-TILT; AVIR; Leopard; ViDoRAG; MLDocRAG; MHier-RAG; M2RAG; Doc-V*; DocR1; VDocRAG; MDocAgent; SV-RAG; VRAG-RL; URaG; DocSeeker | Slide-presentation QA evaluation. ViDoRAG refines this into the ViDoSeek benchmark. |
| ChartQA | VDocRAG (zero-shot); MACT; Texthawk2; DocR1; PDF-WuKong; CoR; Docopilot | Chart QA evaluation. |
| InfographicVQA | Hi-VT5; Arctic-TILT; RAG-DocVQA; MACT; DocR1; CREAM; PDF-WuKong; M3DocRAG; VDocRAG; Texthawk2; InstructDoc; Docopilot | Infographic VQA evaluation. |
| VisualMRC | Hi-VT5; MACT; DocR1; CREAM; InstructDoc | Visual reading comprehension evaluation. |
| TextVQA | Leopard; DocR1; Texthawk2 | Scene-text VQA evaluation. |
| TabFact | DocR1; Texthawk2; InstructDoc | Table-fact verification evaluation. |
| WikiTableQuestions | DocR1; Texthawk2 | Table QA evaluation. |
| MultiHiertt | Leopard; DocR1 | Multi-page hierarchical-table evaluation. |
| MultiChartQA | Leopard; DocR1 | Multi-chart evaluation. |
| FUNSD | LayTokenLLM; InstructDoc | Form-entity extraction held-out evaluation (50 images). |
| CORD | LayTokenLLM; InstructDoc | Receipt-entity extraction held-out evaluation (100 images). |
| SIBR | LayTokenLLM | Real-world VIE evaluation set (400 images). |
| TAT-DQA | Arctic-TILT | Table-aware document QA evaluation. |
| PaperTab | MoLoRAG; SimpleDoc; MDocAgent; DMAP; SLEUTH; VisDoMRAG | Tabular QA from academic papers (393 Qs / 307 docs). VisDoMRAG: included as a sub-task of VisDoMBench. |
| FetaTab | MoLoRAG; SimpleDoc; MDocAgent; DMAP; SLEUTH; VisDoMRAG | Table QA evaluation (1,016 Qs / 871 docs). VisDoMRAG: included as a sub-task of VisDoMBench. |
| PaperText | MDocAgent; DMAP; SLEUTH | Document text understanding evaluation. |
| M3DocVQA† | M3DocRAG | Open-domain multi-doc VQA introduced by M3DocRAG (2,441 Qs / 3,368 PDFs / 41K pages). |
| ViDoSeek† | ViDoRAG; VRAG-RL | Newly introduced visually-rich document retrieval/QA benchmark (~1.2K Qs / 6K images), refined from SlideVQA. VRAG-RL: extraction + logic-reasoning subsets. |
| OpenDocVQA† | VDocRAG | Newly introduced unified open-domain DocVQA benchmark. |
| InstructDoc† | InstructDoc | InstructDoc: 30-source held-out zero-shot evaluation. |
| DocBench | DocAgent; DocDancer; M2RAG | Multi-page evaluation (229 docs / ~1,082–1,102 Qs). |
| HotpotQA | KGP | 500-question multi-document QA evaluation. |
| IIRC | KGP | 477-question multi-document QA evaluation. |
| 2WikiMQA | KGP | 500-question multi-document QA evaluation. |
| MuSiQue | KGP | 500-question multi-document QA evaluation. |
| PDFTriage | KGP | Structural document QA evaluation. |
| KleisterCharity | Arctic-TILT | Financial-document evaluation. |
| Kleister NDA | Arctic-TILT | Legal-document evaluation. |
| VQA-CD | Arctic-TILT | Invoices and purchase-order evaluation. |
| PubMed-Lay | Arctic-TILT | Scientific document summarisation evaluation. |
| ArXivLay | Arctic-TILT | Scientific paper summarisation evaluation. |
| MMMU | Leopard | Multimodal multitask benchmark. |
| MathVista | Leopard; MACT | Mathematical visual reasoning evaluation. |
| ScienceQA | Leopard; MACT | Multimodal science QA evaluation. |
| MIRB | Leopard | Multimodal information retrieval benchmark. |
| MiBench | Leopard | Multimodal benchmark. |
| CharXiv | MACT | Chart-based benchmark. |
| TableVQA-Bench | MACT | Table VQA benchmark. |
| TableBench | MACT | Table reasoning benchmark. |
| RealWorldQA | MACT | Real-world QA benchmark. |
| MathVision | MACT | Mathematical reasoning benchmark. |
| MathVerse | MACT | Mathematical reasoning benchmark. |
| FinRAGBench-V | DocLens | Financial-domain multimodal RAG evaluation with visual citations. |
| MM-NIAH | Docopilot | Long-form PDF understanding ("needle-in-a-haystack") evaluation. |
| NewsVideoQA | DocSLM | Generalisation evaluation for text-rich video QA (qualitative). |
| DocHieNet | MultiDocFusion | Document-hierarchy parsing evaluation. |
| HRDH | MultiDocFusion | Academic-document hierarchy evaluation. |
| MPVQA | MultiDocFusion | Multi-page document QA evaluation. |
| CUAD | MultiDocFusion | Contract understanding evaluation. |
| MOAMOB | MultiDocFusion | Complex document QA evaluation. |
| ArxivFullQA | DocR1 | Scientific paper QA evaluation. |
| TATDoc | DocR1 | Table-aware document reasoning evaluation. |
| OCRBench | Texthawk2 | OCR capability benchmark. |
| RefCOCO / RefCOCO+ / RefCOCOg | Texthawk2 | Referring expression comprehension benchmarks. |
| PaperPDF† | PDF-WuKong | Long-PDF QA test split (introduced alongside the training corpus). |
| MMDocIR | MM-R5 | Multi-modal document retrieval benchmark (10 domains, 313 docs avg 65.1 pages, 1,658 questions). MM-R5's primary benchmark. |
| VisR-Bench† | SV-RAG | SV-RAG: introduced 226-document / 471-QA web-crawled visually-rich benchmark across 9 domains (magazines, flyers, newsletters, manuals, presentations). |
| VisDoMBench† | VisDoMRAG | VisDoMRAG: introduced 2,271-QA multi-document benchmark composed of PaperTab, FetaTab, SciGraphQA, SPIQA, SlideVQA — first multi-doc multi-modal RAG benchmark. |
| SciGraphQA | VisDoMRAG | VisDoMRAG: chart-based QA sub-task of VisDoMBench. |
| SPIQA | VisDoMRAG | VisDoMRAG: chart-and-table scientific QA sub-task of VisDoMBench. |

> Net: 60+ distinct benchmarking datasets across 46 papers. MMLongBench-Doc now appears in **25/46 papers (54%)** — clearly the central long-document benchmark; MP-DocVQA appears in 20/46 (43%); DUDE in 20/46; SlideVQA in 17/46; LongDocURL in 14/46. SlideVQA's rise (from 13/36 to 17/46) is driven by the new papers (SV-RAG, VRAG-RL, URaG, DocSeeker), all of which evaluate on it.

---

## Notes

- **MMLongBench disambiguation.** "MMLongBench" in DMAP, MM-Doc-R1, VRAG-RL, SLEUTH, SV-RAG, and DREAM all cite Ma et al. 2024 — i.e., MMLongBench-Doc, the 47-page-average long-document benchmark. The 2025 "MMLongBench" (Wang et al., NeurIPS-spotlight, 13,331 examples across 5 categories with 8K–128K context lengths) does not appear in any of the 46 papers' usage — only in the survey's own dataset metadata.
- **EviBench training** refers to the unified multi-page training pool that DocR1 builds for its Evidence-Page-Guided GRPO procedure; component datasets are listed individually.
- **EviGRPO** is the GRPO variant in DocSeeker; uses MP-DocVQA + DUDE pool with multi-reward function (format / localization / answer).
- **SPO** (Similarity-based Policy Optimisation) is MM-Doc-R1's GRPO replacement; uses semantic-similarity-weighted baselines.
- **DocStruct4M reuse** — mPLUG-DocOwl2's continued pretraining and SFT phases overlap on `DocStruct4M`; listed under both Pretraining and Fine-Tuning for that paper.
- **Training-free papers** (orchestration only, no parameter updates) — DocReact, MDocAgent, DocAgent, MLDocRAG, MHier-RAG, M3DocRAG, ViDoRAG, SimpleDoc, M2RAG, DMAP, SLEUTH, VisDoMRAG — appear only in the Benchmarking table. MoLoRAG, DocDancer, DFVC, DocR1, CoR, MM-Doc-R1, MM-R5, VRAG-RL, DocSeeker, SV-RAG, URaG, DREAM are *training-light* (adapter / LoRA / RL on top of frozen backbones) — appear in Fine-Tuning + Benchmarking.
- **RL / preference-learning training datasets** — CoR (DPO with 5K pairs); DocR1 (GRPO over EviBench); DocSeeker (EviGRPO over MP-DocVQA + DUDE); MM-R5 (GRPO over 3K resolution-balanced samples from MMDocIR training); MM-Doc-R1 (Similarity-based Policy Optimisation over LongDocURL); VRAG-RL (GRPO over SlideVQA + ViDoSeek + MMLongBench-Doc trajectories). The RL footprint has roughly doubled from v1 (CoR, DocR1) to v2 (six RL papers, four of which arrived in the new batch).
- **Texthawk2 and Leopard** still dominate training-data inventories — none of the 10 new papers introduces a comparably large training corpus.
- **Newly introduced benchmarks among the 10 new papers**: VisR-Bench (SV-RAG), VisDoMBench (VisDoMRAG). MM-Doc-R1 introduces SPO as an algorithmic contribution rather than a dataset.
