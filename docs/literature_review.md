# Literature review closest to the project

Research-only review, checked 11 October 2026. The manuscript and project code were not changed. No training, prediction, or test-record analysis was performed.

## Main finding and evidence limits

The project is a **historical, country-event-level classification study with a separate live-monitoring prototype**. It predicts training-defined affected-population ranges and hazard-relative impact-priority proxies. It does not yet demonstrate real-time impact prediction, expert-validated priority, or operational benefit.

Zhang et al. was retrieved and reviewed first as requested. Five of the eleven requested papers had accessible publisher full text reviewed: Zhang, Gerolimos, Pidurkar, Jones and Aykut. Sun's publisher page provides only an abstract and references, not full text. Albahri, Akshay and Delforge were reviewed at abstract level using indexed abstracts. Chen and Liu could be verified bibliographically but neither an actual abstract nor full text could be retrieved here. Two additional close studies were assessed: Talan et al. in full text and Kalaivani et al. at indexed-abstract level. These access distinctions are essential: **bibliographic verification is not full-text verification**.

ScienceDirect, its article API and linking hub returned access denials in this environment. Legal repository/preprint alternatives were investigated: NTU for Chen, LSU/SSRN for Liu, Research Square/UCL for Delforge, and publisher/author locations for Sun. These did not yield full text. Open-access status in an index did not imply successful access here. No paywall was circumvented. A guessed author PDF URL returned a university webpage and was rejected as a paper source. An arXiv search for Liu returned unrelated papers and none was used. Unretrieved details are explicitly unknown below.

This is a focused, documented search, not an exhaustive systematic review. Numerical results from different targets, populations, units and evaluation protocols are not comparable with our validation scores. “Reported” means stated by the publication, not independently reproduced. No publication-status claim here guarantees paper quality or constitutes a comprehensive retraction audit.

## Our actual task: comparison anchor

Repository evidence: [dataset construction](../../realtime-disaster-ml/ml/preprocessing/build_training_dataset.py), [training and experiment code](../../realtime-disaster-ml/ml/training/train_models.py), [feature report](../../realtime-disaster-ml/docs/emdat_feature_report.txt), and [saved original validation results](../../realtime-disaster-ml/docs/model_evaluation/validation_metrics.json).

- EM-DAT snapshot named `public_emdat_2026-09-09.xlsx`; preprocessing report retains 8,774 country-event rows from 2000–2025 for Earthquake, Flood, Landslide, Storm, Volcanic activity and Wildfire. A country-event record should not automatically be called a unique global disaster.
- Nine predictors: disaster_type, disaster_subtype, iso, region, subregion, season, start_month, magnitude, magnitude_available. Outcomes are excluded from the model inputs. Geography is a location identifier, not a measured social-vulnerability index.
- `affected_range`: low/medium/high using train-only P25/P75 of Total Affected. `impact_priority`: hazard-specific train medians/P75 of Total Deaths and Total Affected; both outcomes must be known. Volcanic activity lacks the minimum 30 complete training observations for priority thresholds. Priority is relative to hazard category, not a globally comparable official response score.
- Temporal development: train 2000–2019; validation 2020–2022; reserved test 2023–2025. Label thresholds and pipeline transformations fit using training only. Test metrics are pending; no final model is selected by this review.
- Original Random Forest: 300 trees, balanced class weights, random_state 42, n_jobs -1, unrestricted depth and leaf size 1. Train/validation labeled samples: affected 5,432/979; priority 3,961/681. Accuracy/macro F1/weighted F1/high recall: affected 0.519918/0.503741/0.517245/0.378151; priority 0.472834/0.463156/0.473155/0.403941. These are project validation results, not literature results.
- Existing validation-only comparisons vary leaf size, depth and class weighting individually. Their purpose is to assess regularization and class trade-offs while keeping features/preprocessing/splits fixed. The original remains the strongest macro-F1 reference for both targets, with some alternatives trading high-class recall against other metrics; this is not evidence of superiority over published models.
- FastAPI ingests GDACS RSS events into SQLite and serves event lists/details/sync; React displays/filter events on a map. Pipeline inference is not integrated. GDACS-to-EM-DAT categories, units, complete feature construction and prediction-time availability are not fully validated. A different input source and event-selection mechanism means domain shift, not a direct interchangeable dataset.
- The report's “no duplicate DisNo.” check does not verify absence of cross-country records of the same underlying disaster across temporal boundaries. Event grouping and timing of historical feature updates remain audit questions, not completed leakage checks. Magnitude is missing in 65% of retained rows and mixes hazard-specific physical units. Retrospective magnitude/availability may differ from what is known at alert time.

The source code takes precedence over prose reports: outcome columns remain in the processed table but are excluded through the explicit nine-feature list; numeric missing values are median-imputed within the training-fitted pipeline. Excluding unknown labels avoids inventing outcomes, but does not establish unbiased sampling (see Jones).

## Bibliographic verification register

Titles/authors/venues below were checked against Crossref DOI records and, where retrieved, publisher pages. Journal articles are published records; Procedia is a published conference-proceedings article. Exact acceptance dates were not obtained for blocked publisher papers. Author spellings below follow deposited records, without inventing expanded initials.

- **[Z]** Tianjie Zhang; Donglei Wang; Yang Lu. **Machine learning-enabled regional multi-hazards risk assessment considering social vulnerability**. Scientific Reports, 13; 1; 13405; registered publication date 2023-8-17. [DOI](https://doi.org/10.1038/s41598-023-40159-9); [metadata](https://api.crossref.org/works/10.1038/s41598-023-40159-9).
- **[1]** Wenjuan Sun; Paolo Bocchini; Brian D. Davison. **Applications of artificial intelligence for disaster management**. Natural Hazards, 103; 3; 2631-2689; registered publication date 2020-7-3. [DOI](https://doi.org/10.1007/s11069-020-04124-3); [metadata](https://api.crossref.org/works/10.1007/s11069-020-04124-3).
- **[2]** A.S. Albahri; Yahya Layth Khaleel; Mustafa Abdulfattah Habeeb; Reem D. Ismael; Qabas A. Hameed; Muhammet Deveci; Raad Z. Homod; O.S. Albahri; A.H. Alamoodi; Laith Alzubaidi. **A systematic review of trustworthy artificial intelligence applications in natural disasters**. Computers and Electrical Engineering, 118; 109409; 109409; registered publication date 2024-9. [DOI](https://doi.org/10.1016/j.compeleceng.2024.109409); [metadata](https://api.crossref.org/works/10.1016/j.compeleceng.2024.109409).
- **[3]** Weiyi Chen; Limao Zhang. **An automated machine learning approach for earthquake casualty rate and economic loss prediction**. Reliability Engineering & System Safety, 225; 108645; 108645; registered publication date 2022-9. [DOI](https://doi.org/10.1016/j.ress.2022.108645); [metadata](https://api.crossref.org/works/10.1016/j.ress.2022.108645).
- **[4]** Akshay S; Meghana Jithuri; Sinchana O S; Mruduhula R Vishwakarma; Aradhya M R. **Predicting Severity of Flood Damage: A Smart Approach to Disaster Management**. Procedia Computer Science, 269; 1424-1433; registered publication date 2025. [DOI](https://doi.org/10.1016/j.procs.2025.09.084); [metadata](https://api.crossref.org/works/10.1016/j.procs.2025.09.084).
- **[5]** Zhewei Liu; Natalie Coleman; Flavia Ioana Patrascu; Kai Yin; Xiangpeng Li; Ali Mostafavi. **Artificial intelligence for flood risk management: A comprehensive state-of-the-art review and future directions**. International Journal of Disaster Risk Reduction, 117; 105110; 105110; registered publication date 2025-2. [DOI](https://doi.org/10.1016/j.ijdrr.2024.105110); [metadata](https://api.crossref.org/works/10.1016/j.ijdrr.2024.105110).
- **[6]** Nikitas Gerolimos; Vasileios Alevizos; Sabrina Edralin; Clark Xu; Georgios Priniotakis; George A. Papakostas; Zongliang Yue. **Autonomous Decision-Making Enhancing Natural Disaster Management through Open World Machine Learning: A Systematic Review**. Human-Centric Intelligent Systems, 5; 2; 269-284; registered publication date 2025-4-25. [DOI](https://doi.org/10.1007/s44230-025-00098-2); [metadata](https://api.crossref.org/works/10.1007/s44230-025-00098-2).
- **[7]** Payal Pidurkar; Roshni. S. Khedgaonkar; Sundeep Reddy Mallu; Nikhil Mangrulkar; Shreyas Rajendra Hole; Sagarkumar S. Badhiye. **Multi hazard risk assessment through geospatial analysis and machine learning**. Discover Global Society, 4; 1; 181; registered publication date 2026-8-12. [DOI](https://doi.org/10.1007/s44282-026-00540-1); [metadata](https://api.crossref.org/works/10.1007/s44282-026-00540-1).
- **[8]** Damien Delforge; Valentin Wathelet; Regina Below; Cinzia Lanfredi Sofia; Margo Tonnelier; Joris A.F. van Loenhout; Niko Speybroeck. **EM-DAT: the Emergency Events Database**. International Journal of Disaster Risk Reduction, 124; 105509; 105509; registered publication date 2025-6. [DOI](https://doi.org/10.1016/j.ijdrr.2025.105509); [metadata](https://api.crossref.org/works/10.1016/j.ijdrr.2025.105509).
- **[9]** Rebecca Louise Jones; Debarati Guha-Sapir; Sandy Tubeuf. **Human and economic impacts of natural disasters: can we trust the global data?**. Scientific Data, 9; 1; 572; registered publication date 2022-9-16. [DOI](https://doi.org/10.1038/s41597-022-01667-x); [metadata](https://api.crossref.org/works/10.1038/s41597-022-01667-x).
- **[10]** Ahmet Aykut; Ertuğ Günsoy; Cem Yıldırım. **Using GDACS to anticipate clinical and operational burden after earthquakes: A global event-level analysis (2020–2024)**. PLOS One, 21; 1; e0339018; registered publication date 2026-1-13. [DOI](https://doi.org/10.1371/journal.pone.0339018); [metadata](https://api.crossref.org/works/10.1371/journal.pone.0339018).
- **[A]** Tarık Talan; Teyfik Giray; Adem Korkmaz; Selahattin Kosunalp; Vedat Tumen; Erdal Akin. **Predicting and Explaining Economic Losses from Natural Disasters: A Post-Hoc Machine Learning and Explainable AI Approach**. Applied Sciences, 16; 19; 9497; registered publication date 2026-9-24. [DOI](https://doi.org/10.3390/app16199497); [metadata](https://api.crossref.org/works/10.3390/app16199497).
- **[B]** A. Kalaivani; V. Asha; J. Sathya; Preetham KM; Rohini Babu T B; Sakshi Saha. **A Machine Learning – Based Framework for Multi-Hazard Natural Disaster Severity Prediction**. 2026 5th International Conference on Sentiment Analysis and Deep Learning (ICSADL), 587-592; registered publication date 2026-2-18. [DOI](https://doi.org/10.1109/icsadl67539.2026.11452016); [metadata](https://api.crossref.org/works/10.1109/icsadl67539.2026.11452016).
- **[C]** Boyi Li; Adu Gong; Tingting Zeng; Wenxuan Bao; Can Xu; Zhiqing Huang. **A Zoning Earthquake Casualty Prediction Model Based on Machine Learning**. Remote Sensing, 14; 1; 30; registered publication date 2021-12-22. [DOI](https://doi.org/10.3390/rs14010030); [metadata](https://api.crossref.org/works/10.3390/rs14010030).

Date qualifications: Sun appeared online 3 July 2020, journal issue September 2020. Gerolimos appeared online 25 April 2025. Pidurkar appeared online **12 August 2026** (publisher issue metadata says December 2026); it is already published online, not an unpublished proposal. Jones appeared 16 September 2022. Aykut appeared 13 January 2026; received 10 September 2025 and accepted 1 December 2025. Liu's journal record is volume 117, February **2025**; a 2024-looking DOI suffix is not a reason to change its citation year. Its SSRN record is separately [10.2139/ssrn.5008577](https://doi.org/10.2139/ssrn.5008577). Delforge's linked preprints are [2023 v1](https://doi.org/10.21203/rs.3.rs-3807553/v1) and [2025 v2](https://doi.org/10.21203/rs.3.rs-3807553/v2), distinct from the 2025 journal paper. Talan appeared 24 September 2026. Li is in the 2022 Remote Sensing volume, published online 22 December 2021; state both dates when necessary.

## Literature-comparison matrix

FT = publisher full text reviewed; AB = abstract only; MD = metadata only. Unknown means not verifiable from material accessed, not necessarily absent from the paper. Detailed profiles immediately below specify legitimate use and limitations.

| ID | Paper / access | Data and scope | Inputs → outputs | Models / evaluation | Reported results / caveat |
|---|---|---|---|---|---|
| Z | Zhang 2023 (primary) / FT | Idaho OEM hazard data + ACS; 84 watersheds/144 cities; floods, wildfire, seismic | Exposure/infrastructure/PGA and 27 social indicators → three-level hazard and social-vulnerability classes, regional composite map | NB/KNN/LR/RF/K-means; 80/20 split, three repetitions; not a temporal design | RF F1: flood .840, wildfire .854, seismic .908, social vulnerability .820; differing constructed spatial targets |
| 1 | Sun 2020 / AB | Review across disaster-management phases; dataset list/geographies not extracted | Diverse AI applications → management support; no single prediction target | Review; no common experiment/split established from abstract | No pooled predictive score verified |
| 2 | Albahri 2024 / AB | Review: 981 retrieved, 108 included, search through Feb 2024; multiple disasters | Trustworthiness taxonomy; XAI/data fusion/ML/DL/fuzzy/MCDM | Systematic synthesis; individual study validation not retrieved | Review counts, not classifier accuracy |
| 3 | Chen & Zhang 2022 / MD | Earthquakes; exact dataset, count and geography UNKNOWN | Title verifies casualty-rate and economic-loss prediction task; inputs UNKNOWN | AutoML approach named in title; component learners/splits UNKNOWN | No quantitative result verified |
| 4 | Akshay et al. 2025 / AB | Historical meteorological/hydrological records; floods; dataset name/size/geography UNKNOWN | Abstract discusses magnitude, economic loss, casualties, displacement; feature/label roles unresolved | Linear regression/RF/XGBoost/SVM; split UNKNOWN | Abstract favors SVM severity classification; numerical scores UNKNOWN |
| 5 | Liu et al. 2025 / MD | Flood-risk-management review; search corpus/geography UNKNOWN | Inputs, reviewed outputs, taxonomy details UNKNOWN | Review title verified; methods/selection strategy not retrieved | No result or study count verified |
| 6 | Gerolimos et al. 2025 / FT | Earthquake/flood/wildfire/hurricane framework; tables name Dataset A/B/C or 1/2/3, not identifiable datasets | Seismic/hydrological/fuel/weather/social criteria → open-world hazard/risk/response concepts | Fisher information + EM + open-world classifier conceptual framework; reproducible split not identified | Table 5 accuracy EQ94.2%, flood93.6%, wildfire94.2%, hurricane92.3%; unsupported as comparable benchmark |
| 7 | Pidurkar et al. 2026 / FT | Ambazari, Nagpur, India; 30m grids; SRTM/Sentinel-2/OSM/weather/historical landslide layers | TWI/elevation/NDVI/water/impervious/roads/rain/wind → four grid risk levels | RF100 trees; random stratified80/20; mentions GBM/baselines but no equivalent detailed comparison | Accuracy82.6%, precision.81, recall.84, F1.82, AUC.84; independent label construction/sample count unclear |
| 8 | Delforge et al. 2025 / AB | Global EM-DAT, natural+technological hazards; 1900–present, systematic recording since1988 | Country-disaster standardized entries → human/economic impact database | Manual monitoring/processing and validation description; not a predictive ML benchmark | Abstract says >26,000 entries; not our snapshot size |
| 9 | Jones et al. 2022 / FT | Global EM-DAT natural hazards1990–2020 + top20 cited empirical studies | Type/year/income and severity → missingness indicators, not severity predictions | Logistic missingness models, probit/fixed-effect robustness; no predictive train/test design | Missing reconstruction96.2%, insured88.1%, totaldamage41.5%; predictors of missingness vary |
| 10 | Aykut et al. 2026 / FT | GDACS Orange/Red earthquakes2020–2024 + WHO/OCHA/national outcome reports;95 raw→85 representatives | GDACS score, magnitude/depth/exposure metadata → mortality association and field-hospital deployment | Spearman/logistic/ridge; Firth supplementary; sensitivity analyses; no independent holdout described | Results ρ=.520; deployment9/17 Red,0/68 Orange; apparent AUC.981, Brier.034 |
| A | Talan et al. 2026 (added) / FT | Global EM-DAT1975–2025; 5,051 rows/4,519 events; eight natural-hazard categories | 21 predictors incl recorded deaths/affected and end dates → adjusted economic loss (post hoc regression) | RF/XGB/LGBM/CatBoost+baselines; event-grouped holdout/CV, strict temporal, region/hazard holdouts | Tuned CatBoost grouped RMSLE1.890, logR².491; temporal RMSLE1.791, logR².464 |
| B | Kalaivani et al. 2026 (added) / AB | Billion Dollar Weather and Climate Disasters (BDWCD); exact filtered sample/geography not full-text verified | Type/duration/affected population/climate region → disaster severity and deaths | RF/gradient boosting/voting regression; validation protocol UNKNOWN | Abstract reports99.37% accuracy; definition/test independence/leakage safety UNKNOWN |
| C | Li et al. 2022 (added) / FT sections reviewed | Mainland China earthquake cases1950–2017 + faults/WorldPop;143 selected,113 training/30 heldout | Magnitude/population density/fault density/GDP → earthquake death count and order-of-magnitude classes | Zoning SVR + RF/BP/LR comparisons; randomized zone-stratified holdout, CV/grid search | Z-SVR reported heldout mean zone RMSE35.61, MAE17.39; excludes >1000 deaths, limiting catastrophe scope |

## Detailed paper reviews and allowable citation uses

### [Z] Zhang, Wang and Lu — primary paper, reviewed first

**Source/access:** [Publisher full text](https://www.nature.com/articles/s41598-023-40159-9), especially Methodology/Dataset/Data processing/Machine learning, Results/Evaluation of machine learning models, Conclusions. Publisher HTML and DOI metadata retained locally. This is the published article, not an author manuscript.

**Dataset, hazards and geography:** Idaho OEM supplies flooding, wildfire and seismic hazard information; ACS supplies social indicators. The framework covers 84 watersheds and 144 cities/county subdivisions in Idaho, USA. Those counts describe spatial units, not a global disaster-event sample. Cities in the same watershed are assumed to share hazard potential.

**Inputs:** Flood risk population, essential facilities in floodplains, dams and levees; wildfire watershed population and wildland–urban-interface structures; seismic high-PGA proportion, population and essential facilities. Social vulnerability uses 27 indicators across demographics, family structure, education/language, housing/transport, employment/economics. Eight PCA components account for 70.25% variance according to Results/Social vulnerability. These explicit vulnerability inputs are absent from our nine-feature model.

**Targets:** Constructed low/medium/high hazard scores and social-vulnerability levels. Hazard scores derive from standardized predictor quantiles; social vulnerability derives from PCA and directional/additive SoVI construction. Hazard and vulnerability levels are combined into a regional multi-hazards index/map. This is regional relative susceptibility/risk classification, not predicting recorded affected-person counts or death-based event priorities. Its labels are constructed from indicators that also enter the modeling process; performance must be read as reproduction/classification of those constructed levels, not external validation against realized future impact.

**Models and evaluation:** NB, KNN, logistic regression, Random Forest, K-means. Methods call the 80/20 partition “training/validation” and later “training/test”; three runs are reported. RF estimator count 7, KNN neighbors 3 and LR penalty None are described as selected settings. The paper does not document a forward temporal split or clearly separate hyperparameter selection from the reported heldout evaluation. Train-only fitting of scaling/PCA/quantile labels is not clearly established in the text; do not adopt this as proof of a leakage-safe protocol.

**Results:** Reported average RF precision/recall/F1: flooding .842/.836/.840; wildfire .867/.868/.854; seismic .928/.927/.908; social vulnerability .797/.789/.820 (Results/Evaluation). These are neither our macro F1 nor a like-for-like benchmark. There are internal inconsistencies: the abstract associates 34.12% with multi-hazards risk and 15.88% with combined high risk; the body assigns 34.12% to most vulnerable cities and 15.29% to high multi-hazards plus high social vulnerability. The body says “all ... over 0.8” despite .797/.789 entries. Equation (8) duplicates FP in the accuracy denominator where FN would normally appear. These are reasons to cite methods and scope carefully, not reproduce ambiguous headline percentages.

**Limitations and relationship:** Small regional spatial units, watershed-level homogeneity assumptions, constructed labels, cross-hazard aggregation and limited external-transfer evidence. The authors suggest replication elsewhere and future updated-data use, rather than a completed global live-feed deployment. The paper legitimately supports: multiple hazard dimensions and social vulnerability can be combined in a regional ML/GIS assessment; RF was compared with other classifiers in this specific setting. It does **not** support: our system measures social vulnerability; our labels reproduce their RMHI; 2023–2025 generalization is verified; or our GDACS integration is complete. It is a **conceptual base**, not a direct experimental baseline or validation target. Our difference is global country-event historical human-impact classification, sparse event metadata, two separate proxy targets and chronological development, rather than Idaho geospatial risk mapping.

### [1] Sun, Bocchini and Davison (2020)

**Source/access:** [Publisher abstract and reference list](https://link.springer.com/article/10.1007/s11069-020-04124-3). The page explicitly labels the content a subscription preview. Full text/author manuscript was not obtained. No full-text-review claim is made.

**What is verified:** A review of AI applications in mitigation, preparedness, response and recovery, with examples of decision-support tools. Its abstract says most applications reviewed focus on response. This is a literature corpus, not a single common disaster dataset. Dataset names/counts, hazard/geographic coverage, exact predictor/target lists, algorithms, study-level validation/results and detailed limitations remain **unverified** from full text.

**Relation/difference/use:** Legitimate abstract-qualified background for the four disaster-management phases and the existence of AI decision support. Our work is one specific tabular classification study plus monitoring software, not a survey spanning all phases. The abstract mentions challenges but cannot support a specific claimed gap in global dual-target EM-DAT classification. Do not attribute an algorithm ranking, pooled accuracy or temporal-validation conclusion to this paper without full text.

### [2] Albahri et al. (2024)

**Source/access:** DOI [10.1016/j.compeleceng.2024.109409](https://doi.org/10.1016/j.compeleceng.2024.109409); [indexed abstract and access locations](https://api.openalex.org/works/https://doi.org/10.1016/j.compeleceng.2024.109409). Published article verified, actual full text not retrieved despite an indexed open-access record.

**Data, methods and findings:** Abstract reports searches in ScienceDirect, Scopus, IEEE Xplore and Web of Science; 981 records retrieved, 108 included, through February 2024. It covers trustworthiness in disaster management/risk/prediction and discusses XAI, fusion, mining, ML/DL, fuzzy logic and multicriteria methods. This is a review corpus, not 108 examples used to train one model. Individual datasets, hazards, geographic coverage, input features, targets, learners, validation strategies and scores are not verified here. The abstract mentions ethics, bias and consequences; its seven open issues cannot be enumerated accurately without the text.

**Relation/difference/use:** Abstract-level support for considering trustworthiness, bias and explanation in disaster AI. Our current forest is neither demonstrated trustworthy nor validated for allocation or autonomous action by citing this review. Background only; cannot serve as a benchmark or establish our narrow novelty claim. Full-text review remains necessary before detailed recommendations are attributed to it.

### [3] Chen and Zhang (2022)

**Source/access:** [DOI](https://doi.org/10.1016/j.ress.2022.108645); [Crossref record](https://api.crossref.org/works/10.1016/j.ress.2022.108645); legal [NTU handle discovered](https://hdl.handle.net/10356/225118). Publisher access failed; NTU handle/page returned 405. Bibliographic/title verification only; neither an actual abstract nor full text was obtained. No author manuscript has been reviewed.

**Verified versus unknown:** Weiyi Chen and Limao Zhang; Reliability Engineering & System Safety 225, 108645, 2022. The title establishes an automated-ML approach to earthquake casualty-rate and economic-loss prediction. Exact dataset/name/size, geographic scope, input variables, output definitions, AutoML library, component estimators, validation/test design, numerical metrics and limitations are **unknown in this review**. Do not infer EM-DAT, global scope or a particular ensemble from the title.

**Relation/difference/use:** A high-priority candidate for direct impact-prediction Related Work, because casualty and loss prediction are closer to our outcomes than susceptibility maps. Our verified task is multiclass, six-hazard, affected-range/relative-priority classification, not the title's earthquake casualty-rate/economic-loss targets. The difference in scope is supported by title and our code; precise methodological differences require the text. It may support only the existence of earthquake AutoML impact prediction at this stage. Hold detailed comparison and performance claims until full text is reviewed.

### [4] Akshay S et al. (2025)

**Source/access:** [DOI](https://doi.org/10.1016/j.procs.2025.09.084), [indexed abstract](https://api.openalex.org/works/https://doi.org/10.1016/j.procs.2025.09.084). Procedia Computer Science 269, 1424–1433 is a published proceedings article. Full text was not retrieved; the specific conference name is not verified here.

**Dataset/hazard/features/output:** Floods, historical meteorological and hydrological records according to the abstract. It discusses magnitude, economic loss, casualties and displacement, but the abstract does not resolve which are predictors versus outputs or exactly how severity is labeled. Dataset source/count, geographic scope, feature table and outcome availability are unknown. In particular, do **not** state that it uses EM-DAT merely because these terms resemble EM-DAT columns.

**Algorithms/evaluation/results:** Abstract names linear regression, RF, XGBoost and SVM and says SVM works best for classifying severity, while RF/XGBoost capture nonlinear relationships. No numerical result, split, tuning design or external evaluation is verified. Any full-text limitation or claim of leakage would be speculative.

**Relation/difference/use:** A close candidate for severity-classification work; our six-hazard, two human-impact-label approach differs from its stated flood-only scope. Supports an abstract-qualified statement that supervised methods have been explored for flood-damage severity. Does not support numerical comparisons, a claim that forests always win, or evidence of live inference. Needs full text before relying on dataset/label/split comparisons.

### [5] Liu et al. (2025)

**Source/access:** [Journal DOI](https://doi.org/10.1016/j.ijdrr.2024.105110), [LSU repository landing page](https://repository.lsu.edu/construction_management_pubs/655), [SSRN record](https://doi.org/10.2139/ssrn.5008577). Legal full-text locations were identified but retrieval failed. The journal abstract was not supplied by the accessed metadata records. **Metadata only**, not a full-text or abstract review.

**Verified:** Zhewei Liu, Natalie Coleman, Flavia Ioana Patrascu, Kai Yin, Xiangpeng Li and Ali Mostafavi; International Journal of Disaster Risk Reduction 117, 105110 (2025); flood-risk-management state-of-the-art review. Corpus/datasets, geographic scope, predictors/targets, algorithms, validation strategies, reported scores, review methodology and specific limitations are unknown from the material retrieved.

**Relation/difference/use:** Potential flood-risk background, not verified evidence about our own experimental design or performance. The title indicates flood-focused review versus our implemented six-hazard classification. Do not use it to claim a specific knowledge gap or future direction until full text is read. A repository's deposited version label does not establish that we read that version.

### [6] Gerolimos et al. (2025)

**Source/access:** [Publisher full text](https://link.springer.com/article/10.1007/s44230-025-00098-2) and linked [Tables 1–5](https://link.springer.com/article/10.1007/s44230-025-00098-2/tables/5). Sections 2–4 and all five tables inspected. Published full text, not an author manuscript.

**Dataset/hazards/geography:** Earthquake, flood, wildfire and hurricane cases with sensors/satellites/historical archives proposed as sources. Tables describe “Dataset 1/2/3” or “Dataset A/B/C” without identifiable data names, counts or sampling geography. No reusable data are provided: Data Availability says there are no additional data/materials beyond the manuscript. Do not turn these generic dataset labels into real dataset identities.

**Features and outputs:** Conceptual inputs include seismic activity/fault/stress; rainfall/drainage/soil; fuel moisture/vegetation/wind; hurricane wind/pressure/SST, plus population, infrastructure or evacuation criteria. Outputs span detection/classification, risk zones and response concepts, not our two EM-DAT human-impact labels. Fisher information and EM equations accompany an open-world classifier/adaptation proposal.

**Validation and results:** Tables report accuracies, sensitivity/specificity and responsiveness. Table 5 lists 94.2%, 93.6%, 94.2%, 92.3% accuracy for earthquake/flood/wildfire/hurricane. The text does not identify a reproducible training/evaluation partition, known-versus-unknown holdout protocol, concrete classifier specification, source for each dataset, or independent operational evaluation. Therefore these are **author-reported numbers with insufficient verification detail**, not defensible numerical comparators. The title calls it a systematic review, but the text combines conceptual formulas and empirical-sounding claims without a clearly documented search/screening or reproducible experimental protocol.

**Limitations and use:** Authors mention quality and complexity; reviewer concern is missing experimental provenance and evaluation detail. It can be cited cautiously as an open-world adaptation proposal; replace it with stronger direct evidence for any performance, autonomy or reliable unknown-hazard claim. Our system is fixed-label closed-set RF; unknown one-hot categories being ignored is not open-world learning. Neither a monitoring loop nor a saved model implements autonomous decision making. At most optional background; not evidence for our novelty or live capability.

### [7] Pidurkar et al. (2026)

**Source/access:** [Published full text](https://link.springer.com/article/10.1007/s44282-026-00540-1), Sections 4–7, especially 4.12/4.14 and Results/Table 2; online 12 August 2026, Discover Global Society 4, article 181.

**Data/hazards/geography:** Ambazari region, Nagpur, Maharashtra, India. 30 m × 30 m grids integrate SRTM DEM, Sentinel-2 NDVI, OSM water/roads/buildings/infrastructure, meteorological inputs and historical landslide susceptibility. The framework discusses flooding, landslides, heatwaves, drought and coastal/weather hazards; do not claim separately validated classifiers for each of those from its single combined-risk result. A precise total grid-observation count/date window is not established in the reviewed text.

**Inputs/output:** Water proximity/density, TWI, elevation, NDVI, impervious surface ratio, landslide susceptibility, coastal proximity, precipitation, wind and anthropogenic infrastructure. Output is spatial multi-hazard risk probability, converted into low/moderate/high/very-high at .25/.50/.75 thresholds. Section 4.14 does not clearly establish independently observed ground-truth label construction for the supervised classifier. It would be inappropriate to equate this map with measured disaster impact.

**Learners/evaluation/results:** RF 100 trees, Gini/bootstrap, automatic depth, square-root feature subsampling; another paragraph mentions GBM and simpler baselines without a comparable detailed result table. Random stratified 80/20 evaluation, rather than temporal or spatial-blocked evaluation. Reported accuracy82.6%, precision.81, recall.84, F1.82, ROC-AUC.84. Averaging conventions/independent label provenance need clarification before interpreting these scores. Adjacent spatial grids may share strong dependence; random-grid separation alone does not establish transfer to new areas. This is an evaluation concern, not a proven leakage finding.

**Relation/difference/use:** Supports the existence of local RF/GIS multi-hazard mapping. Our rows are country-events, targets historical human consequences, inputs sparse tabular metadata, and validation chronological. It does not prove that our nine features measure exposure/vulnerability, that our approach models cascading hazards or that our map is an ML risk map. It is regional background with methodological qualifications, not a direct performance benchmark. Data/code are available only upon request, and live remote sensing is future work in its conclusion.

### [8] Delforge et al. (2025)

**Source/access:** [Journal DOI](https://doi.org/10.1016/j.ijdrr.2025.105509), [indexed abstract](https://api.openalex.org/works/https://doi.org/10.1016/j.ijdrr.2025.105509), [EM-DAT publications page](https://www.emdat.be/publications/). The official publications page lists the DOI. Research Square v1/v2 and UCL alternatives were found, but no full article or author manuscript was retrieved. Abstract-only review.

**Dataset/scope/process:** Abstract describes global natural and technological disaster impacts from1900 onward, systematic recording since1988, country-specific standardized events and >26,000 entries at the paper's database state. It describes monitoring textual sources and manual processing, daily checks and thematic updates, acknowledges inconsistencies/bias and says tabular data are freely available for non-commercial use. It is a database-description paper, not an ML prediction experiment. Thus ML feature set, prediction target, algorithm, heldout split and predictive metrics are **not applicable to its stated purpose**, rather than missing classifier details.

**Relation/difference/use:** Essential dataset-provenance reference. The paper's historical database entry count must not replace our snapshot's actual extraction count. Country-disaster structure informs our unit of analysis and multi-country dependence concerns. Full text is needed before attributing exact schema/protocol details to this article. [Official current documentation](https://doc.emdat.be/docs/introduction/) is an independent primary source for present definitions; it does not substitute for a full-text review of Delforge. The project uses a filtered natural-hazard subset and constructs labels; the article describes the database and does not validate our thresholds or models.

### [9] Jones, Guha-Sapir and Tubeuf (2022)

**Source/access:** [Publisher full text](https://www.nature.com/articles/s41597-022-01667-x), Results/The state of missing data/Explaining missing data, Discussion and Methods. Published full text reviewed.

**Data/scope/input/output:** Global EM-DAT natural hazards1990–2020, including additional variables obtained from CRED beyond the freely downloadable data. Their actual total analytic event count is not asserted here without a clearly extracted denominator. Predictors are disaster type, occurrence year, country income classification and, for the damage-missingness model, log total deaths. Target is whether particular human/economic loss fields are missing. They also review the20 most cited empirical studies from421 eligible papers identified in a larger2,127 search set. This is neither a severity classifier nor a casualty forecaster.

**Methods/evaluation/results:** Missingness descriptions/patterns and logistic regression, with probit and conditional fixed-effect logit robustness; no train/validation/test predictive split. Reported missingness: reconstruction costs96.2%, insured damages88.1%, total estimated damages41.5%, human variables1.3–22.3%. Year, income and disaster type predict missingness. Some regressions omit cases due to perfect prediction, and missing-not-at-random dependence cannot be directly tested. Authors advocate study-specific diagnosis and sensitivity, not a universal imputation recipe.

**Citation caution:** The article's Fig.1 prose/body mistakenly describes “total deaths” as affected plus deaths. Do not copy that definition. Use [official EM-DAT human-impact definitions](https://doc.emdat.be/docs/data-structure-and-content/impact-variables/human/) for deaths/missing/affected components. Their inference that observations are not missing completely at random does not prove a specific mechanism in our newer snapshot; it motivates checking our own selection bias. Their missingness percentages are not our dataset's percentages.

**Relation/difference/use:** Strong support for acknowledging missing-outcome reporting bias and for not silently converting missing impacts into zero. Our nonmissing-target restriction is prudent against fabricated labels but may still select a biased subset. Train-only imputation protects split independence; it does not solve reporting bias. This is a central **dataset-quality reference**, not only broad disaster background. No predictive-accuracy comparison is legitimate.

### [10] Aykut, Günsoy and Yıldırım (2026)

**Source/access:** [Publisher full text](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0339018), Materials and methods/Event selection/Variables and linkage/Statistical analysis, Results, Limitations. Published OA text reviewed. The linked [Zenodo archive](https://doi.org/10.5281/zenodo.17036717) was identified; analyses were not independently rerun.

**Data/hazard/geography:** Global Orange/Red GDACS earthquakes2020–2024.95 raw alerts become85 country-bounded representatives:17 Red/68 Orange. Mortality is abstracted from WHO/OCHA, national authorities and consolidated media with stated source priority; deployment is an operational yes/no proxy. Exposure explicitly “within100km” is available for19 events, only1 Red. This is not an EM-DAT dataset.

**Predictors/outputs/algorithms:** Continuous GDACS score is the primary sole predictor for deployment models; magnitude/depth/alert/exposure metadata describe events. Mortality uses Spearman correlation, deployment maximum-likelihood logistic regression with ridge robustness and Firth supplementary analysis; exploratory negative-binomial rate modeling is mentioned. Outputs are mortality association, deployment probability and exploratory exposure-normalized mortality, not our priority labels.

**Evaluation/results:** No independent chronological holdout or external validation is described. Sensitivity changes deployment definition, composites/raw alerts and time/distance windows; bootstraps are used for some uncertainty/effect-size summaries. Results give score–deathρ=.520,p=3.43×10^-7; deployment9/17 Red versus0/68 Orange; deployment OR42.72 (95%CI4.73–385.71), AUC.981, Brier.034. Treat AUC/Brier as apparent model-fit performance unless the archived protocol establishes otherwise. Brier is a probability-loss score, not classification accuracy.

**Internal issues to preserve:** Abstract givesρ=.522 and calls deployment Firth-modeled, whereas main Methods describe maximum likelihood/ridge and Firth supplementary. Composite definition varies: ±24h/≤150km with aggregated quantities versus48h country-gap clustering selecting a representative. These inconsistencies need resolution before implementation/reproduction. The score may influence deployment decisions itself; this observational association does not demonstrate an independent clinically optimal allocation rule. “Independent predictor” wording does not mean confounding is controlled in a sole-predictor model.

**Limitations/relation/use:** Small deployment count, wide intervals, sparse exposure/geographic strata, selected alerts, heterogeneous secondary outcome reports, unmeasured building/health-system factors and no causal inference. Strong support for evaluating whether GDACS alerts align with outcomes and treating them as preliminary support signals. It cannot certify our EM-DAT-to-GDACS mapping, human-impact label equivalence, real-time inference or dispatch reliability. Our multi-hazard, metadata-based forest task differs from its earthquake-only GDACS-score observational modeling. It is particularly close to the **future live validation question**, though not a direct RF comparator.

### [A] Talan et al. (2026) — additional, directly relevant EM-DAT study

**Source/access:** [Published full text](https://www.mdpi.com/2076-3417/16/19/9497), especially Sections2.1–2.10,3.2–3.6,4–5 and Table1. Article appeared24September2026, before this review. Published text reviewed, not a preprint.

**Data/hazards/geography:** Global EM-DAT1975–2025;25,070raw records→15,869natural→5,059positive-damage→5,051after rare-category filtering;4,519underlying events. Eight categories include earthquake, storm, flood, wet mass movement, wildfire, drought, extreme temperature and volcanic activity. Selected positive-damage records are not a representative sample of all disasters.

**Inputs/output:**21 predictors: Disaster Group/Subgroup/Type/Subtype, Origin, Associated Types; ISO/Country/Subregion/Region; Total Deaths/Total Affected; six start/end date fields; duration/decade/season. Target is2025-price adjusted economic damage, modeled with log1p. Raw magnitude/scale/availability are excluded because physical units differ across hazards; human-impact components are excluded when overlapping Total Affected. The framework explicitly uses post-event human outcomes and end dates and is framed as post hoc, not real-time forecasting. Our model excludes impact outcomes as predictors but still needs a historical-versus-live feature-time audit.

**Models/validation:**RF/XGBoost/LightGBM/CatBoost plus global/type/region-median and linear/ridge/lasso baselines. Event_ID is the country-suffix-stripped DisNo; GroupShuffleSplit gives4,038training/1,013test rows. Grouped5fold selection/stability;5seed sensitivity; event-aware pre2011 versus2011onward temporal evaluation. Temporal tuning uses four forward-chaining folds wholly within pre2011; independent future set unused for selection. Region/hazard holdouts assess transfer. Non-nested grouped CV is explicitly a stability analysis, not an independent final estimate. This is the authors' test data—not our reserved test split.

**Results/limitations:**Tuned CatBoost grouped RMSLE1.890/logR².491; temporal1.791/.464; grouped CV RMSLE1.851±.056. Removing geography increases RMSLE20.9%; removing human outcomes16.9%. Weak region/hazard transfer and large catastrophic-tail errors; reporting/selection bias; geographic identifiers cannot be causally interpreted as vulnerability. All are authors' reported results, not replicated by us. There is a small textual inconsistency in3.2: tuned LightGBM1.922 is described as worse than reference1.931 although it is numerically lower. Do not reproduce that direction error.

**Relation/difference/use:**This substantially overlaps **global multi-hazard, tabular EM-DAT impact modeling and temporal validation**. Those components are not our novelty. Our targets are two human-impact class proxies, rather than economic regression; we use9features rather than21including post-event outcomes. It legitimately supports event-aware evaluation, the distinction between retrospective modeling and prospective operation, hazard-unit caution, and limits on interpreting country IDs. It provides a stronger methodological comparison than regional susceptibility studies. Its own post hoc task means its outcome predictors are not automatically improper leakage; they would be inappropriate for a forecast requiring them before observation.

### [B] Kalaivani et al. (2026) — additional, potential near-overlap

**Source/access:** [DOI](https://doi.org/10.1109/ICSADL67539.2026.11452016), [metadata](https://api.crossref.org/works/10.1109/icsadl67539.2026.11452016), [indexed abstract](https://api.openalex.org/works/https://doi.org/10.1109/icsadl67539.2026.11452016). Published IEEE ICSADL2026 proceedings, pp.587–592. Full text was not retrieved; the IEEE landing-page response was empty. Indexed abstract only.

**What is verified from abstract:**Billion Dollar Weather and Climate Disasters (BDWCD) dataset; disaster type, duration, affected population and climate region are mentioned inputs; outputs severity at daily/monthly scale and event deaths; RF/gradient boosting/voting regression ensemble.99.37% accuracy is reported without accessible definition of accuracy, target derivation or independent evaluation. Geography is not explicitly verified from the text accessed; do not silently equate it with global EM-DAT or assume a specific NOAA release. Exact data size/time span, feature availability, validation/test protocol, leakage safety and limitations remain unknown.

**Relation/difference/use:**This overlaps the broad idea of multi-hazard severity/death prediction. It is an important **novelty-check candidate**, not evidence that our task is unique. Our EM-DAT labels, hazard-relative thresholds, fixed features and chronological protocol differ from what the abstract describes; exact equivalence remains unresolved without full text. Affected-population input would need timing scrutiny for a prospective severity model, but no leakage verdict can be made from the abstract. Do not benchmark its99.37% against our47–52% validation accuracy.

### [C] Li et al. (2022) — additional casualty-prediction comparison

**Source/access:**[Published full text](https://www.mdpi.com/2072-4292/14/1/30). Dataset construction, feature selection, model construction, results and discussion inspected; publisher text available. This additional review focused on empirical sections rather than reproducing the full SVR derivation.

**Data/input/output:**Mainland-China historical earthquakes1950–2017, earthquake reports/fault dataset/WorldPop.152events with death registers are collected;143remain after excluding events with>1000deaths.113training/30heldout records. RF importance initially examines14candidate features; final Z-SVR uses magnitude, population density, fault density and GDP. Population/fault density also define risk zones. Death counts and order-of-magnitude death categories are evaluated; not affected population or multi-hazard priority.

**Validation/results/limitations:**Random zone-stratified approximately80/20 holdout; grid-search and10fold training CV (5fold in sparse medium-risk zone). Z-SVR versus zoning/nonzoning RF/BP/LR/SVR comparisons. Authors report zone-averaged heldout RMSE35.61 and MAE17.39, not a pooled global death-error benchmark. Exclusion of catastrophic events is a major applicability limit. The text repeats full-zone counts49/13/81 when describing113training examples, an internal denominator issue; sourceTable2 should govern replication. Several historical features and modern population layers require temporal-availability scrutiny. Do not assume chronology or train-only feature selection is fully established.

**Relation/difference/use:**Legitimate direct background that observed casualties have been predicted from event and contextual predictors. It illustrates the difference between count regression and our range/priority proxies, and how excluding extremes changes the evaluated task. It is closer to human impacts than generic flood-flow reviews, but is single-hazard, single-country and uses exposure/economic context missing from our features.

## Answers to the eight research questions

### 1. Strongest defensible research gap

**A narrowly defined empirical question:** how much predictive information do a fixed, small set of disaster/location/calendar/magnitude descriptors contain for two explicitly training-defined human-impact class targets across six hazard categories when evaluated on a later historical period, and what class-level trade-offs appear under modest forest changes?

This is a gap in the *specific evaluation we undertake*, not a proven absence of all similar literature. Zhang and Pidurkar evaluate spatial susceptibility/index levels, Chen and Li address earthquake casualties, Akshay addresses flood severity, Aykut studies GDACS earthquake outcomes, Talan already evaluates global multi-hazard EM-DAT loss models temporally, and Kalaivani already describes multi-hazard severity/death prediction. These verified distinctions support a focused positioning; they do **not** establish a “first ever” claim. Metadata-only papers cannot be excluded from novelty considerations simply because their text was inaccessible.

A second, valuable question is whether the historical model can be transferred to event-time GDACS inputs after category/unit/timing mapping. That is currently an **unanswered future validation question**, not a contribution demonstrated by our results. Genuine multi-hazard interaction/cascade modeling is also not implemented: pooling several event types is a multi-disaster dataset, not a learned joint-hazard interaction model.

### 2. Legitimate contribution

Present the current work as a **reproducible applied prototype and validation study**, with these implemented components:

1. A documented EM-DAT natural-hazard extraction and two-target label-construction procedure with thresholds estimated from training only. This makes target meaning inspectable; quantile thresholds themselves are not a new ML method and are not expert ground truth.
2. A fixed nine-feature pipeline, separate Random Forest classifiers and most-frequent baselines, with historical train/validation separation and explicit untouched final-test status.
3. Controlled validation-only comparisons of leaf size, depth and class weighting, reporting class trade-offs rather than accuracy alone. This is an empirical experiment design contribution, not algorithmic novelty or statistically significant superiority.
4. A working GDACS ingestion/API/map monitoring architecture, **separate from** completed offline modeling, plus a documented transfer/integration problem still to resolve.

The saved results justify “outperforms the most-frequent Dummy on this validation set.” They do not justify “outperforms literature/state of the art” or “is useful for real dispatch.” The project can claim transparent task definition and evidence-bound limitations; it cannot yet claim validated real-time impact decision support. Train-derived proxy labels, temporal splits and a web stack are normal methodological/software choices; their combination is a practical implementation distinction, not automatically publishable research novelty. To support publication, independent final evaluation, label validity and stronger analysis will matter more than wording.

### 3. Claims to avoid

- First global/multi-disaster impact predictor; no prior multi-hazard ML; all prior work is single-hazard. Talan and Kalaivani directly challenge those assertions.
- A new Random Forest algorithm, novel temporal validation, or novel use of EM-DAT/GDACS. These components have prior literature.
- Final 2023–2025 performance or a final chosen model; independently validated generalization; significant tuning improvements without uncertainty analysis.
- Complete live ML integration, verified GDACS-to-EM-DAT equivalence, event-onset feature availability, prospective or “real-time” predictive reliability. “Real-time” presently describes feed monitoring.
- Official severity/dispatch priorities, optimal evacuation/resource allocation, lives saved or clinical workload reduction. Our priority target is a historical relative label, not observed response need or expert recommendation.
- Explicit social vulnerability/exposure measurement from country/region IDs; causal interpretations of location or magnitude; cascading/compound hazard modeling.
- Our validation score is worse/better than a published high accuracy with different labels/splits; reuse of review-corpus or database-total counts as our dataset size.
- All leakage has been eliminated merely because thresholds are train-only. That addresses some leakage channels; event grouping, post-event feature timing and domain transfer remain to check.
- Missing-target exclusion makes the retained sample unbiased or mean/median imputation solves EM-DAT reporting bias. Jones shows why this needs diagnosis.

These restrictions follow our code/results and the specific literature profiles, not hypothetical warnings about an otherwise validated deployment.

### 4. Is Zhang an appropriate base paper?

**Yes as the conceptual base for distinguishing hazards, risk and vulnerability, and for motivating comparative ML across hazard categories. No as the closest task-specific base or a directly reproducible benchmark.** Its analysis unit is an Idaho watershed/city; ours is a global country-event record. Its targets are indicator-derived hazard/SoVI levels; ours derive from recorded human outcomes. It has explicit exposure/vulnerability predictors and GIS aggregation; ours does not. Its split is80/20 rather than our forward period split. Its published F1 values are not comparable to our multiclass macro F1. Its methodological/reporting inconsistencies should remain visible.

For final Related Work, keep Zhang as the conceptual starting point, then center the direct comparison on Talan plus casualty/severity/event-alert work. Calling Zhang our “base” must not imply we extended its RMHI, used its data, included social vulnerability or reproduced its evaluation.

### 5. Which five papers are closest to our actual ML task?

A provisional overall shortlist (including the new additions), with strength of evidence stated:

| Rank | Paper | Why close | Important qualification |
|---|---|---|---|
| 1 | Talan et al.2026 [A] | Global, event-aware, multi-hazard EM-DAT impact modeling with temporal evaluation | Economic regression with post-event outcome predictors, not our two human-impact classes; full text reviewed |
| 2 | Kalaivani et al.2026 [B] | Multi-hazard severity and death prediction using tabular ensemble models | Different dataset; abstract only; exact target/split equivalence unresolved |
| 3 | Chen & Zhang2022 [3] | Event impact/casualty-rate and loss prediction with AutoML | Earthquake-only; metadata only, so ranking is task-title based pending text |
| 4 | Akshay et al.2025 [4] | Disaster damage-severity classification, compares RF with other learners | Flood-only; abstract only; dataset/labels/splits require review |
| 5 | Aykut et al.2026 [10] | Country-event outcomes and operational-burden proxy tied to the intended GDACS source | Earthquake-only observational/statistical study, not multiclass forest benchmarking; full text reviewed |

This is a relevance shortlist, not a quality ranking. Li et al.[C] is a strong **fully accessible supplementary human-impact comparator** and can replace an inaccessible shortlist entry for a defensible methods discussion. Among only the eleven user-specified papers, the direct task candidates are Chen/Akshay/Aykut; Zhang/Pidurkar are the next conceptual comparators, but their spatial-risk tasks are substantially farther away. Five equally close direct experimental comparators were not established from that original list.

### 6. Which should remain background references?

- Broad AI/disaster context: Sun, Albahri and Liu. Keep abstract/metadata limitations until full text is read. None should supply a direct numerical benchmark.
- Regional conceptual context: Zhang and Pidurkar, clearly distinguished from event impacts. Zhang remains the designated conceptual base.
- Optional adaptation context: Gerolimos, only with qualifications; do not rely on its unexplained scores as empirical evidence. Prefer a better documented open-world source if that subject becomes important; our system currently does not implement it.
- Dataset/method foundations: Delforge and Jones are **central supporting references**, even though they are not direct predictive comparators. Delforge is provenance, Jones is reporting/missingness evidence.
- Existing flood-flow/economic-damage/social-media papers in our manuscript are context at most. Mosavi2018, Merz2010 and Fekete2009 can support clear scope distinctions; Imran2015 and CrisisMMD2018 are tangential because we do not implement NLP/image classification. Keep them only if the passage genuinely requires those topics. Do not cite many distant papers to make a nearest-task claim look established.

### 7. More closely related papers to add?

**Yes: add Talan[A] to the comparison now; investigate Kalaivani[B] before writing novelty claims; add Li[C] for a concrete casualty-prediction methodology contrast.** Full citation/versions and source scopes are in the register/profiles. They are not interchangeable tasks.

An additional verified-title lead is Ganguly, Nahar and Hossain (2019), *A machine learning-based prediction and analysis of flood affected households: A case study of floods in Bangladesh*, International Journal of Disaster Risk Reduction34,283–294, [DOI10.1016/j.ijdrr.2018.12.002](https://doi.org/10.1016/j.ijdrr.2018.12.002). Metadata was retrieved; abstract/full text was not. It may be close to affected-population outcomes but household-level scope cannot be equated to our country-event labels. Add to the retrieval queue, not as reviewed evidence.

Focused searches used Crossref and OpenAlex queries combining EM-DAT, ML, severity, deaths, affected population and multi-disaster impact, plus title/DOI checks and legal access-location searches. Generic susceptibility maps, social-media severity, country-year analyses, conference abstracts and reviews were not treated as equivalent event-level prediction studies. Search ranking is incomplete and does not support a comprehensive novelty/no-overlap conclusion.

### 8. Does an existing paper already perform essentially the same task?

**The broad task is already covered: global multi-hazard historical impact prediction exists in Talan, and multi-hazard severity/death prediction is described in Kalaivani.** Our broad concept is not new.

**Exact equivalence is not established:** none of the full texts reviewed implements our particular pair of train-defined affected-range and hazard-relative deaths/affected priority targets using the exact nine features and2000–2019/2020–2022 development split. That describes a difference found in this review, not proof nobody else does it. Kalaivani's full text and Chen/Akshay's feature/label definitions remain necessary to rule out much closer overlap. Different threshold labels or a different web framework do not, by themselves, establish a meaningful research advance. The current novelty is best characterized as a modest, transparent application/benchmarking distinction, with online transfer an uncompleted research direction.

## Methodological references to retain selectively

| Reference and verified source | Legitimate role | Scope limitation |
|---|---|---|
| Breiman, *Random Forests*, Machine Learning45:5–32(2001), [DOI10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324); [author manuscript previously reviewed](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf) | Attribute forest algorithm and ensemble ideas | Does not prove forests best for our task or remove need for temporal validation |
| Pedregosa et al., *Scikit-learn: Machine Learning in Python*, JMLR12:2825–2830(2011), [published paper](https://jmlr.org/papers/v12/pedregosa11a.html) | Attribute estimator/software framework | Current settings and pipeline behavior come from our code, not the2011 article |
| Kapoor & Narayanan, *Leakage and the Reproducibility Crisis in ML-Based Science*, [reviewed arXiv2207.07048(2022)](https://arxiv.org/abs/2207.07048) | Train-only transformations, availability, leakage taxonomy | Related published version is Patterns4(9),100804(2023), [DOI10.1016/j.patter.2023.100804](https://doi.org/10.1016/j.patter.2023.100804); metadata verified but journal full text was not the reviewed version |
| Cawley & Talbot, *On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation*, JMLR11:2079–2107(2010), [published paper](https://jmlr.org/papers/v11/cawley10a.html) | Selection bias, preserve independent final evaluation | Does not make our single validation period an independent final test |
| Saito & Rehmsmeier, *The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets*, PLOS ONE10(3),e0118432(2015), [DOI](https://doi.org/10.1371/journal.pone.0118432) | Optional context about imbalance-sensitive interpretation | Binary PR/ROC paper; not direct justification of every multiclass F1 choice or a claim we computed PR curves |

These versions were previously reviewed and recorded in [the repository literature log](../../realtime-disaster-ml/docs/research_literature_review.md). They should complement, not crowd out, nearest-task evidence. No reference's methodological reputation substitutes for checking our code and target validity.

## Evidence/access decisions before manuscript revision

| Requested paper | Bibliographic status | Review status | Decision for manuscript claims |
|---|---|---|---|
| Zhang2023 | VERIFIED published | Publisher full text reviewed | Keep conceptual base; qualify target/split/reporting differences |
| Sun2020 | VERIFIED published | Abstract only | Background; NEEDS FULL-TEXT REVIEW for detailed claims |
| Albahri2024 | VERIFIED published | Indexed abstract only | Trustworthiness background; NEEDS FULL-TEXT REVIEW |
| Chen2022 | VERIFIED published | Metadata only | Close candidate; NEEDS FULL-TEXT REVIEW before substantive methods/results claims |
| Akshay2025 | VERIFIED published proceedings | Indexed abstract only | Close candidate; NEEDS FULL-TEXT REVIEW |
| Liu2025 | VERIFIED journal record; separate SSRN record | Metadata only | Background candidate; NEEDS FULL-TEXT REVIEW |
| Gerolimos2025 | VERIFIED published | Publisher full text/tables reviewed | NEEDS REPLACEMENT as support for empirical performance/autonomous-operation claims; optional conceptual citation |
| Pidurkar2026 | VERIFIED published online | Publisher full text reviewed | Keep regional background with independent-label/evaluation qualifications |
| Delforge2025 | VERIFIED journal record and official EM-DAT listing | Indexed abstract only | Keep provenance; NEEDS FULL-TEXT REVIEW for article-specific protocol/schema details |
| Jones2022 | VERIFIED published | Publisher full text reviewed | Keep missingness evidence; do not copy erroneous total-deaths definition |
| Aykut2026 | VERIFIED published | Publisher full text reviewed | Keep GDACS outcome-validation context; qualify apparent metrics and inconsistent composite description |

“Needs replacement” above concerns a proposed citation **use**, not a claim that the article is fabricated or retracted. Access status is distinct from citation correctness. We have not yet fulfilled full-text review of all eleven requested papers; that limitation must remain visible until those texts become accessible.

## Source record and preservation

Raw metadata, publisher HTML, extracted text, access attempts and search results are retained beside this report under `/workspace/research_sources/closest_review/`. `literature_comparison.csv` provides the editable matrix. Links identify original sources; profile section names identify the passages used. Index abstracts are explicitly identified so they cannot be mistaken for reviewed publisher full text. Literature-review artifacts are outside the checkout, and manuscript Markdown/Word/PDF hashes were checked after writing this report. No repository files, models, evaluation files or test records were changed or evaluated.
