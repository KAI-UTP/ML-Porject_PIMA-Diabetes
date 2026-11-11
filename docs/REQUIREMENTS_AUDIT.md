# Assignment checklist

Checked against all tasks in `ML Porject Requirement.pdf`.

| Outline item | Evidence / status |
|---|---|
| A: Lab dataset and business/data understanding | PIMA from Labs 1 and 3; report introduction, provenance, cohort summary |
| A: Missing data, encoding, normalization | Median imputation, missing indicators, one-hot age bands and standard scaling inside fitted pipelines |
| A: Outlier detection | `outlier_audit.csv`; training IQR fences, counts for all splits, retain-extremes rationale |
| B: One selection method | Embedded Extra Trees; `selected_features.csv` lists retained transformed features |
| B: Feature engineering | Age binning and interaction/ratio features; fixed-C preprocessing ablation |
| C: Classification | Logistic regression, decision tree, random forest, prevalence baseline |
| C: Regression | BMI prediction; Ridge, tree, forest and mean baseline; BMI excluded from inputs |
| C: Feed-forward ANN | PyTorch 32–16–1; ReLU, dropout, Adam, early stopping |
| D: Classification and regression metrics | All required metrics in saved CSVs, report and dashboard |
| D: ANN trends and bias/variance | Loss/accuracy histories, learning curves and measured train/validation gaps |
| D: Four discussion questions | Four explicit report subsections with measured results and limitations |
| Report, maximum 3000 words | Generated Markdown and HTML with tables and plots; generator enforces word limit |
| Fully documented notebook | Executed cleaning and training code, results, provenance and interpretation |
| Turnitin report, AI <40% | Pending: students must obtain the actual institutional report; no score is asserted or guaranteed |
| Presentation, each student presents | Outline supplied; final slides, member allocation and delivery remain student tasks |
| Student/group details | Pending: add names and identifiers before submission |

Technical implementation is covered. Submission is not complete until the pending student tasks are finished.

Method limits: one historical population, one held-out split, exploratory test inspection during development, reused validation for selection/ANN stopping/threshold tuning, no external clinical validation. BMI is a separate secondary regression target. The raw dataset has numeric predictors; categorical age bands are engineered.
