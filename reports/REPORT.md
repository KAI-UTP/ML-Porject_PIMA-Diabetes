# Diabetes Prediction on the PIMA Indians Dataset

**Course:** TDB3123 Machine Learning · **Group members:** _add names and student IDs_

## 1. Introduction

Type 2 diabetes is often detected late. A model that flags high-risk patients from routine measurements can help decide who needs a confirmatory test. This project applies the CRISP-DM process to the PIMA Indians Diabetes dataset (used in Labs 1 and 3) to:

1. **classify** whether a patient has diabetes (main task),
2. **regress** body-mass index (BMI) from the other measurements (secondary task), and
3. compare classical models with a **feed-forward neural network**.

Success is measured with ROC-AUC and recall, always against a baseline, and over-fitting is studied with learning and validation curves.

## 2. Methodology

### 2.1 Data understanding

The dataset has 768 patients (women of Pima heritage aged 21+), 8 numeric measurements and a binary outcome. 268 patients (34.9%) have diabetes, so a model that always says "no" is already 65.1% accurate; accuracy alone is therefore misleading. Several columns contain **impossible zeros**: insulin (374 rows), skin thickness (227), blood pressure (35), BMI (11) and glucose (5). Glucose is the strongest single signal (correlation 0.50 with the outcome; median 140 vs 107 mg/dL).

![Data understanding](figures/1_data.png)

### 2.2 Data preparation

- **Cleaning:** types checked, no duplicates or invalid rows; impossible zeros converted to missing values.
- **Split:** stratified 80/20 split (614 training, 154 test patients, seed 42). The test set is used only once, at the end. All tuning, model selection and threshold choice use **5-fold stratified cross-validation (CV)** on the training set.
- **Missing values:** median imputation plus missing-value indicator columns.
- **Encoding and normalization:** one-hot encoding for binned features; `StandardScaler` for logistic regression, SVM and the ANN (trees are scale-invariant).
- **Outliers:** 1.5 × IQR fences from the training set flag 65 values (mostly pedigree, insulin and blood pressure). They are medically plausible, so they are **kept**; a RobustScaler was tested as an alternative.
- All steps are inside one scikit-learn `Pipeline`, so imputation and scaling are learned from the training folds only (no data leakage).

### 2.3 Feature engineering and selection

Four engineering techniques were applied: **log transform** of insulin and pedigree (insulin skew 2.00 → −0.17), an **interaction term** glucose × BMI, **binning** of age (21–30, 31–40, 41–50, 51+) and BMI (WHO classes), and **missing indicators**. This gives 23 model inputs. Three selection methods (each ≈10 features) were compared: **filter** (mutual information), **wrapper** (recursive feature elimination with logistic regression) and **embedded** (L1/LASSO logistic regression).

| feature | MI | LASSO coef | RF importance | MI kept | RFE kept | LASSO kept |
|---|---|---|---|---|---|---|
| glucose_x_bmi | 0.161 | 0.930 | 0.259 | True | False | True |
| glucose | 0.119 | 0.367 | 0.206 | True | True | True |
| bmi | 0.092 | 0.000 | 0.092 | True | True | False |
| log_insulin | 0.064 | 0.000 | 0.051 | True | False | False |
| age | 0.059 | 0.000 | 0.083 | True | True | False |
| age_band_21-30 | 0.046 | -0.661 | 0.031 | True | True | True |
| bmi_class_normal | 0.037 | 0.000 | 0.012 | True | True | False |
| missingindicator_glucose_x_bmi | 0.033 | 0.000 | 0.000 | True | False | False |
| missingindicator_skin_thickness | 0.031 | 0.000 | 0.005 | True | False | False |
| age_band_31-40 | 0.029 | 0.000 | 0.007 | True | False | False |

All methods keep glucose, and the engineered glucose × BMI feature ranks first for mutual information, LASSO and random-forest importance.

### 2.4 Models

| Task | Models (tuned with GridSearchCV, 5-fold CV) |
|---|---|
| Classification | Baseline (prior), Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVM (RBF) |
| Neural network | PyTorch: inputs → 32 ReLU → dropout 0.2 → 16 ReLU → dropout 0.2 → 1 sigmoid; Adam, binary cross-entropy, batch 32, weight decay, early stopping (patience 25) |
| Regression (BMI) | Baseline (mean), Linear Regression, Ridge, Lasso, Decision Tree, Random Forest |

Feature selection (Mutual information (filter)) is a tuned option in every classifier grid, so CV decides whether it helps. The best classifier is the one with the highest mean ROC-AUC over 5-fold CV repeated 3 times. Its decision threshold maximises F1 on out-of-fold predictions. For regression, BMI and BMI-based features are excluded from the inputs.

## 3. Results

### 3.1 Effect of preprocessing and feature engineering

| variant | Logistic Regression | Random Forest |
|---|---|---|
| 1. Raw data (zeros kept) | 0.830 | 0.827 |
| 2. Zeros -> missing + median imputation | 0.840 | 0.830 |
| 3. + Standard scaling | 0.834 | 0.830 |
| 3b. + Robust scaling (alternative) | 0.840 | 0.830 |
| 4. + Feature engineering | 0.844 | 0.836 |
| 5. + LASSO L1 (embedded) | 0.841 | 0.827 |
| 5. + Mutual information (filter) | 0.849 | 0.831 |
| 5. + RFE (wrapper) | 0.844 | 0.833 |

![Feature ablation](figures/2_feature_ablation.png)

### 3.2 Classification

| model | CV ROC-AUC | CV std | Train ROC-AUC | Test ROC-AUC |
|---|---|---|---|---|
| SVM (RBF) | 0.847 | 0.031 | 0.870 | 0.827 |
| Logistic Regression | 0.846 | 0.037 | 0.870 | 0.835 |
| ANN (PyTorch) | 0.843 | 0.044 | 0.874 | 0.828 |
| Random Forest | 0.839 | 0.033 | 0.918 | 0.805 |
| Gradient Boosting | 0.834 | 0.032 | 0.897 | 0.808 |
| Decision Tree | 0.791 | 0.033 | 0.884 | 0.787 |
| Baseline (prior) | 0.500 | 0.000 | 0.500 | 0.500 |

Test-set metrics at each model's tuned threshold:

| model | threshold | accuracy | precision | recall | f1 | roc_auc |
|---|---|---|---|---|---|---|
| SVM (RBF) | 0.374 | 0.766 | 0.636 | 0.778 | 0.700 | 0.827 |
| Logistic Regression | 0.385 | 0.786 | 0.672 | 0.759 | 0.713 | 0.835 |
| ANN (PyTorch) | 0.297 | 0.747 | 0.597 | 0.852 | 0.702 | 0.828 |
| Random Forest | 0.317 | 0.721 | 0.575 | 0.778 | 0.661 | 0.805 |
| Gradient Boosting | 0.295 | 0.721 | 0.575 | 0.778 | 0.661 | 0.808 |
| Decision Tree | 0.409 | 0.734 | 0.600 | 0.722 | 0.655 | 0.787 |
| Baseline (prior) | 0.500 | 0.649 | 0.000 | 0.000 | 0.000 | 0.500 |

![Classification results](figures/3_classification.png)

The best model by CV is **SVM (RBF)** (CV ROC-AUC 0.847 ± 0.031; test 0.827, 95% bootstrap CI 0.754–0.887). With its threshold of 0.37 it detects 78% of diabetic test patients with 76.6% accuracy (F1 0.700); at the default 0.5 threshold accuracy is 72.1% but fewer diabetics are found. The ANN stopped early at epoch 19. The most important input (permutation importance) is **glucose** (ROC-AUC drop 0.128), followed by bmi and age.

### 3.3 Regression (BMI)

| model | MAE | MSE | RMSE | R² | CV RMSE |
|---|---|---|---|---|---|
| Baseline (mean) | 5.672 | 50.709 | 7.121 | -0.000 | 6.891 |
| Decision Tree | 4.780 | 37.111 | 6.092 | 0.268 | 5.822 |
| Lasso | 4.240 | 28.865 | 5.373 | 0.431 | 5.475 |
| Linear Regression | 4.239 | 28.748 | 5.362 | 0.433 | 5.496 |
| Random Forest | 4.348 | 29.882 | 5.466 | 0.411 | 5.503 |
| Ridge | 4.246 | 28.816 | 5.368 | 0.432 | 5.486 |

![Regression results](figures/5_regression.png)

**Lasso** has the lowest CV RMSE; on the test set it reaches RMSE 5.37 and R² 0.43, compared with RMSE 7.12 for the mean baseline. Skin thickness is the strongest BMI predictor.

### 3.4 Bias–variance

![Bias-variance](figures/4_bias_variance.png)

| model | train_roc_auc | cv_roc_auc | gap |
|---|---|---|---|
| SVM (RBF) | 0.870 | 0.847 | 0.023 |
| Logistic Regression | 0.870 | 0.846 | 0.024 |
| ANN (PyTorch) | 0.874 | 0.843 | 0.030 |
| Gradient Boosting | 0.897 | 0.834 | 0.063 |
| Random Forest | 0.918 | 0.839 | 0.078 |
| Decision Tree | 0.884 | 0.791 | 0.093 |

## 4. Discussion

**How did preprocessing and feature selection affect generalization?**
Treating impossible zeros as missing was the most valuable step: logistic-regression CV ROC-AUC rose from 0.830 to 0.840 and became more stable across folds. Scaling did not change the score of the random forest, confirming that trees do not need it, while it is required for the SVM and ANN. Feature engineering improved both logistic regression (0.844) and random forest (0.836). Mutual-information selection gave the best logistic score (0.849) with fewer inputs. All these gains are smaller than the fold-to-fold standard deviation (≈0.03), so on this small dataset preprocessing mainly made the models simpler and more stable rather than much more accurate.

**Which model type performed best and why?**
SVM (RBF) had the highest CV ROC-AUC, but logistic regression (CV 0.846, test 0.835) and the ANN (CV 0.843) perform the same within the uncertainty. The data are small, tabular and numeric, and the classes are separated by a smooth boundary driven mainly by glucose, so complex models add no benefit. For healthcare use, **logistic regression** is the recommended model: equally accurate, transparent and the least over-fitted. For BMI, the regularised linear models beat the trees.

**How did dataset size, feature quality and data type influence overfitting?**
With only 614 training rows, flexible models over-fit. An unrestricted decision tree reaches train ROC-AUC 1.000 but CV 0.652, while CV peaks at depth 3 (0.793). The tuned tree and random forest still show the largest train–CV gaps (0.09 and 0.08); the random-forest regressor has train R² 0.66 but CV R² 0.36. Feature quality is limited: insulin is missing for 49% of patients and skin thickness for 30%, so they add little signal. Because the inputs are continuous and only moderately informative, the linear models, SVM and ANN have small gaps (≈0.02–0.03): their remaining error is bias and noise, not variance. The learning curves still rise slowly, so more data would help.

**What recommendations would you make for real-world healthcare analytics?**
(1) Train on larger and more diverse data (men, other populations, several hospitals) and validate externally. (2) Prefer interpretable and well-calibrated models, and choose the decision threshold together with clinicians according to the cost of a missed case. (3) Record missing measurements explicitly instead of as zeros. (4) Monitor performance and fairness after deployment and keep a clinician in the loop: the model is a screening aid, not a diagnosis.

## 5. Conclusion

A full CRISP-DM pipeline was built: cleaning of hidden missing values, outlier analysis, four feature-engineering techniques, three feature-selection methods, six tuned classifiers, a PyTorch ANN and six BMI regressors, evaluated with repeated cross-validation, a held-out test set and bias–variance diagnostics. The best classifiers reach a test ROC-AUC of about 0.83 and detect about 78% of diabetic patients at the tuned threshold. Logistic regression offers the best balance of accuracy, stability and interpretability. An interactive Streamlit dashboard presents all results and a prediction lab.

## References

- PIMA Indians Diabetes dataset: <https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv>
- TDB3123 lecture notes, Chapters 2–5 and 8; Labs 1, 3, 4 and 5.
- scikit-learn User Guide: pipelines, cross-validation and common pitfalls. <https://scikit-learn.org/stable/common_pitfalls.html>
- PyTorch tutorials: building and training neural networks. <https://docs.pytorch.org/tutorials/>
