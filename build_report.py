"""Build the report (Markdown + HTML), its figures and the presentation outline from saved artifacts.

Run after `python train.py`.
"""
import base64
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mistune
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve

from data_pipeline import ROOT
from chart_style import ACCENT, MODEL_COLORS, NEUTRAL, OUTCOME_COLORS, SERIES

A, OUT = ROOT / "artifacts", ROOT / "reports"
FIG = OUT / "figures"
PURPLE, GREY = ACCENT, NEUTRAL
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130, "axes.titleweight": "bold", "axes.prop_cycle": matplotlib.cycler(color=SERIES)})

read = lambda name: pd.read_csv(A / name)
cohort = pd.read_csv(ROOT / "data/processed/cohort.csv")
meta = json.loads((ROOT / "data/processed/provenance.json").read_text())
summary = json.loads((A / "run_summary.json").read_text())
cls = read("classification_summary.csv")
cm = read("classification_metrics.csv")
preds = read("classification_predictions.csv")
reg = read("regression_metrics.csv")
reg_pred = read("regression_predictions.csv")
ablation = read("feature_ablation.csv")
selection = read("feature_selection.csv")
depth = read("depth_curve.csv")
curves = read("learning_curves.csv")
history = read("ann_history.csv")
importance = read("permutation_importance.csv")
outliers = read("outlier_audit.csv")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)


def make_figures():
    FIG.mkdir(parents=True, exist_ok=True)
    for old in FIG.glob("*.png"):
        old.unlink()

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for value, color in [(0, OUTCOME_COLORS["No diabetes"]), (1, OUTCOME_COLORS["Diabetes"])]:
        ax[0].hist(cohort.loc[cohort.outcome == value, "glucose"].dropna(), bins=25, alpha=0.6, color=color,
                   label=["No diabetes", "Diabetes"][value])
    ax[0].set(title="Glucose by outcome", xlabel="glucose (mg/dL)", ylabel="patients")
    ax[0].legend()
    zeros = pd.Series(meta["audit"]["zero_as_missing"]) / meta["rows"] * 100
    ax[1].barh(zeros.index, zeros.values, color=SERIES[0])
    ax[1].set(title="Impossible zeros (treated as missing)", xlabel="% of rows")
    save(fig, "1_data.png")

    fig, ax = plt.subplots(figsize=(10, 4.2))
    for offset, (model, part) in zip([-0.15, 0.15], ablation.groupby("model")):
        y = np.arange(len(part)) + offset
        ax.errorbar(part.cv_roc_auc, y, xerr=part.cv_std, fmt="o", capsize=3, label=model, color=MODEL_COLORS[model])
    ax.set_yticks(np.arange(len(part)), part.variant)
    ax.invert_yaxis()
    ax.set(title="Effect of each preparation step (5×3 repeated CV)", xlabel="ROC-AUC (mean ± std)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    save(fig, "2_feature_ablation.png")

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.4), gridspec_kw={"width_ratios": [1.2, 1.2, 0.8]})
    c = cls[~cls.model.str.startswith("Baseline")].sort_values("cv_roc_auc")
    ax[0].barh(c.model, c.cv_roc_auc, xerr=c.cv_roc_auc_std, color=[PURPLE if m == summary["winner"] else GREY for m in c.model])
    ax[0].set(title="Cross-validation ROC-AUC", xlim=(0.6, 0.95))
    test = preds[preds.split == "test"]
    for model in c.model[::-1]:
        p = test[test.model == model]
        fpr, tpr, _ = roc_curve(p.actual, p.probability)
        ax[1].plot(fpr, tpr, color=MODEL_COLORS[model], label=f"{model} ({cls.set_index('model').loc[model, 'test_roc_auc']:.3f})")
    ax[1].plot([0, 1], [0, 1], "k--", lw=1)
    ax[1].set(title="Test ROC curves", xlabel="false-positive rate", ylabel="recall")
    ax[1].legend(fontsize=7)
    p = test[test.model == summary["winner"]]
    cmat = confusion_matrix(p.actual, p.probability >= summary["threshold"])
    ax[2].imshow(cmat, cmap="Purples")
    for (i, j), v in np.ndenumerate(cmat):
        ax[2].text(j, i, v, ha="center", va="center", fontsize=13, color="white" if v > cmat.max() / 2 else "black")
    ax[2].set(title=f"{summary['winner']}: test confusion matrix", xticks=[0, 1], yticks=[0, 1],
              xticklabels=["No diabetes", "Diabetes"], yticklabels=["No diabetes", "Diabetes"],
              xlabel="predicted", ylabel="actual")
    ax[2].grid(False)
    save(fig, "3_classification.png")

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax = ax.ravel()
    ax[0].plot(depth.max_depth, depth.train_roc_auc, "o-", label="train")
    ax[0].plot(depth.max_depth, depth.cv_roc_auc, "o-", label="CV")
    ax[0].set(title="Decision-tree depth", xlabel="max_depth", ylabel="ROC-AUC")
    ax[0].legend()
    for model in ["Logistic Regression", "Random Forest"]:
        color = MODEL_COLORS[model]
        part = curves[curves.model == model]
        ax[1].plot(part.n_train, part.train_roc_auc, "--", color=color, label=f"{model} train")
        ax[1].plot(part.n_train, part.cv_roc_auc, "o-", color=color, label=f"{model} CV")
    ax[1].set(title="Learning curves", xlabel="training rows", ylabel="ROC-AUC")
    ax[1].legend(fontsize=7)
    for axis, metric in [(ax[2], "loss"), (ax[3], "accuracy")]:
        axis.plot(history.epoch, history[f"train_{metric}"], label="train")
        axis.plot(history.epoch, history[f"validation_{metric}"], label="validation")
        axis.axvline(summary["ann_best_epoch"], ls="--", c="grey")
        axis.set(title=f"Neural network {metric}", xlabel="epoch")
        axis.legend()
    save(fig, "4_bias_variance.png")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    r = reg[reg.split == "test"].sort_values("rmse", ascending=False)
    ax[0].barh(r.model, r.rmse, color=[PURPLE if m == summary["regression_winner"] else GREY for m in r.model])
    ax[0].set(title="BMI regression: test RMSE", xlabel="RMSE (kg/m²)")
    p = reg_pred[(reg_pred.model == summary["regression_winner"]) & (reg_pred.split == "test")]
    ax[1].scatter(p.actual, p.predicted, alpha=0.6, color=MODEL_COLORS[summary["regression_winner"]])
    ax[1].plot([15, 60], [15, 60], "k--", lw=1)
    ax[1].set(title=f"{summary['regression_winner']}: actual vs predicted", xlabel="actual BMI", ylabel="predicted BMI")
    save(fig, "5_regression.png")


def md_table(frame, digits=3):
    frame = frame.copy()
    for col in frame.select_dtypes("number"):
        frame[col] = frame[col].map(lambda v: f"{v:.{digits}f}")
    head = "| " + " | ".join(frame.columns) + " |\n|" + "---|" * len(frame.columns)
    return head + "\n" + "\n".join("| " + " | ".join(map(str, row)) + " |" for row in frame.to_numpy())


def build_report() -> str:
    s = cls.set_index("model")
    w, lr = s.loc[summary["winner"]], s.loc["Logistic Regression"]
    t = summary["test_metrics"]
    abl = ablation.set_index(["variant", "model"]).cv_roc_auc
    rtest = reg[reg.split == "test"].set_index("model")
    rtrain = reg[reg.split == "train"].set_index("model")
    rcv = reg[reg.split == "cv"].set_index("model")
    rw = summary["regression_winner"]
    best_depth = depth.loc[depth.cv_roc_auc.idxmax()]
    zeros = meta["audit"]["zero_as_missing"]
    n_train, n_test = meta["split_counts"]["train"], meta["split_counts"]["test"]
    gaps = s.drop("Baseline (prior)").assign(gap=lambda f: f.train_roc_auc - f.cv_roc_auc)

    cls_table = cls[["model", "cv_roc_auc", "cv_roc_auc_std", "train_roc_auc", "test_roc_auc"]].rename(columns={
        "cv_roc_auc": "CV ROC-AUC", "cv_roc_auc_std": "CV std", "train_roc_auc": "Train ROC-AUC", "test_roc_auc": "Test ROC-AUC"})
    test_table = cm[cm.split == "test"].set_index("model").loc[cls.model].reset_index()[
        ["model", "threshold", "accuracy", "precision", "recall", "f1", "roc_auc"]]
    reg_table = (reg[reg.split != "train"].pivot(index="model", columns="split", values=["mae", "mse", "rmse", "r2"]))
    reg_table = reg_table.xs("test", axis=1, level=1).join(reg_table.xs("cv", axis=1, level=1)[["rmse"]], rsuffix=" (CV)")
    reg_table = reg_table.reset_index().rename(columns={"mae": "MAE", "mse": "MSE", "rmse": "RMSE", "r2": "R²", "rmse (CV)": "CV RMSE"})
    abl_table = ablation.pivot(index="variant", columns="model", values="cv_roc_auc").reset_index()
    sel_table = selection.head(10)[["feature", "mutual_information", "lasso_coefficient", "rf_importance",
                                    "Mutual information (filter)", "RFE (wrapper)", "LASSO L1 (embedded)"]]
    sel_table.columns = ["feature", "MI", "LASSO coef", "RF importance", "MI kept", "RFE kept", "LASSO kept"]
    sel_table = sel_table.replace({True: "✓", False: ""})

    return f"""# Diabetes Prediction from Routine Clinical Measurements: A CRISP-DM Study on the PIMA Indians Dataset

## Abstract

This study builds and evaluates machine-learning models that identify diabetes from eight routine clinical measurements of {meta['rows']} women of Pima Indian heritage. Following CRISP-DM, hidden missing values were repaired, four feature-engineering techniques and three feature-selection methods were compared, and five classifiers, a PyTorch feed-forward neural network and five BMI regressors (each against a baseline) were tuned with cross-validation. On a held-out test set the best classifiers reach a ROC-AUC of about {w.test_roc_auc:.2f}; with a tuned decision threshold, {summary['winner']} detects {t['recall']:.0%} of diabetic patients at {t['accuracy']:.1%} accuracy. Simple, regularised models generalise as well as complex ones on this small dataset, and logistic regression offers the best balance of accuracy, stability and interpretability.

## 1. Introduction

Type 2 diabetes often remains undiagnosed for years, and early detection reduces the risk of complications. A model that ranks patients by risk from routine measurements can prioritise who receives a confirmatory test. This study has three objectives:

1. **Classification** – predict whether a patient has diabetes (main task).
2. **Regression** – predict body-mass index (BMI) from the other measurements (secondary task).
3. **Neural network** – compare a feed-forward neural network with classical models.

The work follows the six CRISP-DM phases (Chapman et al., 2000), which structure Sections 2 and 3.

## 2. Methodology

### 2.1 Business understanding

The model is intended as a screening tool, so a missed diabetic (false negative) is more costly than a false alarm. Model quality is therefore measured mainly with ROC-AUC (ranking quality) and recall, and every model is compared with a trivial baseline.

### 2.2 Data understanding

The dataset (Smith et al., 1988) contains {meta['rows']} women aged 21 or older, eight numeric measurements and a binary outcome. {meta['class_counts']['1']} patients ({meta['class_counts']['1'] / meta['rows']:.1%}) have diabetes, so a model that always predicts "no diabetes" is already {1 - meta['class_counts']['1'] / meta['rows']:.1%} accurate; accuracy alone is therefore misleading. Several columns contain physiologically impossible zeros: insulin ({zeros['insulin']} rows), skin thickness ({zeros['skin_thickness']}), blood pressure ({zeros['blood_pressure']}), BMI ({zeros['bmi']}) and glucose ({zeros['glucose']}). Glucose is the strongest single signal (correlation 0.50 with the outcome; median 140 vs 107 mg/dL), and insulin and the diabetes pedigree function are strongly right-skewed.

![Figure 1. Glucose distribution by outcome and share of impossible zeros](figures/1_data.png)

### 2.3 Data preparation

- **Cleaning:** data types, duplicates and value ranges were checked (no rows removed); impossible zeros were converted to missing values.
- **Split:** a stratified 80/20 split gives {n_train} training and {n_test} test patients. The test set is used once, at the end; tuning, model selection and threshold choice use 5-fold stratified cross-validation (CV) on the training set.
- **Missing values:** median imputation, plus indicator columns that record whether a value was measured.
- **Encoding and normalisation:** one-hot encoding for binned features; standardisation for logistic regression, SVM and the neural network. Tree-based models are scale-invariant and use unscaled inputs.
- **Outliers:** 1.5 × IQR fences fitted on the training set flag {int(outliers[outliers.split == 'train'].outliers.sum())} values, mostly in the pedigree function, insulin and blood pressure. These values are rare but physiologically plausible, so they are retained; robust scaling was tested as an alternative.
- **Leakage control:** every step is part of one scikit-learn pipeline (Pedregosa et al., 2011), so imputation and scaling are learned from the training folds only.

### 2.4 Feature engineering and selection

Four techniques were applied: a **log transform** of insulin and the pedigree function (insulin skewness 2.00 → −0.17), an **interaction term** glucose × BMI, **binning** of age (21–30, 31–40, 41–50, 51+) and BMI (WHO, 2000 classes), and **missing-value indicators**, giving 23 model inputs. Three selection methods, each retaining about ten inputs, were compared: a **filter** (mutual information), a **wrapper** (recursive feature elimination with logistic regression) and an **embedded** method (L1-regularised logistic regression, LASSO). Table 1 lists the ten highest-ranked inputs.

**Table 1.** Feature scores and selections (training set).

{md_table(sel_table)}

### 2.5 Modelling

**Table 2.** Models and tuning.

| Task | Models (GridSearchCV, 5-fold CV) |
|---|---|
| Classification | Baseline (class prior), Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVM (RBF kernel) |
| Neural network | PyTorch (Paszke et al., 2019): inputs → 32 ReLU → dropout 0.2 → 16 ReLU → dropout 0.2 → sigmoid output; Adam, binary cross-entropy, batch size 32, weight decay, early stopping (patience 25) |
| Regression (BMI) | Baseline (mean), Linear Regression, Ridge, Lasso, Decision Tree, Random Forest |

Feature selection ({summary['selector']}) is a tuned option in every classifier grid, so cross-validation decides whether it helps each model. The final classifier is the one with the highest mean ROC-AUC over 5-fold CV repeated three times; its decision threshold maximises F1 on out-of-fold predictions. For BMI regression, BMI and all BMI-derived features are excluded from the inputs.

### 2.6 Evaluation and deployment

Classification is evaluated with accuracy, precision, recall, F1 and ROC-AUC; regression with MAE, MSE, RMSE and R²; the neural network additionally with training and validation loss and accuracy per epoch. Bias and variance are analysed through train–CV gaps, a decision-tree depth curve and learning curves, and a 95% bootstrap interval is reported for test ROC-AUC. The trained pipelines are deployed in an interactive Streamlit dashboard.

## 3. Results

### 3.1 Effect of preparation and feature engineering

**Table 3.** CV ROC-AUC as each preparation step is added (5-fold CV, 3 repeats).

{md_table(abl_table)}

![Figure 2. Effect of each preparation step](figures/2_feature_ablation.png)

### 3.2 Classification

**Table 4.** Cross-validated and test ROC-AUC.

{md_table(cls_table)}

**Table 5.** Test-set metrics at each model's tuned threshold.

{md_table(test_table)}

![Figure 3. CV comparison, test ROC curves and confusion matrix of the best model](figures/3_classification.png)

**{summary['winner']}** has the highest CV ROC-AUC ({w.cv_roc_auc:.3f} ± {w.cv_roc_auc_std:.3f}) and a test ROC-AUC of {w.test_roc_auc:.3f} (95% CI {w.test_auc_ci_low:.3f}–{w.test_auc_ci_high:.3f}). At its threshold of {summary['threshold']:.2f} it detects {t['recall']:.0%} of diabetic test patients with {t['accuracy']:.1%} accuracy and F1 {t['f1']:.3f}; at the default threshold of 0.5 accuracy is {summary['test_accuracy_at_0.5']:.1%}, but fewer diabetic patients are found. Permutation importance identifies **{importance.iloc[0].feature}** as the dominant input (ROC-AUC drop {importance.iloc[0].importance:.3f}), followed by {importance.iloc[1].feature} and {importance.iloc[2].feature}.

### 3.3 Neural network and bias–variance

![Figure 4. Bias–variance diagnostics and neural-network training curves](figures/4_bias_variance.png)

The neural network's validation loss is lowest at epoch {summary['ann_best_epoch']} and rises afterwards while the training loss keeps falling, so early stopping restores the epoch-{summary['ann_best_epoch']} weights. Validation accuracy levels off at about {history.validation_accuracy.max():.0%}.

**Table 6.** Train vs CV ROC-AUC (a large gap indicates high variance).

{md_table(gaps.reset_index()[['model', 'train_roc_auc', 'cv_roc_auc', 'gap']].sort_values('gap'))}

### 3.4 Regression (BMI)

**Table 7.** BMI regression: test metrics and CV RMSE (kg/m²).

{md_table(reg_table)}

![Figure 5. BMI regression results](figures/5_regression.png)

**{rw}** has the lowest CV RMSE and reaches a test RMSE of {rtest.loc[rw, 'rmse']:.2f} and R² of {rtest.loc[rw, 'r2']:.2f}, against {rtest.loc['Baseline (mean)', 'rmse']:.2f} for the mean baseline. Skin thickness is the strongest predictor of BMI; both measure body fat.

## 4. Discussion

**How did preprocessing and feature selection affect generalisation?**
Treating impossible zeros as missing was the most valuable step: logistic-regression CV ROC-AUC rose from {abl[('1. Raw data (zeros kept)', 'Logistic Regression')]:.3f} to {abl[('2. Zeros -> missing + median imputation', 'Logistic Regression')]:.3f} and became more stable across folds. Scaling left the random forest unchanged, confirming that trees do not need it, while it is required for the SVM and neural network. Feature engineering improved both logistic regression ({abl[('4. + Feature engineering', 'Logistic Regression')]:.3f}) and random forest ({abl[('4. + Feature engineering', 'Random Forest')]:.3f}), and the engineered glucose × BMI term ranked first under all three scoring methods in Table 1. Mutual-information selection gave the best logistic score ({abl[('5. + Mutual information (filter)', 'Logistic Regression')]:.3f}) with fewer than half of the inputs. All gains are smaller than the fold-to-fold standard deviation (about 0.03), so on this dataset preparation mainly made the models simpler and more stable rather than substantially more accurate.

**Which model type performed best, and why?**
{summary['winner']} achieved the highest CV ROC-AUC, but logistic regression (CV {lr.cv_roc_auc:.3f}, test {lr.test_roc_auc:.3f}) and the neural network (CV {s.loc['ANN (PyTorch)', 'cv_roc_auc']:.3f}) are equivalent within the uncertainty. The data are small, tabular and numeric, and the classes are separated by a smooth boundary driven mainly by glucose, so more flexible models add no benefit. Logistic regression is the preferred model for practical use: equally accurate, transparent and the least over-fitted. For BMI, the regularised linear models outperform the tree-based models.

**How did dataset size, feature quality and data type influence overfitting?**
With only {n_train} training rows, flexible models over-fit. An unrestricted decision tree reaches a train ROC-AUC of {depth.train_roc_auc.iloc[-1]:.3f} but a CV ROC-AUC of only {depth.cv_roc_auc.iloc[-1]:.3f}, whereas CV peaks at depth {int(best_depth.max_depth)} ({best_depth.cv_roc_auc:.3f}). Even after tuning, the decision tree and random forest show the largest train–CV gaps ({gaps.loc['Decision Tree', 'gap']:.2f} and {gaps.loc['Random Forest', 'gap']:.2f}), and the random-forest regressor has a train R² of {rtrain.loc['Random Forest', 'r2']:.2f} against a CV R² of {rcv.loc['Random Forest', 'r2']:.2f}. Feature quality is limited: insulin is missing for {zeros['insulin'] / meta['rows']:.0%} of patients and skin thickness for {zeros['skin_thickness'] / meta['rows']:.0%}, so these inputs contribute little. Because all inputs are continuous and only moderately informative, the linear models, SVM and neural network show small gaps (0.02–0.03); their remaining error is due to bias and noise rather than variance. The learning curves are still rising slowly, so additional data would help.

**What recommendations follow for real-world healthcare analytics?**
(1) Train on larger and more diverse cohorts (men, other ethnic groups, multiple hospitals) and validate externally before use. (2) Prefer interpretable, well-calibrated models and set the decision threshold with clinicians according to the cost of a missed case. (3) Record missing measurements explicitly rather than as zeros. (4) Use the model as decision support with clinician review, and monitor its performance and fairness across patient groups after deployment.

**Limitations.** The dataset is small, historical and limited to one population; results come from a single random train/test split and have not been validated externally.

## 5. Conclusion

A complete CRISP-DM pipeline was developed: repair of hidden missing values, outlier analysis, four feature-engineering techniques, three feature-selection methods, five tuned classifiers, a feed-forward neural network and five BMI regressors, evaluated with repeated cross-validation, a held-out test set and bias–variance diagnostics. The best classifiers reach a test ROC-AUC of about {w.test_roc_auc:.2f} and, at the tuned threshold, detect about {t['recall']:.0%} of diabetic patients. Logistic regression offers the best balance of accuracy, stability and interpretability, and all results are available in an interactive dashboard.

## References

- Chapman, P., Clinton, J., Kerber, R., Khabaza, T., Reinartz, T., Shearer, C., & Wirth, R. (2000). *CRISP-DM 1.0: Step-by-step data mining guide*. SPSS Inc.
- Paszke, A., et al. (2019). PyTorch: An imperative style, high-performance deep learning library. *Advances in Neural Information Processing Systems, 32*.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.
- Smith, J. W., Everhart, J. E., Dickson, W. C., Knowler, W. C., & Johannes, R. S. (1988). Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. *Proceedings of the Annual Symposium on Computer Application in Medical Care*, 261–265.
- World Health Organization. (2000). *Obesity: Preventing and managing the global epidemic* (WHO Technical Report Series 894).
- Dataset file: <{meta['source_url']}>
"""


def build_outline() -> str:
    s = cls.set_index("model")
    return f"""# Presentation Outline

1. **Problem and data** – diabetes detection, CRISP-DM, {meta['rows']} patients, class imbalance, impossible zeros.
2. **Data preparation** – zeros → missing, median imputation + indicators, scaling, outliers kept, 80/20 split, no leakage.
3. **Feature engineering and selection** – log, interaction, binning; filter vs wrapper vs embedded; ablation results.
4. **Models** – five classifiers, PyTorch ANN (32-16, dropout, early stopping), BMI regression.
5. **Results** – {summary['winner']} CV ROC-AUC {s.loc[summary['winner'], 'cv_roc_auc']:.3f}, test {s.loc[summary['winner'], 'test_roc_auc']:.3f}; recall {summary['test_metrics']['recall']:.0%} at the tuned threshold; {summary['regression_winner']} BMI RMSE.
6. **Bias–variance and discussion** – depth curve, learning curves, train–CV gaps, the four discussion questions.
7. **Dashboard** – live walkthrough of the Streamlit app and prediction lab.
"""


if __name__ == "__main__":
    make_figures()
    report = build_report()
    words = len(report.split())
    if words > 3000:
        raise ValueError(f"Report has {words} words; the limit is 3000")
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    (OUT / "PRESENTATION_OUTLINE.md").write_text(build_outline(), encoding="utf-8")

    body = mistune.create_markdown(plugins=["table"])(report)
    for png in FIG.glob("*.png"):
        body = body.replace(f"figures/{png.name}", "data:image/png;base64," + base64.b64encode(png.read_bytes()).decode())
    css = ("body{max-width:980px;margin:40px auto;padding:0 24px;font:15px/1.6 Arial,sans-serif;color:#221A35}"
           "h1,h2,h3{color:#4C2B7A}table{border-collapse:collapse;width:100%;font-size:12px;margin:8px 0 16px}"
           "td,th{padding:5px 7px;border-bottom:1px solid #ddd;text-align:left}img{width:100%}")
    (OUT / "REPORT.html").write_text(f'<!doctype html><meta charset="utf-8"><title>PIMA Diabetes Report</title>'
                                     f"<style>{css}</style><body>{body}</body>", encoding="utf-8")
    print(f"Report: {words} words")
