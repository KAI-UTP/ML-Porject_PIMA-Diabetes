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
from sklearn.metrics import roc_curve

from data_pipeline import ROOT

A, OUT = ROOT / "artifacts", ROOT / "reports"
FIG = OUT / "figures"
PURPLE, ORANGE, TEAL, GREY = "#7147B8", "#E1785C", "#3B8C88", "#A9A0B5"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130, "axes.titleweight": "bold"})

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
    for value, color in [(0, PURPLE), (1, ORANGE)]:
        ax[0].hist(cohort.loc[cohort.outcome == value, "glucose"].dropna(), bins=25, alpha=0.6, color=color,
                   label=["No diabetes", "Diabetes"][value])
    ax[0].set(title="Glucose by outcome", xlabel="glucose (mg/dL)", ylabel="patients")
    ax[0].legend()
    zeros = pd.Series(meta["audit"]["zero_as_missing"]) / meta["rows"] * 100
    ax[1].barh(zeros.index, zeros.values, color=PURPLE)
    ax[1].set(title="Impossible zeros (treated as missing)", xlabel="% of rows")
    save(fig, "1_data.png")

    fig, ax = plt.subplots(figsize=(10, 4.2))
    for offset, (model, part) in zip([-0.15, 0.15], ablation.groupby("model")):
        y = np.arange(len(part)) + offset
        ax.errorbar(part.cv_roc_auc, y, xerr=part.cv_std, fmt="o", capsize=3, label=model)
    ax.set_yticks(np.arange(len(part)), part.variant)
    ax.invert_yaxis()
    ax.set(title="Effect of each preparation step (5×3 repeated CV)", xlabel="ROC-AUC (mean ± std)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    save(fig, "2_feature_ablation.png")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    c = cls[~cls.model.str.startswith("Baseline")].sort_values("cv_roc_auc")
    ax[0].barh(c.model, c.cv_roc_auc, xerr=c.cv_roc_auc_std, color=[PURPLE if m == summary["winner"] else GREY for m in c.model])
    ax[0].set(title="Cross-validation ROC-AUC", xlim=(0.6, 0.95))
    test = preds[preds.split == "test"]
    for model in c.model[::-1]:
        p = test[test.model == model]
        fpr, tpr, _ = roc_curve(p.actual, p.probability)
        ax[1].plot(fpr, tpr, label=f"{model} ({cls.set_index('model').loc[model, 'test_roc_auc']:.3f})")
    ax[1].plot([0, 1], [0, 1], "k--", lw=1)
    ax[1].set(title="Test ROC curves", xlabel="false-positive rate", ylabel="recall")
    ax[1].legend(fontsize=7)
    save(fig, "3_classification.png")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(depth.max_depth, depth.train_roc_auc, "o-", label="train")
    ax[0].plot(depth.max_depth, depth.cv_roc_auc, "o-", label="CV")
    ax[0].set(title="Decision-tree depth", xlabel="max_depth", ylabel="ROC-AUC")
    ax[0].legend()
    for model, color in [("Logistic Regression", PURPLE), ("Random Forest", TEAL)]:
        part = curves[curves.model == model]
        ax[1].plot(part.n_train, part.train_roc_auc, "--", color=color, label=f"{model} train")
        ax[1].plot(part.n_train, part.cv_roc_auc, "o-", color=color, label=f"{model} CV")
    ax[1].set(title="Learning curves", xlabel="training rows")
    ax[1].legend(fontsize=7)
    ax[2].plot(history.epoch, history.train_loss, label="train")
    ax[2].plot(history.epoch, history.validation_loss, label="validation")
    ax[2].axvline(summary["ann_best_epoch"], ls="--", c="grey")
    ax[2].set(title="ANN loss (early stopping)", xlabel="epoch")
    ax[2].legend()
    save(fig, "4_bias_variance.png")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    r = reg[reg.split == "test"].sort_values("rmse", ascending=False)
    ax[0].barh(r.model, r.rmse, color=TEAL)
    ax[0].set(title="BMI regression: test RMSE", xlabel="RMSE (kg/m²)")
    p = reg_pred[(reg_pred.model == summary["regression_winner"]) & (reg_pred.split == "test")]
    ax[1].scatter(p.actual, p.predicted, alpha=0.6, color=TEAL)
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

    return f"""# Diabetes Prediction on the PIMA Indians Dataset

**Course:** TDB3123 Machine Learning · **Group members:** _add names and student IDs_

## 1. Introduction

Type 2 diabetes is often detected late. A model that flags high-risk patients from routine measurements can help decide who needs a confirmatory test. This project applies the CRISP-DM process to the PIMA Indians Diabetes dataset (used in Labs 1 and 3) to:

1. **classify** whether a patient has diabetes (main task),
2. **regress** body-mass index (BMI) from the other measurements (secondary task), and
3. compare classical models with a **feed-forward neural network**.

Success is measured with ROC-AUC and recall, always against a baseline, and over-fitting is studied with learning and validation curves.

## 2. Methodology

### 2.1 Data understanding

The dataset has {meta['rows']} patients (women of Pima heritage aged 21+), 8 numeric measurements and a binary outcome. {meta['class_counts']['1']} patients ({meta['class_counts']['1'] / meta['rows']:.1%}) have diabetes, so a model that always says "no" is already {1 - meta['class_counts']['1'] / meta['rows']:.1%} accurate; accuracy alone is therefore misleading. Several columns contain **impossible zeros**: insulin ({zeros['insulin']} rows), skin thickness ({zeros['skin_thickness']}), blood pressure ({zeros['blood_pressure']}), BMI ({zeros['bmi']}) and glucose ({zeros['glucose']}). Glucose is the strongest single signal (correlation 0.50 with the outcome; median 140 vs 107 mg/dL).

![Data understanding](figures/1_data.png)

### 2.2 Data preparation

- **Cleaning:** types checked, no duplicates or invalid rows; impossible zeros converted to missing values.
- **Split:** stratified 80/20 split ({n_train} training, {n_test} test patients, seed 42). The test set is used only once, at the end. All tuning, model selection and threshold choice use **5-fold stratified cross-validation (CV)** on the training set.
- **Missing values:** median imputation plus missing-value indicator columns.
- **Encoding and normalization:** one-hot encoding for binned features; `StandardScaler` for logistic regression, SVM and the ANN (trees are scale-invariant).
- **Outliers:** 1.5 × IQR fences from the training set flag {int(outliers[outliers.split == 'train'].outliers.sum())} values (mostly pedigree, insulin and blood pressure). They are medically plausible, so they are **kept**; a RobustScaler was tested as an alternative.
- All steps are inside one scikit-learn `Pipeline`, so imputation and scaling are learned from the training folds only (no data leakage).

### 2.3 Feature engineering and selection

Four engineering techniques were applied: **log transform** of insulin and pedigree (insulin skew 2.00 → −0.17), an **interaction term** glucose × BMI, **binning** of age (21–30, 31–40, 41–50, 51+) and BMI (WHO classes), and **missing indicators**. This gives 23 model inputs. Three selection methods (each ≈10 features) were compared: **filter** (mutual information), **wrapper** (recursive feature elimination with logistic regression) and **embedded** (L1/LASSO logistic regression).

{md_table(sel_table)}

All methods keep glucose, and the engineered glucose × BMI feature ranks first for mutual information, LASSO and random-forest importance.

### 2.4 Models

| Task | Models (tuned with GridSearchCV, 5-fold CV) |
|---|---|
| Classification | Baseline (prior), Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVM (RBF) |
| Neural network | PyTorch: inputs → 32 ReLU → dropout 0.2 → 16 ReLU → dropout 0.2 → 1 sigmoid; Adam, binary cross-entropy, batch 32, weight decay, early stopping (patience 25) |
| Regression (BMI) | Baseline (mean), Linear Regression, Ridge, Lasso, Decision Tree, Random Forest |

Feature selection ({summary['selector']}) is a tuned option in every classifier grid, so CV decides whether it helps. The best classifier is the one with the highest mean ROC-AUC over 5-fold CV repeated 3 times. Its decision threshold maximises F1 on out-of-fold predictions. For regression, BMI and BMI-based features are excluded from the inputs.

## 3. Results

### 3.1 Effect of preprocessing and feature engineering

{md_table(abl_table)}

![Feature ablation](figures/2_feature_ablation.png)

### 3.2 Classification

{md_table(cls_table)}

Test-set metrics at each model's tuned threshold:

{md_table(test_table)}

![Classification results](figures/3_classification.png)

The best model by CV is **{summary['winner']}** (CV ROC-AUC {w.cv_roc_auc:.3f} ± {w.cv_roc_auc_std:.3f}; test {w.test_roc_auc:.3f}, 95% bootstrap CI {w.test_auc_ci_low:.3f}–{w.test_auc_ci_high:.3f}). With its threshold of {summary['threshold']:.2f} it detects {t['recall']:.0%} of diabetic test patients with {t['accuracy']:.1%} accuracy (F1 {t['f1']:.3f}); at the default 0.5 threshold accuracy is {summary['test_accuracy_at_0.5']:.1%} but fewer diabetics are found. The ANN stopped early at epoch {summary['ann_best_epoch']}. The most important input (permutation importance) is **{importance.iloc[0].feature}** (ROC-AUC drop {importance.iloc[0].importance:.3f}), followed by {importance.iloc[1].feature} and {importance.iloc[2].feature}.

### 3.3 Regression (BMI)

{md_table(reg_table)}

![Regression results](figures/5_regression.png)

**{rw}** has the lowest CV RMSE; on the test set it reaches RMSE {rtest.loc[rw, 'rmse']:.2f} and R² {rtest.loc[rw, 'r2']:.2f}, compared with RMSE {rtest.loc['Baseline (mean)', 'rmse']:.2f} for the mean baseline. Skin thickness is the strongest BMI predictor.

### 3.4 Bias–variance

![Bias-variance](figures/4_bias_variance.png)

{md_table(gaps.reset_index()[['model', 'train_roc_auc', 'cv_roc_auc', 'gap']].sort_values('gap'))}

## 4. Discussion

**How did preprocessing and feature selection affect generalization?**
Treating impossible zeros as missing was the most valuable step: logistic-regression CV ROC-AUC rose from {abl[('1. Raw data (zeros kept)', 'Logistic Regression')]:.3f} to {abl[('2. Zeros -> missing + median imputation', 'Logistic Regression')]:.3f} and became more stable across folds. Scaling did not change the score of the random forest, confirming that trees do not need it, while it is required for the SVM and ANN. Feature engineering improved both logistic regression ({abl[('4. + Feature engineering', 'Logistic Regression')]:.3f}) and random forest ({abl[('4. + Feature engineering', 'Random Forest')]:.3f}). Mutual-information selection gave the best logistic score ({abl[('5. + Mutual information (filter)', 'Logistic Regression')]:.3f}) with fewer inputs. All these gains are smaller than the fold-to-fold standard deviation (≈0.03), so on this small dataset preprocessing mainly made the models simpler and more stable rather than much more accurate.

**Which model type performed best and why?**
{summary['winner']} had the highest CV ROC-AUC, but logistic regression (CV {lr.cv_roc_auc:.3f}, test {lr.test_roc_auc:.3f}) and the ANN (CV {s.loc['ANN (PyTorch)', 'cv_roc_auc']:.3f}) perform the same within the uncertainty. The data are small, tabular and numeric, and the classes are separated by a smooth boundary driven mainly by glucose, so complex models add no benefit. For healthcare use, **logistic regression** is the recommended model: equally accurate, transparent and the least over-fitted. For BMI, the regularised linear models beat the trees.

**How did dataset size, feature quality and data type influence overfitting?**
With only {n_train} training rows, flexible models over-fit. An unrestricted decision tree reaches train ROC-AUC {depth.train_roc_auc.iloc[-1]:.3f} but CV {depth.cv_roc_auc.iloc[-1]:.3f}, while CV peaks at depth {int(best_depth.max_depth)} ({best_depth.cv_roc_auc:.3f}). The tuned tree and random forest still show the largest train–CV gaps ({gaps.loc['Decision Tree', 'gap']:.2f} and {gaps.loc['Random Forest', 'gap']:.2f}); the random-forest regressor has train R² {rtrain.loc['Random Forest', 'r2']:.2f} but CV R² {rcv.loc['Random Forest', 'r2']:.2f}. Feature quality is limited: insulin is missing for {zeros['insulin'] / meta['rows']:.0%} of patients and skin thickness for {zeros['skin_thickness'] / meta['rows']:.0%}, so they add little signal. Because the inputs are continuous and only moderately informative, the linear models, SVM and ANN have small gaps (≈0.02–0.03): their remaining error is bias and noise, not variance. The learning curves still rise slowly, so more data would help.

**What recommendations would you make for real-world healthcare analytics?**
(1) Train on larger and more diverse data (men, other populations, several hospitals) and validate externally. (2) Prefer interpretable and well-calibrated models, and choose the decision threshold together with clinicians according to the cost of a missed case. (3) Record missing measurements explicitly instead of as zeros. (4) Monitor performance and fairness after deployment and keep a clinician in the loop: the model is a screening aid, not a diagnosis.

## 5. Conclusion

A full CRISP-DM pipeline was built: cleaning of hidden missing values, outlier analysis, four feature-engineering techniques, three feature-selection methods, six tuned classifiers, a PyTorch ANN and six BMI regressors, evaluated with repeated cross-validation, a held-out test set and bias–variance diagnostics. The best classifiers reach a test ROC-AUC of about {w.test_roc_auc:.2f} and detect about {t['recall']:.0%} of diabetic patients at the tuned threshold. Logistic regression offers the best balance of accuracy, stability and interpretability. An interactive Streamlit dashboard presents all results and a prediction lab.

## References

- PIMA Indians Diabetes dataset: <{meta['source_url']}>
- TDB3123 lecture notes, Chapters 2–5 and 8; Labs 1, 3, 4 and 5.
- scikit-learn User Guide: pipelines, cross-validation and common pitfalls. <https://scikit-learn.org/stable/common_pitfalls.html>
- PyTorch tutorials: building and training neural networks. <https://docs.pytorch.org/tutorials/>
"""


def build_outline() -> str:
    s = cls.set_index("model")
    return f"""# Presentation Outline (one part per member)

1. **Problem and data** – diabetes screening, CRISP-DM, {meta['rows']} patients, class imbalance, impossible zeros.
2. **Data preparation** – zeros → missing, median imputation + indicators, scaling, outliers kept, 80/20 split, no leakage.
3. **Feature engineering and selection** – log, interaction, binning; filter vs wrapper vs embedded; ablation results.
4. **Models** – six classifiers, PyTorch ANN (32-16, dropout, early stopping), BMI regression.
5. **Results** – {summary['winner']} CV ROC-AUC {s.loc[summary['winner'], 'cv_roc_auc']:.3f}, test {s.loc[summary['winner'], 'test_roc_auc']:.3f}; recall {summary['test_metrics']['recall']:.0%} at the tuned threshold; {summary['regression_winner']} BMI RMSE.
6. **Bias–variance and discussion** – depth curve, learning curves, train–CV gaps, the four discussion questions.
7. **Live demo** – Streamlit dashboard and prediction lab.
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
