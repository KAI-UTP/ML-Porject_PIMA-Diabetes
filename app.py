"""Streamlit dashboard for the PIMA diabetes project. Run `python train.py` first."""
import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from data_pipeline import FEATURES, ROOT, UNITS, ZERO_AS_MISSING
from chart_style import ACCENT, DIVERGING, MODEL_COLORS, NEUTRAL, OUTCOME_COLORS, SERIES

A = ROOT / "artifacts"
PURPLE, ORANGE, GREY = ACCENT, SERIES[1], NEUTRAL
OUTCOME = {0: "No diabetes", 1: "Diabetes"}
label = lambda name: name.replace("_", " ").capitalize()

st.set_page_config(page_title="PIMA Diabetes ML", page_icon="🩺", layout="wide")
st.markdown("""<style>
.block-container{max-width:1350px;padding-top:2rem}
.hero{padding:28px 34px;border-radius:18px;background:linear-gradient(120deg,#2A174B,#7147B8);color:#fff;margin-bottom:1rem}
.hero h1{color:#fff;margin:0 0 .3rem;font-size:clamp(1.5rem,4vw,2.6rem);word-break:normal;overflow-wrap:normal;hyphens:none}.hero p{margin:0;opacity:.9}
@media (max-width:640px){.hero{padding:18px 18px}}
.step{background:#fff;border:1px solid #e2dced;border-radius:12px;padding:12px 14px;height:100%}
.step b{color:#7147B8}
[data-testid="stMetricValue"]{font-size:clamp(1.3rem,2.4vw,2.2rem)}
[data-testid="stMetricLabel"] p,[data-testid="stMetricDelta"] div{white-space:normal}
</style>""", unsafe_allow_html=True)
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = SERIES


# ---------------------------------------------------------------- Data loading

@st.cache_data
def table(name):
    return pd.read_csv(A / name)


@st.cache_data
def load_core():
    cohort = pd.read_csv(ROOT / "data/processed/cohort.csv")
    meta = json.loads((ROOT / "data/processed/provenance.json").read_text())
    summary = json.loads((A / "run_summary.json").read_text())
    return cohort, meta, summary


@st.cache_resource
def classifiers():
    return joblib.load(A / "classifiers.joblib")


def show(fig, height=None):
    fig.update_layout(margin=dict(l=10, r=20, t=50, b=10), height=height,
                      legend=dict(orientation="h", yanchor="top", y=-0.18, x=0, title=None))
    st.plotly_chart(fig, width="stretch")


def live_metrics(y, p, threshold):
    tn, fp, fn, tp = confusion_matrix(y, p >= threshold, labels=[0, 1]).ravel()
    return {"Accuracy": (tp + tn) / len(y), "Recall": tp / max(tp + fn, 1), "Precision": tp / max(tp + fp, 1),
            "Specificity": tn / max(tn + fp, 1), "F1": 2 * tp / max(2 * tp + fp + fn, 1)}


if not (A / "run_summary.json").exists():
    st.info("No results yet. Run `python train.py` first.")
    st.stop()
cohort, meta, summary = load_core()
cls_summary = table("classification_summary.csv")
WINNER = summary["winner"]
thresholds = cls_summary.set_index("model").threshold.to_dict()


# ---------------------------------------------------------------- Pages

def page_overview():
    best = cls_summary.set_index("model").loc[WINNER]
    t = summary["test_metrics"]
    st.markdown('<div class="hero"><h1>Diabetes Prediction · PIMA Indians</h1>'
                "<p>CRISP-DM pipeline: data cleaning, feature engineering and selection, five classifiers, "
                "a PyTorch neural network and BMI regression, with bias–variance analysis.</p></div>",
                unsafe_allow_html=True)
    off = dict(delta_color="off", delta_arrow="off", border=True)
    c = st.columns(3)
    c[0].metric("Patients", meta["rows"], f"{meta['split_counts']['train']}/{meta['split_counts']['test']} split", **off)
    c[1].metric("With diabetes", meta["class_counts"]["1"], f"{meta['class_counts']['1'] / meta['rows']:.0%} of patients", **off)
    c[2].metric("Missing insulin", f"{meta['audit']['zero_as_missing']['insulin'] / meta['rows']:.0%}", border=True)

    st.subheader(f"Best model: {WINNER}")
    st.caption("Highest mean ROC-AUC in 5-fold cross-validation (3 repeats) on the training set.")
    c = st.columns(3)
    c[0].metric("Test ROC-AUC", f"{best.test_roc_auc:.3f}", f"95% CI {best.test_auc_ci_low:.2f}–{best.test_auc_ci_high:.2f}",
                delta_color="off", delta_arrow="off", border=True)
    c[1].metric("Test recall", f"{t['recall']:.0%}", f"threshold {t['threshold']:.2f}", delta_color="off", delta_arrow="off", border=True)
    c[2].metric("Test accuracy", f"{t['accuracy']:.1%}", f"{summary['test_accuracy_at_0.5']:.1%} at 0.5", delta_color="off", delta_arrow="off", border=True)

    st.subheader("Pipeline")
    steps = [("1 · Clean", "Impossible zeros → missing; types, duplicates and ranges checked"),
             ("2 · Split", "Stratified 80/20; test set used once at the end"),
             ("3 · Prepare", "Median imputation + missing flags, scaling, one-hot encoding"),
             ("4 · Engineer", "Log transform, glucose × BMI, age and BMI bins"),
             ("5 · Select", "Mutual information vs RFE vs LASSO"),
             ("6 · Model", "Grid search with 5-fold CV; F1 threshold from out-of-fold predictions")]
    for row in (steps[:3], steps[3:]):
        for col, (title, text) in zip(st.columns(3), row):
            col.markdown(f'<div class="step"><b>{title}</b><br><small>{text}</small></div>', unsafe_allow_html=True)

    st.subheader("All classifiers")
    c = cls_summary[~cls_summary.model.str.startswith("Baseline")].sort_values("cv_roc_auc")
    fig = go.Figure()
    fig.add_bar(y=c.model, x=c.cv_roc_auc, orientation="h", name="CV ROC-AUC (mean ± std)",
                error_x=dict(array=c.cv_roc_auc_std), marker_color=[PURPLE if m == WINNER else GREY for m in c.model])
    fig.add_scatter(y=c.model, x=c.test_roc_auc, mode="markers", name="Test ROC-AUC",
                    marker=dict(symbol="diamond", size=12, color=ORANGE))
    fig.update_xaxes(range=[0.6, 0.95], title="ROC-AUC")
    show(fig, 380)
    st.caption("Differences between the top models are smaller than the CV standard deviation: SVM, logistic "
               "regression and the ANN perform equally well. Logistic regression is the most interpretable.")


def page_data():
    st.title("Data explorer")
    a, b = st.columns(2)
    splits = a.multiselect("Split", ["train", "test"], default=["train", "test"])
    outcomes = b.multiselect("Outcome", [0, 1], default=[0, 1], format_func=OUTCOME.get)
    d = cohort[cohort.split.isin(splits) & cohort.outcome.isin(outcomes)]
    if d.empty:
        st.info("No rows match the filters.")
        return
    d = d.assign(Outcome=d.outcome.map(OUTCOME))

    feature = st.selectbox("Feature", FEATURES, index=1, format_func=label)
    a, b = st.columns(2)
    with a:
        show(px.histogram(d, x=feature, color="Outcome", nbins=30, barmode="overlay", opacity=0.7, color_discrete_map=OUTCOME_COLORS,
                          title=f"{label(feature)} ({UNITS[feature]}) by outcome"))
    with b:
        show(px.box(d, x="Outcome", y=feature, color="Outcome", points="outliers", color_discrete_map=OUTCOME_COLORS,
                    title=f"{label(feature)}: spread and outliers"))

    a, b = st.columns(2)
    with a:
        missing = pd.DataFrame({"feature": [label(c) for c in ZERO_AS_MISSING],
                                "missing %": [100 * d[c].isna().mean() for c in ZERO_AS_MISSING]}).sort_values("missing %")
        show(px.bar(missing, x="missing %", y="feature", orientation="h", text_auto=".1f",
                    title="Missing values (impossible zeros)"))
    with b:
        show(px.imshow(d[FEATURES + ["outcome"]].corr().round(2), text_auto=True, zmin=-1, zmax=1,
                       color_continuous_scale="RdBu_r", title="Correlation"))

    with st.expander("Outlier audit (1.5 × IQR fences from the training split)"):
        st.dataframe(table("outlier_audit.csv"), hide_index=True, width="stretch")
        st.caption("Outliers are flagged but kept: they are rare but medically plausible values.")
    with st.expander("Cleaning audit"):
        st.json(meta["audit"])
    with st.expander(f"Rows ({len(d)})"):
        st.dataframe(d.drop(columns="Outcome"), hide_index=True, width="stretch")


def page_features():
    st.title("Feature engineering & selection")
    st.markdown("""
| Technique | New feature | Why |
|---|---|---|
| Log transform | `log_insulin`, `log_pedigree` | Reduce strong right skew |
| Interaction | `glucose_x_bmi` | High glucose is more serious with high BMI |
| Binning | `age_band`, `bmi_class` (WHO) | Diabetes rate rises across age and BMI groups |
| Missing indicators | `missingindicator_*` | "Not measured" can itself carry information |
""")
    ab = table("feature_ablation.csv")
    fig = px.scatter(ab, x="cv_roc_auc", y="variant", color="model", error_x="cv_std", color_discrete_map=MODEL_COLORS,
                     title="Impact of each preparation step (5-fold CV × 3 repeats, training split)",
                     labels={"cv_roc_auc": "CV ROC-AUC", "variant": ""})
    fig.update_traces(marker_size=11)
    fig.update_yaxes(autorange="reversed")
    show(fig, 480)
    st.caption(f"Zeros → missing gives the clearest gain for logistic regression; scaling does not change the "
               f"random forest (trees are scale-invariant). Selection method used in the final models: "
               f"**{summary['selector']}** (offered as a tuned option to every classifier).")

    sel = table("feature_selection.csv")
    a, b = st.columns([3, 2])
    with a:
        metric = st.radio("Score", ["mutual_information", "rf_importance", "lasso_coefficient"], horizontal=True,
                          format_func=lambda m: {"mutual_information": "Mutual information (filter)",
                                                 "rf_importance": "Random-forest importance",
                                                 "lasso_coefficient": "LASSO coefficient (embedded)"}[m])
        s = sel.sort_values(metric)
        show(px.bar(s, x=metric, y="feature", orientation="h", title="Feature scores", labels={metric: "", "feature": ""}), 620)
    with b:
        kept = sel.set_index("feature")[["Mutual information (filter)", "RFE (wrapper)", "LASSO L1 (embedded)"]]
        st.markdown("**Features kept by each method**")
        st.dataframe(kept.replace({True: "✅", False: ""}), width="stretch", height=620)

    imp = table("permutation_importance.csv").sort_values("importance")
    show(px.bar(imp, x="importance", y="feature", orientation="h", error_x="std",
                title=f"{WINNER}: drop in test ROC-AUC when one input is shuffled (permutation importance)"), 360)


def page_evaluation():
    st.title("Model evaluation")
    task = st.radio("Task", ["Diabetes classification", "BMI regression"], horizontal=True)
    if task == "BMI regression":
        return regression_view()

    view = cls_summary.copy()
    view["best_params"] = view.best_params.str.replace(r'[{}"]', "", regex=True)
    st.dataframe(view[["model", "cv_roc_auc", "cv_roc_auc_std", "train_roc_auc", "test_roc_auc", "test_auc_ci_low",
                       "test_auc_ci_high", "threshold", "best_params"]].round(3),
                 hide_index=True, width="stretch",
                 column_config={"cv_roc_auc": "CV ROC-AUC", "cv_roc_auc_std": "CV std", "train_roc_auc": "Train ROC-AUC",
                                "test_roc_auc": "Test ROC-AUC", "test_auc_ci_low": "Test CI low",
                                "test_auc_ci_high": "Test CI high", "threshold": "Tuned threshold",
                                "best_params": "Best hyper-parameters"})

    st.subheader("Inspect a model")
    a, b, c = st.columns([2, 1, 2])
    models = cls_summary.model.tolist()
    name = a.selectbox("Model", models, index=models.index(WINNER))
    split = b.radio("Data", ["test", "cv"], format_func={"test": "Test set", "cv": "CV (out-of-fold)"}.get)
    threshold = c.slider("Decision threshold", 0.05, 0.95, float(thresholds[name]), 0.0001, format="%.3f",
                         help="Default is the threshold that maximised F1 on out-of-fold predictions.")
    preds = table("classification_predictions.csv")
    p = preds[(preds.model == name) & (preds.split == split)]

    for col, (metric, value) in zip(st.columns(5), live_metrics(p.actual, p.probability, threshold).items()):
        col.metric(metric, f"{value:.1%}", border=True)

    a, b = st.columns(2)
    with a:
        cm = confusion_matrix(p.actual, p.probability >= threshold, labels=[0, 1])
        show(px.imshow(cm, text_auto=True, color_continuous_scale="Purples", x=list(OUTCOME.values()), y=list(OUTCOME.values()),
                       labels={"x": "Predicted", "y": "Actual"}, title=f"Confusion matrix · threshold {threshold:.2f}"))
    with b:
        fig = go.Figure()
        for model in models:
            q = preds[(preds.model == model) & (preds.split == split)]
            fpr, tpr, _ = roc_curve(q.actual, q.probability)
            fig.add_scatter(x=fpr, y=tpr, name=model, line=dict(width=4 if model == name else 1.5, color=MODEL_COLORS[model]),
                            opacity=1 if model == name else 0.45)
        fig.add_scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="grey"), showlegend=False)
        fig.update_layout(title="ROC curves", xaxis_title="False-positive rate", yaxis_title="Recall")
        show(fig)

    a, b = st.columns(2)
    with a:
        precision, recall, _ = precision_recall_curve(p.actual, p.probability)
        fig = go.Figure(go.Scatter(x=recall, y=precision, name=name, line_color=MODEL_COLORS[name]))
        fig.add_hline(y=p.actual.mean(), line_dash="dash", annotation_text="baseline (prevalence)")
        fig.update_layout(title="Precision–recall", xaxis_title="Recall", yaxis_title="Precision")
        show(fig)
    with b:
        observed, predicted = calibration_curve(p.actual, p.probability, n_bins=6, strategy="quantile")
        fig = go.Figure(go.Scatter(x=predicted, y=observed, mode="lines+markers", name=name, line_color=MODEL_COLORS[name]))
        fig.add_scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="grey"), name="Perfect calibration")
        fig.update_layout(title="Calibration (reliability)", xaxis_title="Mean predicted probability",
                          yaxis_title="Observed diabetes rate", xaxis_range=[0, 1], yaxis_range=[0, 1])
        show(fig)


def regression_view():
    m = table("regression_metrics.csv")
    st.caption("Target: BMI. Inputs: the other seven measurements and the diabetes outcome. Rows without BMI are excluded.")
    pivot = m.pivot(index="model", columns="split", values=["rmse", "r2", "mae"])
    pivot.columns = [f"{metric.upper()} ({split})" for metric, split in pivot.columns]
    st.dataframe(pivot.sort_values("RMSE (cv)").round(3), width="stretch")
    show(px.bar(m[m.split != "train"], x="model", y="rmse", color="split", barmode="group",
                title="RMSE by model (lower is better)", labels={"rmse": "RMSE (kg/m²)", "model": ""}))

    models = m.model.unique().tolist()
    name = st.selectbox("Inspect regressor", models, index=models.index(summary["regression_winner"]))
    p = table("regression_predictions.csv")
    p = p[(p.model == name) & (p.split == "test")]
    a, b = st.columns(2)
    with a:
        fig = px.scatter(p, x="actual", y="predicted", title="Actual vs predicted BMI (test)", opacity=0.7)
        fig.add_scatter(x=[15, 60], y=[15, 60], mode="lines", line=dict(dash="dash", color="grey"), name="Perfect")
        show(fig)
    with b:
        fig = px.scatter(p.assign(residual=p.actual - p.predicted), x="predicted", y="residual",
                         title="Residuals (test)", opacity=0.7)
        fig.add_hline(y=0, line_dash="dash")
        show(fig)


def page_bias_variance():
    st.title("Bias–variance analysis")
    gap = cls_summary[~cls_summary.model.str.startswith("Baseline")].assign(gap=lambda f: f.train_roc_auc - f.cv_roc_auc)
    a, b = st.columns(2)
    with a:
        fig = go.Figure()
        fig.add_bar(y=gap.model, x=gap.train_roc_auc, orientation="h", name="Train", marker_color=GREY)
        fig.add_bar(y=gap.model, x=gap.cv_roc_auc, orientation="h", name="CV", marker_color=PURPLE)
        fig.update_layout(title="Train vs CV ROC-AUC (gap = variance)", barmode="group", xaxis_range=[0.7, 1])
        show(fig)
    with b:
        depth = table("depth_curve.csv")
        fig = go.Figure()
        fig.add_scatter(x=depth.max_depth, y=depth.train_roc_auc, name="Train", mode="lines+markers", line_color=GREY)
        fig.add_scatter(x=depth.max_depth, y=depth.cv_roc_auc, name="CV", mode="lines+markers", line_color=PURPLE,
                        error_y=dict(array=depth.cv_std))
        best = depth.loc[depth.cv_roc_auc.idxmax()]
        fig.add_vline(x=best.max_depth, line_dash="dash", annotation_text=f"best depth {int(best.max_depth)}")
        fig.update_layout(title="Decision-tree depth: underfitting → overfitting", xaxis_title="max_depth",
                          yaxis_title="ROC-AUC")
        show(fig)

    curves = table("learning_curves.csv")
    fig = px.line(curves.melt(id_vars=["model", "n_train"], value_vars=["train_roc_auc", "cv_roc_auc"],
                              var_name="set", value_name="ROC-AUC").replace({"train_roc_auc": "Train", "cv_roc_auc": "CV"}),
                  x="n_train", y="ROC-AUC", color="set", facet_col="model", markers=True,
                  title="Learning curves", labels={"n_train": "training rows"},
                  color_discrete_map={"Train": GREY, "CV": PURPLE})
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    show(fig, 360)

    h = table("ann_history.csv")
    a, b = st.columns(2)
    for col, metric in [(a, "loss"), (b, "accuracy")]:
        with col:
            fig = px.line(h, x="epoch", y=[f"train_{metric}", f"validation_{metric}"], title=f"ANN {metric}",
                          color_discrete_sequence=[GREY, PURPLE])
            fig.add_vline(x=summary["ann_best_epoch"], line_dash="dash", annotation_text="early stop (best)")
            show(fig)
    st.markdown("""
- **High variance:** deep trees and the random forest score much better on training data than in CV.
- **Low variance:** logistic regression, SVM and the ANN have small gaps (≈0.02–0.03); their error is mostly bias and data noise.
- **More data:** CV curves still rise slowly with training size, so more patients would help a little.
- **ANN:** validation loss stops improving after a few dozen epochs; early stopping keeps the best weights.
""")


def page_prediction():
    st.title("Prediction lab")
    models = [m for m in cls_summary.model if not m.startswith("Baseline")]
    name = st.selectbox("Model", models, index=models.index(WINNER))

    with st.form("patient"):
        st.caption("Leave a field empty if it was not measured; it is imputed with the training median.")
        c = st.columns(4)
        values = {
            "pregnancies": c[0].number_input("Pregnancies", 0, 20, 2),
            "glucose": c[1].number_input("Glucose (mg/dL)", 40.0, 250.0, 120.0),
            "blood_pressure": c[2].number_input("Diastolic blood pressure (mm Hg)", 20.0, 140.0, 70.0),
            "skin_thickness": c[3].number_input("Skin thickness (mm)", 5.0, 100.0, None, placeholder="not measured"),
            "insulin": c[0].number_input("Insulin (µU/mL)", 10.0, 900.0, None, placeholder="not measured"),
            "bmi": c[1].number_input("BMI (kg/m²)", 15.0, 70.0, 30.0),
            "diabetes_pedigree": c[2].number_input("Diabetes pedigree", 0.05, 2.5, 0.45),
            "age": c[3].number_input("Age (years)", 21, 90, 35),
        }
        submitted = st.form_submit_button("Predict", type="primary")
    if submitted:
        st.session_state.patient_inputs = values
    if "patient_inputs" not in st.session_state:
        return

    x = pd.DataFrame([st.session_state.patient_inputs], dtype=float)
    fitted = classifiers()
    prob = lambda model, frame: fitted[model].predict_proba(frame)[:, 1]
    p, threshold = float(prob(name, x)[0]), thresholds[name]

    a, b, c = st.columns(3)
    a.metric("Predicted probability", f"{p:.1%}", border=True)
    b.metric("Model threshold", f"{threshold:.1%}", border=True)
    c.metric("Predicted class", OUTCOME[int(p >= threshold)], border=True)

    tab1, tab2, tab3 = st.tabs(["All models", "What-if", "Input sensitivity"])
    with tab1:
        comp = pd.DataFrame({"model": models, "probability": [float(prob(m, x)[0]) for m in models],
                             "threshold": [thresholds[m] for m in models]}).sort_values("probability")
        fig = go.Figure()
        fig.add_bar(y=comp.model, x=comp.probability, orientation="h", text=comp.probability, texttemplate="%{text:.0%}",
                    marker_color=[PURPLE if m == name else GREY for m in comp.model], name="Probability")
        fig.add_scatter(y=comp.model, x=comp.threshold, mode="markers", name="Threshold",
                        marker=dict(symbol="diamond", size=12, color=ORANGE))
        fig.update_layout(title="Same patient, every model", xaxis_range=[0, 1], xaxis_tickformat=".0%")
        show(fig)
        test = table("classification_predictions.csv")
        test = test[(test.model == name) & (test.split == "test")]
        fig = px.histogram(test, x="probability", color=test.actual.map(OUTCOME), nbins=20, barmode="overlay",
                           color_discrete_map=OUTCOME_COLORS, labels={"color": "Outcome"},
                           opacity=0.7, title="This patient among the 154 test patients")
        fig.add_vline(x=p, line_width=3, line_color=PURPLE, annotation_text="this patient")
        fig.add_vline(x=threshold, line_dash="dash", line_color=ORANGE, annotation_text="threshold")
        show(fig)
    with tab2:
        feature = st.selectbox("Change one input", FEATURES, index=1, format_func=label)
        train = cohort[cohort.split == "train"]
        grid = np.linspace(*train[feature].quantile([0.02, 0.98]), 60)
        if feature in ("pregnancies", "age"):
            grid = np.unique(np.round(grid))
        scenarios = pd.concat([x] * len(grid), ignore_index=True).assign(**{feature: grid})
        fig = px.line(x=grid, y=prob(name, scenarios), labels={"x": f"{label(feature)} ({UNITS[feature]})", "y": "Probability"},
                      title=f"{name}: probability as only {label(feature).lower()} changes")
        if pd.notna(x.loc[0, feature]):
            fig.add_scatter(x=[x.loc[0, feature]], y=[p], mode="markers", marker=dict(size=13, color=ORANGE), name="This patient")
        fig.add_hline(y=threshold, line_dash="dash", annotation_text="threshold")
        fig.update_yaxes(range=[0, 1], tickformat=".0%")
        show(fig)
    with tab3:
        medians = cohort[cohort.split == "train"][FEATURES].median()
        rows = [{"feature": label(f), "change": p - float(prob(name, x.assign(**{f: medians[f]}))[0])} for f in FEATURES]
        imp = pd.DataFrame(rows).sort_values("change")
        fig = px.bar(imp, x="change", y="feature", orientation="h", color="change", color_continuous_midpoint=0,
                     color_continuous_scale=DIVERGING,
                     title="How much each input moves the probability vs the training median")
        fig.update_layout(coloraxis_showscale=False)
        fig.update_xaxes(tickformat="+.0%")
        show(fig)
        st.caption("One input is replaced at a time, so the bars show model sensitivity rather than causal effects.")


def page_about():
    st.title("Methodology")
    st.markdown(f"""
**Dataset:** PIMA Indians Diabetes ({meta['rows']} women of Pima heritage aged 21+, 8 clinical measurements, binary outcome).

**CRISP-DM**
1. *Business understanding* – identify patients with diabetes from routine measurements; secondary task: BMI regression.
2. *Data understanding* – class imbalance (35% positive), impossible zeros, skewed insulin and pedigree.
3. *Data preparation* – zeros → missing, median imputation + indicators, scaling, one-hot encoding, IQR outlier audit (kept).
4. *Features* – log transforms, glucose × BMI, age/BMI binning; filter vs wrapper vs embedded selection.
5. *Modeling* – grid search with 5-fold stratified CV; F1-optimal threshold from out-of-fold predictions; PyTorch ANN with early stopping.
6. *Evaluation* – repeated CV for model choice, a held-out test set used once, bootstrap confidence intervals, learning and validation curves.
7. *Deployment* – this dashboard serves the trained pipelines for exploration and prediction.

**Limitations:** small single-population dataset, one random test split, no external validation.

**Reproduce:** `python data_pipeline.py` → `python train.py` → `python build_report.py` → `streamlit run app.py`
(run in {summary['duration_seconds']:.0f} s with scikit-learn {summary['versions']['sklearn']} and PyTorch {summary['versions']['torch']}).
""")
    st.download_button("Download report (HTML)", (ROOT / "reports/REPORT.html").read_bytes(), "PIMA_Diabetes_Report.html",
                       mime="text/html")


PAGES = {"Overview": page_overview, "Data explorer": page_data, "Features": page_features,
         "Model evaluation": page_evaluation, "Bias–variance": page_bias_variance,
         "Prediction lab": page_prediction, "Methodology": page_about}
st.sidebar.title("🩺 PIMA Diabetes")
page = st.sidebar.radio("Page", list(PAGES))
st.sidebar.caption(f"Best model: **{WINNER}**")
PAGES[page]()
