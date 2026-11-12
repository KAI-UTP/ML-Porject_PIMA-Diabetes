"""Experiments for the PIMA project: feature study, classification, BMI regression, ANN, bias-variance.

Each step is a function so the notebook can run and explain it; `run()` executes everything
and saves the artifacts used by the Streamlit app and the report.
"""
from functools import partial
import json
import platform
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (GradientBoostingClassifier, RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.feature_selection import RFE, SelectFromModel, SelectKBest, mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, f1_score,
                             mean_absolute_error, mean_squared_error, precision_recall_curve,
                             precision_score, r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, KFold, RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_predict, cross_validate, learning_curve,
                                     validation_curve)
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from ann import TorchMLPClassifier
from data_pipeline import (FEATURES, REG_FEATURES, REG_TARGET, ROOT, SEED, TARGET,
                           build_preprocessor, load_cohort)

ARTIFACTS = ROOT / "artifacts"
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
REPEATED_CV = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
REG_CV = KFold(n_splits=5, shuffle=True, random_state=SEED)

SELECTORS = {
    "Mutual information (filter)": SelectKBest(partial(mutual_info_classif, random_state=SEED), k=10),
    "RFE (wrapper)": RFE(LogisticRegression(max_iter=5000), n_features_to_select=10),
    "LASSO L1 (embedded)": SelectFromModel(LogisticRegression(penalty="l1", solver="liblinear", C=0.1)),
}


# ---------------------------------------------------------------- Metrics

def classification_metrics(y, p, threshold=0.5) -> dict:
    pred = (np.asarray(p) >= threshold).astype(int)
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred), "f1": f1_score(y, pred, zero_division=0),
            "roc_auc": roc_auc_score(y, p), "average_precision": average_precision_score(y, p),
            "brier": brier_score_loss(y, p), "threshold": threshold}


def regression_metrics(y, pred) -> dict:
    mse = mean_squared_error(y, pred)
    return {"mae": mean_absolute_error(y, pred), "mse": mse, "rmse": np.sqrt(mse), "r2": r2_score(y, pred)}


def best_f1_threshold(y, p) -> float:
    """Cut-off that maximises F1 (chosen on out-of-fold predictions, never on the test set)."""
    precision, recall, thresholds = precision_recall_curve(y, p)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[np.argmax(f1)])


def bootstrap_auc_ci(y, p, n=1000) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    y, p = np.asarray(y), np.asarray(p)
    scores = []
    for _ in range(n):
        idx = rng.integers(len(y), size=len(y))
        if y[idx].min() != y[idx].max():
            scores.append(roc_auc_score(y[idx], p[idx]))
    return tuple(np.quantile(scores, [0.025, 0.975]))


# ---------------------------------------------------------------- Part A: outliers

def outlier_audit(train, test) -> pd.DataFrame:
    """1.5 x IQR fences fitted on the training split; outliers are flagged, not removed."""
    rows = []
    for col in FEATURES:
        q1, q3 = train[col].quantile([0.25, 0.75])
        low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        for split, frame in [("train", train), ("test", test)]:
            x = frame[col].dropna()
            rows.append({"feature": col, "split": split, "lower_fence": low, "upper_fence": high,
                         "outliers": int(((x < low) | (x > high)).sum()), "recorded": len(x)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- Part B: feature engineering & selection

def feature_ablation(train) -> pd.DataFrame:
    """Add one preparation step at a time and measure 5x3 repeated-CV ROC-AUC on the training split."""
    variants = [
        ("1. Raw data (zeros kept)", dict(zeros_as_missing=False, engineered=False, scaler=None), None),
        ("2. Zeros -> missing + median imputation", dict(engineered=False, scaler=None), None),
        ("3. + Standard scaling", dict(engineered=False), None),
        ("3b. + Robust scaling (alternative)", dict(engineered=False, scaler="robust"), None),
        ("4. + Feature engineering", dict(), None),
    ] + [(f"5. + {name}", dict(), name) for name in SELECTORS]
    models = {"Logistic Regression": LogisticRegression(max_iter=5000),
              "Random Forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=SEED)}
    rows = []
    for variant, prep_args, selector in variants:
        for model_name, model in models.items():
            steps = [("prep", build_preprocessor(**prep_args))]
            if selector:
                steps.append(("select", clone(SELECTORS[selector])))
            scores = cross_validate(Pipeline(steps + [("model", clone(model))]), train[FEATURES], train[TARGET],
                                    cv=REPEATED_CV, scoring="roc_auc", return_train_score=True, n_jobs=-1)
            rows.append({"variant": variant, "model": model_name,
                         "cv_roc_auc": scores["test_score"].mean(), "cv_std": scores["test_score"].std(),
                         "train_roc_auc": scores["train_score"].mean()})
    return pd.DataFrame(rows)


def choose_selector(ablation) -> str:
    """Selection method with the best mean CV ROC-AUC across both ablation models."""
    rows = ablation[ablation.variant.str.startswith("5.")]
    return rows.groupby("variant").cv_roc_auc.mean().idxmax().removeprefix("5. + ")


def selection_table(train) -> pd.DataFrame:
    """Score of every engineered feature under each selection method (fitted on training data)."""
    prep = build_preprocessor().fit(train[FEATURES], train[TARGET])
    X, y = prep.transform(train[FEATURES]), train[TARGET]
    names = prep.named_steps["columns"].get_feature_names_out()
    table = pd.DataFrame({"feature": names})
    for name, selector in SELECTORS.items():
        table[name] = clone(selector).fit(X, y).get_support()
    table["mutual_information"] = mutual_info_classif(X, y, random_state=SEED)
    table["lasso_coefficient"] = SELECTORS["LASSO L1 (embedded)"].estimator.fit(X, y).coef_.ravel()
    forest = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=SEED).fit(X, y)
    table["rf_importance"] = forest.feature_importances_
    return table.sort_values("mutual_information", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------- Part C: models

def classifier_specs(selector) -> dict:
    """name -> (pipeline, grid). Feature selection is a tuned option: CV decides whether it helps."""
    def pipe(model, scaler="standard"):
        return Pipeline([("prep", build_preprocessor(scaler=scaler)), ("select", "passthrough"), ("model", model)])
    select = {"select": ["passthrough", selector]}
    return {
        "Baseline (prior)": (pipe(DummyClassifier(strategy="prior")), {}),
        "Logistic Regression": (pipe(LogisticRegression(max_iter=5000)),
                                {**select, "model__C": [0.01, 0.1, 1, 10]}),
        "Decision Tree": (pipe(DecisionTreeClassifier(random_state=SEED), scaler=None),
                          {**select, "model__max_depth": [3, 4, 5, 6, 8], "model__min_samples_leaf": [5, 10, 20]}),
        "Random Forest": (pipe(RandomForestClassifier(n_estimators=300, random_state=SEED), scaler=None),
                          {**select, "model__max_depth": [4, 6, 8, None], "model__min_samples_leaf": [1, 5],
                           "model__max_features": ["sqrt", 0.5]}),
        "Gradient Boosting": (pipe(GradientBoostingClassifier(subsample=0.8, random_state=SEED), scaler=None),
                              {**select, "model__learning_rate": [0.03, 0.1], "model__n_estimators": [100, 200],
                               "model__max_depth": [2, 3]}),
        "SVM (RBF)": (pipe(SVC(probability=True, random_state=SEED)),
                      {**select, "model__C": [0.3, 1, 3], "model__gamma": ["scale", 0.01]}),
        "ANN (PyTorch)": (pipe(TorchMLPClassifier(random_state=SEED)),
                          {**select, "model__hidden": [(32, 16), (64, 32)], "model__weight_decay": [1e-4, 1e-3]}),
    }


def regressor_specs() -> dict:
    def pipe(model, scaler="standard"):
        return Pipeline([("prep", build_preprocessor(scaler=scaler)), ("model", model)])
    return {
        "Baseline (mean)": (pipe(DummyRegressor()), {}),
        "Linear Regression": (pipe(LinearRegression()), {}),
        "Ridge": (pipe(Ridge()), {"model__alpha": [0.1, 1, 10, 100]}),
        "Lasso": (pipe(Lasso(max_iter=20000)), {"model__alpha": [0.01, 0.05, 0.1, 0.5]}),
        "Decision Tree": (pipe(DecisionTreeRegressor(random_state=SEED), scaler=None),
                          {"model__max_depth": [3, 4, 5, 6, 8], "model__min_samples_leaf": [5, 10, 20]}),
        "Random Forest": (pipe(RandomForestRegressor(n_estimators=300, random_state=SEED), scaler=None),
                          {"model__max_depth": [4, 8, None], "model__min_samples_leaf": [3, 5]}),
    }


def tune(specs, X, y, cv, scoring) -> tuple[dict, pd.DataFrame]:
    """Grid search each model with k-fold CV on the training split; return refitted best models."""
    best, rows = {}, []
    for name, (pipe, grid) in specs.items():
        search = GridSearchCV(pipe, grid or {}, cv=cv, scoring=scoring, n_jobs=-1).fit(X, y)
        best[name] = search.best_estimator_
        params = {k.split("__")[-1]: (type(v).__name__ if hasattr(v, "fit") else v)
                  for k, v in search.best_params_.items()}
        rows.append({"model": name, "best_params": json.dumps(params, default=str),
                     "grid_size": len(search.cv_results_["params"])})
    return best, pd.DataFrame(rows)


def evaluate_classifiers(models, train, test):
    """Repeated-CV comparison, out-of-fold threshold, then a single final test evaluation."""
    X, y = train[FEATURES], train[TARGET]
    summary, metrics, predictions = [], [], []
    for name, model in models.items():
        scores = cross_validate(model, X, y, cv=REPEATED_CV, scoring=["roc_auc", "f1", "accuracy"],
                                return_train_score=True, n_jobs=-1)
        oof = cross_val_predict(model, X, y, cv=CV, method="predict_proba", n_jobs=-1)[:, 1]
        threshold = 0.5 if name.startswith("Baseline") else best_f1_threshold(y, oof)
        test_p = model.predict_proba(test[FEATURES])[:, 1]
        low, high = bootstrap_auc_ci(test[TARGET], test_p)
        summary.append({"model": name, "cv_roc_auc": scores["test_roc_auc"].mean(),
                        "cv_roc_auc_std": scores["test_roc_auc"].std(),
                        "train_roc_auc": scores["train_roc_auc"].mean(),
                        "cv_f1": scores["test_f1"].mean(), "cv_accuracy": scores["test_accuracy"].mean(),
                        "threshold": threshold, "test_roc_auc": roc_auc_score(test[TARGET], test_p),
                        "test_auc_ci_low": low, "test_auc_ci_high": high})
        for split, frame, p in [("train", train, model.predict_proba(X)[:, 1]), ("cv", train, oof), ("test", test, test_p)]:
            metrics.append({"model": name, "split": split, **classification_metrics(frame[TARGET], p, threshold)})
            if split != "train":
                predictions.append(pd.DataFrame({"record_id": frame.record_id, "model": name, "split": split,
                                                 "actual": frame[TARGET], "probability": p}))
    summary = pd.DataFrame(summary).sort_values("cv_roc_auc", ascending=False)
    return summary, pd.DataFrame(metrics), pd.concat(predictions, ignore_index=True)


def evaluate_regressors(models, train, test):
    X, y = train[REG_FEATURES], train[REG_TARGET]
    metrics, predictions = [], []
    for name, model in models.items():
        oof = cross_val_predict(model, X, y, cv=REG_CV, n_jobs=-1)
        for split, frame, pred in [("train", train, model.predict(X)), ("cv", train, oof),
                                   ("test", test, model.predict(test[REG_FEATURES]))]:
            metrics.append({"model": name, "split": split, **regression_metrics(frame[REG_TARGET], pred)})
            if split != "train":
                predictions.append(pd.DataFrame({"record_id": frame.record_id, "model": name, "split": split,
                                                 "actual": frame[REG_TARGET], "predicted": pred}))
    return pd.DataFrame(metrics), pd.concat(predictions, ignore_index=True)


# ---------------------------------------------------------------- Part D: bias-variance

def learning_curves(models, train, names) -> pd.DataFrame:
    rows = []
    for name in names:
        sizes, tr, cv = learning_curve(models[name], train[FEATURES], train[TARGET], cv=CV, scoring="roc_auc",
                                       train_sizes=np.linspace(0.2, 1.0, 6), n_jobs=-1)
        for n, a, b in zip(sizes, tr, cv):
            rows.append({"model": name, "n_train": int(n), "train_roc_auc": a.mean(), "train_std": a.std(),
                         "cv_roc_auc": b.mean(), "cv_std": b.std()})
    return pd.DataFrame(rows)


def depth_curve(train) -> pd.DataFrame:
    """Decision-tree depth vs train/CV ROC-AUC: the classic bias-variance trade-off."""
    pipe = Pipeline([("prep", build_preprocessor(scaler=None)), ("model", DecisionTreeClassifier(random_state=SEED))])
    depths = list(range(1, 13))
    tr, cv = validation_curve(pipe, train[FEATURES], train[TARGET], param_name="model__max_depth",
                              param_range=depths, cv=CV, scoring="roc_auc", n_jobs=-1)
    return pd.DataFrame({"max_depth": depths, "train_roc_auc": tr.mean(1), "cv_roc_auc": cv.mean(1),
                         "cv_std": cv.std(1)})


# ---------------------------------------------------------------- Run everything

def run() -> dict:
    start = time.time()
    ARTIFACTS.mkdir(exist_ok=True)
    train, test = load_cohort()
    save = lambda frame, name: frame.to_csv(ARTIFACTS / name, index=False)

    save(outlier_audit(train, test), "outlier_audit.csv")

    ablation = feature_ablation(train)
    selector_name = choose_selector(ablation)
    save(ablation, "feature_ablation.csv")
    save(selection_table(train), "feature_selection.csv")

    classifiers, cls_params = tune(classifier_specs(SELECTORS[selector_name]), train[FEATURES], train[TARGET],
                                   CV, "roc_auc")
    cls_summary, cls_metrics, cls_pred = evaluate_classifiers(classifiers, train, test)
    cls_summary = cls_summary.merge(cls_params, on="model")
    save(cls_summary, "classification_summary.csv")
    save(cls_metrics, "classification_metrics.csv")
    save(cls_pred, "classification_predictions.csv")
    joblib.dump(classifiers, ARTIFACTS / "classifiers.joblib", compress=3)
    save(pd.DataFrame(classifiers["ANN (PyTorch)"].named_steps["model"].history_), "ann_history.csv")

    reg_train, reg_test = train.dropna(subset=[REG_TARGET]), test.dropna(subset=[REG_TARGET])
    regressors, reg_params = tune(regressor_specs(), reg_train[REG_FEATURES], reg_train[REG_TARGET],
                                  REG_CV, "neg_root_mean_squared_error")
    reg_metrics, reg_pred = evaluate_regressors(regressors, reg_train, reg_test)
    save(reg_metrics.merge(reg_params, on="model"), "regression_metrics.csv")
    save(reg_pred, "regression_predictions.csv")
    joblib.dump(regressors, ARTIFACTS / "regressors.joblib", compress=3)

    winner = cls_summary.iloc[0].model
    reg_cv = reg_metrics[reg_metrics.split == "cv"].sort_values("rmse")
    save(learning_curves(classifiers, train, ["Logistic Regression", "Decision Tree", "Random Forest", "ANN (PyTorch)"]),
         "learning_curves.csv")
    save(depth_curve(train), "depth_curve.csv")
    imp = permutation_importance(classifiers[winner], test[FEATURES], test[TARGET], scoring="roc_auc",
                                 n_repeats=30, random_state=SEED)
    save(pd.DataFrame({"feature": FEATURES, "importance": imp.importances_mean, "std": imp.importances_std})
         .sort_values("importance", ascending=False), "permutation_importance.csv")

    best = cls_metrics[(cls_metrics.model == winner) & (cls_metrics.split == "test")].iloc[0]
    ann = classifiers["ANN (PyTorch)"].named_steps["model"]
    summary = {
        "winner": winner,
        "regression_winner": reg_cv.iloc[0].model,
        "selector": selector_name,
        "threshold": float(best.threshold),
        "test_metrics": best.drop(["model", "split"]).astype(float).to_dict(),
        "test_accuracy_at_0.5": float(accuracy_score(test[TARGET], classifiers[winner].predict(test[FEATURES]))),
        "ann_best_epoch": int(ann.best_epoch_), "ann_epochs_run": len(ann.history_),
        "duration_seconds": round(time.time() - start, 1),
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "torch": torch.__version__, "pandas": pd.__version__},
    }
    (ARTIFACTS / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    from data_pipeline import prepare_data
    prepare_data()
    print(json.dumps(run(), indent=2))
