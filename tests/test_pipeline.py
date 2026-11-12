import joblib
import numpy as np
import pandas as pd

from data_pipeline import (COLUMNS, FEATURES, REG_FEATURES, ROOT, ZERO_AS_MISSING, build_preprocessor,
                           clean_data, load_cohort)
from train import classification_metrics


def test_impossible_zeros_become_missing():
    d, audit = clean_data(pd.DataFrame([[1, 0, 0, 0, 0, 0, 0.5, 25, 1]], columns=COLUMNS))
    assert d[ZERO_AS_MISSING].isna().all().all()
    assert audit["zero_as_missing"]["glucose"] == 1


def test_invalid_and_duplicate_rows_removed():
    good = [1, 100, 70, 20, 80, 30, 0.4, 25, 1]
    rows = [good, [1, 100, -70, 20, 80, 30, 0.4, 25, 1], [1.5, 100, 70, 20, 80, 30, 0.4, 25, 1], list(map(str, good))]
    d, audit = clean_data(pd.DataFrame(rows, columns=COLUMNS))
    assert len(d) == 1 and audit["duplicate_rows"] == 1 and audit["invalid_rows_removed"] == 2


def test_bmi_never_an_input_for_bmi_regression():
    assert "bmi" not in REG_FEATURES
    prep = build_preprocessor().fit(pd.DataFrame([[1, 100, 70, 20, 80, 0.4, 25, 1]] * 3, columns=REG_FEATURES))
    assert not any("bmi" in name for name in prep.named_steps["columns"].get_feature_names_out())


def test_split_is_stratified_and_disjoint():
    train, test = load_cohort()
    assert len(train) + len(test) == 768
    assert not set(train.record_id) & set(test.record_id)
    assert abs(train.outcome.mean() - test.outcome.mean()) < 0.01


def test_imputer_learns_from_training_data_only():
    x = pd.DataFrame([[1, 100, 70, 20, 80, 30, 0.4, 25], [2, np.nan, 80, 30, 90, 35, 0.6, 40]], columns=FEATURES)
    prep = build_preprocessor().fit(x)
    before = prep.named_steps["columns"].named_transformers_["num"].named_steps["impute"].statistics_.copy()
    prep.transform(x.assign(glucose=200))
    after = prep.named_steps["columns"].named_transformers_["num"].named_steps["impute"].statistics_
    np.testing.assert_array_equal(before, after)


def test_saved_models_treat_zero_and_missing_the_same():
    train, _ = load_cohort()
    zeros = train[FEATURES].iloc[:3].copy()
    zeros[ZERO_AS_MISSING] = 0
    missing = zeros.replace({c: {0: np.nan} for c in ZERO_AS_MISSING})
    for model in joblib.load(ROOT / "artifacts/classifiers.joblib").values():
        np.testing.assert_allclose(model.predict_proba(zeros), model.predict_proba(missing))


def test_saved_metrics_match_saved_predictions():
    metrics = pd.read_csv(ROOT / "artifacts/classification_metrics.csv")
    preds = pd.read_csv(ROOT / "artifacts/classification_predictions.csv")
    for row in metrics[metrics.split != "train"].itertuples():
        p = preds[(preds.model == row.model) & (preds.split == row.split)]
        expected = classification_metrics(p.actual, p.probability, row.threshold)
        for metric in ["accuracy", "recall", "precision", "f1", "roc_auc"]:
            np.testing.assert_allclose(getattr(row, metric), expected[metric])
