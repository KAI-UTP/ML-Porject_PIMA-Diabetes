"""Prepare the PIMA Indians Diabetes dataset used in Labs 1 and 3."""
from pathlib import Path
import hashlib, json, urllib.request
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

ROOT = Path(__file__).resolve().parent
SEED = 42
SOURCE_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
COLUMNS = ["pregnancies", "glucose", "blood_pressure", "skin_thickness", "insulin", "bmi", "diabetes_pedigree", "age", "outcome"]
CLASS_FEATURES = COLUMNS[:-1]
REG_FEATURES = [c for c in CLASS_FEATURES if c != "bmi"]
ZERO_AS_MISSING = ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, task="classification", enabled=True):
        self.task, self.enabled = task, enabled

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        d = X.copy()
        if self.enabled:
            d["age_band"] = pd.cut(d.age, [20, 30, 40, 50, 60, 100], labels=["21-30", "31-40", "41-50", "51-60", "61+"]).astype(object)
            d["glucose_age"] = d.glucose * d.age
            d["pregnancies_per_age"] = d.pregnancies / d.age
            if self.task == "classification":
                d["glucose_bmi"] = d.glucose * d.bmi
        return d


def clean_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if list(raw.columns) != COLUMNS:
        raise ValueError("Unexpected PIMA schema")
    d = raw.copy()
    audit = {"raw_rows": len(d), "duplicate_rows": int(d.duplicated().sum())}
    d = d.drop_duplicates().reset_index(drop=True)
    for c in COLUMNS:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    audit["invalid_zero_counts"] = {c: int((d[c] == 0).sum()) for c in ZERO_AS_MISSING}
    d[ZERO_AS_MISSING] = d[ZERO_AS_MISSING].replace(0, np.nan)
    valid = d.outcome.isin([0, 1]) & d.pregnancies.ge(0) & d.age.between(21, 100)
    audit["invalid_rows_removed"] = int((~valid).sum())
    d = d.loc[valid].reset_index(drop=True)
    return d, audit


def prepare_data() -> tuple[pd.DataFrame, dict]:
    raw_dir = ROOT / "data" / "raw"
    out_dir = ROOT / "data" / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / "pima-indians-diabetes.csv"
    if not raw_path.exists():
        urllib.request.urlretrieve(SOURCE_URL, raw_path)
    raw = pd.read_csv(raw_path, names=COLUMNS)
    d, audit = clean_data(raw)
    from sklearn.model_selection import train_test_split
    train, remainder = train_test_split(d, test_size=.40, stratify=d.outcome, random_state=SEED)
    validation, test = train_test_split(remainder, test_size=.50, stratify=remainder.outcome, random_state=SEED)
    parts = []
    for name, frame in [("train", train), ("validation", validation), ("test", test)]:
        f = frame.copy(); f["split"] = name; parts.append(f)
    cohort = pd.concat(parts).reset_index(drop=True)
    cohort.insert(0, "record_id", np.arange(1, len(cohort) + 1))
    cohort.to_csv(out_dir / "cohort.csv", index=False)
    meta = {
        "source_url": SOURCE_URL,
        "source_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "seed": SEED,
        "rows": len(cohort),
        "split_counts": cohort.split.value_counts().to_dict(),
        "class_counts": cohort.outcome.value_counts().sort_index().to_dict(),
        "audit": audit,
        "classification_target": "Outcome: diabetes test result (0/1)",
        "regression_target": "BMI, using only rows where BMI was recorded and excluding BMI-derived predictors",
    }
    (out_dir / "provenance.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return cohort, meta


if __name__ == "__main__":
    _, metadata = prepare_data()
    print(json.dumps(metadata, indent=2))
