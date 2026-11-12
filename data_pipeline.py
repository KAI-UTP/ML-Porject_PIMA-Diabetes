"""PIMA Indians Diabetes: loading, cleaning, splitting and feature engineering."""
from pathlib import Path
import hashlib
import json
import urllib.request

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

ROOT = Path(__file__).resolve().parent
SEED = 42
SOURCE_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

COLUMNS = ["pregnancies", "glucose", "blood_pressure", "skin_thickness", "insulin",
           "bmi", "diabetes_pedigree", "age", "outcome"]
FEATURES = COLUMNS[:-1]
TARGET = "outcome"
# A value of 0 is physically impossible for these measurements, so it means "not recorded".
ZERO_AS_MISSING = ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]

# Secondary regression task: predict BMI from the other measurements (BMI itself is excluded).
REG_TARGET = "bmi"
REG_FEATURES = [c for c in FEATURES if c != REG_TARGET] + [TARGET]

UNITS = {"pregnancies": "count", "glucose": "mg/dL", "blood_pressure": "mm Hg", "skin_thickness": "mm",
         "insulin": "µU/mL", "bmi": "kg/m²", "diabetes_pedigree": "score", "age": "years"}


# ---------------------------------------------------------------- 1. Load and clean

def load_raw() -> pd.DataFrame:
    """Read the dataset CSV, downloading it once if it is not cached locally."""
    path = ROOT / "data" / "raw" / "pima-indians-diabetes.csv"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SOURCE_URL, path)
    return pd.read_csv(path, names=COLUMNS)


def clean_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Fix types, drop duplicates/invalid rows and turn impossible zeros into NaN."""
    if list(raw.columns) != COLUMNS:
        raise ValueError("Unexpected PIMA schema")
    d = raw.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    audit = {"raw_rows": len(d), "duplicate_rows": int(d.duplicated().sum())}
    d = d.drop_duplicates().reset_index(drop=True)

    audit["zero_as_missing"] = {c: int((d[c] == 0).sum()) for c in ZERO_AS_MISSING}
    d[ZERO_AS_MISSING] = d[ZERO_AS_MISSING].replace(0, np.nan)

    valid = (d[TARGET].isin([0, 1]) & d.age.between(21, 100) & d.pregnancies.ge(0)
             & d.pregnancies.mod(1).eq(0) & d.age.mod(1).eq(0) & ~d[FEATURES].lt(0).any(axis=1))
    audit["invalid_rows_removed"] = int((~valid).sum())
    d = d.loc[valid].reset_index(drop=True)
    d[TARGET] = d[TARGET].astype(int)
    return d, audit


def prepare_data() -> tuple[pd.DataFrame, dict]:
    """Clean the data, make one stratified 80/20 train/test split and save it."""
    d, audit = clean_data(load_raw())
    train, test = train_test_split(d, test_size=0.20, stratify=d[TARGET], random_state=SEED)
    cohort = pd.concat([train.assign(split="train"), test.assign(split="test")]).reset_index(drop=True)
    cohort.insert(0, "record_id", np.arange(1, len(cohort) + 1))

    out = ROOT / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(out / "cohort.csv", index=False)
    raw_path = ROOT / "data" / "raw" / "pima-indians-diabetes.csv"
    meta = {
        "source_url": SOURCE_URL,
        "source_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "seed": SEED,
        "rows": len(cohort),
        "split_counts": cohort.split.value_counts().to_dict(),
        "class_counts": cohort[TARGET].value_counts().sort_index().to_dict(),
        "audit": audit,
    }
    (out / "provenance.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return cohort, meta


def load_cohort() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the saved (train, test) frames, creating them on first use."""
    path = ROOT / "data" / "processed" / "cohort.csv"
    d = pd.read_csv(path) if path.exists() else prepare_data()[0]
    return d[d.split == "train"].reset_index(drop=True), d[d.split == "test"].reset_index(drop=True)


# ---------------------------------------------------------------- 2. Feature engineering

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Stateless feature step used inside every model pipeline.

    zeros_as_missing=False restores the raw zeros (used only for the "raw data" ablation).
    engineered=True adds: log transforms, an interaction term and two binned features.
    Features whose source column is absent (e.g. BMI in the regression task) are skipped.
    """

    def __init__(self, zeros_as_missing=True, engineered=True):
        self.zeros_as_missing = zeros_as_missing
        self.engineered = engineered

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        d = pd.DataFrame(X).copy()
        cols = [c for c in ZERO_AS_MISSING if c in d]
        d[cols] = d[cols].replace(0, np.nan) if self.zeros_as_missing else d[cols].fillna(0)
        if not self.engineered:
            return d
        # Log transform: insulin and pedigree are strongly right-skewed.
        for col, new in [("insulin", "log_insulin"), ("diabetes_pedigree", "log_pedigree")]:
            if col in d:
                d[new] = np.log1p(d.pop(col))
        # Interaction term: high glucose matters more when BMI is also high.
        if {"glucose", "bmi"} <= set(d):
            d["glucose_x_bmi"] = d.glucose * d.bmi
        # Binning: clinical age groups and WHO BMI classes.
        if "age" in d:
            d["age_band"] = pd.cut(d.age, [0, 30, 40, 50, 200], labels=["21-30", "31-40", "41-50", "51+"]).astype(object)
        if "bmi" in d:
            d["bmi_class"] = pd.cut(d.bmi, [0, 25, 30, 35, 200], right=False,
                                    labels=["normal", "overweight", "obese I", "obese II+"]).astype(object)
        return d


def build_preprocessor(zeros_as_missing=True, engineered=True, scaler="standard") -> Pipeline:
    """Feature engineering -> median imputation (+ missing flags) -> scaling, one-hot encoding."""
    scalers = {"standard": StandardScaler(), "robust": RobustScaler(), None: "passthrough"}
    numeric = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", scalers[scaler])])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    columns = ColumnTransformer([
        ("num", numeric, make_column_selector(dtype_include=np.number)),
        ("cat", categorical, make_column_selector(dtype_include=object)),
    ], verbose_feature_names_out=False)
    return Pipeline([("engineer", FeatureEngineer(zeros_as_missing, engineered)), ("columns", columns)])


if __name__ == "__main__":
    print(json.dumps(prepare_data()[1], indent=2))
