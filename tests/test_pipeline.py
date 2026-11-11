import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from data_pipeline import COLUMNS, ZERO_AS_MISSING, CLASS_FEATURES, REG_FEATURES, clean_data, ROOT
from train import preprocessing

def test_invalid_zeros_become_missing():
    row=[1,0,0,0,0,0,.5,25,1]
    d,a=clean_data(pd.DataFrame([row],columns=COLUMNS))
    assert d[ZERO_AS_MISSING].isna().all().all()
    assert a["invalid_zero_counts"]["glucose"]==1

def test_bmi_is_not_a_regression_input():
    assert "bmi" not in REG_FEATURES
    assert "bmi" in CLASS_FEATURES

def test_imputation_is_fit_on_training_only():
    x=pd.DataFrame([[1,100,70,20,80,30,.4,25],[2,np.nan,80,30,90,35,.6,40]],columns=CLASS_FEATURES)
    prep=Pipeline(preprocessing()).fit(x,[0,1]);before=prep.named_steps["prepare"].named_transformers_["numeric"].named_steps["impute"].statistics_.copy()
    prep.transform(pd.concat([x]*3).assign(glucose=200))
    np.testing.assert_array_equal(before,prep.named_steps["prepare"].named_transformers_["numeric"].named_steps["impute"].statistics_)

def test_split_integrity():
    d=pd.read_csv(ROOT/"data/processed/cohort.csv")
    assert len(d)==768 and not d.record_id.duplicated().any()
    assert set(d.split)=={"train","validation","test"}

def test_invalid_records_and_normalized_duplicates():
    rows=[[1,100,70,20,80,30,.4,25,1],[1,100,-70,20,80,30,.4,25,1],[1.5,100,70,20,80,30,.4,25,1]]
    frame=pd.DataFrame(rows+[list(map(str,rows[0]))],columns=COLUMNS)
    cleaned,audit=clean_data(frame)
    assert len(cleaned)==1 and audit['duplicate_rows']==1
    assert audit['invalid_rows_removed']==2

def test_zero_cleaning_is_identical_in_saved_pipeline():
    import joblib
    d=pd.read_csv(ROOT/'data/processed/cohort.csv')
    x=d[CLASS_FEATURES].iloc[:1].copy(); x.loc[:,ZERO_AS_MISSING]=0
    missing=x.replace({column:{0:np.nan} for column in ZERO_AS_MISSING})
    models=joblib.load(ROOT/'artifacts/classifiers.joblib')
    for model in models.values():
        np.testing.assert_allclose(model.predict_proba(x),model.predict_proba(missing))

def test_operating_metrics_match_saved_predictions():
    from train import cmetrics
    metrics=pd.read_csv(ROOT/'artifacts/operating_metrics.csv')
    predictions=pd.read_csv(ROOT/'artifacts/classification_predictions.csv')
    for row in metrics.itertuples():
        p=predictions[(predictions.model==row.model)&(predictions.split==row.split)]
        actual=cmetrics(p.actual,p.probability,row.threshold)
        for metric in ['accuracy','recall','precision','f1','roc_auc']:
            np.testing.assert_allclose(getattr(row,metric),actual[metric])
