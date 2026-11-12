# Diabetes Prediction · PIMA Indians (TDB3123 ML Project)

CRISP-DM project on the PIMA Indians Diabetes dataset (Labs 1 and 3): data cleaning, feature engineering and
selection, six classifiers, a PyTorch neural network, BMI regression and bias–variance analysis, with a
Streamlit dashboard.

**Live app:** https://ml-porjectpima.streamlit.app/

## Results (held-out test set, 154 patients)

| | Best model | Result |
|---|---|---|
| Classification | SVM (RBF); logistic regression and ANN equal within CV std | ROC-AUC 0.83, recall 78% at the tuned threshold |
| Regression (BMI) | Lasso | RMSE 5.37, R² 0.43 (mean baseline RMSE 7.12) |
| Neural network | PyTorch 32-16, dropout, early stopping | ROC-AUC 0.83, recall 85% |

## Project structure

| File | Purpose |
|---|---|
| `data_pipeline.py` | Load, clean (zeros → missing), 80/20 stratified split, feature engineering, preprocessing pipeline |
| `ann.py` | PyTorch feed-forward network as a scikit-learn classifier |
| `train.py` | Feature ablation and selection, grid search (5-fold CV), evaluation, learning curves; saves `artifacts/` |
| `notebooks/PIMA_Diabetes_Project.ipynb` | Fully documented, executed analysis (Parts A–D and discussion) |
| `build_report.py` | Builds `reports/REPORT.md/.html`, figures and presentation outline from `artifacts/` |
| `app.py` | Streamlit dashboard |
| `tests/` | Data, leakage and app tests |

## Assignment coverage

| Requirement | Where |
|---|---|
| A · CRISP-DM, missing data, encoding, normalization, outliers | Notebook §1–3, `data_pipeline.py`, `outlier_audit` |
| B · Feature selection (filter, wrapper, embedded) | Notebook §5, `selection_table`; mutual information used in final grids |
| B · Feature engineering (log, interaction, binning) | Notebook §4, `FeatureEngineer`; impact in §6 |
| C · Classification, regression, ANN | Notebook §7–9 |
| D · Metrics, learning curves, bias–variance, 4 discussion questions | Notebook §10–14, report §3–4 |
| Report ≤ 3000 words with tables and plots | `reports/REPORT.html` (word limit checked by `build_report.py`) |

Still to do by the group: names/IDs in the report and notebook, Turnitin check, presentation slides.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-validated.txt
.\.venv\Scripts\python.exe train.py            # ~1.5 min: data + all experiments -> artifacts/
.\.venv\Scripts\python.exe build_report.py     # report, figures, outline
.\.venv\Scripts\python.exe -m pytest -q
.\Run Dashboard.ps1                            # http://127.0.0.1:8502
```

`requirements.txt` holds only the app dependencies for Streamlit Community Cloud (entry point `app.py`, Python 3.13).
Retrain after changing scikit-learn or PyTorch versions.

Coursework demonstration only, not a medical diagnosis tool.
