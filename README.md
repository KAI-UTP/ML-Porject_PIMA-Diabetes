# Diabetes Prediction · PIMA Indians

Machine-learning pipeline that predicts diabetes from eight routine clinical measurements, following CRISP-DM:
data cleaning, feature engineering and selection, five tuned classifiers, a PyTorch neural network, BMI regression
and bias–variance analysis, served through a Streamlit dashboard.

**Live app:** https://ml-porjectpima.streamlit.app/

## Results (held-out test set, 154 patients)

| Task | Best model | Result |
|---|---|---|
| Classification | SVM (RBF); logistic regression and the neural network are equivalent within CV uncertainty | ROC-AUC 0.83, recall 78% at the tuned threshold |
| Neural network | PyTorch 32-16, dropout, early stopping | ROC-AUC 0.83, recall 85% |
| Regression (BMI) | Lasso | RMSE 5.37, R² 0.43 (mean baseline RMSE 7.12) |

## Method

1. **Data preparation** – impossible zeros → missing; median imputation with missing indicators; scaling; one-hot encoding; IQR outlier audit; stratified 80/20 split.
2. **Feature engineering** – log transforms, glucose × BMI interaction, age and BMI binning.
3. **Feature selection** – mutual information (filter), RFE (wrapper) and LASSO (embedded), compared in a step-by-step ablation.
4. **Modelling** – grid search with 5-fold stratified CV for logistic regression, decision tree, random forest, gradient boosting, SVM and the neural network; Linear, Ridge, Lasso, tree and forest regressors for BMI.
5. **Evaluation** – repeated CV for model selection, F1-optimal threshold from out-of-fold predictions, single test evaluation with bootstrap CIs, learning and validation curves.

## Project structure

| File | Purpose |
|---|---|
| `data_pipeline.py` | Loading, cleaning, split, feature engineering, preprocessing pipeline |
| `ann.py` | PyTorch feed-forward network as a scikit-learn classifier |
| `train.py` | All experiments; saves results to `artifacts/` |
| `notebooks/PIMA_Diabetes_Project.ipynb` | Documented, executed analysis |
| `build_report.py` | Builds `reports/REPORT.md` / `REPORT.html`, figures and presentation outline |
| `app.py` | Streamlit dashboard |
| `tests/` | Data, leakage and dashboard tests |

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-validated.txt
.\.venv\Scripts\python.exe train.py            # data + all experiments (~1.5 min)
.\.venv\Scripts\python.exe build_report.py     # report, figures, outline
.\.venv\Scripts\python.exe -m pytest -q
.\Run Dashboard.ps1                            # http://127.0.0.1:8502
```

`requirements.txt` lists only the dashboard dependencies (used by Streamlit Community Cloud, Python 3.13).
Retrain after changing the scikit-learn or PyTorch version.
