# PIMA Diabetes machine-learning project

This project rebuilds the coursework around the PIMA dataset used in Labs 1 and 3.

## Run the dashboard

```powershell
.\Run Dashboard.ps1
```

Then open http://127.0.0.1:8502.

## Reproduce

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe data_pipeline.py
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe build_deliverables.py
.\.venv\Scripts\python.exe -m pytest -q
```

## Main files

- `data_pipeline.py`: download, validation, missing values, and train/validation/test split.
- `train.py`: classification, BMI regression, ANN, feature experiments, and saved results.
- `app.py`: Streamlit dashboard.
- `notebooks/PIMA_Diabetes_Project.ipynb`: executed analysis.
- `reports/REPORT.html`: report with measured results.
- `reports/PRESENTATION_OUTLINE.md`: concise presentation structure.

The prediction page is for coursework demonstration, not diagnosis. The report needs student names and group review before submission.
