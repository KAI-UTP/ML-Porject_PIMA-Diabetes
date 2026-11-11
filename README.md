# PIMA Diabetes machine-learning project

This project rebuilds the coursework around the PIMA dataset used in Labs 1 and 3.

## Run the dashboard

```powershell
.\Run Dashboard.ps1
```

Then open http://127.0.0.1:8502.

## Deploy on Streamlit Community Cloud

1. Open https://share.streamlit.io and connect the GitHub account with repository access.
2. Create an app and select `KAI-UTP/ML-Porject_PIMA-Diabetes`.
3. Use branch `main` and entrypoint `app.py`.
4. In Advanced settings, select Python 3.13. No secrets are required.
5. Deploy and wait for the dependency installation and health check.

The root `requirements.txt` contains only the pinned app runtime dependencies. Community Cloud redeploys after future pushes to `main`.

## Reproduce

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-validated.txt
.\.venv\Scripts\python.exe data_pipeline.py
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe build_deliverables.py
.\.venv\Scripts\python.exe -m pytest -q
```

## Main files

- `data_pipeline.py`: download, validation, missing values, and train/validation/test split.
- `train.py`: classification, BMI regression, ANN, feature experiments, and saved results.
- `app.py`: Streamlit dashboard with model evaluation, patient-level model comparison, sensitivity, and what-if analysis.
- `notebooks/PIMA_Diabetes_Project.ipynb`: executed analysis.
- `reports/REPORT.html`: report with measured results.
- `reports/PRESENTATION_OUTLINE.md`: concise presentation structure.

The prediction page is for coursework demonstration, not diagnosis. The report needs student names and group review before submission.

`requirements-validated.txt` records the tested direct dependency versions. `requirements.txt` gives broader supported ranges; retrain before loading saved models after changing scikit-learn versions. The tested PyTorch build was CPU-only. This is not a complete transitive lockfile or a fresh-environment installation certification.

The executed notebook reruns data preparation and training. The report generator executes that notebook, so run it after training. Start the dashboard after generation finishes; restart a running dashboard after retraining to reload all artifacts together.

See `docs/REQUIREMENTS_AUDIT.md` for evidence against every assignment item and the remaining student submission tasks. Threshold-dependent metrics use the selected operating policy consistently. Controlled feature experiments use fixed C=1; the separately tuned models form the prediction leaderboard.
