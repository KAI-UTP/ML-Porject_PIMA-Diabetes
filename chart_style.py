"""Shared chart colours for the app, report and notebook.

SERIES is a colour-blind-validated categorical order; each model keeps the same colour in every chart.
"""
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
ACCENT = "#7147B8"   # brand colour: highlights the selected / best item
NEUTRAL = "#a8a6a0"  # context marks (training scores, other models)

MODEL_COLORS = {
    "Logistic Regression": SERIES[0], "Decision Tree": SERIES[1], "Random Forest": SERIES[2],
    "Gradient Boosting": SERIES[3], "SVM (RBF)": SERIES[4], "ANN (PyTorch)": SERIES[5],
    "Baseline (prior)": NEUTRAL,
    "Linear Regression": SERIES[0], "Ridge": SERIES[1], "Lasso": SERIES[2], "Baseline (mean)": NEUTRAL,
}
OUTCOME_COLORS = {"No diabetes": SERIES[0], "Diabetes": SERIES[1]}
DIVERGING = [SERIES[0], "#f0efec", SERIES[7]]  # blue - neutral grey - red
