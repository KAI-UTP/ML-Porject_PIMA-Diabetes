# Course alignment

| Requirement | Implementation |
|---|---|
| CRISP-DM | Diabetes-risk question, data audit, preparation, modeling, evaluation, local demonstration |
| Cleaning | Invalid zero measurements become missing; duplicate and range checks |
| Feature engineering | Age band, glucose × age, pregnancies ÷ age, and glucose × BMI for classification |
| Feature selection | Embedded Extra Trees selection inside cross-validation pipelines |
| Classification | Logistic regression, decision tree, random forest, and baseline |
| Regression | BMI prediction with Ridge, decision tree, random forest, and mean baseline |
| Neural network | PyTorch 32–16 feed-forward ANN with dropout and early stopping |
| Evaluation | Accuracy, precision, recall, F1, ROC-AUC, AP, Brier, MAE, MSE, RMSE, and R² |
| Bias–variance | Train/validation comparison, learning curves, and ANN histories |
| Visualization | Streamlit and Plotly dashboard |

Classification predicts the original diabetes outcome. Regression predicts observed BMI from the other measurements; BMI and BMI-derived features are excluded from regression inputs. The tasks are separate experiments.

The dataset is small and covers women aged at least 21 from a specific population. Results do not establish clinical validity or generalize automatically to other populations.
