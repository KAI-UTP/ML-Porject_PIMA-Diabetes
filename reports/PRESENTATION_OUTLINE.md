# Presentation Outline

1. **Problem and data** – diabetes detection, CRISP-DM, 768 patients, class imbalance, impossible zeros.
2. **Data preparation** – zeros → missing, median imputation + indicators, scaling, outliers kept, 80/20 split, no leakage.
3. **Feature engineering and selection** – log, interaction, binning; filter vs wrapper vs embedded; ablation results.
4. **Models** – five classifiers, PyTorch ANN (32-16, dropout, early stopping), BMI regression.
5. **Results** – SVM (RBF) CV ROC-AUC 0.847, test 0.827; recall 78% at the tuned threshold; Lasso BMI RMSE.
6. **Bias–variance and discussion** – depth curve, learning curves, train–CV gaps, the four discussion questions.
7. **Dashboard** – live walkthrough of the Streamlit app and prediction lab.
