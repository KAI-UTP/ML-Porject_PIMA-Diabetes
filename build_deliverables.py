from pathlib import Path
import base64, json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd, numpy as np, nbformat, mistune
from nbclient import NotebookClient
from jupyter_client import KernelManager
from data_pipeline import ROOT, CLASS_FEATURES

A=ROOT/"artifacts"; OUT=ROOT/"reports"; FIG=OUT/"figures"; FIG.mkdir(parents=True,exist_ok=True); (ROOT/"notebooks").mkdir(exist_ok=True)
d=pd.read_csv(ROOT/"data/processed/cohort.csv"); meta=json.loads((ROOT/"data/processed/provenance.json").read_text()); summary=json.loads((A/"run_summary.json").read_text()); c=pd.read_csv(A/"classification_metrics.csv"); r=pd.read_csv(A/"regression_metrics.csv"); h=pd.read_csv(A/"ann_history.csv")
plt.rcParams.update({"font.family":"DejaVu Sans","axes.spines.top":False,"axes.spines.right":False,"figure.dpi":130})
fig,ax=plt.subplots(1,2,figsize=(11,4));
for target,color in [(0,"#7147B8"),(1,"#E1785C")]: ax[0].hist(d.loc[d.outcome==target,"glucose"].dropna(),bins=20,alpha=.6,label=str(target),color=color)
ax[0].set(title="Glucose by diabetes outcome",xlabel="Glucose",ylabel="Records");ax[0].legend(title="Outcome")
f=c[c.split=="test"].sort_values("roc_auc");ax[1].barh(f.model,f.roc_auc,color="#7147B8");ax[1].set(title="Test ROC-AUC",xlabel="ROC-AUC");plt.tight_layout();plt.savefig(FIG/"01_results.png",bbox_inches="tight");plt.close()
fig,ax=plt.subplots(1,2,figsize=(11,4));ax[0].plot(h.epoch,h.train_loss,label="train");ax[0].plot(h.epoch,h.validation_loss,label="validation");ax[0].set(title="ANN loss",xlabel="Epoch");ax[0].legend();rf=r[r.split=="test"];ax[1].barh(rf.model,rf.rmse,color="#3B8C88");ax[1].set(title="BMI regression test RMSE",xlabel="RMSE");plt.tight_layout();plt.savefig(FIG/"02_learning_regression.png",bbox_inches="tight");plt.close()

best=c[(c.model==summary["winner"])&(c.split=="test")].iloc[0];reg=r[(r.model==summary["regression_winner"])&(r.split=="test")].iloc[0]
def table(f):
    f=f.copy();
    for col in f.select_dtypes("number"): f[col]=f[col].map(lambda x:f"{x:.3f}")
    return "| "+" | ".join(f.columns)+" |\n| "+" | ".join(["---"]*len(f.columns))+" |\n"+"\n".join("| "+" | ".join(map(str,row))+" |" for row in f.to_numpy())
report=f'''# PIMA Diabetes Prediction and BMI Regression

**Group details:** add before submission

## Introduction

This project applies CRISP-DM to the PIMA Indians Diabetes dataset used in Labs 1 and 3. Classification predicts the documented diabetes outcome. A separate regression experiment predicts BMI from the remaining measurements. A PyTorch neural network provides the required neural model. The objective is coursework comparison, not clinical diagnosis.

## Data understanding and preparation

The dataset contains {len(d)} records, eight numeric predictors, and one binary outcome. There are {int(d.outcome.sum())} positive cases. The documented population consists of women aged at least 21 of Pima Indian heritage, limiting generalization.

Zeros in glucose, blood pressure, skin thickness, insulin, and BMI are physiologically invalid and are converted to missing values. Median imputation, missingness indicators, scaling, encoding, and selection are learned from training data inside pipelines. The stratified split contains {meta['split_counts']['train']} training, {meta['split_counts']['validation']} validation, and {meta['split_counts']['test']} test records.

Engineered classification features are age bands, glucose × age, pregnancies ÷ age, and glucose × BMI. Embedded Extra Trees selection is compared with all engineered features. BMI regression excludes BMI and every BMI-derived feature from its inputs. Only recorded BMI values form its target.

## Models and evaluation

Classification compares a prevalence baseline, four logistic variants, decision tree, random forest, and a 32–16 PyTorch ANN. Five-fold stratified cross-validation tunes classical classifiers. The validation set selects the model by ROC-AUC and selects an F1 threshold. The test set is reserved for reporting.

Regression compares a mean baseline, Ridge, decision tree, and random forest using five-fold cross-validation. Metrics are accuracy, precision, recall, F1, ROC-AUC, average precision, Brier score, MAE, MSE, RMSE, and R².

## Results

{table(c[c.split=='test'][['model','accuracy','precision','recall','f1','roc_auc','average_precision']])}

The validation-selected classifier is **{summary['winner']}**. Its test ROC-AUC is **{best.roc_auc:.3f}**, average precision **{best.average_precision:.3f}**, and recall **{best.recall:.3f}** at threshold 0.50. At the validation-selected threshold {summary['threshold']:.3f}, test F1 is {summary['test_at_selected_threshold']['f1']:.3f}. The bootstrap 95% ROC-AUC interval is {summary['bootstrap_95pct']['roc_auc'][0]:.3f}–{summary['bootstrap_95pct']['roc_auc'][1]:.3f}.

![Classification results](figures/01_results.png)

{table(r[r.split=='test'][['model','mae','mse','rmse','r2']])}

The selected BMI regressor is **{summary['regression_winner']}**, with test RMSE **{reg.rmse:.3f}** and R² **{reg.r2:.3f}**. Performance must be judged against the mean baseline. BMI prediction is an academic secondary task, not a replacement for measurement.

![Learning and regression](figures/02_learning_regression.png)

## Discussion

The ablation shows whether scaling, engineered features, or embedded selection changes generalization. Feature selection can simplify a model without guaranteeing higher performance. Learning curves compare training and validation behavior as the training sample grows. The ANN uses dropout and early stopping; its best checkpoint is epoch {summary['ann_best_epoch']}.

The dataset is small, historical, and population-specific. Missing measurements may not be missing at random. A random split does not test future or external populations. Real deployment would require broader data, external validation, calibration, subgroup evaluation, and clinical review.

## Conclusion

The project implements the complete required pipeline: cleaning, feature engineering, embedded selection, classification, regression, neural-network training, bias–variance diagnostics, and Streamlit visualization. Results support comparison of methods on this dataset only.

## References

- [PIMA dataset CSV]({meta['source_url']})
- [Dataset documentation](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.names)
- Local Machine Learning Labs 1 and 3 and project brief.
- [scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)
- [PyTorch optimization tutorial](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
'''
(OUT/"REPORT.md").write_text(report,encoding="utf-8")
(OUT/"PRESENTATION_OUTLINE.md").write_text(f'''# Presentation outline

1. Problem, dataset, and CRISP-DM.
2. Invalid zeros, missing values, and leakage prevention.
3. Feature engineering and embedded selection.
4. Classification, BMI regression, and ANN architectures.
5. Results: {summary['winner']} test ROC-AUC {best.roc_auc:.3f}; {summary['regression_winner']} BMI RMSE {reg.rmse:.3f}.
6. Bias–variance evidence, limitations, and Streamlit demonstration.
''',encoding="utf-8")

nb=nbformat.v4.new_notebook();nb.cells=[
 nbformat.v4.new_markdown_cell("# PIMA Diabetes Machine-Learning Project\n\nExecuted CRISP-DM analysis. The reusable implementation is in `data_pipeline.py` and `train.py`."),
 nbformat.v4.new_code_cell("from pathlib import Path\nimport json, pandas as pd\nfrom IPython.display import display, Image, Markdown\nROOT=Path.cwd()\nif not (ROOT/'train.py').exists(): ROOT=ROOT.parent\nd=pd.read_csv(ROOT/'data/processed/cohort.csv')\nsummary=json.loads((ROOT/'artifacts/run_summary.json').read_text())\ndisplay(d.describe(include='all'))"),
 nbformat.v4.new_markdown_cell("## Data preparation\n\nInvalid clinical zeros become missing. Imputation, scaling, engineering, and selection are fit inside training pipelines."),
 nbformat.v4.new_code_cell("display(pd.read_csv(ROOT/'artifacts/classification_metrics.csv').round(4))\ndisplay(Image(filename=str(ROOT/'reports/figures/01_results.png')))"),
 nbformat.v4.new_markdown_cell("## Regression and neural network\n\nBMI is excluded from regression inputs. The ANN uses 32 and 16 hidden units with dropout and early stopping."),
 nbformat.v4.new_code_cell("display(pd.read_csv(ROOT/'artifacts/regression_metrics.csv').round(4))\ndisplay(pd.read_csv(ROOT/'artifacts/ann_history.csv').tail())\ndisplay(Image(filename=str(ROOT/'reports/figures/02_learning_regression.png')))"),
 nbformat.v4.new_markdown_cell("## Feature experiments and conclusions"),
 nbformat.v4.new_code_cell("display(pd.read_csv(ROOT/'artifacts/permutation_importance.csv'))\ndisplay(pd.read_csv(ROOT/'artifacts/learning_curves.csv'))\ndisplay(Markdown((ROOT/'reports/REPORT.md').read_text(encoding='utf-8').split('## Discussion')[1]))")]
nb.metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":sys.version.split()[0]}}
path=ROOT/"notebooks/PIMA_Diabetes_Project.ipynb";nbformat.write(nb,path);km=KernelManager();km.kernel_spec.argv=[sys.executable,"-m","ipykernel_launcher","-f","{connection_file}"];NotebookClient(nb,timeout=180,resources={"metadata":{"path":str(ROOT)}},km=km).execute();nbformat.write(nb,path)
render=mistune.create_markdown(plugins=["table"]);body=render(report)
for f in FIG.glob("*.png"): body=body.replace("figures/"+f.name,"data:image/png;base64,"+base64.b64encode(f.read_bytes()).decode())
css="body{max-width:1000px;margin:45px auto;padding:0 28px;font:16px/1.6 Arial;color:#221A35}h1,h2{color:#4C2B7A}table{border-collapse:collapse;width:100%;font-size:12px}td,th{padding:7px;border-bottom:1px solid #ddd}img{width:100%}"
(OUT/"REPORT.html").write_text('<!doctype html><meta charset="utf-8"><style>'+css+'</style><body>'+body+'</body>',encoding="utf-8")
print("Report words:",len(report.split()),"Notebook executed")
