"""Reproducible PIMA classification, BMI regression, and PyTorch ANN experiments."""
import copy, json, platform, time
import joblib, numpy as np, pandas as pd, sklearn, torch
from torch import nn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold, cross_val_score
from sklearn.metrics import *
from sklearn.inspection import permutation_importance
from data_pipeline import ROOT, SEED, CLASS_FEATURES, REG_FEATURES, FeatureEngineer, prepare_data


def preprocessing(task="classification", engineered=True, scaled=True, selected=False):
    base = CLASS_FEATURES if task == "classification" else REG_FEATURES
    numeric = list(base) + (["glucose_age", "pregnancies_per_age"] if engineered else []) + (["glucose_bmi"] if engineered and task == "classification" else [])
    categorical = ["age_band"] if engineered else []
    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler() if scaled else "passthrough")])
    transformers = [("numeric", num_pipe, numeric)]
    if categorical:
        transformers.append(("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical))
    steps = [("engineer", FeatureEngineer(task, engineered)), ("prepare", ColumnTransformer(transformers))]
    if selected and task == "classification":
        selector = ExtraTreesClassifier(n_estimators=150, random_state=SEED, min_samples_leaf=4)
        steps.append(("select", SelectFromModel(selector, threshold="median")))
    return steps


def cmetrics(y, p, threshold=.5):
    pred = np.asarray(p) >= threshold
    return {"accuracy":accuracy_score(y,pred),"precision":precision_score(y,pred,zero_division=0),"recall":recall_score(y,pred,zero_division=0),"f1":f1_score(y,pred,zero_division=0),"roc_auc":roc_auc_score(y,p),"average_precision":average_precision_score(y,p),"brier":brier_score_loss(y,p),"n":len(y),"positives":int(np.sum(y)),"threshold":threshold}


def rmetrics(y, p):
    return {"mae":mean_absolute_error(y,p),"mse":mean_squared_error(y,p),"rmse":np.sqrt(mean_squared_error(y,p)),"r2":r2_score(y,p),"n":len(y)}


def make_network(dim):
    return nn.Sequential(nn.Linear(dim,32),nn.ReLU(),nn.Dropout(.2),nn.Linear(32,16),nn.ReLU(),nn.Linear(16,1))


def train_ann(train, validation, epochs=150):
    torch.manual_seed(SEED)
    prep=Pipeline(preprocessing("classification",True,True,False))
    x=prep.fit_transform(train[CLASS_FEATURES]).astype("float32"); v=prep.transform(validation[CLASS_FEATURES]).astype("float32")
    xt=torch.tensor(x); yt=torch.tensor(train.outcome.to_numpy(dtype="float32")).reshape(-1,1)
    xv=torch.tensor(v); yv=torch.tensor(validation.outcome.to_numpy(dtype="float32")).reshape(-1,1)
    model=make_network(x.shape[1]); opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=1e-4); lossfn=nn.BCEWithLogitsLoss()
    history=[]; best=float("inf"); stale=0; best_epoch=0
    for epoch in range(1,epochs+1):
        model.train(); opt.zero_grad(); loss=lossfn(model(xt),yt); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            a=model(xt); b=model(xv); tl=lossfn(a,yt).item(); vl=lossfn(b,yv).item()
            ta=((a.sigmoid()>=.5)==yt.bool()).float().mean().item(); va=((b.sigmoid()>=.5)==yv.bool()).float().mean().item()
        history.append({"epoch":epoch,"train_loss":tl,"validation_loss":vl,"train_accuracy":ta,"validation_accuracy":va})
        if vl < best-1e-5: best=vl; state=copy.deepcopy(model.state_dict()); best_epoch=epoch; stale=0
        else: stale+=1
        if stale>=15: break
    model.load_state_dict(state); model.eval(); return prep,model,history,best_epoch


def ann_prob(prep, model, X):
    with torch.no_grad(): return model(torch.tensor(prep.transform(X).astype("float32"))).sigmoid().numpy().ravel()


def run():
    start=time.time(); out=ROOT/"artifacts"; out.mkdir(exist_ok=True)
    if not (ROOT/"data/processed/cohort.csv").exists(): prepare_data()
    d=pd.read_csv(ROOT/"data/processed/cohort.csv"); parts={s:d[d.split==s].reset_index(drop=True) for s in ["train","validation","test"]}; tr,va,te=parts.values()
    # Outlier fences are estimated from training data; unusual records are retained.
    outliers=[]
    for feature in CLASS_FEATURES:
        q1,q3=tr[feature].quantile([.25,.75]); iqr=q3-q1
        low,high=q1-1.5*iqr,q3+1.5*iqr
        for split,frame in parts.items():
            observed=frame[feature].dropna()
            outliers.append(dict(feature=feature,split=split,lower=low,upper=high,flagged=int(((observed<low)|(observed>high)).sum()),observed=len(observed)))
    pd.DataFrame(outliers).to_csv(out/"outlier_audit.csv",index=False)
    cv=StratifiedKFold(5,shuffle=True,random_state=SEED); rows=[]; preds=[]; models={}; tuning=[]
    specs={
      "Prevalence baseline":(False,True,False,DummyClassifier(strategy="prior"),{}),
      "Logistic / raw":(False,False,False,LogisticRegression(max_iter=3000),{}),
      "Logistic / scaled":(False,True,False,LogisticRegression(max_iter=3000),{}),
      "Logistic / engineered":(True,True,False,LogisticRegression(max_iter=3000),{"model__C":[.1,1,10]}),
      "Logistic / selected":(True,True,True,LogisticRegression(max_iter=3000),{"model__C":[.1,1,10]}),
      "Decision tree":(True,True,False,DecisionTreeClassifier(random_state=SEED,min_samples_leaf=8),{"model__max_depth":[3,5,8]}),
      "Random forest":(True,True,False,RandomForestClassifier(random_state=SEED,n_estimators=300,min_samples_leaf=4),{"model__max_depth":[4,8,None]})}
    for name,(eng,scale,sel,est,grid) in specs.items():
        pipe=Pipeline(preprocessing("classification",eng,scale,sel)+[("model",est)])
        if grid:
            gs=GridSearchCV(pipe,grid,cv=cv,scoring="roc_auc",return_train_score=True); gs.fit(tr[CLASS_FEATURES],tr.outcome); pipe=gs.best_estimator_
            for i,p in enumerate(gs.cv_results_["params"]): tuning.append({"task":"classification","model":name,"params":str(p),"cv_score":gs.cv_results_["mean_test_score"][i]})
        else: pipe.fit(tr[CLASS_FEATURES],tr.outcome)
        models[name]=pipe
        for split,f in parts.items():
            pr=pipe.predict_proba(f[CLASS_FEATURES])[:,1]; rows.append({"model":name,"split":split,**cmetrics(f.outcome,pr)}); preds.append(pd.DataFrame({"record_id":f.record_id,"model":name,"split":split,"actual":f.outcome,"probability":pr}))
    prep,ann,history,best_epoch=train_ann(tr,va)
    for split,f in parts.items():
        pr=ann_prob(prep,ann,f[CLASS_FEATURES]); rows.append({"model":"PyTorch ANN","split":split,**cmetrics(f.outcome,pr)}); preds.append(pd.DataFrame({"record_id":f.record_id,"model":"PyTorch ANN","split":split,"actual":f.outcome,"probability":pr}))
    cm=pd.DataFrame(rows); cp=pd.concat(preds); winner=cm[(cm.split=="validation")&(cm.model!="Prevalence baseline")].sort_values("roc_auc",ascending=False).iloc[0].model
    ablations=[]
    for label,eng,scaled,selected in [("Raw",False,False,False),("Scaled",False,True,False),("Engineered",True,True,False),("Selected",True,True,True)]:
        estimator=Pipeline(preprocessing("classification",eng,scaled,selected)+[("model",LogisticRegression(C=1,max_iter=3000))])
        scores=cross_val_score(estimator,tr[CLASS_FEATURES],tr.outcome,cv=cv,scoring="roc_auc")
        estimator.fit(tr[CLASS_FEATURES],tr.outcome)
        for split,frame in [("train",tr),("validation",va)]:
            ablations.append(dict(variant=label,split=split,roc_auc=roc_auc_score(frame.outcome,estimator.predict_proba(frame[CLASS_FEATURES])[:,1]),cv_mean=scores.mean(),cv_std=scores.std()))
    pd.DataFrame(ablations).to_csv(out/"preprocessing_ablation.csv",index=False)
    selected_pipe=models["Logistic / selected"]
    feature_names=selected_pipe.named_steps["prepare"].get_feature_names_out()
    selection=selected_pipe.named_steps["select"]
    pd.DataFrame(dict(feature=feature_names,selected=selection.get_support(),importance=selection.estimator_.feature_importances_)).to_csv(out/"selected_features.csv",index=False)
    vp=cp[(cp.split=="validation")&(cp.model==winner)]; precision,recall,thresholds=precision_recall_curve(vp.actual,vp.probability); fs=2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-12); threshold=float(thresholds[np.argmax(fs)])
    tp=cp[(cp.split=="test")&(cp.model==winner)]; selected_test=cmetrics(tp.actual,tp.probability,threshold)
    cm.to_csv(out/"classification_metrics.csv",index=False); cp.to_csv(out/"classification_predictions.csv",index=False); pd.DataFrame(history).to_csv(out/"ann_history.csv",index=False)
    operating=[]
    for (model,split),frame in cp.groupby(["model","split"]):
        cutoff=threshold if model==winner else .5
        operating.append(dict(model=model,split=split,**cmetrics(frame.actual,frame.probability,cutoff)))
    pd.DataFrame(operating).to_csv(out/"operating_metrics.csv",index=False)
    joblib.dump(models,out/"classifiers.joblib"); joblib.dump(prep,out/"ann_preprocessor.joblib"); torch.save({"input_dim":ann[0].in_features,"state_dict":ann.state_dict()},out/"ann.pt")
    imp=permutation_importance(models["Random forest"],va[CLASS_FEATURES],va.outcome,n_repeats=20,random_state=SEED,scoring="roc_auc"); pd.DataFrame({"feature":CLASS_FEATURES,"importance":imp.importances_mean,"std":imp.importances_std}).sort_values("importance",ascending=False).to_csv(out/"permutation_importance.csv",index=False)
    curves=[]
    for frac in [.25,.5,.75,1.]:
        sub=tr.sample(frac=frac,random_state=SEED).sort_index()
        for name in ["Logistic / engineered","Random forest"]:
            model=clone(models[name]).fit(sub[CLASS_FEATURES],sub.outcome)
            for split,f in [("train",sub),("validation",va)]: curves.append({"model":name,"n_train":len(sub),"split":split,"roc_auc":roc_auc_score(f.outcome,model.predict_proba(f[CLASS_FEATURES])[:,1])})
    pd.DataFrame(curves).to_csv(out/"learning_curves.csv",index=False)
    # BMI regression uses only observed BMI and excludes BMI plus every BMI-derived feature.
    rparts={s:f.dropna(subset=["bmi"]).reset_index(drop=True) for s,f in parts.items()}; rtr=rparts["train"]; rcv=KFold(5,shuffle=True,random_state=SEED); rrows=[]; rpred=[]; regressors={}
    rspecs={"Mean baseline":(DummyRegressor(),{}),"Ridge":(Ridge(),{"model__alpha":[.1,1,10,100]}),"Decision tree regressor":(DecisionTreeRegressor(random_state=SEED,min_samples_leaf=8),{"model__max_depth":[3,5,8]}),"Random forest regressor":(RandomForestRegressor(random_state=SEED,n_estimators=300,min_samples_leaf=5),{"model__max_depth":[4,8,None]})}
    for name,(est,grid) in rspecs.items():
        pipe=Pipeline(preprocessing("regression",True,True,False)+[("model",est)])
        if grid:
            gs=GridSearchCV(pipe,grid,cv=rcv,scoring="neg_root_mean_squared_error"); gs.fit(rtr[REG_FEATURES],rtr.bmi); pipe=gs.best_estimator_
            for i,p in enumerate(gs.cv_results_["params"]): tuning.append({"task":"regression","model":name,"params":str(p),"cv_score":-gs.cv_results_["mean_test_score"][i]})
        else: pipe.fit(rtr[REG_FEATURES],rtr.bmi)
        regressors[name]=pipe
        for split,f in rparts.items():
            pr=pipe.predict(f[REG_FEATURES]); rrows.append({"model":name,"split":split,**rmetrics(f.bmi,pr)}); rpred.append(pd.DataFrame({"record_id":f.record_id,"model":name,"split":split,"actual":f.bmi,"predicted":pr,"residual":f.bmi-pr}))
    rm=pd.DataFrame(rrows); rm.to_csv(out/"regression_metrics.csv",index=False); pd.concat(rpred).to_csv(out/"regression_predictions.csv",index=False); joblib.dump(regressors,out/"regressors.joblib"); pd.DataFrame(tuning).to_csv(out/"cv_results.csv",index=False)
    rw=rm[rm.split=="validation"].sort_values("rmse").iloc[0].model
    rng=np.random.default_rng(SEED); bs=[]; y=tp.actual.to_numpy(); pr=tp.probability.to_numpy()
    for _ in range(500):
        ix=rng.integers(len(y),size=len(y)); bs.append([roc_auc_score(y[ix],pr[ix]),average_precision_score(y[ix],pr[ix])])
    summary={"winner":winner,"regression_winner":rw,"threshold":threshold,"test_at_selected_threshold":selected_test,"bootstrap_95pct":{"roc_auc":np.quantile(np.array(bs)[:,0],[.025,.975]).tolist(),"average_precision":np.quantile(np.array(bs)[:,1],[.025,.975]).tolist()},"ann_best_epoch":best_epoch,"ann_epochs_run":len(history),"duration_seconds":round(time.time()-start,1),"versions":{"python":platform.python_version(),"sklearn":sklearn.__version__,"torch":torch.__version__,"pandas":pd.__version__}}
    (out/"run_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8"); print(json.dumps(summary,indent=2)); return summary


if __name__ == "__main__": run()
