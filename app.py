import json, joblib, numpy as np, pandas as pd, plotly.express as px, plotly.graph_objects as go, streamlit as st
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
from data_pipeline import ROOT, CLASS_FEATURES, REG_FEATURES, ZERO_AS_MISSING

st.set_page_config(page_title="PIMA Diabetes ML", page_icon="◈", layout="wide")
st.markdown("""<style>.block-container{max-width:1400px;padding-top:2rem}.hero{padding:30px 36px;border-radius:20px;background:linear-gradient(120deg,#2A174B,#7147B8);color:white}.hero h1{color:white}[data-testid=stMetric]{background:white;border:1px solid #e2dced;border-radius:14px;padding:15px}</style>""", unsafe_allow_html=True)
px.defaults.template="plotly_white"; px.defaults.color_discrete_sequence=["#7147B8","#E1785C","#3B8C88","#CF9A36"]

@st.cache_data
def load_data():
    d=pd.read_csv(ROOT/"data/processed/cohort.csv")
    meta=json.loads((ROOT/"data/processed/provenance.json").read_text())
    summary=json.loads((ROOT/"artifacts/run_summary.json").read_text())
    return d,meta,summary

@st.cache_data
def table(name): return pd.read_csv(ROOT/"artifacts"/name)

@st.cache_resource
def classifiers(): return joblib.load(ROOT/"artifacts/classifiers.joblib")

@st.cache_resource
def ann_model():
    import torch
    from train import make_network
    obj=torch.load(ROOT/"artifacts/ann.pt",map_location="cpu",weights_only=True)
    m=make_network(obj["input_dim"]);m.load_state_dict(obj["state_dict"]);m.eval()
    return joblib.load(ROOT/"artifacts/ann_preprocessor.joblib"),m

def clean_profile(frame):
    x=frame.copy()
    x[ZERO_AS_MISSING]=x[ZERO_AS_MISSING].replace(0,np.nan)
    return x

def predict_probability(name,frame):
    x=clean_profile(frame)
    if name=="PyTorch ANN":
        from train import ann_prob
        prep,model=ann_model()
        return np.asarray(ann_prob(prep,model,x),dtype=float)
    return classifiers()[name].predict_proba(x)[:,1]

def chart(fig):
    fig.update_layout(margin=dict(l=15,r=15,t=55,b=15),font=dict(family="Arial",color="#221A35"),paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig,width="stretch")

if not (ROOT/"artifacts/run_summary.json").exists():
    st.info("Run data_pipeline.py and train.py first.");st.stop()
d,meta,summary=load_data()
st.sidebar.title("◈ PIMA Diabetes")
page=st.sidebar.radio("Workspace",["Overview","Data explorer","Model evaluation","Feature experiments","Prediction lab","Methodology"])
st.sidebar.caption("Lab 1 and Lab 3 dataset · 768 records")
st.sidebar.info("Coursework demonstration. Not for clinical diagnosis.")

if page=="Overview":
    st.markdown('<div class="hero"><h1>PIMA Diabetes Machine Learning</h1><p>Classification, BMI regression, neural network, and interactive evaluation.</p></div>',unsafe_allow_html=True)
    a,b,c,e=st.columns(4);a.metric("Records",len(d));b.metric("Positive outcomes",int(d.outcome.sum()));c.metric("Positive rate",f"{d.outcome.mean():.1%}");e.metric("Classification inputs",len(CLASS_FEATURES))
    m=table("classification_metrics.csv");best=m[(m.model==summary["winner"])&(m.split=="test")].iloc[0]
    st.subheader("Measured result");a,b,c=st.columns(3);a.metric("Selected model",summary["winner"]);b.metric("Test ROC-AUC",f"{best.roc_auc:.3f}");c.metric("Test recall",f"{best.recall:.3f}")
    left,right=st.columns(2)
    with left: chart(px.histogram(d,x="glucose",color=d.outcome.map({0:"Negative",1:"Positive"}),nbins=25,title="Glucose distribution"))
    with right: chart(px.scatter(d,x="bmi",y="glucose",color=d.outcome.map({0:"Negative",1:"Positive"}),title="BMI and glucose"))

elif page=="Data explorer":
    st.title("Data explorer")
    a,b=st.columns(2);split=a.multiselect("Split",["train","validation","test"],default=["train"]);outcome=b.multiselect("Outcome",[0,1],default=[0,1])
    f=d[d.split.isin(split)&d.outcome.isin(outcome)]
    if f.empty: st.info("No matching rows.");st.stop()
    missing=pd.DataFrame({"feature":CLASS_FEATURES,"missing_pct":[100*f[c].isna().mean() for c in CLASS_FEATURES]})
    a,b=st.columns(2)
    with a: chart(px.bar(missing,x="missing_pct",y="feature",orientation="h",title="Missing after invalid zeros are removed"))
    with b: chart(px.box(f,x="outcome",y="age",color="outcome",title="Age by outcome"))
    st.dataframe(f.head(200),width="stretch",hide_index=True)
    with st.expander("Cleaning audit"): st.json(meta)

elif page=="Model evaluation":
    st.title("Model evaluation")
    task=st.radio("Task",["Diabetes classification","BMI regression"],horizontal=True);split=st.radio("Split",["validation","test"],horizontal=True)
    if task=="Diabetes classification":
        m=table("classification_metrics.csv");m=m[m.split==split];metric=st.selectbox("Metric",["roc_auc","average_precision","recall","precision","f1","accuracy","brier"])
        chart(px.bar(m.sort_values(metric),x=metric,y="model",orientation="h",color="model",title=f"{split.title()} model comparison"));st.dataframe(m.round(4),width="stretch",hide_index=True)
        name=st.selectbox("Inspect model",m.model.tolist(),index=m.model.tolist().index(summary["winner"]));p=table("classification_predictions.csv");p=p[(p.split==split)&(p.model==name)]
        threshold=summary["threshold"] if name==summary["winner"] else .5;cm=confusion_matrix(p.actual,p.probability>=threshold)
        a,b=st.columns(2)
        with a: chart(px.imshow(cm,text_auto=True,x=["Negative","Positive"],y=["Negative","Positive"],labels={"x":"Predicted","y":"Actual"},title=f"Confusion matrix · threshold {threshold:.2f}"))
        with b:
            fpr,tpr,_=roc_curve(p.actual,p.probability);fig=go.Figure(go.Scatter(x=fpr,y=tpr,name=name));fig.add_scatter(x=[0,1],y=[0,1],line=dict(dash="dash"),name="Chance");fig.update_layout(title="ROC curve",xaxis_title="False-positive rate",yaxis_title="Recall");chart(fig)
    else:
        m=table("regression_metrics.csv");m=m[m.split==split];chart(px.bar(m,x="model",y="rmse",color="model",title="BMI prediction RMSE"));st.dataframe(m.round(4),width="stretch",hide_index=True)
        name=st.selectbox("Inspect regressor",m.model.tolist());p=table("regression_predictions.csv");p=p[(p.split==split)&(p.model==name)]
        chart(px.scatter(p,x="actual",y="predicted",trendline="ols",title="Observed vs predicted BMI"))

elif page=="Feature experiments":
    st.title("Feature experiments")
    tab=st.radio("View",["Preprocessing","Learning curves","Neural network"],horizontal=True)
    if tab=="Preprocessing":
        m=table("classification_metrics.csv");f=m[m.model.str.startswith("Logistic")];chart(px.bar(f,x="model",y="roc_auc",color="split",barmode="group",title="Raw, scaled, engineered, and selected features"));chart(px.bar(table("permutation_importance.csv").sort_values("importance"),x="importance",y="feature",error_x="std",orientation="h",title="Validation permutation importance"))
    elif tab=="Learning curves":
        c=table("learning_curves.csv");chart(px.line(c,x="n_train",y="roc_auc",color="model",line_dash="split",markers=True,title="Training-size learning curves"))
    else:
        h=table("ann_history.csv");a,b=st.columns(2)
        with a: chart(px.line(h,x="epoch",y=["train_loss","validation_loss"],title="ANN loss"))
        with b: chart(px.line(h,x="epoch",y=["train_accuracy","validation_accuracy"],title="ANN accuracy"))
        st.caption(f"32 → 16 → 1 network; best epoch {summary['ann_best_epoch']}.")

elif page=="Prediction lab":
    st.title("Prediction lab");st.warning("Educational model output, not a diagnosis.")
    names=[n for n in classifiers() if n!="Prevalence baseline"]+["PyTorch ANN"];name=st.selectbox("Model",names,index=names.index(summary["winner"]))
    with st.form("patient"):
        c1,c2,c3=st.columns(3)
        values={"pregnancies":c1.number_input("Pregnancies",0,20,2),"glucose":c2.number_input("Glucose",0.0,250.0,120.0),"blood_pressure":c3.number_input("Blood pressure",0.0,150.0,70.0),"skin_thickness":c1.number_input("Skin thickness",0.0,100.0,25.0),"insulin":c2.number_input("Insulin",0.0,900.0,80.0),"bmi":c3.number_input("BMI",0.0,70.0,30.0),"diabetes_pedigree":c1.number_input("Diabetes pedigree",0.0,3.0,.5),"age":c2.number_input("Age",21,100,35)}
        run_prediction=st.form_submit_button("Calculate model score",type="primary")
    if run_prediction: st.session_state.prediction_ready=True
    if st.session_state.get("prediction_ready"):
        x=pd.DataFrame([values]);pr=float(predict_probability(name,x)[0]);threshold=float(summary["threshold"] if name==summary["winner"] else .5)
        invalid=[c.replace("_"," ").title() for c in ZERO_AS_MISSING if values[c]==0]
        if invalid: st.info("Treated as missing: "+", ".join(invalid))

        a,b,c=st.columns(3)
        a.metric("Model score",f"{pr:.1%}",border=True)
        b.metric("Decision cutoff",f"{threshold:.1%}",border=True)
        c.metric("Model output","Above cutoff" if pr>=threshold else "Below cutoff",border=True)

        comparison_names=list(classifiers())+["PyTorch ANN"]
        comparison=pd.DataFrame({"model":comparison_names,"score":[float(predict_probability(model,x)[0]) for model in comparison_names]})
        metrics=table("classification_metrics.csv")
        validation=metrics[metrics.split=="validation"].set_index("model")
        test=metrics[metrics.split=="test"].set_index("model")
        comparison["cutoff"]=comparison.model.map(lambda model:float(summary["threshold"] if model==summary["winner"] else .5))
        comparison["output"]=np.where(comparison.score>=comparison.cutoff,"Above","Below")
        comparison["validation_roc_auc"]=comparison.model.map(validation.roc_auc)
        comparison["test_roc_auc"]=comparison.model.map(test.roc_auc)

        score_tab,compare_tab,context_tab,explain_tab,whatif_tab=st.tabs(["Score","Model comparison","Cohort context","Model sensitivity","What-if curve"])
        with score_tab:
            fig=go.Figure(go.Indicator(mode="gauge+number",value=100*pr,number={"suffix":"%","valueformat":".1f"},gauge={"axis":{"range":[0,100],"ticksuffix":"%"},"bar":{"color":"#7147B8"},"threshold":{"line":{"color":"#E1785C","width":4},"value":100*threshold}}))
            fig.update_layout(title="Predicted probability and decision cutoff",height=330);chart(fig)
            st.caption("The marker is the model's decision cutoff. It is not a medical diagnostic threshold.")

        with compare_tab:
            winner=summary["winner"];winner_val=float(validation.loc[winner,"roc_auc"]);winner_test=float(test.loc[winner,"roc_auc"])
            a,b,c=st.columns(3)
            a.metric("Best validated model",winner,border=True)
            b.metric("Validation ROC-AUC",f"{winner_val:.3f}",border=True)
            c.metric("Test ROC-AUC",f"{winner_test:.3f}",border=True)

            ordered=comparison.sort_values("score")
            colors=["#7147B8" if model==winner else "#A9A0B5" for model in ordered.model]
            fig=go.Figure()
            fig.add_bar(x=ordered.score,y=ordered.model,orientation="h",text=ordered.score,texttemplate="%{text:.1%}",textposition="outside",marker_color=colors,name="Patient score")
            fig.add_scatter(x=ordered.cutoff,y=ordered.model,mode="markers",marker={"symbol":"diamond","size":11,"color":"#E1785C"},name="Decision cutoff")
            fig.update_layout(title="Same patient scored by every model",xaxis_title="Predicted probability",yaxis_title="",xaxis_tickformat=".0%",xaxis_range=[0,1],barmode="overlay");chart(fig)

            leaderboard=metrics[metrics.split=="validation"].copy().sort_values("roc_auc")
            leaderboard["status"]=np.where(leaderboard.model==winner,"Selected best","Other model")
            fig=px.bar(leaderboard,x="roc_auc",y="model",orientation="h",color="status",text="roc_auc",color_discrete_map={"Selected best":"#7147B8","Other model":"#A9A0B5"},title="Validation ROC-AUC determines the best model",labels={"roc_auc":"Validation ROC-AUC","model":"","status":""})
            fig.update_traces(texttemplate="%{text:.3f}",textposition="outside");fig.update_xaxes(range=[0,1]);chart(fig)

            display=comparison.sort_values("validation_roc_auc",ascending=False).rename(columns={"model":"Model","score":"Patient score","cutoff":"Cutoff","output":"Output","validation_roc_auc":"Validation ROC-AUC","test_roc_auc":"Test ROC-AUC"})
            st.dataframe(display[["Model","Patient score","Cutoff","Output","Validation ROC-AUC","Test ROC-AUC"]],column_config={"Patient score":st.column_config.ProgressColumn(format="percent",min_value=0,max_value=1),"Cutoff":st.column_config.NumberColumn(format="percent"),"Validation ROC-AUC":st.column_config.NumberColumn(format="%.3f"),"Test ROC-AUC":st.column_config.NumberColumn(format="%.3f")},hide_index=True,width="stretch")
            st.caption("Best means highest validation ROC-AUC. Patient probability is not used to choose the winning model.")

        with context_tab:
            preds=table("classification_predictions.csv");preds=preds[(preds.split=="test")&(preds.model==name)]
            if not preds.empty:
                fig=px.histogram(preds,x="probability",color=preds.actual.map({0:"Negative",1:"Positive"}),nbins=20,barmode="overlay",opacity=.7,title="Score compared with held-out test predictions",labels={"probability":"Predicted probability","color":"Outcome"})
                fig.add_vline(x=pr,line_width=3,line_color="#7147B8",annotation_text="Current profile");fig.add_vline(x=threshold,line_dash="dash",line_color="#E1785C",annotation_text="Cutoff");chart(fig)
            train=d[d.split=="train"]
            percentiles=[]
            for feature in CLASS_FEATURES:
                value=clean_profile(x).iloc[0][feature];observed=train[feature].dropna()
                percentiles.append(np.nan if pd.isna(value) else 100*float((observed<=value).mean()))
            pct=pd.DataFrame({"feature":[f.replace("_"," ").title() for f in CLASS_FEATURES],"percentile":percentiles}).dropna()
            fig=px.bar(pct.sort_values("percentile"),x="percentile",y="feature",orientation="h",text="percentile",range_x=[0,100],title="Input percentile within the training cohort")
            fig.update_traces(texttemplate="%{text:.0f}th",textposition="outside");fig.add_vline(x=50,line_dash="dot",line_color="#777");chart(fig)

        with explain_tab:
            train=d[d.split=="train"];rows=[]
            for feature in CLASS_FEATURES:
                changed=x.copy();changed.loc[0,feature]=train[feature].median();reference=float(predict_probability(name,changed)[0])
                rows.append({"feature":feature.replace("_"," ").title(),"change":pr-reference})
            impacts=pd.DataFrame(rows).sort_values("change")
            fig=px.bar(impacts,x="change",y="feature",orientation="h",color="change",color_continuous_scale=["#3B8C88","#E8E2EF","#E1785C"],color_continuous_midpoint=0,title="Score change versus replacing one input with its training median",labels={"change":"Probability change"})
            fig.update_layout(coloraxis_showscale=False);fig.update_xaxes(tickformat="+.1%",zeroline=True,zerolinewidth=2);chart(fig)
            st.caption("One input changes at a time. This shows model sensitivity, not cause or medical importance.")

        with whatif_tab:
            feature=st.selectbox("Change one input",CLASS_FEATURES,index=CLASS_FEATURES.index("glucose"),format_func=lambda v:v.replace("_"," ").title())
            observed=d.loc[d.split=="train",feature].dropna();low,high=observed.quantile([.02,.98]);grid=np.linspace(low,high,60)
            scenarios=pd.concat([x]*len(grid),ignore_index=True);scenarios[feature]=grid;scenario_prob=predict_probability(name,scenarios)
            curve=pd.DataFrame({feature:grid,"probability":scenario_prob})
            fig=px.line(curve,x=feature,y="probability",title=f"Model response while only {feature.replace('_',' ')} changes",labels={"probability":"Predicted probability"})
            current=clean_profile(x).iloc[0][feature]
            if not pd.isna(current): fig.add_scatter(x=[current],y=[pr],mode="markers",marker={"size":12,"color":"#E1785C"},name="Current profile")
            fig.add_hline(y=threshold,line_dash="dash",line_color="#E1785C",annotation_text="Cutoff");fig.update_yaxes(range=[0,1],tickformat=".0%");chart(fig)
            st.caption("All other inputs stay fixed. The curve is a model response, not a treatment recommendation.")

else:
    st.title("Methodology")
    st.markdown((ROOT/"docs/COURSE_ALIGNMENT.md").read_text(encoding="utf-8"))
    st.markdown((ROOT/"docs/SOURCES.md").read_text(encoding="utf-8"))
    st.download_button("Download report",(ROOT/"reports/REPORT.md").read_text(encoding="utf-8"),"PIMA_Diabetes_Report.md")
