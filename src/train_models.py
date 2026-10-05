from pathlib import Path
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,average_precision_score,roc_auc_score,confusion_matrix

SEED=42
BASE=["Type","Air temperature [K]","Process temperature [K]","Rotational speed [rpm]","Torque [Nm]","Tool wear [min]"]
def features(df):
    x=df[BASE].copy(); x["Temperature difference [K]"]=x["Process temperature [K]"]-x["Air temperature [K]"]
    x["Power proxy"]=x["Torque [Nm]"]*x["Rotational speed [rpm]"]; x["Torque x wear"]=x["Torque [Nm]"]*x["Tool wear [min]"]; return x

def score(y,pred,prob):
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(Accuracy=accuracy_score(y,pred),Precision=precision_score(y,pred,zero_division=0),Recall=recall_score(y,pred,zero_division=0),F1=f1_score(y,pred,zero_division=0),PR_AUC=average_precision_score(y,prob),ROC_AUC=roc_auc_score(y,prob),TN=tn,FP=fp,FN=fn,TP=tp)

def main():
    root=Path(__file__).resolve().parents[1]; df=pd.read_csv(root/"data/raw/ai4i2020.csv"); X=features(df); y=df["Machine failure"].astype(int)
    Xtv,_,ytv,_=train_test_split(X,y,test_size=.2,stratify=y,random_state=SEED)
    Xtr,Xv,ytr,yv=train_test_split(Xtv,ytv,test_size=.2,stratify=ytv,random_state=SEED)
    cat=["Type"]; num=[c for c in X.columns if c not in cat]
    pre=ColumnTransformer([("num",Pipeline([("imp",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num),("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",OneHotEncoder(handle_unknown="ignore"))]),cat)])
    models={"Dummy majority":DummyClassifier(strategy="most_frequent"),"Logistic Regression":LogisticRegression(max_iter=3000,class_weight="balanced",random_state=SEED),"Decision Tree":DecisionTreeClassifier(max_depth=6,min_samples_leaf=8,class_weight="balanced",random_state=SEED),"Random Forest":RandomForestClassifier(n_estimators=400,min_samples_leaf=2,class_weight="balanced",random_state=SEED,n_jobs=-1)}
    rows=[]
    for name,est in models.items():
        p=Pipeline([("prep",pre),("model",est)]); p.fit(Xtr,ytr); pred=p.predict(Xv); prob=p.predict_proba(Xv)[:,1] if hasattr(p,"predict_proba") else np.repeat(ytr.mean(),len(yv)); rows.append({"Model":name,**score(yv,pred,prob)})
    out=root/"results/tables"; out.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(out/"validation_metrics.csv",index=False); print(pd.DataFrame(rows).to_string(index=False))
if __name__=="__main__": main()
