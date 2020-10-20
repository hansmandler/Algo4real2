import pandas as pd 
import numpy as np
from sklearn.linear_model import    LogisticRegression 
import statsmodels.api as sn 
import statsmodels.discrete.discrete_model as sm 
from sklearn.metrics import confusion_matrix 

df =pd.read_csv("testnew7.csv",delimiter=",",header=0,index_col=0)

X=df.loc[:,df.columns!="LANGL"]
y=df["LANGL"]

cfl_lr = LogisticRegression()

cfl_lr.fit(X,y)
#print(cfl_rs.coef_)
#print(cfl_rs.intercept_)
X_cons=sn.add_constant(X)

#print(X_cons.head())

logit =sm.Logit(y,X_cons).fit()
#print(logit.summary())

#print(cfl_lr.predict_proba(X))
y_pred=cfl_lr.predict(X)
#print(y_pred)
y_pred_03=(cfl_lr.predict_proba(X)[:,0]>=0.3)
#print(y_pred_03)
confusion_matrix(y,y_pred)

