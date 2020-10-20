import pandas as pd 
import numpy as np
from sklearn.linear_model import    LogisticRegression 
import statsmodels.api as sn 
import statsmodels.discrete.discrete_model as sm 

df =pd.read_csv("testnew7.csv",delimiter=",",header=0,index_col=0)

X=df[["WOCHENTAG"]]
y=df["LANGL"]

cfl_rs = LogisticRegression()

cfl_rs.fit(X,y)
#print(cfl_rs.coef_)
#print(cfl_rs.intercept_)
X_cons=sn.add_constant(X)

#print(X_cons.head())

logit =sm.Logit(y,X_cons).fit()
print(logit.summary())