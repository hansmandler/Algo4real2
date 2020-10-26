from matplotlib.pyplot import grid
from sklearn.model_selection import train_test_split
import pandas as pd 
import numpy as np
from sklearn.linear_model import    LogisticRegression 
import statsmodels.api as sn 
import statsmodels.discrete.discrete_model as sm 
from sklearn.metrics import confusion_matrix, accuracy_score , mean_squared_error, r2_score
from sklearn.svm import SVR

df =pd.read_csv("testnew5.csv",error_bad_lines=False,encoding = 'unicode_escape',delimiter=",",header=0,index_col=0)
X=df.loc[:,df.columnsgit !="LANGL"]
y=(df["LANGL"]==1)
X_train, X_test,y_train,y_test =train_test_split(X,y, test_size=0.2, random_state=0)

svr=SVR(kernel="linear",C=1000)

svr.fit(X_train,y_train)

print(svr.predict(X_train))
print(svr.predict(X_train))