from sklearn.model_selection import train_test_split
import pandas as pd 
import numpy as np
from sklearn.linear_model import    LogisticRegression 
import statsmodels.api as sn 
import statsmodels.discrete.discrete_model as sm 
from sklearn.metrics import confusion_matrix, accuracy_score , mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier


df =pd.read_csv("testnew5.csv",error_bad_lines=False,encoding = 'unicode_escape',delimiter=",",header=0,index_col=0)



X=df.loc[:,df.columns !="LANGL"]
y=(df["LANGL"]==1)


X_train, X_test,y_train,y_test =train_test_split(X,y, test_size=0.2, random_state=0)
rf_clf=RandomForestClassifier(n_estimators=1000,n_jobs=-1,random_state=42)
rf_clf.fit(X_train,y_train)


print(confusion_matrix(y_test,rf_clf.predict(X_test)))
print(accuracy_score(y_test,rf_clf.predict(X_test)))

'''
testnew7:
[[13598   222]
 [ 1377   213]]
0.8962362102530824
'''