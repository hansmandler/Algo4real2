from matplotlib.pyplot import grid
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
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
import xgboost as xgb

df =pd.read_csv("testnew5.csv",error_bad_lines=False,encoding = 'unicode_escape',delimiter=",",header=0,index_col=0)



X=df.loc[:,df.columns !="LANGL"]
y=(df["LANGL"]==1)


X_train, X_test,y_train,y_test =train_test_split(X,y, test_size=0.2, random_state=0)

xgb_clf=xgb.XGBClassifier(n_estimators=250,learning_rate=0.1,random_state=42)
param_test1={
            "max_depth":range(3,10,2),
            "gamma":[0.1,0.2,0.3],
            "subsample":[0.8,0.9],
            "colsample_bytree":[0.8,0.9],
            "reg_alpha":[1e-2,0.1,1]
}
grid_search=GridSearchCV(xgb_clf,param_test1,n_jobs=-1,cv=5,scoring="accuracy")
grid_search.fit(X_train,y_train)
cvxg_clf=grid_search.best_estimator_
print(accuracy_score(y_test,cvxg_clf.predict(X_test)))
print(confusion_matrix(y_test,cvxg_clf.predict(X_test)))
print(grid_search.best_params_)




'''
xgb_clf=xgb.XGBClassifier(max_depth=5,n_estimators=1000,learning_rate=0.3,n_jobs=-1)
xgb_clf.fit(X_train,y_train)
print(accuracy_score(y_test,xgb_clf.predict(X_test)))
print(confusion_matrix(y_test,xgb_clf.predict(X_test)))
print(xgb.plot_importance(xgb_clf))
testnew 5 
0.8933160285528877
[[13505   315]
 [ 1329   261]]'''