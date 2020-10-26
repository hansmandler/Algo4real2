from sklearn.model_selection import train_test_split
import pandas as pd 
import numpy as np
from sklearn.linear_model import    LogisticRegression 
import statsmodels.api as sn 
import statsmodels.discrete.discrete_model as sm 
from sklearn.metrics import confusion_matrix, accuracy_score 
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV


df =pd.read_csv("testnew7.csv",error_bad_lines=False,encoding = 'unicode_escape',delimiter=",",header=0,index_col=0)



X=df.loc[:,df.columns !="LANGL"]
y=(df["LANGL"]==1)


X_train, X_test,y_train,y_test =train_test_split(X,y, test_size=0.2, random_state=0)
scaler=preprocessing.StandardScaler().fit(X_train)
X_train_s=scaler.transform(X_train)

scaler=preprocessing.StandardScaler().fit(X_test)
X_test_s=scaler.transform(X_test)
params = {"n_neighbors": [1,2,3,45,6,7,8,9,10]}
grid_search_cv= GridSearchCV(KNeighborsClassifier(),params )

grid_search_cv.fit(X_train_s,y_train)
grid_search_cv.best_params_

optimised_KNN=grid_search_cv.best_estimator_

y_test_pred=optimised_KNN.predict(X_test_s)

print(confusion_matrix(y_test,y_test_pred))
print(accuracy_score(y_test,y_test_pred))



'''
scaler=preprocessing.StandardScaler().fit(X_train)
X_train_s=scaler.transform(X_train)

scaler=preprocessing.StandardScaler().fit(X_test)
X_test_s=scaler.transform(X_test)

clf_knn_1=KNeighborsClassifier(n_neighbors=1)
clf_knn_1.fit(X_train_s,y_train)
'''
#print(confusion_matrix(y_test,clf_knn_1.predict(X_test_s)))
#print(accuracy_score(y_test,clf_knn_1.predict(X_test_s)))
