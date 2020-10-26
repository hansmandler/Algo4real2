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


df =pd.read_csv("testnew5.csv",error_bad_lines=False,encoding = 'unicode_escape',delimiter=",",header=0,index_col=0)



X=df.loc[:,df.columns !="LANGL"]
y=(df["LANGL"]==1)


X_train, X_test,y_train,y_test =train_test_split(X,y, test_size=0.2, random_state=0)
clftree =tree.DecisionTreeClassifier(max_depth=3)

clftree.fit(X_train,y_train)

y_train_pred=clftree.predict(X_train)
y_test_pred=clftree.predict(X_test)

print(confusion_matrix(y_train,y_train_pred))
print(confusion_matrix(y_test,y_test_pred))

print(accuracy_score(y_test,y_test_pred))
'''
test new7:
[[55209   169]
 [ 6060   198]]
[[13769    51]
 [ 1538    52]]
0.8968851395197923
test new5:
[[55137   241]
 [ 5919   339]]
[[13760    60]
 [ 1510    80]]
0.8981181051265412
'''