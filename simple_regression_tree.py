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


df =pd.read_csv("testnew7.csv",error_bad_lines=False,encoding = 'unicode_escape',delimiter=",",header=0,index_col=0)



X=df.loc[:,df.columns !="LANGL"]
y=df["LANGL"]


X_train, X_test,y_train,y_test =train_test_split(X,y, test_size=0.2, random_state=0)
regtree =tree.DecisionTreeRegressor(min_samples_split=50, max_depth=4 )  
'''min_sample_split mindest anzahl an samples um zu splitten sonst wird nichtgespölitetet  ''' #max_depth=3 maximum numbers of level in tree 
regtree.fit(X_train,y_train)

y_train_pred = regtree.predict(X_train)
y_test_pred=regtree.predict(X_test)

print(mean_squared_error(y_test,y_test_pred)) # MSE ist nur zum vergleichen von models alleine ncht aussage kräftig 
print(r2_score(y_train,y_train_pred))
print(r2_score(y_test,y_test_pred))