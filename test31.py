#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import seaborn as sns
#import statsmodels.api as sm
#from patsy import dmatrices
#from scipy import stats
#from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
#from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
#from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import GridSearchCV
#from sklearn.model_selection import RandomizedSearchCV
import time

start_time = time.time()

df = pd.read_csv('HousingData.csv')  

df.shape

df.head(10)

df.describe()

data = df.copy()



print(data.columns[data.isnull().any()])  
print(data.isnull().sum())  

data['CRIM'] = data['CRIM'].fillna(data['CRIM'].median())
data['ZN'] = data['ZN'].fillna(data['ZN'].median())
data['INDUS'] = data['INDUS'].fillna(data['INDUS'].median())
data['LSTAT'] = data['LSTAT'].fillna(data['LSTAT'].median())
data['CHAS'] = data['CHAS'].fillna(data['CHAS'].median())
data['AGE'] = data['AGE'].fillna(data['AGE'].median())
print(data.isna().sum())



X = data.iloc[:, :-1]
y = data.iloc[:, -1]

print("Dataset shape: ", X.shape)

X_1, X_val, y_1, y_val = train_test_split(X, y, test_size=0.1, random_state=2)
X_train, X_test, y_train, y_test = train_test_split(X_1, y_1, test_size=0.1, random_state=2)
print(X_train.shape)
print(X_test.shape)
print(X_val.shape)
print(y_train.shape)
print(y_test.shape)
print(y_val.shape)
    



rf = RandomForestRegressor(max_depth=4,n_estimators=100)

param_grid = {
    
    'max_depth': [2,4,6],
    
    
    
    'n_estimators': [i for i in range(100, 401, 100)],
    
}


rf.fit(X_train,y_train)
#model2 = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3)
#model2.fit(X_train, y_train)


# In[ ]:




