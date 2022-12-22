#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns
from scipy import stats
df = pd.read_csv('HousingData.csv')
data = df.copy()

# Missing Value Treatment

print(data.columns[data.isnull().any()])  # Getting names of columns having null values
print(data.isnull().sum())  # Getting the count of missing values in each column

data['CRIM'] = data['CRIM'].fillna(data['CRIM'].median())
data['ZN'] = data['ZN'].fillna(data['ZN'].median())
data['INDUS'] = data['INDUS'].fillna(data['INDUS'].median())
data['LSTAT'] = data['LSTAT'].fillna(data['LSTAT'].median())
data['CHAS'] = data['CHAS'].fillna(data['CHAS'].median())
data['AGE'] = data['AGE'].fillna(data['AGE'].median())
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

z = np.abs(stats.zscore(data))
print(z)

data_outlier = data[(z < 3).all(axis=1)]
data_outlier.shape




print("Dataset shape: ", X.shape)

X_1, X_val, y_1, y_val = train_test_split(X, y, test_size=0.1, random_state=2)
X_train, X_test, y_train, y_test = train_test_split(X_1, y_1, test_size=0.1, random_state=2)
print(X_train.shape)
print(X_test.shape)
print(X_val.shape)
print(y_train.shape)
print(y_test.shape)
print(y_val.shape)
rf = RandomForestRegressor()
rf.fit(X_train,y_train)
y_predicted2 = rf.predict(X_test)


# In[1]:


X


# In[ ]:




