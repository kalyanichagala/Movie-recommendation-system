#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns
from patsy import dmatrices
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
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



Q1 = data.quantile(0.25)  # Getting the value of First Quartile
Q3 = data.quantile(0.75)  # Getting the value of Third Quartile
IQR = Q3 - Q1  # Calculating the interquartile range or IQR value
print(IQR)

data_outlier_IQ = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]

data_outlier_IQ.shape

X = data_outlier_IQ.drop("MEDV", 1)  # data  --- name of the dataset without missing values
y = data_outlier_IQ["MEDV"]

# VIF


d1 = 'CRIM' + '+ZN' + '+INDUS' + '+CHAS' + '+NOX' + '+RM' + '+AGE' + '+DIS' + '+RAD' + '+TAX' + '+PTRATIO' + '+B'      + '+LSTAT '
y1, X1 = dmatrices('MEDV ~' + d1, data, return_type='dataframe')
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X1.values, i) for i in range(X1.shape[1])]
vif["Feature"] = X1.columns
vif.round(2)

d2 = data_outlier_IQ.drop('TAX', axis=1)  # As TAX column VIF is more we remove the column
d2

d1 = 'CRIM' + '+ZN' + '+INDUS' + '+CHAS' + '+NOX' + '+RM' + '+AGE' + '+DIS' + '+RAD' + '+PTRATIO' + '+B' + '+LSTAT'
y1, X1 = dmatrices('MEDV ~' + d1, d2, return_type='dataframe')
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X1.values, i) for i in range(X1.shape[1])]
vif["Feature"] = X1.columns
vif.round(2)

d2 = d2.drop('CRIM', axis=1)  # As TAX column VIF is more we remove the column
d2

d1 = 'ZN' + '+INDUS' + '+CHAS' + '+NOX' + '+RM' + '+AGE' + '+DIS' + '+RAD' + '+PTRATIO' + '+B' + '+LSTAT'
y1, X1 = dmatrices('MEDV ~' + d1, d2, return_type='dataframe')
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X1.values, i) for i in range(X1.shape[1])]
vif["Feature"] = X1.columns
vif.round(2)

d2 = d2.drop("MEDV", 1)
d2  # DATASET WITHOUT TARGET VARIABLE, TAX, CRIM VARIABLE

# Backward elimination method

# Adding constant column of ones, mandatory for sm.OLS model
X_1 = sm.add_constant(d2)
# Fitting sm.OLS model
model = sm.OLS(y, X_1).fit()
model.pvalues

cols = list(d2.columns)
pmax = 1
while len(cols) > 0:
    p = []
    X_1 = d2[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y, X_1).fit()
    p = pd.Series(model.pvalues.values[1:], index=cols)
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if pmax > 0.05:
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print(selected_features_BE)

data = d2.drop(['NOX', 'RAD', 'B'], axis=1)
data


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




