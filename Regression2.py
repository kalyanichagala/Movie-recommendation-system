import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from patsy import dmatrices
from scipy import stats
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
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





num = [f for f in data.columns if data.dtypes[f] != 'object']
nd = pd.melt(data, value_vars=num)
n1 = sns.FacetGrid(nd, col='variable', col_wrap=4, sharex=False, sharey=False)
n1 = n1.map(sns.boxplot, 'value')
n1



z = np.abs(stats.zscore(data))
print(z)

data_outlier = data[(z < 3).all(axis=1)]
data_outlier.shape



Q1 = data.quantile(0.25)  
Q3 = data.quantile(0.75)  
IQR = Q3 - Q1  
print(IQR)

data_outlier_IQR = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]

data_outlier_IQR.shape





corr = data.corr()
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(corr, annot=True, ax=ax)



X = data_outlier_IQR.drop("MEDV", 1)  
y = data_outlier_IQR["MEDV"]




d1 = 'CRIM' + '+ZN' + '+INDUS' + '+CHAS' + '+NOX' + '+RM' + '+AGE' + '+DIS' + '+RAD' + '+TAX' + '+PTRATIO' + '+B'     + '+LSTAT '
y1, X1 = dmatrices('MEDV ~' + d1, data, return_type='dataframe')
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X1.values, i) for i in range(X1.shape[1])]
vif["Feature"] = X1.columns
vif.round(2)

d2 = data_outlier_IQR.drop('TAX', axis=1)  
d2

d1 = 'CRIM' + '+ZN' + '+INDUS' + '+CHAS' + '+NOX' + '+RM' + '+AGE' + '+DIS' + '+RAD' + '+PTRATIO' + '+B' + '+LSTAT'
y1, X1 = dmatrices('MEDV ~' + d1, d2, return_type='dataframe')
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X1.values, i) for i in range(X1.shape[1])]
vif["Feature"] = X1.columns
vif.round(2)

d2 = d2.drop('CRIM', axis=1)  
d2

d1 = 'ZN' + '+INDUS' + '+CHAS' + '+NOX' + '+RM' + '+AGE' + '+DIS' + '+RAD' + '+PTRATIO' + '+B' + '+LSTAT'
y1, X1 = dmatrices('MEDV ~' + d1, d2, return_type='dataframe')
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X1.values, i) for i in range(X1.shape[1])]
vif["Feature"] = X1.columns
vif.round(2)

d2 = d2.drop("MEDV", 1)
d2  




X_1 = sm.add_constant(d2)

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
    



rf = RandomForestRegressor()

print(rf.get_params())

param_distributions = {
    'max_depth': [2,4,6],
    'max_features': ['auto', 'sqrt', 'log2'],
    
    
    'n_estimators': [i for i in range(100, 401, 100)],
    
}

param_grid = {
    
    'max_depth': [2,4,6],
    
    
    
    'n_estimators': [i for i in range(100, 401, 100)],
    
}

print("Running Gridsearch CV")

model2 = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3)
model2.fit(X_train, y_train)
model2.best_params_
model2_score = model2.score(X_train, y_train)


print("coefficient of determination R^2 of the prediction.: ", model2_score)
y_predicted2 = model2.predict(X_test)


print("Mean Squared Error: %.2f" % mean_squared_error(y_test, y_predicted2))





print('Test Variance score: %.2f' % r2_score(y_test, y_predicted2))



fig, ax = plt.subplots()
ax.scatter(y_test, y_predicted2, edgecolors=(0, 0, 0))
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title("Ground Truth vs Predicted")


print("Execution time {} seconds: ".format(np.round(time.time() - start_time, 2)))