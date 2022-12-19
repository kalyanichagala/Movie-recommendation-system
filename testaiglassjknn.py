#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd


# In[21]:


import numpy as np


# In[22]:


df=pd.read_csv("Classification_Model_data.csv")


# In[23]:


df.head()


# In[24]:


df["price_range"].value_counts()


# In[25]:


#df1=pd.get_dummies(df, columns=['clock_speed', 'mobile_wt'])


# In[26]:


#df1.head()

#from sklearn.preprocessing import OneHotEncoder#enc=OneHotEncoder()
#enc_data=pd.DataFrame(enc.fit_transform(df[['battery_power']]).toarray())
#df=df.join(enc_data)
# In[27]:


df.head()


# In[28]:


X_train=df[df.columns[0:-1]]


# In[29]:


X_train.shape


# In[30]:


y_train=df[df.columns[-1]]


# In[31]:


y_train


# In[32]:


#df=df1


# In[33]:


# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()


# In[34]:


# sc.fit(X_train)
# df = sc.transform(X_train)


# In[35]:


from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


# In[36]:


from sklearn.ensemble import RandomForestClassifier


# In[37]:


# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train, y_train)


# In[59]:


#from sklearn.ensemble import RandomForestClassifier


# In[60]:


#clf = RandomForestClassifier(max_depth=2, random_state=0)


# In[29]:


#y_train=df[-1]


# In[38]:


df


# In[39]:


X_train.shape


# In[40]:


#clf.fit(X_train,y_train)


# In[41]:


X_test=X_train


# In[42]:


y_test=rf_random.predict(X_test)


# In[43]:


y_test


# In[44]:


X_val=X_train


# In[45]:


y_val=y_train


# In[46]:


X_val.head()


# In[47]:


# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()


# In[48]:


# from sklearn.preprocessing import MaxAbsScaler
# ma = MaxAbsScaler()


# In[49]:


# from sklearn.preprocessing import RobustScaler

# rs = RobustScaler()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




