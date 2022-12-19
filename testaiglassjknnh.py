#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


df=pd.read_csv("Classification_Model_data.csv")


# In[4]:


df.head()


# In[5]:


df["price_range"].value_counts()


# In[6]:


#df1=pd.get_dummies(df, columns=['clock_speed', 'mobile_wt'])


# In[7]:


#df1.head()

#from sklearn.preprocessing import OneHotEncoder#enc=OneHotEncoder()
#enc_data=pd.DataFrame(enc.fit_transform(df[['battery_power']]).toarray())
#df=df.join(enc_data)
# In[8]:


df.head()


# In[9]:


X_train=df[df.columns[0:-1]]


# In[10]:


X_train.shape


# In[11]:


y_train=df[df.columns[-1]]


# In[12]:


y_train


# In[13]:


#df=df1


# In[14]:


# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()


# In[15]:


# sc.fit(X_train)
# df = sc.transform(X_train)


# In[38]:


from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 300, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 55, num = 11)]
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


# In[39]:


from sklearn.ensemble import RandomForestClassifier


# In[40]:


# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 10,cv=2, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train, y_train)


# In[59]:


#from sklearn.ensemble import RandomForestClassifier


# In[60]:


#clf = RandomForestClassifier(max_depth=2, random_state=0)


# In[29]:


#y_train=df[-1]


# In[19]:


df


# In[20]:


X_train.shape


# In[21]:


#clf.fit(X_train,y_train)


# In[22]:


X_test=X_train


# In[23]:


y_test=rf_random.predict(X_test)


# In[24]:


y_test


# In[25]:


X_val=X_train


# In[26]:


y_val=y_train


# In[27]:


X_val.head()


# In[28]:


# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()


# In[29]:


# from sklearn.preprocessing import MaxAbsScaler
# ma = MaxAbsScaler()


# In[30]:


# from sklearn.preprocessing import RobustScaler

# rs = RobustScaler()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




