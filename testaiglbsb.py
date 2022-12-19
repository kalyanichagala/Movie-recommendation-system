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


# In[41]:


# from sklearn.model_selection import RandomizedSearchCV
# # Number of trees in random forest
# n_estimators = [int(x) for x in np.linspace(start = 200, stop = 300, num = 10)]
# # Number of features to consider at every split
# # Maximum number of levels in tree
# max_depth = [int(x) for x in np.linspace(10, 55, num = 11)]
# max_depth.append(None)
# # Minimum number of samples required to split a node
# min_samples_split = [2, 5, 10]
# # Minimum number of samples required at each leaf node
# min_samples_leaf = [1, 2, 4]
# # Method of selecting samples for training each tree
# bootstrap = [True, False]
# # Create the random grid
# random_grid = {'n_estimators': n_estimators,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap}


# In[42]:


from sklearn.ensemble import RandomForestClassifier


# In[44]:


# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
#rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 10,cv=2, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf.fit(X_train, y_train)


# In[45]:


#from sklearn.ensemble import RandomForestClassifier


# In[46]:


#clf = RandomForestClassifier(max_depth=2, random_state=0)


# In[47]:


#y_train=df[-1]


# In[48]:


df


# In[49]:


X_train.shape


# In[50]:


#clf.fit(X_train,y_train)


# In[51]:


X_test=X_train


# In[52]:


y_test=rf_random.predict(X_test)


# In[53]:


y_test


# In[54]:


X_val=X_train


# In[55]:


y_val=y_train


# In[56]:


X_val.head()


# In[57]:


# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()


# In[58]:


# from sklearn.preprocessing import MaxAbsScaler
# ma = MaxAbsScaler()


# In[59]:


# from sklearn.preprocessing import RobustScaler

# rs = RobustScaler()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




