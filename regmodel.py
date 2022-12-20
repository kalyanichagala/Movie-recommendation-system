#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[10]:


from sklearn.model_selection import train_test_split


# In[14]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


# In[3]:


df=pd.read_csv("HousingData.csv")


# In[4]:


df.head()


# In[12]:


X=df.iloc[:,:-1]
y=df.iloc[:,-1]


# In[8]:


df.fillna(0,inplace=True)


# In[13]:


X_1, X_val, y_1, y_val = train_test_split(X, y, test_size=0.1, random_state=2)
X_train, X_test, y_train, y_test = train_test_split(X_1, y_1, test_size=0.1, random_state=2)


# In[15]:


rf=RandomForestRegressor()
rf.fit(X_train,y_train)


# In[ ]:




