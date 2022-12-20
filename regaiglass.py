#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


df=pd.read_csv("HousingData.csv")


# In[4]:


df.head()


# In[12]:


df.dropna(inplace=True)


# In[13]:


X_train=df[df.columns[0:-1]]


# In[14]:


y_train=df[df.columns[-1]]


# In[15]:


from sklearn.ensemble import RandomForestRegressor


# In[16]:


rf=RandomForestRegressor()


# In[17]:


rf.fit(X_train,y_train)


# In[18]:


X_test=X_train


# In[20]:


y_test=rf.predict(X_test)


# In[21]:


X_val=X_train


# In[22]:


y_val=y_train


# In[ ]:




