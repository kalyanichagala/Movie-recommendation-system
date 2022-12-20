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


# In[5]:


df.dropna(inplace=True)


# In[6]:


df


# In[7]:


X_train=df.drop("LSTAT",axis=1)


# In[8]:


y_train=df["LSTAT"]


# In[9]:


from sklearn.ensemble import RandomForestRegressor


# In[10]:


from sklearn.linear_model import LinearRegression


# In[11]:


reg = LinearRegression().fit(X_train, y_train)


# In[ ]:





# In[12]:


#rf=RandomForestRegressor()


# In[13]:


#rf.fit(X_train,y_train)


# In[14]:


X_test=X_train


# In[15]:


y_test=reg.predict(X_test)


# In[16]:


X_val=X_train


# In[17]:


y_val=y_train


# In[ ]:




