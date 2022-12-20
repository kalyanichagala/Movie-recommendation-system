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


df.fillna(0,inplace=True)


# In[6]:


df


# In[22]:


X_train=df.iloc[:,:-1]


# In[23]:


y_train=df.iloc[:,-1]


# In[24]:


X_train.head()


# In[25]:


from sklearn.ensemble import RandomForestRegressor


# In[26]:


from sklearn.linear_model import LinearRegression


# In[27]:


reg = LinearRegression().fit(X_train, y_train)


# In[28]:


#rf=RandomForestRegressor()


# In[29]:


#rf.fit(X_train,y_train)


# In[30]:


X_test=X_train


# In[31]:


y_test=reg.predict(X_test)


# In[32]:


X_val=X_train


# In[33]:


y_val=y_train


# In[ ]:





# In[ ]:





# In[ ]:




