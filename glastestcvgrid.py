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


# In[16]:


from sklearn.model_selection import GridSearchCV


# In[17]:


from sklearn.ensemble import RandomForestClassifier


# In[18]:


rfc=RandomForestClassifier(random_state=42)


# In[22]:


param_grid = { 
    'max_depth' : [4,5,6,7,8],
}


# In[24]:


CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=2)
CV_rfc.fit(X_train, y_train)


# In[25]:


#from sklearn.ensemble import RandomForestClassifier


# In[26]:


#clf = RandomForeCV_rfcstClassifier(max_depth=2, random_state=0)


# In[27]:


#y_train=df[-1]


# In[28]:


df


# In[29]:


X_train.shape


# In[30]:


#clf.fit(X_train,y_train)


# In[31]:


X_test=X_train


# In[32]:


y_test=CV_rfc.predict(X_test)


# In[33]:


y_test


# In[34]:


X_val=X_train


# In[35]:


y_val=y_train


# In[36]:


X_val.head()


# In[37]:


# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()


# In[38]:


# from sklearn.preprocessing import MaxAbsScaler
# ma = MaxAbsScaler()


# In[39]:


# from sklearn.preprocessing import RobustScaler

# rs = RobustScaler()
from sklearn.model_selection import RandomizedSearchCV


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




