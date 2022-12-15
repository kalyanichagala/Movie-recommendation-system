#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv("Classification_Model_data.csv")


# In[3]:


df.head()


# In[4]:


df["price_range"].value_counts()


# In[15]:


#df1=pd.get_dummies(df, columns=['clock_speed', 'mobile_wt'])


# In[16]:


df1.head()


# In[17]:


#df=df1
from sklearn.preprocessing import OneHotEncoder

enc=OneHotEncoder()
enc_data=pd.DataFrame(enc.fit_transform(df[['battery_power']]).toarray())
df=df.join(enc_data)


# In[18]:


from sklearn.ensemble import RandomForestClassifier


# In[19]:


clf = RandomForestClassifier(max_depth=2, random_state=0)


# In[20]:


X_train=df.drop("price_range",axis=1)


# In[21]:


y_train=df["price_range"]


# In[22]:


clf.fit(X_train,y_train)


# In[23]:


X_test=X_train


# In[24]:


y_test=clf.predict(X_test)


# In[25]:


y_test


# In[26]:


X_val=X_train


# In[27]:


y_val=y_train


# In[28]:


X_val.head()

from sklearn.preprocessing import OneHotEncoder

enc=OneHotEncoder()
enc_data=pd.DataFrame(enc.fit_transform(df[[' battery_power']]).toarray())
New_df=df.join(enc_data)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




