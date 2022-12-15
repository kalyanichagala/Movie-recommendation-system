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


#df1.head()


# In[30]:


from sklearn.preprocessing import OneHotEncoder


# In[32]:


enc=OneHotEncoder()
enc_data=pd.DataFrame(enc.fit_transform(df[['battery_power']]).toarray())
df=df.join(enc_data)


# In[33]:


df.head()


# In[17]:


#df=df1


# In[ ]:





# In[34]:


from sklearn.ensemble import RandomForestClassifier


# In[35]:


clf = RandomForestClassifier(max_depth=2, random_state=0)


# In[36]:


X_train=df.drop("price_range",axis=1)


# In[37]:


y_train=df["price_range"]


# In[38]:


clf.fit(X_train,y_train)


# In[ ]:





# In[39]:


X_test=X_train


# In[40]:


y_test=clf.predict(X_test)


# In[41]:


y_test


# In[42]:


X_val=X_train


# In[43]:


y_val=y_train


# In[44]:


X_val.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




