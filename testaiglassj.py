#!/usr/bin/env python
# coding: utf-8

# In[45]:


import pandas as pd


# In[46]:


df=pd.read_csv("Classification_Model_data.csv")


# In[47]:


df.head()


# In[48]:


df["price_range"].value_counts()


# In[5]:


#df1=pd.get_dummies(df, columns=['clock_speed', 'mobile_wt'])


# In[6]:


#df1.head()

#from sklearn.preprocessing import OneHotEncoder#enc=OneHotEncoder()
#enc_data=pd.DataFrame(enc.fit_transform(df[['battery_power']]).toarray())
#df=df.join(enc_data)
# In[49]:


df.head()


# In[53]:


X_train=df[df.columns[0:-1]]


# In[54]:


X_train.shape


# In[55]:


y_train=df[df.columns[-1]]


# In[56]:


y_train


# In[10]:


#df=df1


# In[57]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()


# In[58]:


sc.fit(X_train)
df = sc.transform(X_train)


# In[59]:


from sklearn.ensemble import RandomForestClassifier


# In[60]:


clf = RandomForestClassifier(max_depth=2, random_state=0)


# In[29]:


#y_train=df[-1]


# In[69]:


df


# In[76]:


X_train.shape


# In[61]:


clf.fit(X_train,y_train)


# In[ ]:





# In[62]:


X_test=X_train


# In[63]:


y_test=clf.predict(X_test)


# In[64]:


y_test


# In[65]:


X_val=X_train


# In[66]:


y_val=y_train


# In[67]:


X_val.head()


# In[73]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


# In[74]:


from sklearn.preprocessing import MaxAbsScaler
ma = MaxAbsScaler()


# In[75]:


#from sklearn.preprocessing import RobustScaler

#rs = RobustScaler()


# In[ ]:

from sklearn.decomposition import PCA
pca = PCA(n_components=2)



# In[ ]:




