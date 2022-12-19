#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[8]:


df=pd.read_csv("Classification_Model_data.csv")


# In[9]:


df.head()


# In[10]:


df["price_range"].value_counts()


# In[6]:


from sklearn.ensemble import RandomForestClassifier


# In[7]:


clf = RandomForestClassifier(max_depth=2, random_state=0)


# In[11]:


X_train=df.drop("price_range",axis=1)


# In[12]:


y_train=df["price_range"]


# In[15]:


clf.fit(X_train,y_train)


# In[13]:


X_test=X_train


# In[16]:


y_test=clf.predict(X_test)


# In[17]:


y_test


# In[18]:


X_val=X_train


# In[19]:


y_val=y_train



# In[ ]:
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder()

k=pd.get_dummies(y_train)
from sklearn.preprocessing import OneHotEncoder

enc=OneHotEncoder()





