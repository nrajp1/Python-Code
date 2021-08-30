#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import statsmodels.api as sm
from sklearn.model_selection import train_test_split


# In[18]:


mission_hospital = pd.read_csv("/Users/neharajput/Downloads/MH.csv")


# In[19]:


mission_hospital.info()


# In[20]:


mission_hospital.iloc[0:5,0:10]


# In[21]:


X_features = mission_hospital.columns


# In[22]:


X_features.drop('SL.')


# In[23]:


X_features.drop('IMPLANT USED (Y/N)')


# In[24]:


categorical_features = ['GENDER','MARITAL STATUS','KEY COMPLAINTS -CODE','PAST MEDICAL HISTORY CODE','MODE OF ARRIVAL','STATE AT THE TIME OF ARRIVAL','TYPE OF ADMSN']


# In[25]:


mission_hospital= pd.get_dummies(mission_hospital[X_features],columns=categorical_features,drop_first=True)


# In[26]:


mission_hospital.columns


# In[27]:


X_features=mission_hospital.columns


# In[64]:


mission_hospital.head(10)


# In[37]:


X= sm.add_constant(mission_hospital)
Y= mission_hospital['TOTAL COST TO HOSPITAL ']
train_X,test_X,train_y,test_y=train_test_split(X,Y,train_size=0.8,random_state=42)


# In[45]:


mission_hospital.drop('SL.',inplace=True,axis=1)


# In[46]:


mission_hospital.drop('IMPLANT USED (Y/N)',inplace=True,axis=1)


# In[47]:


mission_hospital.dtypes


# In[54]:


mission_hospital.iloc[:,15:20]


# In[58]:


mission_hospital.iloc[:, 15:39]= mission_hospital.iloc[:,15:39].astype('category')


# In[59]:


mission_hospital.dtypes


# In[66]:


np.asarray(train_X).dtype


# In[63]:


mh = sm.OLS(train_y,train_X).fit()


# In[ ]:





# In[ ]:





# In[ ]:




