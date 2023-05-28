#!/usr/bin/env python
# coding: utf-8

# # Email Spam Detection 

# ## Importing Libraries

# In[71]:


get_ipython().run_line_magic('pylab', '')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.figure import Figure
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


# In[72]:


spam_df=pd.read_csv(r"C:\Users\sp924\Downloads\spam.csv",encoding="ISO-8859-1")
spam_df.head()


# In[73]:


spam_df.sample(5)


# In[74]:


spam_df.tail()


# ### EDA

# In[75]:


spam_df.shape


# In[76]:


spam_df.columns


# In[77]:


spam_df.info()


# In[78]:


##Checking NaN values
spam_df.isnull().sum()


# In[79]:


##Checking duplicate values
spam_df[spam_df.duplicated()]


# In[80]:


## Dropping Columnns
spam_df.drop(labels=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1,inplace=True)


# In[81]:


spam_df.sample(5)


# In[82]:


## Renameing columns
spam_df.rename(columns={'v1': 'Target','v2' : 'SMS'},inplace=True)


# In[83]:


spam_df.columns


# In[84]:


spam_df.describe()


# In[85]:


spam_df.isnull().sum()


# In[86]:


spam_df['Target'].value_counts()


# In[87]:


## Checking duplicates
spam_df.duplicated().sum()


# In[88]:


spam_df[spam_df.duplicated()]


# In[89]:


## Dropping duplicate values
spam_df.drop_duplicates(keep='first',inplace=True)


# In[90]:


spam_df.shape


# In[91]:


##Encoding Target columns value
le=LabelEncoder()
spam_df['Target']=le.fit_transform(spam_df['Target'])
spam_df['Target']


# ### Model Training

# In[92]:


## Segregating Independent and dependent columns
X = spam_df['SMS']
Y =spam_df['Target']


# In[93]:


X_train, X_test, Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state=42)


# In[94]:


##feature Extraction
cv = CountVectorizer()


# In[95]:


X_train_cv = cv.fit_transform(X_train)
X_test_cv = cv.transform(X_test)


# ### Logistic Regression

# In[96]:


LR = LogisticRegression()


# In[97]:


LR.fit(X_train_cv, Y_train)


# In[98]:


Y_pred=LR.predict(X_train_cv)


# In[99]:


print('Train',accuracy_score(Y_train,Y_pred)*100)


# In[100]:


Y_pred_test=LR.predict(X_test_cv)


# In[101]:


print('Test',accuracy_score(Y_test,Y_pred_test)*100)

