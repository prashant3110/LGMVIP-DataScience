#!/usr/bin/env python
# coding: utf-8

# # Data Science intern - Lets Grow More
#    Author: PRASHANT
# 

# # Beginner Level Task
# 
# 

# # Task 1: Iris Flowers Classification ML Project 

# In[ ]:


# importing libraries
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


#loading dataset
df = pd.read_csv("C:\\Users\lenovo\Desktop\iris.csv")


# In[9]:


df.head()


# In[10]:


df.tail()


# In[14]:


df.describe()


# In[16]:


df.isnull().sum()


# In[18]:


df.shape


# # Visualizing the dataset

# In[19]:


#plot 1
df.plot(x='sepal_length',y='sepal_width', style='og')
plt.xlabel("sepal_length")
plt.ylabel("sepal_width")
plt.title("sepal length vs sepal width")
plt.grid()
plt.show()


# In[21]:


#plot 2
df.plot(x='petal_length',y='petal_width', style='*b')
plt.xlabel("petal_length")
plt.ylabel("petal_width")
plt.title("petal length vs petal width")
plt.grid()
plt.show()


# # Train Data

# In[22]:


X=df.drop('class',axis=1)
y=df['class']


# In[25]:


X.head()


# In[26]:


y.head()


# In[45]:


from sklearn.model_selection import train_test_split

X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2)


# In[28]:


print(X_train.shape , X_test.shape , y_train.shape , y_test.shape)


# # Training Model

# In[30]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(X_train,y_train)
print("Training complete")


# # Predicting Result

# In[32]:


y_pred=lr.predict(X_test)


# In[40]:


y_pred


# In[41]:


df2=pd.DataFrame({'Actual' : y_test,'Predicted' : y_pred})


# In[50]:


df2.head(15)


# In[51]:


from sklearn import metrics
print('Accuracy:',metrics.accuracy_score(y_pred,y_test))


# In[52]:


print(lr.score(X_test,y_test))


# In[ ]:




