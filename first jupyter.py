#!/usr/bin/env python
# coding: utf-8

# In[2]:


a = 3
b = 4
print(a+b)


# In[3]:


get_ipython().system('pip install numpy')


# In[4]:


get_ipython().system('pip install pandas')


# In[5]:


import numpy as np


# In[22]:


import pandas as pd


# data exploratio

# In[6]:


get_ipython().system('pip install matplotlib')
get_ipython().system('pip install scikit-learn')


# In[7]:


import sklearn


# In[8]:


from sklearn.datasets import load_iris


# In[9]:


import matplotlib.pyplot as plt


# In[15]:


iris = load_iris()


# In[16]:


iris['data']


# In[17]:


features = iris['data']


# In[18]:


label = iris['target']


# In[34]:


feature_names = iris['feature_names']


# In[36]:


df = pd.DataFrame(features, columns = feature_names)


# In[37]:


df.head(3)


# In[38]:


df.tail(3)


# In[40]:


df['target'] = label 


# In[41]:


df.head(4)


# In[42]:


df.describe()


# In[43]:


df0 = df[df['target'] == 0]
df1 = df[df['target'] == 1]
df2 = df[df['target'] == 2]
x = np.arange(len(df0))


# In[48]:


df0['sepal length (cm)']


# # Setosa:

# # Sepal length graphs:

# In[68]:


plt.figure(figsize=(6,4))
plt.hist(df0['sepal length (cm)'])
plt.title('sepal length (cm) of Iris Setosa')
plt.show()

plt.figure(figsize=(6,4))
plt.plot(df0['sepal length (cm)'])
plt.title('sepal length (cm) of Iris Setosa')
plt.show()

plt.figure(figsize=(6,4))
plt.bar(x, df0['sepal length (cm)'])
plt.title('sepal length (cm) of Iris Setosa')
plt.show

plt.figure(figsize=(6,4))
fig, ax = plt.subplots()
ax.boxplot(df0['sepal length (cm)'])
plt.title('boxplot of sepal length (cm) of Iris Setosa')
plt.show()


# # sepal width graphs:

# In[64]:


plt.figure(figsize=(6,4))
plt.hist(df0['sepal width (cm)'])
plt.title('sepal width (cm) of Iris Setosa')
plt.show()

plt.figure(figsize=(6,4))
plt.plot(df0['sepal width (cm)'])
plt.title('sepal width (cm) of Iris Setosa')
plt.show()

plt.figure(figsize=(6,4))
plt.bar(x, df0['sepal width (cm)'])
plt.title('sepal width (cm) of Iris Setosa')
plt.show

plt.figure(figsize=(6,4))
fig, ax = plt.subplots()
ax.boxplot(df0['sepal width (cm)'])
plt.title('boxplot of sepal width (cm) of Iris Setosa')
plt.show()


# # Petal length graphs:

# In[63]:


plt.figure(figsize=(6,4))
plt.hist(df0['petal length (cm)'])
plt.title('petal length (cm) Iris Setosa')
plt.show()

plt.figure(figsize=(6,4))
plt.plot(df0['petal length (cm)'])
plt.title('petal length (cm) of Iris Setosa')
plt.show()

plt.figure(figsize=(6,4))
plt.bar(x, df0['petal length (cm)'])
plt.title('petal length (cm) of Iris Setosa')
plt.show

plt.figure(figsize=(6,4))
fig, ax = plt.subplots()
ax.boxplot(df0['petal length (cm)'])
plt.title('boxplot of petal length (cm) of Iris Setosa')
plt.show()


# # Petal width graphs:

# In[66]:


plt.figure(figsize=(6,4))
plt.hist(df0['petal width (cm)'])
plt.title('petal width (cm) of Iris Setosa')
plt.show()


plt.figure(figsize=(6,4))
plt.plot(df0['petal width (cm)'])
plt.title('petal width (cm) of Iris Setosa')
plt.show()


plt.figure(figsize=(6,4))
plt.bar(x, df0['petal width (cm)'])
plt.title('petal width (cm) of Iris Setosa')
plt.show


plt.figure(figsize=(6,4))
fig, ax = plt.subplots()
ax.boxplot(df0['petal width (cm)'])
plt.title('boxplot of petal width (cm) of Iris Setosa')
plt.show()


# # Versicolor:

# In[75]:


x = np.arange(len(df1))


# # sepal length graphs:

# In[76]:


plt.figure(figsize=(6,4))
plt.hist(df1['sepal length (cm)'])
plt.title('sepal length (cm) of Iris Versicolor')
plt.show()


plt.figure(figsize=(6,4))
plt.plot(df1['sepal length (cm)'])
plt.title('sepal length (cm) of Iris Versicolor')
plt.show()


plt.figure(figsize=(6,4))
plt.bar(x, df1['sepal length (cm)'])
plt.title('sepal length (cm) of Iris Versicolor')
plt.show


plt.figure(figsize=(6,4))
fig, ax = plt.subplots()
ax.boxplot(df1['sepal length (cm)'])
plt.title('boxplot of sepal length (cm) of Iris Versicolor')
plt.show()


# # Sepal width graphs:

# In[77]:


plt.figure(figsize=(6,4))
plt.hist(df1['sepal width (cm)'])
plt.title('sepal width (cm) of Iris Versicolor')
plt.show()


plt.figure(figsize=(6,4))
plt.plot(df1['sepal width (cm)'])
plt.title('sepal width (cm) of Iris Versicolor')
plt.show()


plt.figure(figsize=(6,4))
plt.bar(x, df1['sepal width (cm)'])
plt.title('sepal width (cm) of Iris Versicolor')
plt.show


plt.figure(figsize=(6,4))
fig, ax = plt.subplots()
ax.boxplot(df1['sepal width (cm)'])
plt.title('boxplot of sepal width (cm) of Iris Versicolor')
plt.show()


# # Petal length graphs:

# In[78]:


plt.figure(figsize=(6,4))
plt.hist(df1['petal length (cm)'])
plt.title('petal length (cm) of Iris Versicolor')
plt.show()


plt.figure(figsize=(6,4))
plt.plot(df1['petal length (cm)'])
plt.title('petal length (cm) of Iris Versicolor')
plt.show()


plt.figure(figsize=(6,4))
plt.bar(x, df1['petal length (cm)'])
plt.title('petal length (cm) of Iris Versicolor')
plt.show


plt.figure(figsize=(6,4))
fig, ax = plt.subplots()
ax.boxplot(df1['petal length (cm)'])
plt.title('boxplot of petal length (cm) of Iris Versicolor')
plt.show()


# # Petal width graphs:

# In[79]:


plt.figure(figsize=(6,4))
plt.hist(df1['petal width (cm)'])
plt.title('petal width (cm) of Iris Versicolor')
plt.show()


plt.figure(figsize=(6,4))
plt.plot(df1['petal width (cm)'])
plt.title('petal width (cm) of Iris Versicolor')
plt.show()


plt.figure(figsize=(6,4))
plt.bar(x, df1['petal width (cm)'])
plt.title('petal width (cm) of Iris Versicolor')
plt.show


plt.figure(figsize=(6,4))
fig, ax = plt.subplots()
ax.boxplot(df1['petal width (cm)'])
plt.title('boxplot of petal width (cm) of Iris Versicolor')
plt.show()


# # Verginica:

# # sepal length graphs:

# In[84]:


x = np.arange(len(df2))


# In[83]:


plt.figure(figsize=(6,4))
plt.hist(df2['sepal length (cm)'])
plt.title('sepal length (cm) of Iris Versicolor')
plt.show()


plt.figure(figsize=(6,4))
plt.plot(df2['sepal length (cm)'])
plt.title('sepal length (cm) of Iris Versicolor')
plt.show()


plt.figure(figsize=(6,4))
plt.bar(x, df2['sepal length (cm)'])
plt.title('sepal length (cm) of Iris Versicolor')
plt.show


plt.figure(figsize=(6,4))
fig, ax = plt.subplots()
ax.boxplot(df2['sepal length (cm)'])
plt.title('boxplot of sepal length (cm) of Iris Versicolor')
plt.show()


# # Sepal width graphs:

# In[89]:


plt.figure(figsize=(6,4))
plt.hist(df2['sepal width (cm)'])
plt.title('sepal width (cm) of Iris Verginica')
plt.show()


plt.figure(figsize=(6,4))
plt.plot(df2['sepal width (cm)'])
plt.title('sepal width (cm) of Iris Verginica')
plt.show()


plt.figure(figsize=(6,4))
plt.bar(x, df2['sepal width (cm)'])
plt.title('sepal width (cm) of Iris Verginica')
plt.show


plt.figure(figsize=(6,4))
fig, ax = plt.subplots()
ax.boxplot(df2['sepal width (cm)'])
plt.title('boxplot of sepal width (cm) of Iris Verginica')
plt.show()


# # Petal length graphs:

# In[91]:


plt.hist(df2['petal length (cm)'])
plt.title('petal length (cm) of Iris Verginica')
plt.show()


plt.figure(figsize=(6,4))
plt.plot(df2['petal length (cm)'])
plt.title('petal length (cm) of Iris Verginica')
plt.show()


plt.figure(figsize=(6,4))
plt.bar(x, df2['petal length (cm)'])
plt.title('petal length (cm) of Iris Verginica')
plt.show


plt.figure(figsize=(6,4))
fig, ax = plt.subplots()
ax.boxplot(df2['petal length (cm)'])
plt.title('boxplot of petal length (cm) of Iris Verginica')
plt.show()


# # Petal width graphs:

# In[92]:


plt.figure(figsize=(6,4))
plt.hist(df2['petal width (cm)'])
plt.title('petal width (cm) of Iris Verginica')
plt.show()


plt.figure(figsize=(6,4))
plt.plot(df2['petal width (cm)'])
plt.title('petal width (cm) of Iris Verginica')
plt.show()


plt.figure(figsize=(6,4))
plt.bar(x, df2['petal width (cm)'])
plt.title('petal width (cm) of Iris Verginica')
plt.show


plt.figure(figsize=(6,4))
fig, ax = plt.subplots()
ax.boxplot(df2['petal width (cm)'])
plt.title('boxplot of petal width (cm) of Iris Verginica')
plt.show()


# In[ ]:




