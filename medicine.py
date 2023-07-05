#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 


# In[2]:


medicines = pd.read_csv('medicine.csv')


# In[3]:


medicines.head()


# In[4]:


medicines.shape


# In[5]:


medicines.isnull().sum()


# In[6]:


medicines.duplicated().sum()


# In[7]:


medicines['Description']


# In[8]:


medicines['Reason'] = medicines['Reason'].apply(lambda x:x.split())
medicines['Description'] = medicines['Description'].apply(lambda x:x.split())
medicines['Description']


# In[9]:


medicines['Description'] = medicines['Description'].apply(lambda x:[i.replace(" ","") for i in x])
medicines['Description']


# In[10]:


medicines['tags'] = medicines['Description'] + medicines['Reason']
medicines['tags']


# In[11]:


final = medicines[['index','Drug_Name','tags']]


# In[12]:


final


# In[13]:


final['tags'].apply(lambda x:" ".join(x))


# In[14]:


final['tags'] = final['tags'].apply(lambda x:" ".join(x))


# In[15]:


final


# In[16]:


final['tags'] = final['tags'].apply(lambda x:x.lower())


# In[17]:


final


# In[18]:


get_ipython().system('pip install nltk')


# In[19]:


import nltk


# In[21]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[22]:


pip install FuzzyTM


# In[23]:


get_ipython().system('pip install -U scikit-learn scipy matplotlib')


# In[24]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(stop_words='english',max_features=5000)


# In[25]:


def stem(text):
  y = []

  for i in text.split():
    y.append(ps.stem(i))

  return " ".join(y)


# In[26]:


final['tags'] = final['tags'].apply(stem)


# In[27]:


final['tags']


# In[28]:


cv.fit_transform(final['tags']).toarray().shape


# In[29]:


vectors = cv.fit_transform(final['tags']).toarray()


# In[30]:


from sklearn.metrics.pairwise import cosine_similarity


# In[31]:


similarity = cosine_similarity(vectors)


# In[32]:


similarity[1]


# In[33]:


def recommend(medicine):
    medicine_index = final[final['Drug_Name'] == medicine].index[0]
    distances = similarity[medicine_index]
    medicines_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]

    for i in medicines_list:
        print(final.iloc[i[0]].Drug_Name)


# In[34]:


recommend("Paracetamol 125mg Syrup 60mlParacetamol 500mg Tablet 10'S")

