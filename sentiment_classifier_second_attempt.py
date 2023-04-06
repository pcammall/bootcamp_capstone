#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import train_test_split

import re
import string
import nltk
from nltk.stem import WordNetLemmatizer


# In[21]:


input_data = pd.read_csv(r'C:\Users\Patrick\Documents\GitHub\bootcamp_capstone\kaggle_dataset\sentiment_analysis_financial_news\all-data.csv'
                , encoding = "ISO-8859-1", header=None, names=['sentiment', 'text'])

input_data = pd.read_csv(r'C:\Users\Patrick\Documents\GitHub\bootcamp_capstone\kaggle_dataset\stock-market_sentiment\stock_data.csv',
                        encoding="ISO-8859-1", header=1, names=['text', 'sentiment'] )

input_data.head()


# In[ ]:





# In[22]:


#install nltk package
import sys
#get_ipython().system('{sys.executable} -m pip install nltk')


# In[25]:


#download the necessary data 
nltk.download("stopwords")


# In[26]:


#Data Cleaning
#first, remove stopwords
stopwords = nltk.corpus.stopwords.words('english')
lemmatizer = WordNetLemmatizer()


# In[27]:


#data cleaning preprocessing
pattern = r'[^a-zA-Z0-9\s\%]'
cleaned_buffer = []
for x in input_data['text']:
    temp = re.sub(pattern, " ", x)
    temp = temp.lower()
    temp = temp.split()
    temp = [lemmatizer.lemmatize(word) for word in temp if not word in set(stopwords)]
    temp = ' '.join(temp)
    cleaned_buffer.append(temp)


# In[28]:


cleaned_buffer


# In[29]:


input_data['cleaned'] = cleaned_buffer
input_data.head()


# In[30]:


#split into training and test data sets
xtrain, xtest, ytrain, ytest = train_test_split( input_data['cleaned'], input_data['sentiment'],
                                                               test_size=.4, random_state=10)
xtrain


# In[31]:


tfidf = TfidfVectorizer(ngram_range=(1,3))
xtrain_tf = tfidf.fit_transform(xtrain)
print("nsamples: %d, nfeatures: %d" % xtrain_tf.shape)

xtest_tf = tfidf.transform(xtest)
print("nsamples: %d, nfeatures: %d" % xtest_tf.shape)


# In[32]:


print(xtest_tf)


# In[33]:


#bayes classification
nb_classify = MultinomialNB()
nb_classify.fit(xtrain_tf, ytrain)


# In[34]:


predictions = nb_classify.predict(xtest_tf)


# In[35]:


results = metrics.classification_report(ytest, predictions)
print(results) #naive bays


# In[36]:


print(metrics.confusion_matrix(ytest, predictions))


# In[ ]:





# In[ ]:


#Now do with random forest


# In[ ]:





# In[41]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


# In[40]:


randforest = RandomForestClassifier()
scores = cross_val_score(randforest, xtrain_tf, ytrain.values.ravel(), cv=5)
print(scores)


# In[43]:


params = { 'n_estimators' : [5, 10, 25, 50, 100], 'max_depth' : [2, 5, 10, 20, None]}

grid_search = GridSearchCV (randforest, params)
grid_search.fit(xtrain_tf, ytrain.values.ravel())


# In[55]:


all_means = grid_search.cv_results_['mean_test_score']
all_std_dev = grid_search.cv_results_['std_test_score']
all_params = grid_search.cv_results_['params']
for x in range(0, len(all_means)):
    print(all_params[x], "\t", all_means[x], "\t", all_std_dev[x])


# In[54]:


grid_search.best_estimator_


# In[56]:


forest1 = RandomForestClassifier(n_estimators=50, max_depth=5)
forest2 = RandomForestClassifier(n_estimators=25, max_depth=20)
forest3 = RandomForestClassifier(n_estimators=100, max_depth=None)

forest1.fit(xtrain_tf, ytrain.values.ravel())
forest2.fit(xtrain_tf, ytrain.values.ravel())
forest3.fit(xtrain_tf, ytrain.values.ravel())


# In[59]:


for x in [forest1, forest2, forest3]:
    predictions = x.predict(xtest_tf)
    results = metrics.classification_report(ytest, predictions)
    print(results)
    print("+++++++++++++++++++++++++++++++++++")


# In[ ]:


#turn into pickle file to load to the interwebs
#https://towardsdatascience.com/3-ways-to-deploy-machine-learning-models-in-production-cdba15b00e

