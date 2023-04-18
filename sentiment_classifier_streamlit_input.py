#streamlit version of classifier

#do all the imports
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import train_test_split
#used for second classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

import re
import string
import nltk
from nltk.stem import WordNetLemmatizer

#setup streamlit, which makes it pretty for display
import streamlit as st
st.title("Sentiment Classifier trained on financial headlines")




#now setup nltk
nltk.download("stopwords")
stopwords = nltk.corpus.stopwords.words('english')
lemmatizer = WordNetLemmatizer()


#load training data
@st.cache_data
def load_training_data():
    data_load_text = st.text("Loading trianing data")
    input_data = pd.read_csv(r'C:\Users\Patrick\Documents\GitHub\bootcamp_capstone\kaggle_dataset\stock-market_sentiment\stock_data.csv',
                            encoding="ISO-8859-1", header=1, names=['text', 'sentiment'] )
    #input_data.head() #print out to see that it laoded correctly
    data_load_text = st.text("Cleaning training data")
    #data cleaning
    #preprocessing
    pattern = r'[^a-zA-Z0-9\s\%]'
    cleaned_buffer = []
    for x in input_data['text']:
        temp = re.sub(pattern, " ", x)
        temp = temp.lower()
        temp = temp.split()
        temp = [lemmatizer.lemmatize(word) for word in temp if not word in set(stopwords)]
        temp = ' '.join(temp)
        cleaned_buffer.append(temp)
        
    #cleaned_buffer #print to verify cleaning has changed data
    data_load_text = st.text("Done")
    input_data['cleaned'] = cleaned_buffer

    
    return input_data

#split into train/test data sets
input_data = load_training_data()
if st.checkbox("Show Cleaned data"):
    st.subheader("Cleaned training data")
    st.dataframe(input_data)
xtrain, xtest, ytrain, ytest = train_test_split(input_data['cleaned'], input_data['sentiment'],
                                test_size=.4, random_state=10)
      

st.write(xtrain)

#use TFIDF vectorizer to convert words into numbers. That is, get count of words as they appear
#near other words and such

tfidf = TfidfVectorizer(ngram_range=(1,3))
#use the slider at the user level below?
#x = st.slider(min_value=1, max_value=3, label='Max NGram Value')
#tfidf = TfidfVectorizer(ngram_range=(1, x))
xtrain_tf = tfidf.fit_transform(xtrain)
xtest_tf = tfidf.transform(xtest)
st.subheader("Data after applying TFIDF")
#st.write(xtrain_tf)

#print("nsamples: %d, nfeatures: %d" % xtrain_tf.shape) #check results of vectorizer


#if i need to include MultiNB then it would go here.
#Except my version was crap and didn't predict anything well. Something tuning?




#random forest method
st.header("Building Random Forest Classifier")
rand_forest = RandomForestClassifier()
#scores = cross_val_score(rand_forest, xtrain_tf, ytrain.values.ravel(), cv=5)
#print(scores)


#use various paramenters with Random Forest to find which provides best results
#figure out how to do this with a slider
# params = {'n_estimators' : [5, 10, 25, 50, 100], 'max_depth': [2, 5, 10, 20, None]}

# grid_search = GridSearchCV(rand_forest, params)
# grid_search.fit(xtrain_tf, ytrain.values.ravel())

#print results
# all_means = grid_search.cv_results_['mean_test_score']
# all_stdev = grid_search.cv_results_['std_test_score']
# all_params = grid_search.cv_results_['params']



# grid_results = pd.DataFrame(columns=['Depth', 'Estimators', 'Mean', 'StdDev'])
# for x in range(0, len(all_means)):
    # grid_results = grid_results.append({'Depth':all_params[x]['max_depth'], "Estimators":all_params[x]['n_estimators'], 'Mean':all_means[x], 'StdDev':all_stdev[x]}, ignore_index=True)
    #print(all_params[x], "\t", all_means[x], "\t", all_stdev[x])
    
#st.dataframe(grid_results)
    
#grid_search.best_estimator_
#st.write("Best estimator was ", grid_search.best_estimator_)

#forest1 = RandomForestClassifier(n_estimators=50, max_depth=5)
#forest2 = RandomForestClassifier(n_estimators=25, max_depth=20)
#forest3 = RandomForestClassifier(n_estimators=100, max_depth=None)


#forest1.fit(xtrain_tf, ytrain.values.ravel())
#forest2.fit(xtrain_tf, ytrain.values.ravel())
#forest3.fit(xtrain_tf, ytrain.values.ravel())

#so instead of comparing 3 trees with different values, we're going to take the best estimator (50), with no max depth to use as the official one.
#this one will be feed user input
forest4 = RandomForestClassifier(n_estimators=50, max_depth=None)
forest4.fit(xtrain_tf, ytrain)
forest4.predict(xtest_tf)
st.write("scores:", forest4.score(xtest_tf, ytest))



user_input = st.text_input("Enter a finance headline to classify", "This is a happy sample string")
#clean the user input

pattern = r'[^a-zA-Z0-9\s\%]'
cleaned_buffer = []
temp = re.sub(pattern, " ", user_input)
temp = temp.lower()
temp = temp.split()
temp = [lemmatizer.lemmatize(word) for word in temp if not word in set(stopwords)]
temp = ' '.join(temp)
cleaned_buffer.append(temp)
    

    
cleaned_buffer
# tfidf = TfidfVectorizer(ngram_range=(1,3))
# user_input_tf = tfidf.fit_transform(cleaned_buffer)    
user_input_tf = tfidf.transform(cleaned_buffer) #<<<< Use transform
# user_input_tf.shape
print(user_input_tf)   

result = forest4.predict(user_input_tf)

if result >= 0:
    st.write("Negative")
else:
    st.write("Positive")
    
st.write("end file")

#run training seperately. Then load to streamlit and do the prediction input in streamlit. Only the prediction input.
#save both classifer and vectorizer in a seperate thing that gets loaded. The training is done offline, and then you load them to do the prediction