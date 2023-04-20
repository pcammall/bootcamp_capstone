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

import pickle


import re
import string
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')



#setup streamlit, which makes it pretty for display
import streamlit as st
st.title("Sentiment Classifier trained on financial headlines")




@st.cache_data
def clean_text(user_input):

    stopwords = nltk.corpus.stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    
    pattern = r'[^a-zA-Z0-9\s\%]'
    cleaned_buffer = []
    temp = re.sub(pattern, " ", user_input)
    temp = temp.lower()
    temp = temp.split()
    temp = [lemmatizer.lemmatize(word) for word in temp if not word in set(stopwords)]
    temp = ' '.join(temp)
    cleaned_buffer.append(temp)
    return cleaned_buffer

@st.cache_data
def make_tfidf(input):
    filename = "my_saved_tfidf"
    tfidf = pickle.load(open(filename, 'rb'))
    return tfidf.transform(input)
    
    
    
def main(user_input):
    filename = "my_saved_model"
    loaded_model = pickle.load(open(filename, "rb"))
    
    filename = "my_saved_tfidf"
    loaded_tfidf = pickle.load(open(filename, "rb"))

    
    #user_input = st.text_input("Enter a finance headline to classify", "This is a happy sample string")
    #clean the user input

    cleaned_input = clean_text(user_input)
    
    tfidf_input = loaded_tfidf.transform(cleaned_input)
    prediction = loaded_model.predict(tfidf_input)
    #st.write("input: " , cleaned_input)
    st.write("Prediction:", prediction)
    if prediction[0] > 0:
        st.write("Positive")
    else:
        st.write("Negative Sentiment")
    
    #st.write(score) write the score after the sentiment is done
    #only need the input box, and the score. and then done.
    
input = st.text_input("Enter Finance Headline to classify", "sample string")    
main(input)