
#pip install streamlit
#pip install spacy
#python -m spacy download ru_core_news_sm
#pip install pymorphy2

import streamlit as st
import pickle
import numpy as np
import json


import spacy
import re

def tokenize(joke:str): # ret list of tokens
    
    # Load the Russian SpaCy model
    nlp = spacy.load("ru_core_news_sm/ru_core_news_sm-3.8.0")
    valid_characters_pattern = re.compile(r'[^\u0400-\u04FFa-zA-Z]+')

    def lemmatize_and_tokenize(text):
        """Tokenizes, lemmatizes, and cleans tokens to contain only alphabetic characters."""
        doc = nlp(text)  # Process text with SpaCy
        tokens = []
        for token in doc:
            if token.text and not token.is_stop:
                tokens.append(token.lemma_)
        return tokens

    def clear_(text):
        temp = re.sub(r'[^\u0400-\u04FFa-zA-Z\s]+', '', text.lower())
        return re.sub(r'\s+', ' ', temp)

    jokeC = clear_(joke)
    return lemmatize_and_tokenize(jokeC)

def load_from_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

d = load_from_json("vocdic.json")
dc = load_from_json("catdic.json")

def OneHot(tokens):

    vector = np.zeros(len(d)+1)  # Initialize vector of 0's + len
    st.write(len(d)) #debug

    for token in tokens:
        if token in d:  # Only encode tokens in the vocabulary
            vector[d[token]] = 1
            vector[-1] += 1
    return vector.reshape((1,-1))


# Load the trained model

try:
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    print("Error loading model:", e)

# Streamlit app title
st.title("Machine Learning Model Deployment")

# Input fields for user input
st.write("Enter a joke in Russian:")
joke = st.text_input("joke")



# Prediction button
if st.button("Predict"):
    joket = tokenize(joke)
    X = OneHot(joket)
    st.write(X.shape)

    y_pred = np.argmax(model.predict(X), axis=1)
    st.write(f"Prediction: {dc[str(y_pred[0])]}")

