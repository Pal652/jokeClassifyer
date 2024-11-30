
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
    
    valid_characters_pattern = re.compile(r'[^\u0400-\u04FFa-zA-Z]+')

    def lemmatize_and_tokenize(text):
        """Tokenizes, lemmatizes, and cleans tokens to contain only alphabetic characters."""
        doc = nlp(text)  # Process text with SpaCy
        tokens = []
        print("lol01")
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

def OneHot(tokens):

    vector = np.zeros(len(d)+1)  # Initialize vector of 0's + len
    st.write(len(d)) #debug

    for token in tokens:
        if token in d:  # Only encode tokens in the vocabulary
            vector[d[token]] = 1
            vector[-1] += 1
    return vector.reshape((1,-1))

@st.cache_resource
def load_model():
    print("lol0")
    with open('model.pkl', 'rb') as file:
        return pickle.load(file)

@st.cache_resource
def load_vocabularies():
    d = load_from_json("vocdic.json")
    dc = load_from_json("catdic.json")
    return d, dc

@st.cache_resource
def get_spacy_model():
    return spacy.load("ru_core_news_sm/ru_core_news_sm-3.8.0")

model = load_model()
d, dc = load_vocabularies()
nlp = get_spacy_model()


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
    print("lol3")

    y_pred = np.argmax(model.predict(X), axis=1)
    st.write(f"Prediction: {dc[str(y_pred[0])]}")

print("lol2")
