import nltk

# Add the custom NLTK data path
# nltk.data.path.append('/app/.heroku/nltk_data')  # Add the path where the data is located on Heroku
# from nltk.tokenize import word_tokenize

import streamlit as st
import pickle
from nltk.corpus import stopwords
import string
nltk.download('stopwords')
from nltk.stem import PorterStemmer
ps = PorterStemmer()

stop_words = set(nltk.corpus.stopwords.words('english'))
def transform_text(text):
  text = text.lower()
  text = nltk.word_tokenize(text)

  y = []
  for i in text:
    if i.isalnum():
      y.append(i)

  text = y[:]
  y.clear()

  for i in text:
    if i not in stop_words and i not in string.punctuation:
      y.append(i)

  text= y[:]
  y.clear()

  for i in text:
    y.append(ps.stem(i))

  return " ".join(text)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/Sms Spam Classifier")

input_sms = st.text_input("Enter the message")

if st.button("Predict"):

  # 1. Preprocessing
  transformed_sms = transform_text(input_sms)

  # 2. Vectorize
  vector_input = tfidf.transform([transformed_sms])

  # 3. Predict
  result = model.predict(vector_input)

  # 4. Display
  if result == 1:
    st.header("Spam")
  else:
    st.header("Not Spam")

