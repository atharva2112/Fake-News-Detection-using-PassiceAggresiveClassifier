#%%
import pickle
import matplotlib.pyplot as plt
import string
import os
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import PassiveAggressiveClassifier
import requests
#%%
filename = 'fake_news_model.sav'
model = pickle.load(open(filename, 'rb'))
url = 'https://www.cnn.com/2023/03/02/world/brain-computer-organoids-scn/index.html'
article_label = 1
response = requests.get(url)

soup = BeautifulSoup(response.content, "html.parser")

article_content = ""
article = soup.find("article")
if article:
    paragraphs = article.find_all("p")
    for paragraph in paragraphs:
        article_content += paragraph.get_text()

article_tfidf = tfidf_vect.transform([clean_text(article_content)])
prediction = model.predict(article_tfidf)

# Print the prediction
if prediction[0] == article_label:
    print(f"Prediction: True News (Correct)")
else:
    print(f"Prediction: Fake News (Incorrect)")
    model.partial_fit(article_tfidf, [article_label])
#%%
# Load the saved model
# loaded_model = pickle.load(open(filename, 'rb'))
prediction = model.predict(article_tfidf)
print(f"Prediction: {'Fake News' if prediction[0]==0 else 'True News'}")