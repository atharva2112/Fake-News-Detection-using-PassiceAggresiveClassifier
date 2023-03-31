# #%%
# import matplotlib.pyplot as plt
# import string
# import os
# from bs4 import BeautifulSoup
# import pandas as pd
# import numpy as np
# import re
# import nltk
# from nltk.corpus import stopwords
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.linear_model import PassiveAggressiveClassifier
# import requests
# from sentence_transformers import SentenceTransformer
# #%%
# nltk.download('stopwords')
# nltk.download('wordnet')
# wn = nltk.WordNetLemmatizer()
# ps = nltk.PorterStemmer()
# stoplist = set(stopwords.words("english"))
#
# def clean_text(text):
#     text = "".join([word.lower() for word in text if word not in string.punctuation])
#     tokens = re.split('\W+', text)
#     text = [ps.stem(wn.lemmatize(word)) for word in tokens if word not in stoplist]
#     return " ".join(text)
#
#
# true = pd.read_csv("True.csv")
# true["label"] = 1
# fake = pd.read_csv("Fake.csv")
# fake["label"] = 0
#
# #%%
# # Concat the true and false news articles to one dataframe.
# data = pd.concat([true, fake], ignore_index=True)
#
# # Clean the data using the NLP techniques.
# data['text'] = data['text'].apply(clean_text)
#
# #%%
# # Train test split
# X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)
# #%%
# # Vectorise the data
# tfidf_vect = TfidfVectorizer()
# X_train_tfidf = tfidf_vect.fit_transform(data["text"])
#
# #%%
# # Define the model and train it using the train data.
# model = PassiveAggressiveClassifier()
# model.fit(X_train_tfidf, data.label)
# print("Train Accuracy:",model.score(X_train_tfidf, data.label))
################################################################################################
#%%
import string
import os
import requests
import pandas as pd
import numpy as np
import re
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression
from sklearn.pipeline import Pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
#%%
nltk.download('stopwords')
nltk.download('wordnet')
wn = nltk.WordNetLemmatizer()
ps = nltk.PorterStemmer()
stoplist = set(stopwords.words("english"))

def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    text = [ps.stem(wn.lemmatize(word)) for word in tokens if word not in stoplist]
    return " ".join(text)

true = pd.read_csv("True.csv")
true["label"] = 1
fake = pd.read_csv("Fake.csv")
fake["label"] = 0

data = pd.concat([true, fake], ignore_index=True)
data['text'] = data['text'].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)
#%%
def sentiment_features(texts):
    sentiment_analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = [sentiment_analyzer.polarity_scores(text) for text in texts]
    sentiment_scores_df = pd.DataFrame(sentiment_scores)
    return sentiment_scores_df[['neg', 'neu', 'pos', 'compound']].values
#%%
embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v1")
#%%
def embed_text(texts):
    return embedder.encode(texts)
#%%
column_transformer = ColumnTransformer(
    transformers=[
        ('sentiment', FunctionTransformer(sentiment_features, validate=False), 'text'),
        ('embedding', FunctionTransformer(embed_text, validate=False), 'text')
    ],
    remainder='drop'
)
#%%
model_pipeline = Pipeline([
    ('features', column_transformer),
    ('model', PassiveAggressiveClassifier())
])
#%%
param_grid = {
    'model__C': [0.1, 1, 10],
    'model__max_iter': [1000, 2000]
}
#%%
grid_search = GridSearchCV(model_pipeline, param_grid, cv=5)
grid_search.fit(X_train.to_frame(), y_train)

print("Best hyperparameters:", grid_search.best_params_)
print("Train accuracy:", grid_search.best_score_)
#%%
y_pred = grid_search.predict(X_test.to_frame())
print("Test accuracy:", accuracy_score(y_test, y_pred))
print("Classification report:", classification_report(y_test, y_pred))

