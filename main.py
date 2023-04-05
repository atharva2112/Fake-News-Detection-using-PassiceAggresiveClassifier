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
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression
from sklearn.pipeline import Pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.decomposition import TruncatedSVD
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

X_train, X_test, y_train, y_test = train_test_split(data[['text']], data['label'], test_size=0.2, random_state=42)
#%%
class SentimentTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        sentiment_analyzer = SentimentIntensityAnalyzer()
        sentiment_scores = [sentiment_analyzer.polarity_scores(text) for text in X['text']]
        sentiment_scores_df = pd.DataFrame(sentiment_scores)
        return sentiment_scores_df[['neg', 'neu', 'pos', 'compound']].values

#%%
class EmbeddingTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        embedder = SentenceTransformer("sentence-transformers/distilbert-base-nli-mean-tokens")
        return embedder.encode(X['text'].tolist())

column_transformer = ColumnTransformer(
    transformers=[
        ('sentiment', SentimentTransformer(), ['text']),
        ('embedding', EmbeddingTransformer(), ['text'])
    ],
    remainder='drop'
)
#%%
model_pipeline = Pipeline([
    ('features', column_transformer),
    ('dimensionality_reduction', TruncatedSVD(n_components=100)),
    ('model', PassiveAggressiveClassifier())
])

param_grid = {
    'model__C': [0.1, 1, 10],
    'model__max_iter': [1000, 2000]
}
#%%
n_iter_search = 1
random_search = RandomizedSearchCV(model_pipeline, param_distributions=param_grid, n_iter=n_iter_search, cv=3, random_state=42, n_jobs=-1, error_score='raise')
random_search.fit(X_train, y_train)

print("Best hyperparameters:", random_search.best_params_)
print("Train accuracy:", random_search.best_score_)
#%%
y_pred = random_search.predict(X_test)
print("Test accuracy:", accuracy_score(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred))
