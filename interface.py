import argparse
import requests
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

# Download stopwords and wordnet if necessary
nltk.download('stopwords')
nltk.download('wordnet')

# Create Lemmatizer and Stemmer
wn = nltk.WordNetLemmatizer()
ps = nltk.PorterStemmer()

# Define the set of stopwords
stoplist = set(stopwords.words("english"))


def clean_text(text):
    """
    Tokenizes and cleans text, removing punctuation, stopwords, and performing stemming and lemmatization.
    """
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    text = [ps.stem(wn.lemmatize(word)) for word in tokens if word not in stoplist]
    return " ".join(text)


def predict_article_real_or_fake(article_url, model, tfidf_vect, article_label=1):
    """
    Predicts whether the news article at the specified URL is real or fake using the provided trained model and TF-IDF vectorizer.
    """
    response = requests.get(article_url)
    soup = BeautifulSoup(response.content, "html.parser")
    article_content = ""
    article = soup.find("article")
    if article:
        paragraphs = article.find_all("p")
        for paragraph in paragraphs:
            article_content += paragraph.get_text()
    article_tfidf = tfidf_vect.transform([clean_text(article_content)])
    prediction = model.predict(article_tfidf)
    if prediction[0] == article_label:
        print(f"Prediction: True News (Correct)")
    else:
        print(f"Prediction: Fake News (Incorrect)")
        model.partial_fit(article_tfidf, [article_label])


def main():
    # Load the dataset and concatenate the true and fake news data
    true = pd.read_csv("True.csv")
    true["label"] = 1
    fake = pd.read_csv("Fake.csv")
    fake["label"] = 0
    data = pd.concat([true, fake], ignore_index=True)

    # Clean the text data
    data['text'] = data['text'].apply(clean_text)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

    # Train a Passive Aggressive Classifier model using TF-IDF vectors
    tfidf_vect = TfidfVectorizer()
    X_train_tfidf = tfidf_vect.fit_transform(X_train)
    model = PassiveAggressiveClassifier()
    model.fit(X_train_tfidf, y_train)

    # Create command-line interface for predicting news article validity
    parser = argparse.ArgumentParser(description='Predict whether a news article is real or fake.')
    parser.add_argument('article_url', type=str, help='The URL of the news article to predict.')
    args = parser.parse_args()

    # Call the prediction function with the specified article URL and print the prediction
    predict_article_real_or_fake(args.article_url, model, tfidf_vect)


if __name__ == "__main__":
    main()
