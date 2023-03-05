#%%
import pandas as pd
import matplotlib.pyplot as plt
import calendar
import string
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import PassiveAggressiveClassifier
import os
from sklearn.metrics import classification_report, accuracy_score
import requests
#%%
def month_to_num(df):
    months = list(calendar.month_name)[1:]
    for i in range(len(months)):
        df.date = df.date.replace("",f"{i+1},").replace(", ","/")
    return df
def count_punct(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count/(len(text) - text.count(" ")), 3)*100
#%%
true = pd.read_csv("True.csv")
true["label"] = 1
fake = pd.read_csv("Fake.csv")
fake["label"] = 0
#%%
data = [true,fake]
data = pd.concat(data,ignore_index=True)
data = data.drop(["date","subject"],axis=1)
# data["length"] = data['text'].apply(lambda x: len(x) - x.count(" "))

# main.index = pd.to_datetime(main.date)
#%%
print("Input data has {} rows and {} columns".format(len(data), len(data.columns)))
print("Out of {} rows, {} are spam, {} are ham".format(len(data),len(data[data['label']==0]),len(data[data['label']==1])))
print("Number of null in label: {}".format(data['label'].isnull().sum()))
print("Number of null in text: {}".format(data['text'].isnull().sum()))
#%%
# # Removing all the punctuations from the sentences
# def remove_punct(text):
#     text_nopunct = "".join([char for char in text if char not in string.punctuation])
#     return text_nopunct
#
# main['text_no_punct'] = main['text'].apply(lambda x: remove_punct(x))
#
# main.head()
# #%%
# # Separating each words of the sentence as the element of the list.
# def tokenize(text):
#     tokens = re.split('\W+', text)
#     return tokens
#
# main["text_cleaned"] = main.text_no_punct.apply(lambda x: tokenize(x.lower()))
# #%%
# # Removing stop words from the data
# stopword = nltk.corpus.stopwords.words('english')
# def remove_stopwords(tokenized_list):
#     text = [word for word in tokenized_list if word not in stopword]
#     return text
#
# main['text_nostop'] = main['text_cleaned'].apply(lambda x: remove_stopwords(x))
# main.head()
# #%%
# # Lemmatizing the text
# wn = nltk.WordNetLemmatizer()
# def lemmatizing(tokenized_text):
#     text = [wn.lemmatize(word) for word in tokenized_text]
#     return text
# main['text_lemmatized'] = main['text_nostop'].apply(lambda x: lemmatizing(x))
# main.head(10)
#%%
wn = nltk.WordNetLemmatizer()
ps = nltk.PorterStemmer()
nltk.download('stopwords')
stoplist = set(stopwords.words("english"))
def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    text = [ps.stem(word) for word in tokens if word not in stoplist]
    return text
#%%
nltk.download('wordnet')
tfidf_vect = TfidfVectorizer(analyzer=clean_text)
#%%
X_tfidf = tfidf_vect.fit_transform(data['text'])
#%%
print(X_tfidf.shape)
print(tfidf_vect.get_feature_names_out())
#%%
X_tfidf_df = pd.DataFrame(X_tfidf.toarray())
X_tfidf_df.columns = tfidf_vect.get_feature_names_out()
X_tfidf_df

#%%
X_train, X_test, y_train, y_test = train_test_split(X_tfidf_df, data['label'], test_size=0.2, random_state=42)
#%%
model = PassiveAggressiveClassifier(C=0.5, random_state=5)

# Fitting model
model.fit(X_train, y_train)

# Making prediction on test set
test_pred = model.predict(X_test)

# Model evaluation
print(f"Test Set Accuracy : {accuracy_score(y_test, test_pred) * 100} %\n\n")
print(f"Classification Report : \n\n{classification_report(y_test, test_pred)}")

#%%
test = data.text[1]
test_vect = TfidfVectorizer(analyzer=clean_text,vocabulary=tfidf_vect.vocabulary_)
test1 = test_vect.fit_transform([test])
test_df = pd.DataFrame(test1.toarray())
test_df.columns = test_vect.get_feature_names_out()
test_df
#%%
url = 'https://www.bbc.com/news/world-europe-60447272'
article = requests.get(url).text
article_tfidf = tfidf_vect.transform([clean_text(article)])
prediction = model.predict(article_tfidf)

# Print the prediction
print(f"Prediction: {prediction[0]}")
#%%
test1 = tfidf_vect.fit_transform([test])
test_df = pd.DataFrame(test1.toarray())
test_df.columns = tfidf_vect.get_feature_names_out()
# Make a prediction using the trained model
prediction = model.predict(test_df)
print(prediction)
#%%
# # Visualizing the true data
# true.subject.value_counts().plot.bar()
# plt.title("True news articles for different subjects")
# plt.xticks(rotation =0)
# plt.show()
# #%%
# fake.subject.value_counts().plot.bar()
# plt.title("Fake news articles for different subjects")
# plt.xticks(rotation = 45)
# plt.tight_layout()
# plt.show()
# #%%
# bins = np.linspace(0, 10000, 40)
# plt.hist(main[main['label']==0]['length'], bins, alpha=0.5, label='fake',density = True)
# plt.hist(main[main['label']==1]['length'], bins, alpha=0.5, label='true',density = True)
# plt.legend()
# plt.show()
# #%%
# # Split the dataset into training and testing sets
# import pickle
# X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)
#
# # Vectorize the text using TF-IDF Vectorizer
# tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
# tfidf_train = tfidf_vectorizer.fit_transform(X_train)
# tfidf_test = tfidf_vectorizer.transform(X_test)
# #%%
# # Train the model
# pac = PassiveAggressiveClassifier(max_iter=50)
# for i in range(10):
#     pac.partial_fit(tfidf_train, y_train, classes=['FAKE', 'REAL'])
#     y_pred = pac.predict(tfidf_test)
#     score = accuracy_score(y_test, y_pred)
#     print(f"Iteration {i}: Accuracy: {score}")
#
# # Save the model
# with open('pac_model.pkl', 'wb') as f:
#     pickle.dump(pac, f)
# #%%
# print(data["label"])
#%%
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

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Define functions for text preprocessing
wn = nltk.WordNetLemmatizer()
ps = nltk.PorterStemmer()
stoplist = set(stopwords.words("english"))

def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    text = [ps.stem(wn.lemmatize(word)) for word in tokens if word not in stoplist]
    return " ".join(text)

# Load the data
true = pd.read_csv("True.csv")
true["label"] = 1
fake = pd.read_csv("Fake.csv")
fake["label"] = 0

# Concatenate the data
data = pd.concat([true, fake], ignore_index=True)

# Clean the text
data['text'] = data['text'].apply(clean_text)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# Vectorize the text data
tfidf_vect = TfidfVectorizer()
X_train_tfidf = tfidf_vect.fit_transform(X_train)

# Train the model
model = PassiveAggressiveClassifier()
model.fit(X_train_tfidf, y_train)
#%%
# Test the model on a random article from the internet
# url = ''
# article = requests.get(url).text
article = "The Pentagon is considering a Boeing proposal to supply Ukraine with cheap, small precision bombs fit ted on to abundantly available rockets, allowing Kyiv to strike far behind Russian lines, according t o a Reuters report. US and allied military inventories are shrinking, and Ukraine faces an increasin g need for more sophisticated weapons as the war drags on. Boeing's proposed system, dubbed Ground-La unched Small Diameter Bomb (GLSDB), is one of about a half-dozen plans for getting new munitions into production for Ukraine and America's eastern European allies, industry sources told the news agency. GLSDB could be delivered as early as spring 2023, according to a document reviewed by Reuters and thr ee people familiar with the plan. It combines the GBU-39 Small Diameter Bomb (SDB) withith the M26 rocket motor, both of which are common in US inventories. Although a handful of GLSDB units have already been made, there are many logistical obstacles to formal procurement. The Boeing plan requires a price discovery waiver, exempting the contractor from an in-depth review that ensur es the Pentagon is getting the best deal possible. Any arrangement would also require at least six suppliers to expedite shipments of their parts and services to produce the weapon quickly. Althoug h the US has rebuffed requests for the 185-mile (297km) range Atacms missile, the GLSDB's 94-mile (150km) range would allow Ukraine to hit valuable military targets that have been out of reach and help it continue pressing its counterattacks by disrupting Russian rear areas. GLSDB is made joint ly by Saab AB and Boeing Co and has been in development since 2019, well before the invasion, which Russia calls a special operation. In October, SAAB chief executive Micael Johansson said of the G LSDB:We are imminently shortly expecting contracts on that. According to the document - a Boein g proposal to US European - has small, folding wings that allow it to glide more than 100km if drop ped from an aircraft and targets as small as 3ft in diameter."
article_tfidf = tfidf_vect.transform([clean_text(article)])
prediction = model.predict(article_tfidf)
# Print the prediction
print(f"Prediction: {prediction[0]}")
