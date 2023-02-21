#%%
import pandas as pd
import matplotlib.pyplot as plt
import calendar
import string
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
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
main = [true,fake]
main = pd.concat(main,ignore_index=True)
main = main.drop(["date","subject"],axis=1)
main["length"] = main['text'].apply(lambda x: len(x) - x.count(" "))

# main.index = pd.to_datetime(main.date)
#%%
print("Input data has {} rows and {} columns".format(len(main), len(main.columns)))
print("Out of {} rows, {} are spam, {} are ham".format(len(main),len(main[main['label']==0]),len(main[main['label']==1])))
print("Number of null in label: {}".format(main['label'].isnull().sum()))
print("Number of null in text: {}".format(main['text'].isnull().sum()))
#%%
# Removing all the punctuations from the sentences
def remove_punct(text):
    text_nopunct = "".join([char for char in text if char not in string.punctuation])
    return text_nopunct

main['text_no_punct'] = main['text'].apply(lambda x: remove_punct(x))

main.head()
#%%
# Separating each words of the sentence as the element of the list.
def tokenize(text):
    tokens = re.split('\W+', text)
    return tokens

main["text_cleaned"] = main.text_no_punct.apply(lambda x: tokenize(x.lower()))
#%%
# Removing stop words from the data
stopword = nltk.corpus.stopwords.words('english')
def remove_stopwords(tokenized_list):
    text = [word for word in tokenized_list if word not in stopword]
    return text

main['text_nostop'] = main['text_cleaned'].apply(lambda x: remove_stopwords(x))
main.head()
#%%
# Lemmatizing the text
wn = nltk.WordNetLemmatizer()
def lemmatizing(tokenized_text):
    text = [wn.lemmatize(word) for word in tokenized_text]
    return text
main['text_lemmatized'] = main['text_nostop'].apply(lambda x: lemmatizing(x))
main.head(10)
#%%
# Visualizing the true data
true.subject.value_counts().plot.bar()
plt.title("True news articles for different subjects")
plt.xticks(rotation =0)
plt.show()
#%%
fake.subject.value_counts().plot.bar()
plt.title("Fake news articles for different subjects")
plt.xticks(rotation = 45)
plt.tight_layout()
plt.show()
#%%
bins = np.linspace(0, 10000, 40)
plt.hist(main[main['label']==0]['length'], bins, alpha=0.5, label='fake',density = True)
plt.hist(main[main['label']==1]['length'], bins, alpha=0.5, label='true',density = True)
plt.legend()
plt.show()
#%%
