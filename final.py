#%%
import matplotlib.pyplot as plt
import string
import os
import pandas as pd
import numpy as np
import re
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier
from sentence_transformers import SentenceTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, accuracy_score
#%%
# Read the data
new_data = pd.read_csv("train.csv")
new_data = new_data.drop(columns=["id", 'title', "author"])
new_data = new_data.dropna().reset_index()
new_data = new_data.drop(columns=["index"])

true = pd.read_csv("True.csv")
true["label"] = 1
fake = pd.read_csv("Fake.csv")
fake["label"] = 0
data = pd.concat([true, fake,new_data], ignore_index=True)
#%%



####################################################################################################################
# Approach 1
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
#%%
# Clean the data using the NLP techniques.
data['text'] = data['text'].apply(clean_text)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)
#%%
# Vectorise the data
tfidf_vect = TfidfVectorizer()
X_train_tfidf = tfidf_vect.fit_transform(X_train)

# Define the model and train it using the train data.
model_pac_tfidf = PassiveAggressiveClassifier()
model_pac_tfidf.fit(X_train_tfidf, y_train)

# Evaluate the model
X_test_tfidf = tfidf_vect.transform(X_test)
y_pred_pac_tfidf = model_pac_tfidf.predict(X_test_tfidf)
print("Approach 1 - PAC-TFIDF Test Accuracy:", accuracy_score(y_test, y_pred_pac_tfidf))
#%%



####################################################################################################################
# Approach 2
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
    ('model', PassiveAggressiveClassifier(C=0.01, max_iter=1000))
])

param_grid = {
    'model__C': [0.1, 1, 10],
    'model__max_iter': [1000, 2000]
}
#%%
model_pipeline.fit(X_train, y_train)
#%%
y_pred_pac_sentiment = model_pipeline.predict(X_test)
print("Test accuracy:", accuracy_score(y_test, y_pred_pac_sentiment))
print("Classification report:\n", classification_report(y_test, y_pred_pac_sentiment))
#%%



####################################################################################################################
# Approach 3
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
    ('model', RandomForestClassifier(n_estimators=100, random_state=42))
])

param_grid = {
    'model__C': [0.1, 1, 10],
    'model__max_iter': [1000, 2000]
}
#%%
model_pipeline.fit(X_train, y_train)
#%%
y_pred_rf_sentiment = model_pipeline.predict(X_test)
print("Test accuracy:", accuracy_score(y_test, y_pred_rf_sentiment))
print("Classification report:\n", classification_report(y_test, y_pred_rf_sentiment))
#%%



####################################################################################################################
# Approach 4
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, AdamW
epochs = 2
batch = 20
learning_r = 5e-5
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        inputs = self.tokenizer(text, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
        inputs['input_ids'] = inputs['input_ids'].squeeze()
        inputs['attention_mask'] = inputs['attention_mask'].squeeze()
        inputs['labels'] = torch.tensor(label, dtype=torch.long)
        return inputs

#%%
# Load the pretrained DistilBert tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
#%%
# Prepare the datasets
train_dataset = NewsDataset(X_train['text'].tolist(), y_train.tolist(), tokenizer)
test_dataset = NewsDataset(X_test['text'].tolist(), y_test.tolist(), tokenizer)
#%%
# Create DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch, shuffle=False)
#%%
# Load the DistilBert model
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
#%%
# Prepare the optimizer
optimizer = AdamW(model.parameters(), lr=learning_r)
#%%
# Train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
#%%
from tqdm import tqdm

# Train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
#%%
from tqdm import tqdm

# Train the model
train_loss_values = []
train_acc_values = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0

    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
    for batch in progress_bar:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        correct += (preds == batch['labels']).sum().item()
        total += len(batch['labels'])

        epoch_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})

    epoch_loss /= len(train_dataloader)
    epoch_acc = correct / total
    train_loss_values.append(epoch_loss)
    train_acc_values.append(epoch_acc)
    print(f"Epoch {epoch + 1}, Loss: {epoch_loss}, Accuracy: {epoch_acc}")

# Plot the metrics
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_loss_values, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_acc_values, label="Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.show()
#%%
# Evaluate the model on the test dataset
model.eval()
correct = 0
total = 0

for batch in test_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
    logits = outputs.logits
    y_pred_bert = torch.argmax(logits, dim=1)
    correct += (y_pred_bert == batch['labels']).sum().item()
    total += len(batch['labels'])

print("Accuracy on test dataset:", correct / total)
#%%
# Save the BERT model
save_directory = "saved_model"

if not os.path.exists(save_directory):
    os.makedirs(save_directory)

model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)
#%%
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

save_directory = "saved_model"

# Load the saved tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained(save_directory)

# Load the saved model
model = DistilBertForSequenceClassification.from_pretrained(save_directory)
#%%
# Evaluate the model
# Load the new dataset
dataset = pd.read_csv("fake_news_data.csv")

# Keep only English articles
english_articles = dataset[dataset["language"] == "english"]

# Assign labels (1 for fake, 0 for others)
english_articles["label"] = english_articles["type"].apply(lambda x: 1 if x == "fake" else 0)
english_articles = english_articles.dropna()
#%%
# Load the tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# Prepare the dataset
test_dataset = NewsDataset(english_articles["text"].tolist(), english_articles["label"].tolist(), tokenizer)

# Create DataLoader
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Load the pretrained model and tokenizer
model_directory = "saved_model"
model = DistilBertForSequenceClassification.from_pretrained(model_directory)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_directory)
#%%
# Move the model to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
#%%
# Evaluate the model on the new dataset
model.eval()
correct = 0
total = 0

for batch in test_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
    logits = outputs.logits
    preds = torch.argmax(logits, dim=1)
    correct += (preds == batch['labels']).sum().item()
    total += len(batch['labels'])

print("Accuracy on new dataset:", correct / total)
#%%



####################################################################################################################
# Approach 5
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, AdamW
#%%
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        inputs = self.tokenizer(text, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
        inputs['input_ids'] = inputs['input_ids'].squeeze()
        inputs['attention_mask'] = inputs['attention_mask'].squeeze()
        inputs['labels'] = torch.tensor(label, dtype=torch.long)
        return inputs
#%%
# Load the pretrained RoBERTa tokenizer
tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')

# Prepare the datasets
train_dataset = NewsDataset(X_train['text'].tolist(), y_train.tolist(), tokenizer)
test_dataset = NewsDataset(X_test['text'].tolist(), y_test.tolist(), tokenizer)
#%%
# Create DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=7, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=7, shuffle=False)
#%%
# Load the RoBERTa model
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

# Prepare the optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
#%%
losses = []
accuracies = []

for epoch in range(3):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        correct += (preds == batch['labels']).sum().item()
        total += len(batch['labels'])

    losses.append(epoch_loss / len(train_dataloader))
    accuracies.append(correct / total)

    # Print the accuracy for the current epoch
    print(f"Epoch {epoch + 1} accuracy: {accuracies[-1]:.4f}")

# Plot loss and accuracy
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(losses)
ax1.set_title("Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax2.plot(accuracies)
ax2.set_title("Accuracy")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
plt.show()
#%%
# Evaluate the model
model.eval()
correct = 0
total = 0
for batch in test_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
    logits = outputs.logits
    y_pred_roberta = torch.argmax(logits, dim=1)
    correct += (y_pred_roberta == batch['labels']).sum().item()
    total += len(batch['labels'])

print("Test accuracy:", correct / total)
#%%


####################################################################################################################
# Plotting
y_preds = [y_pred_pac_tfidf, y_pred_pac_sentiment, y_pred_rf_sentiment, y_pred_bert, y_pred_roberta]
approaches = ['PAC-TFIDF', 'PAC-Sentiment', 'RF-Sentiment', 'BERT', 'RoBERTa']

def plot_metrics(y_true, y_preds, approaches):
    plt.figure(figsize=(20, 5))

    for i, (y_pred, approach) in enumerate(zip(y_preds, approaches)):
        cm = confusion_matrix(y_true, y_pred)
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        precision, recall, _ = precision_recall_curve(y_true, y_pred)

        plt.subplot(1, 3, 1)
        plt.plot(fpr, tpr, label=f"{approach} (AUC = {round(np.trapz(tpr, fpr), 2)})")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(recall, precision, label=f"{approach} (AUC = {round(np.trapz(precision, recall), 2)})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()

    plt.subplot(1, 3, 3)
    for i, (y_pred, approach) in enumerate(zip(y_preds, approaches)):
        plt.bar(i, accuracy_score(y_true, y_pred), label=approach)
    plt.xticks(range(len(approaches)), approaches)
    plt.xlabel("Approach")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Comparison")
    plt.legend()

    plt.show()

plot_metrics(y_test, y_preds, approaches)