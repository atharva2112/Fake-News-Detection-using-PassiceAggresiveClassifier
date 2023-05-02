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
# from sentence_transformers import SentenceTransformer
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import numpy as np
import glob
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import learning_curve
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, accuracy_score, auc, RocCurveDisplay
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
# Visualising the Data
# Countplot
fig, ax = plt.subplots()
sns.countplot(data=data, x='label', ax=ax)
ax.set_title('Class Distribution')
ax.set_xlabel('Label')
ax.set_ylabel('Count')
ax.set_xticklabels(['Fake', 'True'])
fig.patch.set_facecolor('none')
ax.set_facecolor('none')
plt.show()


# Word Cloud
true_text = ' '.join(data[data['label'] == 1]['text'])
fake_text = ' '.join(data[data['label'] == 0]['text'])

true_wordcloud = WordCloud(width=1800, height=1800, background_color='white').generate(true_text)
fake_wordcloud = WordCloud(width=1800, height=1800, background_color='white').generate(fake_text)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.imshow(true_wordcloud, interpolation='bilinear')
ax1.axis('off')
ax1.set_title('True Articles')

ax2.imshow(fake_wordcloud, interpolation='bilinear')
ax2.axis('off')
ax2.set_title('Fake Articles')

plt.show()


def get_top_ngrams(corpus, ngram_range=(1, 1), top_n=10):
    vectorizer = CountVectorizer(ngram_range=ngram_range)
    X = vectorizer.fit_transform(corpus)
    freqs = np.sum(X, axis=0).A1
    index = np.argsort(freqs)[-top_n:]
    ngrams = [key for key, value in vectorizer.vocabulary_.items() if value in index]
    return ngrams, freqs[index]

true_articles = data[data['label'] == 1]['text']
fake_articles = data[data['label'] == 0]['text']

true_ngrams, true_freqs = get_top_ngrams(true_articles, ngram_range=(1, 2), top_n=10)
fake_ngrams, fake_freqs = get_top_ngrams(fake_articles, ngram_range=(1, 2), top_n=10)

true_ngram_df = pd.DataFrame({'ngram': true_ngrams, 'freq': true_freqs}).sort_values('freq', ascending=False)
fake_ngram_df = pd.DataFrame({'ngram': fake_ngrams, 'freq': fake_freqs}).sort_values('freq', ascending=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

sns.barplot(data=true_ngram_df, x='freq', y='ngram', ax=ax1, palette='viridis')
ax1.set_title('Top N-grams in True Articles')
ax1.set_xlabel('Frequency')

sns.barplot(data=fake_ngram_df, x='freq', y='ngram', ax=ax2, palette='viridis')
ax2.set_title('Top N-grams in Fake Articles')
ax2.set_xlabel('Frequency')

fig.patch.set_facecolor('none')
ax1.set_facecolor('none')
ax2.set_facecolor('none')

plt.tight_layout()
plt.show()

data['unique_words_count'] = data['text'].apply(lambda x: len(set(x.split())))

plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='text_length', y='unique_words_count', hue='label', alpha=0.7)
plt.title('Scatter plot of article text length and unique words count for true and fake articles')
plt.xlabel('Text Length')
plt.ylabel('Unique Words Count')
plt.show()


#%%
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
X_test_tfidf = tfidf_vect.transform(X_test)
y_pred_pac_tfidf = model_pac_tfidf.predict(X_test_tfidf)

# Create confusion matrix and normalize it
cm = confusion_matrix(y_test, y_pred_pac_tfidf)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Plot normalized confusion matrix
fig, ax = plt.subplots()
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=['Fake', 'True'], yticklabels=['Fake', 'True'])
ax.set_title('Normalized Confusion Matrix')
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
plt.show()
#%%


####################################################################################################################
# Approach 2
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training accuracy")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation accuracy")

    plt.legend(loc="best")
    return plt
X_train, X_test, y_train, y_test = train_test_split(data[['text']], data['label'], test_size=0.2, random_state=42)
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
model_pipeline_senti = Pipeline([
    ('features', column_transformer),
    ('dimensionality_reduction', TruncatedSVD(n_components=100)),
    ('model', PassiveAggressiveClassifier(C=1, max_iter=10))
])

param_grid = {
    'model__C': [0.1, 1, 10],
    'model__max_iter': [1000, 2000]
}
#%%
model_pipeline_senti.fit(X_train, y_train)

y_pred_pac_sentiment = model_pipeline_senti.predict(X_test)
print("Test accuracy:", accuracy_score(y_test, y_pred_pac_sentiment))
print("Classification report:\n", classification_report(y_test, y_pred_pac_sentiment))
#%%
# Plot the confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes, title='Confusion matrix', normalize=True):
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    fmt = '.2f' if normalize else 'd'
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2. else "black")
    fig.tight_layout()
    plt.show()

y_true = y_test
y_pred = y_pred_pac_sentiment
classes = ['Fake', 'True']
plot_confusion_matrix(y_true, y_pred, classes, normalize=True)
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
X_train, X_test, y_train, y_test = train_test_split(data[['text']], data['label'], test_size=0.2, random_state=42)
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
# Create confusion matrix and normalize it
cm = confusion_matrix(y_test, y_pred_rf_sentiment)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Plot normalized confusion matrix
fig, ax = plt.subplots()
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=['Fake', 'True'], yticklabels=['Fake', 'True'])
ax.set_title('Normalized Confusion Matrix (RandomForestClassifier)')
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
plt.show()
#%%

####################################################################################################################
# Approach 4
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, AdamW, RobertaModel

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



#%%
# Approach 6
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizerFast
from torch.utils.data import Dataset, DataLoader, Subset
from torch import nn
from torch.nn import Dropout
from sklearn.model_selection import KFold
import torch
from transformers import RobertaModel
from transformers import AdamW
from tqdm import tqdm
import os
#%%
# Load and preprocess data
new_data = pd.read_csv("train.csv")
new_data = new_data.drop(columns=["id", 'title', "author"])
new_data = new_data.dropna().reset_index()
new_data = new_data.drop(columns=["index"])

true = pd.read_csv("True.csv")
true["label"] = 1
fake = pd.read_csv("Fake.csv")
fake["label"] = 0
data = pd.concat([true, fake,new_data], ignore_index=True)

X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=["label"]), data["label"], test_size=0.2, random_state=42)
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
tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
train_dataset = NewsDataset(X_train['text'].tolist(), y_train.tolist(), tokenizer)
test_dataset = NewsDataset(X_test['text'].tolist(), y_test.tolist(), tokenizer)

class CustomRobertaModel(nn.Module):
    def __init__(self, num_labels, dropout_prob):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.dropout = Dropout(dropout_prob)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        dropped_output = self.dropout(pooled_output)
        logits = self.classifier(dropped_output)
        return logits
#%%
kf = KFold(n_splits=5, shuffle=True, random_state=42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 2
batch_size = 7
learning_rate = 2e-5
num_labels = 2
#%%
save_checkpoint_path = "checkpoints"
if not os.path.exists(save_checkpoint_path):
    os.makedirs(save_checkpoint_path)

losses = []
accuracies = []
for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
    print(f"Fold {fold + 1}")

    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(train_dataset, val_idx)

    train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    model = CustomRobertaModel(num_labels, dropout_prob=0.1).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']

            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)

        train_acc = correct / total
        losses.append(epoch_loss / len(train_dataloader))
        accuracies.append(correct / total)
        print(f"Training accuracy after epoch {epoch + 1}: {train_acc:.4f}")

        # Validation loop
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']

                logits = model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += len(labels)

        val_acc = val_correct / val_total
        print(f"Validation accuracy after epoch {epoch + 1}: {val_acc:.4f}")

    print(f"Fold {fold + 1} finished")
#%%
def load_checkpoint(checkpoint_path, model, optimizer):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return epoch
#%%
# Continue training
for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
    print(f"Fold {fold + 1}")

    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(train_dataset, val_idx)

    train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    model = CustomRobertaModel(num_labels, dropout_prob=0.1).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    starting_epoch = 0

    # Load the checkpoint
    checkpoint_pattern = f"checkpoints/checkpoint_fold_{fold + 1}_epoch_*.pth"
    checkpoint_files = glob.glob(checkpoint_pattern)
    if checkpoint_files:
        last_checkpoint_path = max(checkpoint_files, key=os.path.getctime)  # Get the most recent checkpoint
        starting_epoch = load_checkpoint(last_checkpoint_path, model, optimizer)
        print(f"Loaded checkpoint for fold {fold + 1} from {last_checkpoint_path}")
    else:
        print(f"No checkpoint found for fold {fold + 1}")

    for epoch in range(starting_epoch, num_epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']

            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)

        train_acc = correct / total
        losses.append(epoch_loss / len(train_dataloader))
        accuracies.append(correct / total)
        print(f"Training accuracy after epoch {epoch + 1}: {train_acc:.4f}")

        # Validation loop
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']

                logits = model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += len(labels)

            val_acc = val_correct / val_total
            print(f"Validation accuracy after epoch {epoch + 1}: {val_acc:.4f}")

        print(f"Fold {fold + 1} finished")
#%%

if not os.path.exists("models"):
    os.makedirs("models")
model_save_path = "models/trained_model.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved at {model_save_path}")

#%%
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

####################################################################################################################
# Plotting
# y_preds = [y_pred_pac_tfidf, y_pred_pac_sentiment, y_pred_rf_sentiment, y_pred_bert, y_pred_roberta]
# approaches = ['PAC-TFIDF', 'PAC-Sentiment', 'RF-Sentiment', 'BERT', 'RoBERTa']
#
# def plot_metrics(y_true, y_preds, approaches):
#     plt.figure(figsize=(20, 5))
#
#     for i, (y_pred, approach) in enumerate(zip(y_preds, approaches)):
#         cm = confusion_matrix(y_true, y_pred)
#         fpr, tpr, _ = roc_curve(y_true, y_pred)
#         precision, recall, _ = precision_recall_curve(y_true, y_pred)
#
#         plt.subplot(1, 3, 1)
#         plt.plot(fpr, tpr, label=f"{approach} (AUC = {round(np.trapz(tpr, fpr), 2)})")
#         plt.xlabel("False Positive Rate")
#         plt.ylabel("True Positive Rate")
#         plt.title("ROC Curve")
#         plt.legend()
#
#         plt.subplot(1, 3, 2)
#         plt.plot(recall, precision, label=f"{approach} (AUC = {round(np.trapz(precision, recall), 2)})")
#         plt.xlabel("Recall")
#         plt.ylabel("Precision")
#         plt.title("Precision-Recall Curve")
#         plt.legend()
#
#     plt.subplot(1, 3, 3)
#     for i, (y_pred, approach) in enumerate(zip(y_preds, approaches)):
#         plt.bar(i, accuracy_score(y_true, y_pred), label=approach)
#     plt.xticks(range(len(approaches)), approaches)
#     plt.xlabel("Approach")
#     plt.ylabel("Accuracy")
#     plt.title("Accuracy Comparison")
#     plt.legend()
#
#     plt.show()
#
# plot_metrics(y_test, y_preds, approaches)
#%%
models = ['Passive Aggressive\n(TF-IDF)','Passive Aggressive\n(Sentiment Analysis)', 'Random Forest\n(Sentiment Analysis)', 'BERT', 'RoBERTa']
training_times = [30, 65, 68, 180,520]   # Replace these with your actual training times
colors = ['blue', 'green', 'orange','yellow','red']

# Create a horizontal bar chart
fig, ax = plt.subplots(figsize=(20, 6))
bars = ax.barh(models, training_times, color=colors)

# Set the labels and title
ax.set_xlabel('Training Time (in minutes)')
ax.set_ylabel('Models')
ax.set_title('Training Time Comparison')

# Annotate the bars with their values
for i, bar in enumerate(bars):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
            str(training_times[i]) + ' min', fontsize=12, va='center')
plt.tight_layout()
# Show the chart
plt.show()