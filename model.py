import os
import streamlit as st
import numpy as np
import pandas as pd
import re
import string
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from PyPDF2 import PdfReader
import pathway as pw
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
nltk_data_path = os.path.expanduser('~/nltk_data')
if not os.path.exists(nltk_data_path):
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
exclude = string.punctuation
lemmatizer = WordNetLemmatizer()


# Function Definitions (Text extraction, cleaning, etc.)
def text_extractor_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def lemmatize_words(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

def remove_stopwords(text):
    new_text = []
    for word in text.split():
        if word in stopwords.words('english'):
            new_text.append('')
        else:
            new_text.append(word)
    x = new_text[:]
    new_text.clear()
    return " ".join(x)

def preprocessed_text(text):
    text = text.lower()  # Lowercase Conversion
    text = re.sub(r'<.*?>', '', text)  # Removing html tags
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Removing URLs
    text = text.translate(str.maketrans('', '', exclude))  # Removing punctuations
    text = remove_stopwords(text)  # Removing stop words
    text = lemmatize_words(text)  # Lemmatizing
    return text

# Reading PDFs and preprocessing
non_publishable_folder = 'Reference/Non-Publishable/'
non_publishable_texts = []
for file in os.listdir(non_publishable_folder):
    if file.endswith('.pdf'):
        pdf_path = os.path.join(non_publishable_folder, file)
        non_publishable_texts.append(preprocessed_text(text_extractor_from_pdf(pdf_path)))

publishable_folder = ['Reference/Publishable/CVPR/', 'Reference/Publishable/EMNLP/', 'Reference/Publishable/KDD/', 'Reference/Publishable/NeurIPS/', 'Reference/Publishable/TMLR/']
publishable_texts = []
for folder in publishable_folder:
    for file in os.listdir(folder):
        if file.endswith('.pdf'):
            pdf_path = os.path.join(folder, file)
            publishable_texts.append(preprocessed_text(text_extractor_from_pdf(pdf_path)))

# Data Preparation
data = {
    "Text": publishable_texts + non_publishable_texts,
    "Publishable": [1] * len(publishable_texts) + [0] * len(non_publishable_texts)
}
df = pd.DataFrame(data)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df_publishable = df[df['Publishable'] == 1]
df_non_publishable = df[df['Publishable'] == 0]

df_publishable_downsampled = df_publishable.sample(df_non_publishable.shape[0], random_state=42)
df_balanced = pd.concat([df_publishable_downsampled, df_non_publishable])
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

X = df_balanced.iloc[:, 0:1]
y = df_balanced['Publishable'].to_numpy(dtype=np.int32)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
cv = CountVectorizer()
X_train_bow = cv.fit_transform(X_train['Text']).toarray()
X_test_bow = cv.transform(X_test['Text']).toarray()

# Training classifiers
gnb = GaussianNB()
gnb.fit(X_train_bow, y_train)

clf = LogisticRegression(random_state=42)
clf.fit(X_train_bow, y_train)

svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train_bow, y_train)

voting_clf = VotingClassifier(estimators=[('gnb', gnb), ('lr', clf), ('svm', svm)], voting='hard')
voting_clf.fit(X_train_bow, y_train)