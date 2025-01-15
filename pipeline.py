import os
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
from sklearn.ensemble import VotingClassifier  # type: ignore
import time
import requests
import google.generativeai as genai
import json
from google.api_core import retry
import threading
import logging
import warnings
# Suppress logs from specific libraries
logging.getLogger('pathway_engine').setLevel(logging.WARNING)
logging.getLogger('aiohttp.access').setLevel(logging.WARNING)
logging.getLogger('root').setLevel(logging.WARNING)  # General log level for all root logs
logging.getLogger('requests').setLevel(logging.WARNING)
logging.basicConfig(level=logging.WARNING)
GEMINI_API_KEY="AIzaSyBsjjQRqoo40I6pxLHJ4zpukOt5e1lg8C0"

# Environment setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"
nltk.download('stopwords')
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

# Pathway and server setup
from pathway.xpacks.llm.embedders import GeminiEmbedder
from pathway.xpacks.llm.splitters import TokenCountSplitter
from pathway.xpacks.llm.vector_store import VectorStoreClient, VectorStoreServer
from pathway.xpacks.llm.parsers import ParseUnstructured

text_splitter = TokenCountSplitter()
embedder = GeminiEmbedder(api_key=GEMINI_API_KEY)
parser = ParseUnstructured(mode='single', post_processors=[preprocessed_text])

# Pathway Reference Folders and Server Setup
conference_folders = {
    "CVPR": r"Reference/Publishable/CVPR",
    "EMNLP": r"Reference/Publishable/EMNLP",
    "KDD": r"Reference/Publishable/KDD",
    "NeurIPS": r"Reference/Publishable/NeurIPS",
    "TMLR": r"Reference/Publishable/TMLR"
}

reference_sources = []
for conference_name, folder_path in conference_folders.items():
    table = pw.io.fs.read(
        path=folder_path + "/*.pdf",  # Glob pattern to match all PDF files
        format="binary",
        with_metadata=True,
        mode="static",  # Static mode to ingest data once
    )
    reference_sources.append(table)

vector_server = VectorStoreServer(
    *reference_sources,
    parser=parser,
    embedder=embedder,
    splitter=text_splitter,
)


def run_vector_server():
    # The server is now a long-running process; ensure the program doesn't try to restart it.
    vector_server.run_server(host="127.0.0.1", port=8000)
    
resource_client = VectorStoreClient(
    host="127.0.0.1",
    port=8000,
)
# No stop_server() function is available, so we ensure no thread starts during shutdown
def classify_paper(paper_text,conferences, genai_model):
    # Perform similarity search
    result = resource_client.query(query=[paper_text], k=1)
    
    if not result:
        return "Could not classify the paper.", ""

    closest_match = result[0]
    metadata = closest_match["metadata"]
    # classification = metadata["conference"]
    # # Extract relevant passage
    relevant_passages = closest_match["text"]
    passage_oneline = " ".join(relevant_passages).replace("\n", " ")

    # Generate classification and rationale
    prompt = (
        f"Classify the following paper into one of these conferences: {', '.join(conferences)}.\n"
        f"Paper Content (trimmed): {paper_text}...\n"
        f"Relevant Passage: {passage_oneline}\n"
        f"Provide the closest match classification from the listed conferences. The classification must not be 'None of the above'\n"
        f"Additionally, provide a separate rationale for the classification (not more than 100 words). Start your rationale with 'Rationale:'."
)


    response = genai_model.generate_content(prompt).parts[0].text

    # Split response into classification and rationale
    if "Rationale:" in response:
        classification, rationale = response.split("Rationale:", maxsplit=1)
        classification = classification.replace("**Classification:**", "").replace("**","").strip()
        rationale = rationale.replace("**", "").strip()
    else:
        classification = response.strip()
        rationale = "Rationale not provided."

    return classification, rationale


genai.configure(api_key=GEMINI_API_KEY)

genai_model = genai.GenerativeModel("gemini-1.5-pro")

def classify_rationale(paper_path):
    if not os.path.exists(paper_path) or not paper_path.endswith(".pdf"):
        print("Invalid file. Please try again.")
    
    with open(paper_path, "rb") as file:
        pdf_reader = PdfReader(file)
        paper_text = " ".join(page.extract_text() for page in pdf_reader.pages)
        classification, rationale = classify_paper(paper_text, list(conference_folders.keys()), genai_model)
    return classification,rationale

# Run vector server and handle graceful termination
if __name__ == "__main__":
    try:
        vector_server_thread = threading.Thread(target=run_vector_server)
        vector_server_thread.start()

        # paper_path = "sample_paper.pdf"
        # classification, rationale = classify_paper(paper_path, list(conference_folders.keys()), genai_model)
        # print(f"Classification: {classification}")
        # print(f"Rationale: {rationale}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
    
    finally:
        # Wait for the thread to finish before exit
        if vector_server_thread.is_alive():
            vector_server_thread.join()
