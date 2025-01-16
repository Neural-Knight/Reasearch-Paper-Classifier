# import os
# import streamlit as st
# import numpy as np
# import pandas as pd
# import re
# import string
# import nltk
# from nltk.stem.porter import PorterStemmer
# from nltk.stem import WordNetLemmatizer
# from nltk.corpus import stopwords
# from PyPDF2 import PdfReader
# import pathway as pw
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.naive_bayes import GaussianNB
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import cross_val_score
# from sklearn.metrics import accuracy_score, f1_score
# from sklearn.svm import SVC
# from sklearn.ensemble import VotingClassifier  # type: ignore
# import time
# import requests
# import google.generativeai as genai
# import json
# from google.api_core import retry
# import threading
# import logging
# import warnings
# # Suppress logs from specific libraries
# logging.getLogger('pathway_engine').setLevel(logging.WARNING)
# logging.getLogger('aiohttp.access').setLevel(logging.WARNING)
# logging.getLogger('root').setLevel(logging.WARNING)  # General log level for all root logs
# logging.getLogger('requests').setLevel(logging.WARNING)
# logging.basicConfig(level=logging.WARNING)
# GEMINI_API_KEY="AIzaSyBsjjQRqoo40I6pxLHJ4zpukOt5e1lg8C0"

# # Environment setup
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# nltk_data_path = os.path.expanduser('~/nltk_data')
# if not os.path.exists(nltk_data_path):
#     nltk.download('stopwords')
#     nltk.download('wordnet')
#     nltk.download('omw-1.4')
# exclude = string.punctuation
# lemmatizer = WordNetLemmatizer()

# # Function Definitions (Text extraction, cleaning, etc.)
# def text_extractor_from_pdf(pdf_path):
#     reader = PdfReader(pdf_path)
#     text = ""
#     for page in reader.pages:
#         text += page.extract_text()
#     return text

# def lemmatize_words(text):
#     return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

# def remove_stopwords(text):
#     new_text = []
#     for word in text.split():
#         if word in stopwords.words('english'):
#             new_text.append('')
#         else:
#             new_text.append(word)
#     x = new_text[:]
#     new_text.clear()
#     return " ".join(x)

# def preprocessed_text(text):
#     text = text.lower()  # Lowercase Conversion
#     text = re.sub(r'<.*?>', '', text)  # Removing html tags
#     text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Removing URLs
#     text = text.translate(str.maketrans('', '', exclude))  # Removing punctuations
#     text = remove_stopwords(text)  # Removing stop words
#     text = lemmatize_words(text)  # Lemmatizing
#     return text

# # Reading PDFs and preprocessing
# non_publishable_folder = 'Reference/Non-Publishable/'
# non_publishable_texts = []
# for file in os.listdir(non_publishable_folder):
#     if file.endswith('.pdf'):
#         pdf_path = os.path.join(non_publishable_folder, file)
#         non_publishable_texts.append(preprocessed_text(text_extractor_from_pdf(pdf_path)))

# publishable_folder = ['Reference/Publishable/CVPR/', 'Reference/Publishable/EMNLP/', 'Reference/Publishable/KDD/', 'Reference/Publishable/NeurIPS/', 'Reference/Publishable/TMLR/']
# publishable_texts = []
# for folder in publishable_folder:
#     for file in os.listdir(folder):
#         if file.endswith('.pdf'):
#             pdf_path = os.path.join(folder, file)
#             publishable_texts.append(preprocessed_text(text_extractor_from_pdf(pdf_path)))

# # Data Preparation
# data = {
#     "Text": publishable_texts + non_publishable_texts,
#     "Publishable": [1] * len(publishable_texts) + [0] * len(non_publishable_texts)
# }
# df = pd.DataFrame(data)
# df = df.sample(frac=1, random_state=42).reset_index(drop=True)
# df_publishable = df[df['Publishable'] == 1]
# df_non_publishable = df[df['Publishable'] == 0]

# df_publishable_downsampled = df_publishable.sample(df_non_publishable.shape[0], random_state=42)
# df_balanced = pd.concat([df_publishable_downsampled, df_non_publishable])
# df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# X = df_balanced.iloc[:, 0:1]
# y = df_balanced['Publishable'].to_numpy(dtype=np.int32)
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
# cv = CountVectorizer()
# X_train_bow = cv.fit_transform(X_train['Text']).toarray()
# X_test_bow = cv.transform(X_test['Text']).toarray()

# # Training classifiers
# gnb = GaussianNB()
# gnb.fit(X_train_bow, y_train)

# clf = LogisticRegression(random_state=42)
# clf.fit(X_train_bow, y_train)

# svm = SVC(kernel='linear', random_state=42)
# svm.fit(X_train_bow, y_train)

# voting_clf = VotingClassifier(estimators=[('gnb', gnb), ('lr', clf), ('svm', svm)], voting='hard')
# voting_clf.fit(X_train_bow, y_train)

# # Pathway and server setup
# from pathway.xpacks.llm.embedders import GeminiEmbedder
# from pathway.xpacks.llm.splitters import TokenCountSplitter
# from pathway.xpacks.llm.vector_store import VectorStoreClient, VectorStoreServer
# from pathway.xpacks.llm.parsers import ParseUnstructured

# text_splitter = TokenCountSplitter()
# embedder = GeminiEmbedder(api_key=GEMINI_API_KEY)
# parser = ParseUnstructured(mode='single', post_processors=[preprocessed_text])

# # Pathway Reference Folders and Server Setup
# conference_folders = {
#     "CVPR": r"Reference/Publishable/CVPR",
#     "EMNLP": r"Reference/Publishable/EMNLP",
#     "KDD": r"Reference/Publishable/KDD",
#     "NeurIPS": r"Reference/Publishable/NeurIPS",
#     "TMLR": r"Reference/Publishable/TMLR"
# }

# reference_sources = []
# for conference_name, folder_path in conference_folders.items():
#     table = pw.io.fs.read(
#         path=folder_path + "/*.pdf",  # Glob pattern to match all PDF files
#         format="binary",
#         with_metadata=True,
#         mode="static",  # Static mode to ingest data once
#     )
#     reference_sources.append(table)

# vector_server = VectorStoreServer(
#     *reference_sources,
#     parser=parser,
#     embedder=embedder,
#     splitter=text_splitter,
# )


# # No stop_server() function is available, so we ensure no thread starts during shutdown
# def classify_paper(paper_text,conferences, genai_model):
#     # Perform similarity search
#     result = resource_client.query(query=[paper_text], k=1)
    
#     if not result:
#         return "Could not classify the paper.", ""

#     closest_match = result[0]
#     metadata = closest_match["metadata"]
#     # classification = metadata["conference"]
#     # # Extract relevant passage
#     relevant_passages = closest_match["text"]
#     passage_oneline = " ".join(relevant_passages).replace("\n", " ")

#     # Generate classification and rationale
#     prompt = (
#         f"Classify the following paper into one of these conferences: {', '.join(conferences)}.\n"
#         f"Paper Content (trimmed): {paper_text}...\n"
#         f"Relevant Passage: {passage_oneline}\n"
#         f"Provide the closest match classification from the listed conferences. The classification must not be 'None of the above'\n"
#         f"Additionally, provide a separate rationale for the classification (not more than 100 words). Start your rationale with 'Rationale:'."
# )


#     response = genai_model.generate_content(prompt).parts[0].text

#     # Split response into classification and rationale
#     if "Rationale:" in response:
#         classification, rationale = response.split("Rationale:", maxsplit=1)
#         classification = classification.replace("**Classification:**", "").replace("**","").strip()
#         rationale = rationale.replace("**", "").strip()
#     else:
#         classification = response.strip()
#         rationale = "Rationale not provided."

#     return classification, rationale


# genai.configure(api_key=GEMINI_API_KEY)

# genai_model = genai.GenerativeModel("gemini-1.5-pro")

# def classify_rationale(paper_path):
#     if not os.path.exists(paper_path) or not paper_path.endswith(".pdf"):
#         print("Invalid file. Please try again.")
    
#     with open(paper_path, "rb") as file:
#         pdf_reader = PdfReader(file)
#         paper_text = " ".join(page.extract_text() for page in pdf_reader.pages)
#         classification, rationale = classify_paper(paper_text, list(conference_folders.keys()), genai_model)
#     return classification,rationale



# def run_vector_server():
#     # The server is now a long-running process; ensure the program doesn't try to restart it.
#     vector_server.run_server(host="127.0.0.1", port=8000)

# vector_server_thread = threading.Thread(target=run_vector_server)
# vector_server_thread.daemon=True
# vector_server_thread.start()
# time.sleep(5)
# resource_client= VectorStoreClient(host="127.0.0.1", port=8000)
# # # Run vector server and handle graceful termination
# # if __name__ == "__main__":
# #     try:
# #         vector_server_thread = threading.Thread(target=run_vector_server)
# #         vector_server_thread.start()

# #         # paper_path = "sample_paper.pdf"
# #         # classification, rationale = classify_paper(paper_path, list(conference_folders.keys()), genai_model)
# #         # print(f"Classification: {classification}")
# #         # print(f"Rationale: {rationale}")
        
# #     except Exception as e:
# #         print(f"Error occurred: {str(e)}")
    
# #     finally:
# #         # Wait for the thread to finish before exit
# #         if vector_server_thread.is_alive():
# #             vector_server_thread.join()

# #------------------------------------------------------------------------------------------------------

# # Apply custom CSS for better UI
# st.markdown(
#     """
#     <style>
#     body {
#         background: linear-gradient(to bottom, #1c1c1c, #2c3e50);
#         color: #fff;
#         font-family: 'Helvetica', sans-serif;
#     }
#     h1 {
#         font-family: 'Montserrat', sans-serif;
#         color: #00c853;
#         text-align: center;
#         margin-bottom: 0.5em;
#     }
#     h3 {
#         font-family: 'Montserrat', sans-serif;
#     }
#     .file-uploader {
#         text-align: center;
#     }
#     .result-box {
#         background-color: #333;
#         padding: 20px;
#         border-radius: 10px;
#         margin-top: 20px;
#         text-align: center;
#     }
#     /* Target Conference Styling */
#     .target-conference {
#         font-size: 24px;
#         font-weight: bold;
#         color: #ffd700; /* Gold color for Target Conference */
#         text-align: center;
#         margin-top: 25px;
#     }
#     .conference-name {
#         color: #1e90ff; /* Dodger Blue for Conference Name */
#         font-weight: bold;
#     }
#     /* Rationale Styling */
#     .rationale-header {
#         font-size: 20px;
#         font-weight: bold;
#         color: #ff4500; /* Orange-Red for Rationale Header */
#     }
#     .pin-icon {
#         font-size: 26px; /* Adjust size of the pin */
#         color: #e74c3c; /* Red for the pin */
#         transform: rotate(45deg); /* Rotate the pin slightly */
#     }
#     .rationale-text {
#         font-size: 18px;
#         color: #ffffff; /* White for Rationale Text */
#         background-color: #444;
#         padding: 10px;
#         border-radius: 8px;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )
# # App Title
# st.title("📚 Research Paper Classifier and Publishability Checker")
# st.write(
#     "<h4 style='text-align: center; color: #ddd;'>Upload a research paper PDF to classify its publishability and target conference.</h4>",
#     unsafe_allow_html=True,
# )

# # File Upload
# uploaded_file = st.file_uploader("Upload Your Research Paper (PDF)", type="pdf", label_visibility="collapsed")

# if uploaded_file:
#     # Display the upload button
#     st.markdown("<div class='file-uploader'>📂 File Selected: Click 'Upload Paper' to process.</div>", unsafe_allow_html=True)
#     if st.button("Upload Paper"):
#         with st.spinner("Processing your file... ⏳"):
#             # Save uploaded file
#             temp_dir = "uploaded_files"
#             os.makedirs(temp_dir, exist_ok=True)
#             file_path = os.path.join(temp_dir, uploaded_file.name)
#             with open(file_path, "wb") as f:
#                 f.write(uploaded_file.getbuffer())
            
#             # Extract text from PDF
#             pdf_reader = PdfReader(file_path)
#             paper_text = " ".join(page.extract_text() for page in pdf_reader.pages)
            
#             # Preprocess and classify
#             processed_text = preprocessed_text(paper_text)
#             processed_text_bow = cv.transform([processed_text]).toarray()
#             publishable = voting_clf.predict(processed_text_bow)[0]

#             # Display results dynamically
#             if publishable:
#                 st.markdown(
#                     """
#                     <div class='result-box'>
#                         <h3 style='color: #4CAF50;'>✅ Publishable</h3>
#                         <p style='font-size: 20px; color: #fff;'>The paper is likely <strong>Publishable</strong>.</p>
#                     </div>
#                     """,
#                     unsafe_allow_html=True,
#                 )

#                 # Generate classification and rationale
#                 classification, rationale = classify_rationale(file_path)
#                 st.markdown(
#                     f"<div class='target-conference'>"
#                     f"<span class='pin-icon'>&#128204;</span>"
#                     f"Target Conference: <span class='conference-name'>{classification}</span>",
#                     unsafe_allow_html=True,
#                 )
#                 st.markdown(
#                     f"<div class='rationale-header'>Rationale:</div>"
#                     f"<div class='rationale-text'>{rationale}</div>",
#                     unsafe_allow_html=True,
#                 )
#             else:
#                 st.markdown(
#                     """
#                     <div class='result-box'>
#                         <h3 style='color: #FF5252;'>❌ Not Publishable</h3>
#                         <p style='font-size: 20px; color: #fff;'>The paper is likely <strong>Not Publishable</strong>.</p>
#                     </div>
#                     """,
#                     unsafe_allow_html=True,
#                 )

#         # Offer download option
#         st.write("📥 **Download the Results**")
#         df = pd.DataFrame(
#             [[uploaded_file.name, publishable, classification if publishable else "N/A", rationale if publishable else "N/A"]],
#             columns=["Paper Name", "Publishable", "Conference", "Rationale"]
#         )
#         st.download_button(
#             label="Download Results as CSV",
#             data=df.to_csv(index=False),
#             file_name="classification_results.csv",
#             mime="text/csv"
#         )

# # Sidebar Information
# st.sidebar.title("About This App")
# st.sidebar.info(
#     """
#     This application processes research papers to:
#     - Check publishability.
#     - Classify papers into target conferences.
#     """
# )
# st.sidebar.markdown(
#     """
#     **Technologies Used:**
#     - Streamlit for the user interface.
#     - Machine Learning for classification.
#     """
# )




import os
import streamlit as st
import numpy as np
import pandas as pd
import re
import string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
import subprocess
import time
import requests
import google.generativeai as genai
import socket
import threading

# Suppress unnecessary warnings and logs
import logging
logging.getLogger('pathway_engine').setLevel(logging.WARNING)
logging.getLogger('aiohttp.access').setLevel(logging.WARNING)
logging.getLogger('root').setLevel(logging.WARNING)
logging.basicConfig(level=logging.WARNING)

# Constants
GEMINI_API_KEY = "AIzaSyBsjjQRqoo40I6pxLHJ4zpukOt5e1lg8C0"
VECTOR_SERVER_HOST = "127.0.0.1"
VECTOR_SERVER_PORT = 8000
VECTOR_SERVER_SCRIPT = "vector_server_script.py" # Replace with your actual script name

# Initialize environment
os.environ["TOKENIZERS_PARALLELISM"] = "false"
nltk_data_path = os.path.expanduser('~/nltk_data')
if not os.path.exists(nltk_data_path):
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
exclude = string.punctuation
lemmatizer = WordNetLemmatizer()

# Utility functions
def text_extractor_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def lemmatize_words(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

def remove_stopwords(text):
    return " ".join(word for word in text.split() if word not in stopwords.words('english'))

def preprocessed_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = text.translate(str.maketrans('', '', exclude))
    text = remove_stopwords(text)
    text = lemmatize_words(text)
    return text

# Data Preparation
non_publishable_folder = 'Reference/Non-Publishable/'
non_publishable_texts = [preprocessed_text(text_extractor_from_pdf(os.path.join(non_publishable_folder, file)))
                         for file in os.listdir(non_publishable_folder) if file.endswith('.pdf')]

publishable_folders = ['Reference/Publishable/CVPR/', 'Reference/Publishable/EMNLP/', 
                       'Reference/Publishable/KDD/', 'Reference/Publishable/NeurIPS/', 
                       'Reference/Publishable/TMLR/']

publishable_texts = []
for folder in publishable_folders:
    publishable_texts += [preprocessed_text(text_extractor_from_pdf(os.path.join(folder, file)))
                          for file in os.listdir(folder) if file.endswith('.pdf')]

data = {"Text": publishable_texts + non_publishable_texts,
        "Publishable": [1] * len(publishable_texts) + [0] * len(non_publishable_texts)}
df = pd.DataFrame(data).sample(frac=1, random_state=42).reset_index(drop=True)

# Model Training
X_train, X_test, y_train, y_test = train_test_split(df['Text'], df['Publishable'], random_state=1)
cv = CountVectorizer()
X_train_bow = cv.fit_transform(X_train).toarray()
X_test_bow = cv.transform(X_test).toarray()

gnb = GaussianNB().fit(X_train_bow, y_train)
clf = LogisticRegression(random_state=42).fit(X_train_bow, y_train)
svm = SVC(kernel='linear', random_state=42).fit(X_train_bow, y_train)
voting_clf = VotingClassifier(estimators=[('gnb', gnb), ('lr', clf), ('svm', svm)], voting='hard').fit(X_train_bow, y_train)

conference_folders = {
    "CVPR": r"Reference/Publishable/CVPR",
    "EMNLP": r"Reference/Publishable/EMNLP",
    "KDD": r"Reference/Publishable/KDD",
    "NeurIPS": r"Reference/Publishable/NeurIPS",
    "TMLR": r"Reference/Publishable/TMLR"
}
# Server Management Functions
def is_server_running(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0

def start_vector_server():
    if not is_server_running(VECTOR_SERVER_HOST, VECTOR_SERVER_PORT):
        subprocess.Popen(
            ["python", VECTOR_SERVER_SCRIPT],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        # Wait for the server to start
        start_time = time.time()
        while not is_server_running(VECTOR_SERVER_HOST, VECTOR_SERVER_PORT):
            if time.time() - start_time > 30:
                raise TimeoutError("Vector server did not start in time.")
            time.sleep(1)

# Classification Functions
def classify_paper(paper_text, conferences, genai_model):
    result = requests.post(
        f"http://{VECTOR_SERVER_HOST}:{VECTOR_SERVER_PORT}/v1/retrieve",
        json={"query": [paper_text], "k": 1},
        timeout=60
    ).json()

    if not result or "text" not in result[0]:
        return "Could not classify the paper.", ""

    relevant_passages = result[0]["text"]
    passage_oneline = " ".join(relevant_passages).replace("\n", " ")

    prompt = (
        f"Classify the following paper into one of these conferences: {', '.join(conferences)}.\n"
        f"Paper Content (trimmed): {paper_text}...\n"
        f"Relevant Passage: {passage_oneline}\n"
        f"Provide the closest match classification and a rationale (not more than 100 words)."
    )

    response = genai_model.generate_content(prompt).parts[0].text

    if "Rationale:" in response:
        classification, rationale = response.split("Rationale:", maxsplit=1)
        classification = classification.replace("**Classification:**", "").replace("**","").strip()
        rationale = rationale.replace("**", "").strip()
    else:
        classification = response.strip()
        rationale = "Rationale not provided."

    return classification, rationale

def classify_rationale(paper_path):
    if not os.path.exists(paper_path) or not paper_path.endswith(".pdf"):
        print("Invalid file. Please try again.")
    
    with open(paper_path, "rb") as file:
        pdf_reader = PdfReader(file)
        paper_text = " ".join(page.extract_text() for page in pdf_reader.pages)
        classification, rationale = classify_paper(paper_text, list(conference_folders.keys()), genai_model)
    return classification,rationale

# Initialize Gemini API
genai.configure(api_key=GEMINI_API_KEY)
genai_model = genai.GenerativeModel("gemini-1.5-pro")

# Start Server
try:
    start_vector_server()
except Exception as e:
    st.error(f"Failed to start vector server: {str(e)}")




# Streamlit UI
st.title("📚 Research Paper Classifier and Publishability Checker")
st.write("Upload a research paper PDF to classify its publishability and target conference.")

uploaded_file = st.file_uploader("Upload Your Research Paper (PDF)", type="pdf")

if uploaded_file:
    if st.button("Upload Paper"):
        with st.spinner("Processing..."):
            temp_dir = "uploaded_files"
            os.makedirs(temp_dir, exist_ok=True)
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            processed_text = preprocessed_text(text_extractor_from_pdf(file_path))
            publishable = voting_clf.predict(cv.transform([processed_text]).toarray())[0]

            if publishable:
                st.success("✅ Publishable")
                classification, rationale = classify_rationale(file_path)
                st.write(f"**Target Conference:** {classification}")
                st.write(f"**Rationale:** {rationale}")
            else:
                st.error("❌ Not Publishable")



# # Apply custom CSS for better UI
# st.markdown(
#     """
#     <style>
#     body {
#         background: linear-gradient(to bottom, #1c1c1c, #2c3e50);
#         color: #fff;
#         font-family: 'Helvetica', sans-serif;
#     }
#     h1 {
#         font-family: 'Montserrat', sans-serif;
#         color: #00c853;
#         text-align: center;
#         margin-bottom: 0.5em;
#     }
#     h3 {
#         font-family: 'Montserrat', sans-serif;
#     }
#     .file-uploader {
#         text-align: center;
#     }
#     .result-box {
#         background-color: #333;
#         padding: 20px;
#         border-radius: 10px;
#         margin-top: 20px;
#         text-align: center;
#     }
#     /* Target Conference Styling */
#     .target-conference {
#         font-size: 24px;
#         font-weight: bold;
#         color: #ffd700; /* Gold color for Target Conference */
#         text-align: center;
#         margin-top: 25px;
#     }
#     .conference-name {
#         color: #1e90ff; /* Dodger Blue for Conference Name */
#         font-weight: bold;
#     }
#     /* Rationale Styling */
#     .rationale-header {
#         font-size: 20px;
#         font-weight: bold;
#         color: #ff4500; /* Orange-Red for Rationale Header */
#     }
#     .pin-icon {
#         font-size: 26px; /* Adjust size of the pin */
#         color: #e74c3c; /* Red for the pin */
#         transform: rotate(45deg); /* Rotate the pin slightly */
#     }
#     .rationale-text {
#         font-size: 18px;
#         color: #ffffff; /* White for Rationale Text */
#         background-color: #444;
#         padding: 10px;
#         border-radius: 8px;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )
# # App Title
# st.title("📚 Research Paper Classifier and Publishability Checker")
# st.write(
#     "<h4 style='text-align: center; color: #ddd;'>Upload a research paper PDF to classify its publishability and target conference.</h4>",
#     unsafe_allow_html=True,
# )

# # File Upload
# uploaded_file = st.file_uploader("Upload Your Research Paper (PDF)", type="pdf", label_visibility="collapsed")

# if uploaded_file:
#     # Display the upload button
#     st.markdown("<div class='file-uploader'>📂 File Selected: Click 'Upload Paper' to process.</div>", unsafe_allow_html=True)
#     if st.button("Upload Paper"):
#         with st.spinner("Processing your file... ⏳"):
#             # Save uploaded file
#             temp_dir = "uploaded_files"
#             os.makedirs(temp_dir, exist_ok=True)
#             file_path = os.path.join(temp_dir, uploaded_file.name)
#             with open(file_path, "wb") as f:
#                 f.write(uploaded_file.getbuffer())
            
#             # Extract text from PDF
#             pdf_reader = PdfReader(file_path)
#             paper_text = " ".join(page.extract_text() for page in pdf_reader.pages)
            
#             # Preprocess and classify
#             processed_text = preprocessed_text(paper_text)
#             processed_text_bow = cv.transform([processed_text]).toarray()
#             publishable = voting_clf.predict(processed_text_bow)[0]

#             # Display results dynamically
#             if publishable:
#                 st.markdown(
#                     """
#                     <div class='result-box'>
#                         <h3 style='color: #4CAF50;'>✅ Publishable</h3>
#                         <p style='font-size: 20px; color: #fff;'>The paper is likely <strong>Publishable</strong>.</p>
#                     </div>
#                     """,
#                     unsafe_allow_html=True,
#                 )

#                 # Generate classification and rationale
#                 classification, rationale = classify_rationale(file_path)
#                 st.markdown(
#                     f"<div class='target-conference'>"
#                     f"<span class='pin-icon'>&#128204;</span>"
#                     f"Target Conference: <span class='conference-name'>{classification}</span>",
#                     unsafe_allow_html=True,
#                 )
#                 st.markdown(
#                     f"<div class='rationale-header'>Rationale:</div>"
#                     f"<div class='rationale-text'>{rationale}</div>",
#                     unsafe_allow_html=True,
#                 )
#             else:
#                 st.markdown(
#                     """
#                     <div class='result-box'>
#                         <h3 style='color: #FF5252;'>❌ Not Publishable</h3>
#                         <p style='font-size: 20px; color: #fff;'>The paper is likely <strong>Not Publishable</strong>.</p>
#                     </div>
#                     """,
#                     unsafe_allow_html=True,
#                 )

#         # Offer download option
#         st.write("📥 **Download the Results**")
#         df = pd.DataFrame(
#             [[uploaded_file.name, publishable, classification if publishable else "N/A", rationale if publishable else "N/A"]],
#             columns=["Paper Name", "Publishable", "Conference", "Rationale"]
#         )
#         st.download_button(
#             label="Download Results as CSV",
#             data=df.to_csv(index=False),
#             file_name="classification_results.csv",
#             mime="text/csv"
#         )

# # Sidebar Information
# st.sidebar.title("About This App")
# st.sidebar.info(
#     """
#     This application processes research papers to:
#     - Check publishability.
#     - Classify papers into target conferences.
#     """
# )
# st.sidebar.markdown(
#     """
#     **Technologies Used:**
#     - Streamlit for the user interface.
#     - Machine Learning for classification.
#     """
# )
# #------------------------------------------------------------------------------------------
