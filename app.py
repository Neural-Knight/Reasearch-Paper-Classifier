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
from pathway.xpacks.llm.embedders import GeminiEmbedder
from pathway.xpacks.llm.splitters import TokenCountSplitter
from pathway.xpacks.llm.vector_store import VectorStoreClient, VectorStoreServer
from pathway.xpacks.llm.parsers import ParseUnstructured
import time
import requests
import google.generativeai as genai
import json
from google.api_core import retry
import threading
import logging
import warnings
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv
from model import voting_clf,preprocessed_text,cv

# Suppress logs from specific libraries
logging.getLogger('pathway_engine').setLevel(logging.WARNING)
logging.getLogger('aiohttp.access').setLevel(logging.WARNING)
logging.getLogger('root').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)
logging.basicConfig(level=logging.WARNING)

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

conference_folders = {
    "CVPR": r"Reference/Publishable/CVPR",
    "EMNLP": r"Reference/Publishable/EMNLP",
    "KDD": r"Reference/Publishable/KDD",
    "NeurIPS": r"Reference/Publishable/NeurIPS",
    "TMLR": r"Reference/Publishable/TMLR"
}
class VectorStoreManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VectorStoreManager, cls).__new__(cls)
            cls._instance.server = None
            cls._instance.client = None
            cls._instance.server_thread = None
            cls._instance.is_running = False
            cls._instance.setup_vector_store()
        return cls._instance

    def setup_vector_store(self):
        if not self.is_running:
            text_splitter = TokenCountSplitter()
            embedder = GeminiEmbedder(api_key=GEMINI_API_KEY)
            parser = ParseUnstructured(mode='single', post_processors=[preprocessed_text])

            reference_sources = []
            for conference_name, folder_path in conference_folders.items():
                table = pw.io.fs.read(
                    path=folder_path + "/*.pdf",
                    format="binary",
                    with_metadata=True,
                    mode="static",
                )
                reference_sources.append(table)

            self.server = VectorStoreServer(
                *reference_sources,
                parser=parser,
                embedder=embedder,
                splitter=text_splitter,
            )

    def start_server(self):
        if not self.is_running:
            try:
                self.server_thread = threading.Thread(target=self._run_server)
                self.server_thread.daemon = True
                self.server_thread.start()
                time.sleep(2)  # Wait for server to start
                self.client = VectorStoreClient(host="127.0.0.1", port=8000, timeout=30)
                self.is_running = True
            except Exception as e:
                st.error(f"Error starting server: {str(e)}")
                self.is_running = False

    def _run_server(self):
        try:
            self.server.run_server(host="127.0.0.1", port=8000)
        except OSError as e:
            if e.errno == 98:  # Address already in use
                st.warning("Server is already running")
                self.is_running = True
            else:
                raise e

    def restart_server(self):
        if self.is_running:
            self.is_running = False
            time.sleep(1)
            self.setup_vector_store()
            self.start_server()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def query(self, text):
        if not self.is_running:
            self.start_server()
        try:
            return self.client.query(query=[text], k=1)
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError):
            self.restart_server()
            return self.client.query(query=[text], k=1)
        
# Initialize vector store manager
vector_store_manager = VectorStoreManager()
vector_store_manager.start_server()

# Modified classification functions
def classify_paper(paper_text, conferences, genai_model):
    try:
        result = vector_store_manager.query(paper_text)
        
        if not result:
            return "Could not classify the paper.", ""

        closest_match = result[0]
        metadata = closest_match["metadata"]
        relevant_passages = closest_match["text"]
        passage_oneline = " ".join(relevant_passages).replace("\n", " ")

        prompt = (
            f"Classify the following paper into one of these conferences: {', '.join(conferences)}.\n"
            f"Paper Content (trimmed): {paper_text}...\n"
            f"Relevant Passage: {passage_oneline}\n"
            f"Provide the closest match classification from the listed conferences. The classification must not be 'None of the above'\n"
            f"Additionally, provide a separate rationale for the classification (not more than 100 words). Start your rationale with 'Rationale:'."
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
    except Exception as e:
        st.error(f"Error during classification: {str(e)}")
        return "Classification failed", "An error occurred during the classification process."

def classify_rationale(paper_path):
    if not os.path.exists(paper_path) or not paper_path.endswith(".pdf"):
        return "Invalid file", "Please upload a valid PDF file."
    
    try:
        with open(paper_path, "rb") as file:
            pdf_reader = PdfReader(file)
            paper_text = " ".join(page.extract_text() for page in pdf_reader.pages)
            return classify_paper(paper_text, list(conference_folders.keys()), genai_model)
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return "Processing failed", "An error occurred while processing the PDF file."

# Initialize Gemini
genai.configure(api_key=GEMINI_API_KEY)
genai_model = genai.GenerativeModel("gemini-1.5-pro")

#------------------------------------------------------------------------------------------------------

# Apply custom CSS for better UI
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(to bottom, #1c1c1c, #2c3e50);
        color: #fff;
        font-family: 'Helvetica', sans-serif;
    }
    h1 {
        font-family: 'Montserrat', sans-serif;
        color: #00c853;
        text-align: center;
        margin-bottom: 0.5em;
    }
    h3 {
        font-family: 'Montserrat', sans-serif;
    }
    .file-uploader {
        text-align: center;
    }
    .result-box {
        background-color: #333;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        text-align: center;
    }
    /* Target Conference Styling */
    .target-conference {
        font-size: 24px;
        font-weight: bold;
        color: #ffd700; /* Gold color for Target Conference */
        text-align: center;
        margin-top: 25px;
    }
    .conference-name {
        color: #1e90ff; /* Dodger Blue for Conference Name */
        font-weight: bold;
    }
    /* Rationale Styling */
    .rationale-header {
        font-size: 20px;
        font-weight: bold;
        color: #ff4500; /* Orange-Red for Rationale Header */
    }
    .pin-icon {
        font-size: 26px; /* Adjust size of the pin */
        color: #e74c3c; /* Red for the pin */
        transform: rotate(45deg); /* Rotate the pin slightly */
    }
    .rationale-text {
        font-size: 18px;
        color: #ffffff; /* White for Rationale Text */
        background-color: #444;
        padding: 10px;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# App Title
st.title("📚 Research Paper Classifier and Publishability Checker")
st.write(
    "<h4 style='text-align: center; color: #ddd;'>Upload a research paper PDF to classify its publishability and target conference.</h4>",
    unsafe_allow_html=True,
)

# File Upload
uploaded_file = st.file_uploader("Upload Your Research Paper (PDF)", type="pdf", label_visibility="collapsed")

if uploaded_file:
    # Display the upload button
    st.markdown("<div class='file-uploader'>📂 File Selected: Click 'Upload Paper' to process.</div>", unsafe_allow_html=True)
    if st.button("Upload Paper"):
        with st.spinner("Processing your file... ⏳"):
            # Save uploaded file
            temp_dir = "uploaded_files"
            os.makedirs(temp_dir, exist_ok=True)
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Extract text from PDF
            pdf_reader = PdfReader(file_path)
            paper_text = " ".join(page.extract_text() for page in pdf_reader.pages)
            
            # Preprocess and classify
            processed_text = preprocessed_text(paper_text)
            processed_text_bow = cv.transform([processed_text]).toarray()
            publishable = voting_clf.predict(processed_text_bow)[0]

            # Display results dynamically
            if publishable:
                st.markdown(
                    """
                    <div class='result-box'>
                        <h3 style='color: #4CAF50;'>✅ Publishable</h3>
                        <p style='font-size: 20px; color: #fff;'>The paper is likely <strong>Publishable</strong>.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Generate classification and rationale
                classification, rationale = classify_rationale(file_path)
                st.markdown(
                    f"<div class='target-conference'>"
                    f"<span class='pin-icon'>&#128204;</span>"
                    f"Target Conference: <span class='conference-name'>{classification}</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div class='rationale-header'>Rationale:</div>"
                    f"<div class='rationale-text'>{rationale}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div class='result-box'>
                        <h3 style='color: #FF5252;'>❌ Not Publishable</h3>
                        <p style='font-size: 20px; color: #fff;'>The paper is likely <strong>Not Publishable</strong>.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Offer download option
        st.write("📥 **Download the Results**")
        df = pd.DataFrame(
            [[uploaded_file.name, publishable, classification if publishable else "N/A", rationale if publishable else "N/A"]],
            columns=["Paper Name", "Publishable", "Conference", "Rationale"]
        )
        st.download_button(
            label="Download Results as CSV",
            data=df.to_csv(index=False),
            file_name="classification_results.csv",
            mime="text/csv"
        )

# Sidebar Information
st.sidebar.title("About This App")
st.sidebar.info(
    """
    This application processes research papers to:
    - Check publishability.
    - Classify papers into target conferences.
    """
)
st.sidebar.markdown(
    """
    **Technologies Used:**
    - Streamlit for the user interface.
    - Machine Learning for classification.
    """
)
