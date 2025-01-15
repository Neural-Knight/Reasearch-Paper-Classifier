import streamlit as st
import os
from PyPDF2 import PdfReader
import pandas as pd
from pipeline import preprocessed_text, classify_rationale, cv, voting_clf

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