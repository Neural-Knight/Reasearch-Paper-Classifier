import os
import logging
import pathway as pw
from pathway.xpacks.llm.embedders import GeminiEmbedder
from pathway.xpacks.llm.splitters import TokenCountSplitter
from pathway.xpacks.llm.vector_store import VectorStoreServer
from pathway.xpacks.llm.parsers import ParseUnstructured

# Suppress logs from specific libraries
logging.getLogger('pathway_engine').setLevel(logging.WARNING)

# Define constants
GEMINI_API_KEY = "AIzaSyBsjjQRqoo40I6pxLHJ4zpukOt5e1lg8C0"
VECTOR_SERVER_HOST = "127.0.0.1"
VECTOR_SERVER_PORT = 8000

# Function to initialize and start the VectorStoreServer
def start_vector_server():
    # Path to the folders containing reference PDFs for each conference
    conference_folders = {
        "CVPR": r"Reference/Publishable/CVPR",
        "EMNLP": r"Reference/Publishable/EMNLP",
        "KDD": r"Reference/Publishable/KDD",
        "NeurIPS": r"Reference/Publishable/NeurIPS",
        "TMLR": r"Reference/Publishable/TMLR"
    }

    # Initialize components
    text_splitter = TokenCountSplitter()
    embedder = GeminiEmbedder(api_key=GEMINI_API_KEY)
    parser = ParseUnstructured(mode='single')

    # Read reference data
    reference_sources = []
    for folder_name, folder_path in conference_folders.items():
        table = pw.io.fs.read(
            path=os.path.join(folder_path, "*.pdf"),
            format="binary",
            with_metadata=True,
            mode="static",  # Load the data only once
        )
        reference_sources.append(table)

    # Initialize VectorStoreServer
    vector_server = VectorStoreServer(
        *reference_sources,
        parser=parser,
        embedder=embedder,
        splitter=text_splitter,
    )

    # Start the server
    vector_server.run_server(host=VECTOR_SERVER_HOST, port=VECTOR_SERVER_PORT)
    
if __name__ == "__main__":
    try:
        print(f"Starting VectorStoreServer on {VECTOR_SERVER_HOST}:{VECTOR_SERVER_PORT}...")
        start_vector_server()
    except Exception as e:
        print(f"Failed to start VectorStoreServer: {str(e)}")