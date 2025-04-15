import streamlit as st
import chardet
import nltk
import fitz  # PyMuPDF for extracting text from PDFs
import google.generativeai as genai
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from dotenv import load_dotenv
import os

load_dotenv()
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
# Ensure NLTK is correctly set up
nltk.data.path.append("C:/Users/cbec/AppData/Roaming/nltk_data")
nltk.download("punkt", quiet=True)  # Download silently

# Configure Gemini API with API key in code
genai.configure(api_key=GOOGLE_API_KEY)  # Replace with your actual API key
model = genai.GenerativeModel("gemini-1.5-pro-latest")

def extractive_summary(text, num_sentences=10):
    """Perform extractive summarization using LexRank."""
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LexRankSummarizer()
        summary = summarizer(parser.document, num_sentences)
        return " ".join(str(sentence) for sentence in summary)
    except Exception as e:
        return f"Extractive Summarization Error: {e}"

def abstractive_summary(text):
    """Perform abstractive summarization using Gemini API."""
    try:
        response = model.generate_content(f"Summarize the following text:\n\n{text}")
        return response.text.strip() if response.text else "Abstractive summarization failed."
    except Exception as e:
        return f"Abstractive Summarization Error: {e}"

def hybrid_summarization(text):
    """Combine extractive and abstractive summarization."""
    extracted_text = extractive_summary(text)
    if not extracted_text.strip():
        return "Extractive summarization returned an empty result."
    return abstractive_summary(extracted_text)

def extract_text_from_pdf(pdf_bytes):
    """Extract text from a PDF file."""
    try:
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = "\n".join([page.get_text("text") for page in pdf_document])
        return text.strip() if text else None
    except Exception as e:
        return f"PDF Extraction Error: {e}"

# Streamlit UI
st.title("Hybrid Text Summarizer")
uploaded_file = st.file_uploader("Upload a PDF or Text File", type=["pdf", "txt"])

if uploaded_file is not None:
    raw_bytes = uploaded_file.read()
    
    if uploaded_file.type == "application/pdf":
        # Extract text from PDF
        raw_text = extract_text_from_pdf(raw_bytes)
    else:
        # Detect file encoding
        result = chardet.detect(raw_bytes)
        encoding = result.get("encoding") or "utf-8"

        try:
            # Decode text with proper encoding
            raw_text = raw_bytes.decode(encoding, errors="replace")
        except Exception as e:
            st.error(f"Error processing file: {e}")
            raw_text = None

    if raw_text:
        summary = hybrid_summarization(raw_text)
        st.subheader("Summary:")
        st.write(summary)
    else:
        st.error("The uploaded file contains no readable text.")
