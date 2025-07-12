"""
Module for loading and splitting uploaded text or PDF files into smaller chunks
using LangChain document loaders and text splitter.
"""
import tempfile
import os

from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_and_split_file(uploaded_file):
    """
    Load an uploaded text or PDF file, then split its content into smaller chunks.

    The function:
    - Saves the uploaded file temporarily
    - Uses appropriate LangChain document loader based on file extension
    - Splits the loaded document into chunks using RecursiveCharacterTextSplitter

    Args:
        uploaded_file: A file-like object uploaded via Streamlit or similar,
                       with attributes like `.name` and `.read()` method.

    Returns:
        List of document chunks split by RecursiveCharacterTextSplitter.

    Raises:
        ValueError: If the uploaded file type is not supported (not pdf, txt, or md).
    """
    suffix = os.path.splitext(uploaded_file.name)[-1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    if suffix == ".pdf":
        loader = PyMuPDFLoader(tmp_path)
    elif suffix in [".txt", ".md"]:
        loader = TextLoader(tmp_path)
    else:
        raise ValueError("지원하지 않는 파일 형식입니다.")

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_documents(documents)
