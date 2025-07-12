"""
Module for saving and loading document embeddings using FAISS vector store.
"""

import os

from langchain_community.vectorstores import FAISS

from config.parameters import DB_PATH, get_embeddings


def save_to_faiss(documents):
    """
    Save a list of documents to a local FAISS vector store.

    Args:
        documents (list): List of documents to be embedded and saved.
    """

    db = FAISS.from_documents(documents, embedding=get_embeddings())
    db.save_local(DB_PATH)


def load_faiss():
    """
    Load the FAISS vector store from the local directory.

    Returns:
        FAISS: Loaded FAISS vector store instance.

    Raises:
        FileNotFoundError: If the FAISS index does not exist locally.
    """
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError("FAISS 인덱스가 존재하지 않습니다. 문서를 먼저 업로드하세요.")

    return FAISS.load_local(
        folder_path=DB_PATH,
        embeddings=get_embeddings(),
        allow_dangerous_deserialization=True
        )
