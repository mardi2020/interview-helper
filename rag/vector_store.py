from langchain_community.vectorstores import FAISS

import os

from config.parameters import DB_PATH, get_embeddings


def save_to_faiss(documents):
    db = FAISS.from_documents(documents, embedding=get_embeddings())
    db.save_local(DB_PATH)


def load_faiss():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError("FAISS 인덱스가 존재하지 않습니다. 문서를 먼저 업로드하세요.")

    return FAISS.load_local(
        folder_path=DB_PATH,
        embeddings=get_embeddings(),
        allow_dangerous_deserialization=True
    )