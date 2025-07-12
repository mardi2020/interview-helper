from langchain.document_loaders import PyMuPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
import os


def load_and_split_file(uploaded_file):
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
