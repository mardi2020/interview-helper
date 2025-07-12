import os

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()
MODEL = os.getenv('MODEL')
EMBEDDING = os.getenv('EMBEDDING')

DB_PATH = "vectorstore/index"


def get_llm():
    return ChatOpenAI(
        model=MODEL,
        temperature=0
    )


def get_embeddings():
    return OpenAIEmbeddings(model=EMBEDDING)
