"""
🌟 This module handles initialization of the OpenAI language model
and embedding model instances used for the interview helper chatbot.

It loads model configuration from environment variables and provides
convenient functions to get the ChatOpenAI and OpenAIEmbeddings objects.
"""

import os

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()
MODEL = os.getenv('MODEL')
EMBEDDING = os.getenv('EMBEDDING')

DB_PATH = "vectorstore/index"


def get_llm():
    """
    Create and return a ChatOpenAI instance with specified model and zero temperature.
    """
    return ChatOpenAI(
        model=MODEL,
        temperature=0
    )


def get_embeddings():
    """
    Create and return an OpenAIEmbeddings instance with the specified embedding model.
    """
    return OpenAIEmbeddings(model=EMBEDDING)
