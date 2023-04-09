"""Load html from files, clean up, split, ingest into Weaviate."""
import pickle

from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

#load dot env
from dotenv import load_dotenv
load_dotenv()

def load_docs():
    # loader = ReadTheDocsLoader("langchain.readthedocs.io/en/latest/")
    loader = DirectoryLoader("data", glob="*.txt")
    # loader = UnstructuredHTMLLoader("index.html")
    raw_documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    documents = text_splitter.split_documents(raw_documents)
    return documents

def ingest_docs():
    """Get documents from web pages."""
    documents = load_docs()

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Save vectorstore
    with open("vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore, f)


if __name__ == "__main__":
    ingest_docs()
