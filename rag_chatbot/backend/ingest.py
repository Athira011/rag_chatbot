import os
from typing import List
import langchain_community
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import chromadb
from chromadb.config import Settings

# Directory where ChromaDB will store the vector database
CHROMA_DIR = os.getenv("CHROMA_DIRECTORY", "./chroma_db")

def load_documents(file_paths: List[str]):
    """Load documents from given file paths into LangChain Document format."""
    documents = []
    for file_path in file_paths:
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
        elif file_path.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        else:
            print(f"❌ Skipping unsupported file: {file_path}")
            continue

        documents.extend(loader.load())
    return documents


def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Split documents into smaller chunks for better embeddings & retrieval."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)


def ingest_files(file_paths: List[str], collection_name="docs"):
    """Load, split, and embed documents into Chroma vectorstore."""
    print("📥 Loading documents...")
    documents = load_documents(file_paths)

    print("✂️ Splitting documents...")
    chunks = split_documents(documents)

    print("🔢 Generating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print(f"💾 Storing embeddings in ChromaDB at {CHROMA_DIR} ...")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=collection_name
    )

    vectordb.persist()
    print("✅ Ingestion complete!")
    return vectordb


if __name__ == "__main__":
    # Example usage: run `python ingest.py` to ingest sample files
    sample_files = ["temp_upload.pdf"]
    ingest_files(sample_files)
