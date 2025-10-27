# qa_chain.py
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.chat_models import Ollama
import os

# Directory to store Chroma vectorstore
VECTORSTORE_PATH = "vectorstore"

def build_qa_chain():
    """
    Build a QA chain using Ollama LLM and HuggingFace embeddings (CPU).
    """
    # Step 1: HuggingFace embeddings (force CPU)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}  # important to avoid meta tensor error
    )

    # Step 2: Chroma vectorstore
    vectorstore = Chroma(
        persist_directory=VECTORSTORE_PATH,
        embedding_function=embeddings
    )

    # Step 3: Ollama LLM
    llm = Ollama(model="llama2")  # replace "llama2" with your Ollama model name

    # Step 4: Build QA chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff",
        return_source_documents=True
    )

    return qa

if __name__ == "__main__":
    qa_chain = build_qa_chain()
    print("QA Chain with Ollama successfully built!")
