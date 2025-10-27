import os
import streamlit as st

from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ----------------------
# Streamlit UI
# ----------------------
st.title("RAG Chatbot with Ollama + Chroma")

uploaded_file = st.file_uploader("Upload a DOCX or PDF file", type=["docx", "pdf"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    file_ext = os.path.splitext(uploaded_file.name)[1]
    file_path = "temp_upload" + file_ext
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"{uploaded_file.name} uploaded successfully!")

    # Load document based on file type
    if file_ext == ".docx":
        loader = Docx2txtLoader(file_path)
    elif file_ext == ".pdf":
        loader = PyPDFLoader(file_path)
    else:
        st.error("Unsupported file type")
        st.stop()

    documents = loader.load()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # CPU-friendly

    # Build Chroma vectorstore
    vectordb = Chroma.from_documents(docs, embeddings)

    # Initialize Ollama chat model
    llm = ChatOllama(model="llama2")  # Change model if needed

    # Create RAG QA chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever()
    )

    # User question input
    question = st.text_input("Ask a question about the document:")

    if question:
        answer = qa.run(question)
        st.write("**Answer:**", answer)
