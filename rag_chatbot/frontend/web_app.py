import os
import streamlit as st
from langchain_ollama import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# ----------------------
# Streamlit UI
# ----------------------
st.title("RAG Chatbot with Ollama + Chroma")

uploaded_file = st.file_uploader("Upload a DOCX or PDF file", type=["docx", "pdf"])

if uploaded_file:
    file_ext = os.path.splitext(uploaded_file.name)[1]
    file_path = "temp_upload" + file_ext
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"{uploaded_file.name} uploaded successfully!")

    # Load document
    if file_ext == ".docx":
        loader = Docx2txtLoader(file_path)
    elif file_ext == ".pdf":
        loader = PyPDFLoader(file_path)
    else:
        st.error("Unsupported file type")
        st.stop()

    documents = loader.load()

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(docs, embeddings)

    # Initialize Ollama LLM
    llm = ChatOllama(model="llama2")

    # Prompt template
    prompt = ChatPromptTemplate.from_template(
        "Answer the question based only on the context below.\n\nContext:\n{context}\n\nQuestion: {question}"
    )

    # Build retrieval-based pipeline manually
    retriever = vectordb.as_retriever()

    # The new 1.x style RAG pipeline
    from langchain_core.runnables import RunnableLambda, RunnableMap

    def combine_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        RunnableMap({"context": retriever | RunnableLambda(combine_docs), "question": RunnablePassthrough()})
        | prompt
        | llm
    )

    # Ask question
    question = st.text_input("Ask a question about the document:")

    if question:
        result = rag_chain.invoke(question)
        st.write("**Answer:**", result.content)
