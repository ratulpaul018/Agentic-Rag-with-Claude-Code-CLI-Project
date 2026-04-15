import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
import os

# 1. Cache the PDF loading and vectorization so it only happens ONCE
@st.cache_resource


def initialize_vectorstore():
    # The 'r' at the start is the magic fix for Windows paths!
    path = r"C:\RAG venv (09.04.2026)\ragenv\resume.pdf"
    loader = PyPDFLoader(path).load()
    
    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(loader)
    
    # Embed and store
    embeddings = OllamaEmbeddings(model="llama3.2:latest")
    # In-memory vectorstore for faster local performance
    return Chroma.from_documents(chunks, embeddings)

vectorstore = initialize_vectorstore()

# 2. Setup the LLM Chain
model = OllamaLLM(model="llama3.2:latest")
template = """
You are a helpful assistant who answers questions about resume based on the provided context.
Context: {context}

Question: {question}
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# 3. Streamlit UI
st.title("⚛️ Quantum Physics QA System")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if user_query := st.chat_input("Ask a question about quantum physics:"):
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)
    
    with st.chat_message("assistant"):
        with st.spinner("Searching textbook and thinking..."):
            # RETRIEVAL: Correct method to get documents
            docs = vectorstore.similarity_search(user_query, k=3)
            context_text = "\n\n".join([doc.page_content for doc in docs])
            
            # GENERATION: Run the chain
            response = chain.invoke({"context": context_text, "question": user_query})
            
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})