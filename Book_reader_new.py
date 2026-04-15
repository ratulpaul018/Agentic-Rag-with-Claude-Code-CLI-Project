import streamlit as st
import os

# Updated imports for better PDF handling and vector storage
from langchain_community.document_loaders import PyMuPDFLoader 
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

# 1. Page Configuration
st.set_page_config(page_title="Book AI", page_icon="⚛️")
st.title("⚛️ Book QA System")
st.markdown("Retrieving answers from Provided book.")

# 2. Caching the Vectorstore
# This prevents the app from re-loading the PDF on every user click
@st.cache_resource
def initialize_vectorstore():
    # Use the absolute path you provided
    path = r"C:\RAG venv (09.04.2026)\ragenv\010-CAPTAIN-FANTASTIC-Free-Childrens-Book-By-Monkey-Pen.pdf"
    
    if not os.path.exists(path):
        st.error(f"PDF not found at: {path}")
        st.stop()

    with st.status("Reading textbook and building knowledge base..."):
        # PyMuPDFLoader handles physics equations much better than PyPDF
        loader = PyMuPDFLoader(path).load()
        
        # Split into manageable chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        chunks = splitter.split_documents(loader)
        
        # Embed and store in an in-memory Chroma database
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vectorstore = Chroma.from_documents(chunks, embeddings)
        return vectorstore

# Initialize the vectorstore
vectorstore = initialize_vectorstore()

# 3. Setup the LLM and Prompt Template
model = OllamaLLM(model="gemma4:e4b")

template = """
You are a helpful expert assistant specializing in reading a provided book. 
Use the following pieces of retrieved context from the book to answer the question. 
If you don't know the answer based on the context, say that you don't know. 

Context:
{context}

Question: 
{question}

Answer:
"""
prompt_template = ChatPromptTemplate.from_template(template)
chain = prompt_template | model

# 4. Chat Interface Logic
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# User Input
if user_query := st.chat_input("Ask a question about the provided book:"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)
    
    with st.chat_message("assistant"):
        with st.spinner("Analyzing the book..."):
            # A. Retrieve relevant document chunks
            docs = vectorstore.similarity_search(user_query, k=4)
            context_text = "\n\n".join([doc.page_content for doc in docs])
            
            # B. Generate answer using the chain
            response = chain.invoke({
                "context": context_text, 
                "question": user_query
            })
            
            # C. Display and Save response
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})