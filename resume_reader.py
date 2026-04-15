import streamlit as st
import os
import tempfile

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

# 1. Page Configuration
st.set_page_config(page_title="Resume Comparator AI", page_icon="⚛️")

# Custom background color
st.markdown("""
    <style>
        .stApp {
            background-color: #1a1a2e;
        }
        .stChatMessage, .stChatInput {
            background-color: #16213e;
        }
        h1, h2, h3, p, label, .stMarkdown {
            color: #e0e0e0 !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Resume Comparator AI")
st.markdown("Upload two resumes and ask questions to compare them.")

# 2. File uploaders
col1, col2 = st.columns(2)
with col1:
    file1 = st.file_uploader("Upload Resume 1", type=["pdf"], key="resume1")
with col2:
    file2 = st.file_uploader("Upload Resume 2", type=["pdf"], key="resume2")

# 3. Build vectorstore from both uploaded PDFs
@st.cache_resource
def build_vectorstore(name1, name2, bytes1, bytes2):
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    all_chunks = []

    for label, data in [("Resume 1", bytes1), ("Resume 2", bytes2)]:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        docs = PyMuPDFLoader(tmp_path).load()
        os.unlink(tmp_path)
        chunks = splitter.split_documents(docs)
        for chunk in chunks:
            chunk.metadata["source_label"] = label
        all_chunks.extend(chunks)

    return Chroma.from_documents(all_chunks, embeddings)

# 4. LLM and comparison chain — cached
@st.cache_resource
def initialize_chain():
    model = OllamaLLM(model="gemma4:e4b")
    template = """You are an expert recruiter and resume analyst.
You have been given content from two resumes labelled "Resume 1" and "Resume 2".
Use the context below to answer the question, comparing the two candidates where relevant.
If you cannot find the answer in the context, say so.
Keep your answer concise and structured.

Context:
{context}

Question:
{question}

Answer:"""
    prompt_template = ChatPromptTemplate.from_template(template)
    return prompt_template | model

chain = initialize_chain()

# 5. Only proceed once both files are uploaded
if file1 and file2:
    with st.spinner("Processing resumes and building knowledge base..."):
        vectorstore = build_vectorstore(
            file1.name, file2.name,
            file1.read(), file2.read()
        )
    st.success(f"Ready! Comparing **{file1.name}** vs **{file2.name}**")

    # 6. Chat Interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if user_query := st.chat_input("Ask a question to compare the two resumes:"):
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.chat_message("user").write(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Analysing resumes..."):
                docs = vectorstore.similarity_search(user_query, k=4)
                context_text = "\n\n".join(
                    [f"[{d.metadata.get('source_label', 'Unknown')}]\n{d.page_content}" for d in docs]
                )

            thinking = st.empty()
            thinking.markdown("⏳ *Generating answer...*")

            def stream_with_clear():
                first = True
                for chunk in chain.stream({"context": context_text, "question": user_query}):
                    if first:
                        thinking.empty()
                        first = False
                    yield chunk

            response = st.write_stream(stream_with_clear())
            st.session_state.messages.append({"role": "assistant", "content": response})

else:
    st.info("Please upload both resumes above to get started.")
