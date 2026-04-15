import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

# Configuration
BOOK_PATH = "book.pdf"
OLLAMA_MODEL = "llama3.2:latest"  # Using Llama model
EMBEDDING_MODEL = "nomic-embed-text"
CHUNK_SIZE = 900  # Optimized for resumes - preserves section boundaries
CHUNK_OVERLAP = 250  # Good overlap for document continuity
VECTOR_DB_PATH = "./chroma_db"
RETRIEVAL_K = 15  # Get 15 most relevant chunks for better coverage and comparisons


def load_and_chunk_book(pdf_path: str):
    """Load PDF and split into chunks with document metadata"""
    print(f"Loading document from {pdf_path}...")

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Document not found at {pdf_path}")

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages")

    # Add document filename to metadata
    filename = os.path.basename(pdf_path)
    for doc in documents:
        doc.metadata['source_document'] = filename

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks from {filename}")

    return chunks


def merge_and_chunk_all_books(upload_folder: str):
    """Merge all PDFs in folder into one document, then chunk"""
    print(f"Loading and merging all PDFs from {upload_folder}...")

    if not os.path.exists(upload_folder):
        raise FileNotFoundError(f"Upload folder not found at {upload_folder}")

    # Collect all PDF files
    pdf_files = [f for f in os.listdir(upload_folder) if f.endswith('.pdf')]

    if not pdf_files:
        raise FileNotFoundError("No PDF files found in upload folder")

    print(f"Found {len(pdf_files)} PDF files to merge")

    # Load all PDFs and merge their content
    merged_text = ""
    all_documents = []

    for pdf_file in sorted(pdf_files):  # Sort for consistent order
        pdf_path = os.path.join(upload_folder, pdf_file)
        print(f"  Loading {pdf_file}...")

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        print(f"    Loaded {len(documents)} pages from {pdf_file}")

        # Add document source metadata
        for doc in documents:
            doc.metadata['source_document'] = pdf_file

        all_documents.extend(documents)
        # Add separator between documents
        merged_text += f"\n\n[Document: {pdf_file}]\n\n"
        merged_text += "\n".join([doc.page_content for doc in documents])
        merged_text += "\n\n"

    print(f"Total pages loaded: {len(all_documents)}")
    print(f"Merged text length: {len(merged_text)} characters")

    # Now chunk the merged text
    print(f"Chunking merged document with chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )

    # Create chunks from all_documents which preserves metadata
    chunks = text_splitter.split_documents(all_documents)
    print(f"Created {len(chunks)} chunks from merged documents")

    return chunks


def create_vector_store(chunks):
    """Create vector store with embeddings"""
    print(f"Creating embeddings using {EMBEDDING_MODEL}...")

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_PATH,
        collection_name="book_qa"
    )
    print("Vector store created successfully")

    return vector_store


def load_vector_store():
    """Load existing vector store"""
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vector_store = Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embeddings,
        collection_name="book_qa"
    )
    return vector_store


def create_rag_chain(vector_store):
    """Create RAG chain using LCEL"""
    print(f"Initializing {OLLAMA_MODEL} model...")

    llm = Ollama(model=OLLAMA_MODEL)

    # Custom prompt template with structured formatting instructions
    prompt_template = """You are an expert document analyst.
You have been given content from multiple documents.
Use the context below to answer the question accurately and comprehensively.
If you cannot find the answer in the context, say so clearly.
Keep your answer concise and well-structured using paragraphs, bullet points, or tables where appropriate.

Context:
{context}

Question:
{question}

Answer:"""
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_K})

    # Build chain using LCEL
    def format_context_with_sources(docs):
        """Format docs with source document information for comparison"""
        formatted = []
        for doc in docs:
            source = doc.metadata.get('source_document', 'Unknown')
            page = doc.metadata.get('page', 'N/A')
            formatted.append(f"[From: {source}, Page {page}]\n{doc.page_content}")
        return "\n\n---\n\n".join(formatted)

    chain = (
        RunnableParallel(
            context=retriever | format_context_with_sources,
            question=RunnablePassthrough()
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    # Wrap chain to also return source documents
    class RAGChain:
        def __init__(self, chain, retriever):
            self.chain = chain
            self.retriever = retriever

        def invoke(self, input_dict):
            question = input_dict.get("query", "")
            answer = self.chain.invoke(question)
            source_documents = self.retriever.invoke(question)

            return {
                "result": answer,
                "source_documents": source_documents
            }

    return RAGChain(chain, retriever)


def setup_rag(pdf_path: str):
    """Setup RAG system from scratch"""
    chunks = load_and_chunk_book(pdf_path)
    vector_store = create_vector_store(chunks)
    qa_chain = create_rag_chain(vector_store)
    return qa_chain


def load_rag(vector_store_path: str = VECTOR_DB_PATH):
    """Load existing RAG system"""
    if not os.path.exists(vector_store_path):
        raise FileNotFoundError(f"Vector store not found at {vector_store_path}")

    vector_store = load_vector_store()
    qa_chain = create_rag_chain(vector_store)
    return qa_chain


def answer_question(qa_chain, question: str):
    """Ask a question to the RAG system"""
    print(f"\nQuestion: {question}")
    print("-" * 50)

    result = qa_chain.invoke({"query": question})

    print(f"Answer: {result['result']}")
    print("\nSource documents:")
    for i, doc in enumerate(result['source_documents'], 1):
        print(f"  {i}. Page {doc.metadata.get('page', 'N/A')}: {doc.page_content[:100]}...")


def main():
    """Main function"""
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  First time: python book_qa.py setup <path_to_book.pdf>")
        print("  Ask questions: python book_qa.py ask <question>")
        print("\nExample:")
        print("  python book_qa.py setup ./my_book.pdf")
        print("  python book_qa.py ask 'What is the main theme of the book?'")
        return

    command = sys.argv[1]

    if command == "setup":
        if len(sys.argv) < 3:
            print("Error: Please provide path to PDF book")
            print("Usage: python book_qa.py setup <path_to_book.pdf>")
            return

        pdf_path = sys.argv[2]
        qa_chain = setup_rag(pdf_path)
        print("\n✓ RAG system ready! You can now ask questions.")

    elif command == "ask":
        if len(sys.argv) < 3:
            print("Error: Please provide a question")
            print("Usage: python book_qa.py ask '<question>'")
            return

        question = " ".join(sys.argv[2:])
        qa_chain = load_rag()
        answer_question(qa_chain, question)

    elif command == "interactive":
        qa_chain = load_rag()
        print("RAG system loaded. Type 'exit' to quit.\n")

        while True:
            question = input("Question: ").strip()
            if question.lower() == "exit":
                break
            if question:
                answer_question(qa_chain, question)

    else:
        print(f"Unknown command: {command}")
        print("Use 'setup', 'ask', or 'interactive'")


if __name__ == "__main__":
    main()
