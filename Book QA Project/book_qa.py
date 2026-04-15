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
CHUNK_SIZE = 500  # Smaller chunks to capture more content from all pages
CHUNK_OVERLAP = 150  # Good overlap to maintain context
VECTOR_DB_PATH = "./chroma_db"
RETRIEVAL_K = 5  # Get 5 most relevant chunks instead of 3


def load_and_chunk_book(pdf_path: str):
    """Load PDF and split into chunks"""
    print(f"Loading book from {pdf_path}...")

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Book not found at {pdf_path}")

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")

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

    llm = Ollama(model=OLLAMA_MODEL, temperature=0.7)

    # Custom prompt template with structured formatting instructions
    prompt_template = """
    You are expert in reading each pages of a book provided in the context top to bottom and answering questions based on that content.
    
    Use the following pieces of context to answer the question comprehensively and clearly.

IMPORTANT: Format your answer in a well-structured way:
- Use paragraphs for detailed explanations
- Use bullet points (•) for lists of items, features, or key points
- Use numbered lists (1. 2. 3.) for sequential steps or processes
- Use tables (with | separators) for comparing items or showing data
- Use bold text (**text**) for important terms or headings
- Use clear headings with bold text (**text**) for different sections if the answer has multiple parts

answer questions based on the provided book content and do not answer outside of the book."

Context:
{context}

Question: {question}

Structured Answer:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_K})

    # Build chain using LCEL
    chain = (
        RunnableParallel(
            context=retriever | (lambda docs: "\n\n".join([doc.page_content for doc in docs])),
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
