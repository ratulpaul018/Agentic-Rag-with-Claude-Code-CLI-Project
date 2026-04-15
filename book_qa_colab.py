import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
import torch

# Configuration
BOOK_PATH = "book.pdf"
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"  # Lightweight model suitable for T4
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Fast embedding model
CHUNK_SIZE = 500
CHUNK_OVERLAP = 150
VECTOR_DB_PATH = "./chroma_db"
RETRIEVAL_K = 5

# Check if GPU is available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


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

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": DEVICE}
    )

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
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": DEVICE}
    )
    vector_store = Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embeddings,
        collection_name="book_qa"
    )
    return vector_store


def create_rag_chain(vector_store):
    """Create RAG chain using LCEL with Hugging Face models"""
    print(f"Initializing {LLM_MODEL} model...")

    # Create Hugging Face pipeline for LLM
    llm = HuggingFacePipeline(
        model_id=LLM_MODEL,
        task="text-generation",
        pipeline_kwargs={
            "torch_dtype": torch.float16,
            "device": DEVICE,
            "max_length": 512,
            "temperature": 0.7,
        }
    )

    # Custom prompt template
    prompt_template = """Use the following pieces of context to answer the question comprehensively and clearly.

IMPORTANT: Format your answer in a well-structured way:
- Use paragraphs for detailed explanations
- Use bullet points (•) for lists of items, features, or key points
- Use numbered lists (1. 2. 3.) for sequential steps or processes
- Use tables (with | separators) for comparing items or showing data
- Use bold text (**text**) for important terms or headings
- Use clear headings with ## for different sections if the answer has multiple parts

If you don't know the answer from the context, say "I don't know based on the provided book content."

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
        print("  First time: python book_qa_colab.py setup <path_to_book.pdf>")
        print("  Ask questions: python book_qa_colab.py ask <question>")
        print("\nExample:")
        print("  python book_qa_colab.py setup ./my_book.pdf")
        print("  python book_qa_colab.py ask 'What is the main theme of the book?'")
        return

    command = sys.argv[1]

    if command == "setup":
        if len(sys.argv) < 3:
            print("Error: Please provide path to PDF book")
            print("Usage: python book_qa_colab.py setup <path_to_book.pdf>")
            return

        pdf_path = sys.argv[2]
        qa_chain = setup_rag(pdf_path)
        print("\n✓ RAG system ready! You can now ask questions.")

    elif command == "ask":
        if len(sys.argv) < 3:
            print("Error: Please provide a question")
            print("Usage: python book_qa_colab.py ask '<question>'")
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
