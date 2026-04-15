# Book QA RAG System Setup Guide

## Prerequisites
- **Ollama**: Installed and running
- **Gemma 4**: Pull with `ollama pull gemma:4`
- **Embedding Model**: Pull with `ollama pull nomic-embed-text`

## Installation

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Prepare your book**:
   - Place your PDF file in the same directory as `book_qa.py`
   - Update `BOOK_PATH` in the script if it's named differently

## Usage

### First Time: Create Vector Store
```bash
python book_qa.py setup ./my_book.pdf
```
This will:
- Load the PDF
- Split it into chunks
- Create embeddings
- Store in ChromaDB

### Ask Questions
After setup, ask questions using:

**Command mode**:
```bash
python book_qa.py ask "What is the main theme of the book?"
```

**Interactive mode**:
```bash
python book_qa.py interactive
```
Then type your questions. Type `exit` to quit.

## Configuration

Edit these values in `book_qa.py` to customize:
- `CHUNK_SIZE`: Size of text chunks (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)
- `OLLAMA_MODEL`: Model to use (default: "gemma:4")
- `EMBEDDING_MODEL`: Embedding model (default: "nomic-embed-text")

## Troubleshooting

**"Ollama connection refused"**:
- Make sure Ollama is running: `ollama serve`

**"Model not found"**:
- Pull the model: `ollama pull gemma:4` and `ollama pull nomic-embed-text`

**"No module named langchain"**:
- Install requirements: `pip install -r requirements.txt`

**Large PDF taking too long**:
- Increase `CHUNK_SIZE` to create fewer chunks
- Reduce `EMBEDDING_MODEL` to "all-minilm" (smaller, faster)
