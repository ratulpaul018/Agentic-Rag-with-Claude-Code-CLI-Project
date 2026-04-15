# RAG Document Analysis System

A local RAG system for uploading PDFs and asking questions about them using LangChain, Ollama, and ChromaDB.

## Features

- Multi-document PDF upload and analysis
- Semantic search with embeddings (nomic-embed-text)
- Local LLM inference (llama3.2:latest)
- Dark-themed web interface
- PDF modal viewer with page navigation
- Structured Q&A responses

## Quick Start

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Pull Ollama models
ollama pull llama3.2:latest
ollama pull nomic-embed-text
```

### Run

```bash
python web_app.py
# Open http://localhost:5000
```

## Usage

1. Open browser to `http://localhost:5000`
2. Drag & drop PDF files
3. Ask questions about the documents
4. Click page numbers to view source PDFs

## How It Works

```
PDF Upload → Merge Documents → Chunk (900 chars, 250 overlap)
    ↓
Generate Embeddings (nomic-embed-text)
    ↓
Store in ChromaDB Vector Database
    ↓
Retrieve Top-K Chunks + Pass to LLM
    ↓
Generate Answer with Sources
```

## Configuration

Edit `Book QA Project/book_qa.py`:

```python
CHUNK_SIZE = 900          # Characters per chunk
CHUNK_OVERLAP = 250       # Overlap between chunks
RETRIEVAL_K = 15          # Number of chunks to retrieve
OLLAMA_MODEL = "llama3.2:latest"
EMBEDDING_MODEL = "nomic-embed-text"
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Failed to fetch" | Restart server: `python web_app.py` |
| Ollama 404 | Run: `ollama pull llama3.2:latest nomic-embed-text` |
| "No book loaded" | Upload a PDF first |
| Slow responses | Ensure Ollama is running |

## Architecture

- **Backend**: Flask (Python)
- **RAG**: LangChain + LCEL
- **Vector DB**: ChromaDB
- **Embeddings**: nomic-embed-text
- **LLM**: Ollama (llama3.2)
- **Frontend**: Vanilla JS + Dark CSS

## Files

```
Book QA Project/
├── book_qa.py       # Core RAG logic
├── CLAUDE.md        # Detailed guidelines

web_app.py           # Flask API
templates/index.html # Web UI
static/
├── script.js        # Frontend logic
└── style.css        # Dark theme
requirements.txt     # Dependencies
```

## Privacy

✅ 100% local processing - no cloud or external services
✅ All documents stay on your machine

## Author

Ratul Paul ([@ratulpaul018](https://github.com/ratulpaul018))
