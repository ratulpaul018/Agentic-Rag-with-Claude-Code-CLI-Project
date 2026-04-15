# Book QA Web Application Guide

A professional, fully-featured web interface for your RAG system. Upload PDFs and ask questions with instant answers backed by source citations.

## Features

✨ **Professional UI**
- Clean, corporate design with responsive layout
- Dark/light mode ready
- Smooth animations and transitions

📤 **File Upload**
- Drag & drop PDF support
- Single-click processing
- Visual progress indicators

💬 **Interactive Q&A**
- Real-time chat interface
- Keyboard shortcuts (Enter to send)
- Message history display

📌 **Source Citations**
- Automatic source document extraction
- Page references for each answer
- Context preview from source material

⚙️ **System Management**
- Reset button to clear books and data
- Status indicators
- Auto-detection of existing vector stores

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- Flask (web framework)
- Werkzeug (WSGI utilities)
- LangChain + community packages
- ChromaDB (vector store)
- Ollama client
- PyPDF (PDF processing)

### 2. Verify Ollama Setup

Make sure Ollama is running and models are pulled:

```bash
# In a separate terminal, start Ollama
ollama serve

# In another terminal, pull models if needed
ollama pull gemma:4
ollama pull nomic-embed-text
```

### 3. Start the Web App

```bash
python web_app.py
```

The app will start at: **http://localhost:5000**

## Usage

### 1. Upload a Book

**Option A: Drag & Drop**
- Drag your PDF file onto the upload area
- Or click to select from your computer

**Option B: Automatic Vector Store**
- If you've processed a book before, it will auto-load
- The app detects existing vector stores on startup

### 2. Ask Questions

Once a book is loaded:
1. Type your question in the text input
2. Press **Enter** or click **Send**
3. Get instant answers with source citations

**Example Questions:**
- "What is the main theme?"
- "Summarize chapter 3"
- "Who are the main characters?"
- "What happens at the end?"

### 3. View Sources

Below the Q&A section, you'll see **Source References** showing:
- Page number where answer came from
- Context snippet (first 200 characters)
- Multiple sources for comprehensive answers

### 4. Reset System

Click the **Reset** button in the header to:
- Clear the loaded book
- Delete all vector store data
- Reset the Q&A interface

## Project Structure

```
ragenv/
├── web_app.py                 # Flask application & API endpoints
├── book_qa.py                 # RAG logic (shared with CLI)
├── requirements.txt           # Python dependencies
├── templates/
│   └── index.html            # Main web page
└── static/
    ├── style.css             # Professional styling
    ├── script.js             # Frontend interactivity
    └── chroma_db/            # Vector store (created on first use)
```

## API Endpoints

### `POST /api/upload`
Upload and process a PDF book
- **Body**: multipart/form-data with 'file' field
- **Response**: `{ success: bool, message: str, chunks: int }`

### `POST /api/ask`
Ask a question about the loaded book
- **Body**: `{ "question": "string" }`
- **Response**: `{ success: bool, answer: str, sources: array }`

### `GET /api/status`
Check system status
- **Response**: `{ vector_store_loaded: bool, qa_chain_ready: bool }`

### `POST /api/reset`
Clear book and vector store
- **Response**: `{ success: bool, message: str }`

### `POST /api/load-existing`
Load existing vector store (auto-called on startup)
- **Response**: `{ success: bool, message: str }`

## Configuration

Edit `web_app.py` to customize:

```python
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # Max file size
```

Edit `book_qa.py` to customize RAG behavior:

```python
CHUNK_SIZE = 1000              # Text chunk size
CHUNK_OVERLAP = 200            # Overlap between chunks
OLLAMA_MODEL = "gemma:4"       # LLM to use
EMBEDDING_MODEL = "nomic-embed-text"  # Embedding model
```

## Troubleshooting

### "Connection refused" / "Ollama not running"
```bash
ollama serve
```
Start Ollama in a separate terminal

### "Model not found"
```bash
ollama pull gemma:4
ollama pull nomic-embed-text
```
Pull the required models

### "ModuleNotFoundError"
```bash
pip install -r requirements.txt
```
Reinstall dependencies

### Slow response times
- Increase `CHUNK_SIZE` in `book_qa.py` (faster but less precise)
- Use smaller embedding model: `ollama pull all-minilm`
- Ensure Ollama has enough system resources

### Large PDF taking too long
- The first embedding run is always slower
- Subsequent queries will be faster
- Large books (500+ pages) may take 5-15 minutes

## Performance Tips

1. **First-time processing**: Larger PDFs take longer. Let it finish.
2. **Query speed**: Typically 2-5 seconds per question (depends on model size)
3. **Memory usage**: Keep ~4GB RAM free for Ollama
4. **Concurrent users**: Single-instance Flask is for development only

## Deployment

For production, use a production WSGI server:

```bash
# Install Gunicorn
pip install gunicorn

# Run with Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 web_app:app
```

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

Requires JavaScript enabled.

## Tips & Tricks

🎯 **Better answers**: Ask specific questions about content
💡 **Follow-ups**: Ask follow-up questions naturally
📖 **Summaries**: Ask for chapter or section summaries
🔍 **Search**: Use specific keywords from the book
⏱️ **Wait**: Let Ollama process fully on first run

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Enter` | Send question |
| `Shift+Enter` | New line (in input) |

## Next Steps

- Upload your favorite book
- Ask it questions
- Integrate with your applications
- Deploy to production when ready

Enjoy your intelligent book companion! 📚
