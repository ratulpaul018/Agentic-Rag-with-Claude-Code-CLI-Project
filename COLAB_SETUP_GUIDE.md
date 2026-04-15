# RAG System on Google Colab - Complete Setup Guide

## Overview

This guide helps you run your RAG (Retrieval-Augmented Generation) system on Google Colab with a **T4 GPU** for fast inference. The system uses:

- **Hugging Face Models** (Mistral-7B) with GPU acceleration
- **ChromaDB** for vector storage
- **Flask** web interface exposed via **ngrok**
- **T4 GPU** (12GB VRAM) for inference

## Prerequisites

1. **Google Account** (for Colab)
2. **ngrok Account** (free at https://ngrok.com)
3. **PDF file** to analyze

## Step-by-Step Instructions

### 1. Get Your ngrok Auth Token (5 minutes)

1. Go to https://ngrok.com/signup
2. Sign up with your Google account (or email)
3. After signup, go to https://dashboard.ngrok.com/auth/your-authtoken
4. Copy your **Auth Token** (keep it secret!)

### 2. Open Google Colab Notebook

1. Upload the notebook to Colab:
   - Go to https://colab.research.google.com
   - Click "Upload" tab
   - Upload `RAG_Colab_Setup.ipynb`

2. **OR** open the notebook directly:
   - Go to https://colab.research.google.com
   - Paste the notebook URL

### 3. Configure GPU (Important!)

Before running any cells:

1. Click **Runtime** → **Change runtime type**
2. Select **GPU** from the dropdown
3. Choose **T4** (recommended)
4. Click **Save**

Your notebook header should show: ⚡ GPU (T4)

### 4. Run the Notebook Cells in Order

#### Cell 1: Install Dependencies
```
⏳ Takes ~2-3 minutes
Downloads: PyTorch, LangChain, transformers, ChromaDB, Flask
```

#### Cell 2: Check GPU Setup
```
✓ Should show:
- GPU Available: True
- GPU Name: Tesla T4
- GPU Memory: 16.0 GB
```

#### Cell 3: Setup ngrok
```
⚠️ IMPORTANT:
- Update NGROK_AUTH_TOKEN with your token from step 1
- Don't share your token with anyone!
```

#### Cell 4 & 5: Create RAG Code
```
Automatically creates:
- book_qa_colab.py (RAG system)
- web_app_colab.py (Flask app)
```

#### Cell 6: Upload Your PDF
```
Click "Choose Files" when prompted
Select your PDF to analyze
```

#### Cell 7: Start the Application
```
⏳ Takes ~30 seconds
🌐 Shows: PUBLIC URL - this is your application link!
Keep this cell running to maintain the connection
```

#### Cell 8: Test (Optional)
```
Runs a quick test to verify everything works
```

### 5. Access Your Application

When Cell 7 completes, you'll see:
```
🌐 PUBLIC URL: https://abcd-1234-efgh.ngrok.io
📱 Open this link in your browser
```

**The URL is publicly accessible for 2 hours!**

### 6. Using the Application

#### Upload a Book:
1. Click the upload area or drag-and-drop your PDF
2. Click "Process Book"
3. Wait for ⏳ "Processing PDF..." status
4. See ✓ "Book processed!" when complete

#### Ask Questions:
1. Type your question in the input box
2. Press Enter or click Send
3. Wait for the AI response (10-30 seconds)
4. View answers and source references

## Important Notes

### GPU Memory
- **Mistral-7B**: Uses ~7GB VRAM
- **T4 GPU**: Has 16GB total
- You have ~9GB free for embeddings and processing

### Response Times
- **First run**: 15-30 seconds (model loading)
- **Subsequent**: 10-20 seconds (inference only)
- Depends on question complexity and answer length

### Vector Store Persistence
- Saved in `chroma_db/` folder
- **Within session**: Available until you disconnect
- **Across sessions**: Use Google Drive mounting (advanced)

### ngrok URL Expiration
- **Free tier**: Expires after 2 hours of inactivity
- **Solution**: Refresh/restart Cell 7
- **Premium**: Get static URL with paid ngrok plan

## Troubleshooting

### "CUDA out of memory" Error
```
Solution:
1. Reduce CHUNK_SIZE in book_qa_colab.py (from 500 to 256)
2. Restart runtime and try again
3. Try a smaller model (DistilBERT for embeddings)
```

### "No module named" Error
```
Solution:
1. Re-run Cell 1 to reinstall dependencies
2. Restart runtime (Runtime → Restart runtime)
3. Run cells again in order
```

### ngrok Connection Failed
```
Solution:
1. Check your ngrok token is correct
2. Verify internet connection
3. Disconnect other ngrok sessions (new one required)
```

### "Connection Timeout" from Browser
```
Solution:
1. Verify Cell 7 is still running (shouldn't show "⚡" cell icon)
2. Wait a bit longer (model might still be loading)
3. Refresh the browser page
```

## Advanced Tips

### Use Multiple PDFs

The system supports multiple books in the same vector store:

```python
# In a new cell, after first book is processed:
# Upload another PDF and run Cell 6 again
# It will add to the existing vector store
```

### Increase Response Quality

Edit `book_qa_colab.py` before running:

```python
# More context = better answers but slower responses
RETRIEVAL_K = 10  # Get 10 most relevant chunks (default: 5)

# Smaller chunks = more granular retrieval
CHUNK_SIZE = 256  # Default: 500
```

### Save Vector Store to Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')

# Copy vector store to Drive
import shutil
shutil.copytree('chroma_db', '/content/drive/My Drive/chroma_db')
```

### Use Different LLM Models

```python
# In book_qa_colab.py, change:
LLM_MODEL = "meta-llama/Llama-2-7b-chat"  # Other options:
# - "meta-llama/Llama-2-13b-chat" (slower, better)
# - "mistralai/Mistral-7B" (faster, less context aware)
# - "teknium/OpenHermes-2.5-Mistral-7B" (balanced)
```

## Performance Comparison

| Aspect | Local | Colab T4 |
|--------|-------|----------|
| First Response | N/A (requires Ollama) | 15-30s |
| Subsequent | N/A | 10-20s |
| GPU Memory | CPU only | 16GB (7GB used) |
| Cost | Free (local) | Free (limited quota) |
| Accessibility | Local only | Public URL |
| Uptime | Always (local) | While session active |

## Getting Help

1. Check error messages in cell output
2. Look at GPU memory usage: `!nvidia-smi`
3. Check server logs above Cell 7
4. Restart runtime and try again: Runtime → Restart runtime

## Summary

You now have a powerful RAG system running on Google's T4 GPU! The system:

✅ Processes PDFs with GPU acceleration
✅ Answers questions in structured format
✅ Shows source references with page numbers
✅ Accessible via public URL
✅ Completely free (with Colab limits)

Enjoy your AI-powered document analysis! 🚀
