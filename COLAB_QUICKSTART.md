# 🚀 Google Colab Quick Start (5 minutes)

## Pre-Flight Checklist

- [ ] Have ngrok token ready? Get free: https://ngrok.com/signup
- [ ] Have a PDF file ready to upload
- [ ] Gmail account (for Colab)

## Step 1: Get ngrok Token (2 min)

```
1. Go: https://ngrok.com/signup
2. Sign up with Google
3. Go: https://dashboard.ngrok.com/auth/your-authtoken
4. Copy the token (keep secret!)
```

## Step 2: Open Colab Notebook (1 min)

```
Method A - Direct:
1. Go: https://colab.research.google.com
2. Click "File" → "Open notebook"
3. Click "Upload"
4. Upload: RAG_Colab_Setup.ipynb

Method B - From Google Drive:
1. Upload notebook file to your Drive
2. Right-click → "Open with" → "Google Colaboratory"
```

## Step 3: Set GPU (1 min) ⚡

```
1. Click "Runtime" menu
2. Click "Change runtime type"
3. Select: GPU → T4
4. Click "Save"

You should see ⚡ next to "Colab" in top-left
```

## Step 4: Run Notebook (1 min) ▶️

**Run these cells IN ORDER:**

### Cell 1: Install Dependencies
```python
# Just click ▶️ and wait 2-3 minutes
```

### Cell 2: Check GPU
```python
# Should show: GPU Available: True, GPU Name: Tesla T4
```

### Cell 3: Setup ngrok
```python
NGROK_AUTH_TOKEN = "YOUR_TOKEN_HERE"  # ⚠️ UPDATE THIS!
# Then click ▶️
```

### Cell 4: Create RAG Code
```python
# Just click ▶️
```

### Cell 5: Create Flask App
```python
# Just click ▶️
```

### Cell 6: Upload PDF
```python
# Click ▶️ and upload your PDF when prompted
```

### Cell 7: Start Application ✨
```python
# Click ▶️ and wait for:
# 🌐 PUBLIC URL: https://xxxxx.ngrok.io
# This is your application! Open in browser.
```

## What You'll See

```
✓ Server logs showing requests
✓ Your public URL (changes each session)
✓ "Waiting for connections..." message
```

## Access Your App

**Click the PUBLIC URL** from Cell 7 output:
```
🌐 https://abcd-1234-efgh.ngrok.io
```

You'll see the web interface with:
- 📄 Upload area on the left
- ❓ Q&A area on the right

## Workflow

```
1. Drop PDF in upload area
2. Click "Process Book"
3. Wait for ⏳ → ✓ "Book processed!"
4. Type a question
5. Click Send or press Enter
6. Get AI-powered answer!
```

## Keep It Running

- **Cell 7 must stay running** for the app to work
- Colab will disconnect after 30 mins of no activity
- If disconnected, just run Cell 7 again

## Common Issues & Quick Fixes

| Problem | Fix |
|---------|-----|
| GPU not showing | Restart: Runtime → Restart runtime |
| Module not found | Re-run Cell 1 |
| ngrok URL not working | Check token in Cell 3 |
| Connection timeout | Wait 10s, then refresh browser |

## Next Steps

📖 **For detailed info**: Read `COLAB_SETUP_GUIDE.md`

🔧 **To customize**:
- Change LLM model in book_qa_colab.py
- Adjust chunk size for better/worse accuracy
- Use multiple PDFs in one session

## Free Colab Limits

- **12 hours per session** (with activity)
- **GPU availability** (limited in high-demand hours)
- **Internet** (full bandwidth)
- **Storage** (upload/download limits)

---

**You're all set!** 🎉 Start with Cell 1 and follow the steps.

Questions? Check the full guide or Colab documentation.
