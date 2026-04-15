#!/usr/bin/env python
"""Test script to diagnose setup issues"""

import sys
import subprocess

print("=" * 50)
print("RAG System Setup Diagnostic")
print("=" * 50)

# 1. Check Python version
print("\n✓ Python version:")
print(f"  {sys.version}")

# 2. Check required packages
print("\n✓ Checking required packages...")
required_packages = [
    'flask',
    'langchain',
    'langchain_community',
    'pypdf',
    'chromadb',
    'ollama'
]

missing = []
for package in required_packages:
    try:
        __import__(package)
        print(f"  ✓ {package}")
    except ImportError:
        print(f"  ✗ {package} - MISSING")
        missing.append(package)

if missing:
    print(f"\n⚠️  Missing packages: {', '.join(missing)}")
    print(f"Install with: pip install -r requirements.txt")
    sys.exit(1)

# 3. Check Ollama connection
print("\n✓ Checking Ollama connection...")
try:
    result = subprocess.run(
        ['ollama', 'list'],
        capture_output=True,
        timeout=5
    )
    if result.returncode == 0:
        print("  ✓ Ollama is running")
        models = result.stdout.decode().strip()
        print(f"\n  Available models:\n{models}")
    else:
        print("  ✗ Ollama command failed")
        print("  Start Ollama with: ollama serve")
except FileNotFoundError:
    print("  ✗ Ollama not found - install from https://ollama.ai")
except subprocess.TimeoutExpired:
    print("  ✗ Ollama timeout - is it running?")
except Exception as e:
    print(f"  ✗ Error: {e}")

# 4. Check Flask app
print("\n✓ Checking Flask app...")
try:
    from web_app import app
    print("  ✓ Flask app loads successfully")
    print(f"  Routes available: {len(app.url_map._rules)}")
except Exception as e:
    print(f"  ✗ Flask app error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 50)
print("✓ All checks complete!")
print("=" * 50)
