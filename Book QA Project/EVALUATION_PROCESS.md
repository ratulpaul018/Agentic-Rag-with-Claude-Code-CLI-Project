# Book QA Project - Evaluation Process

This document outlines the evaluation methodology, metrics, and procedures for assessing the quality and performance of the Book QA RAG (Retrieval-Augmented Generation) system.

## Table of Contents

1. [Overview](#overview)
2. [Evaluation Metrics](#evaluation-metrics)
3. [Evaluation Dataset](#evaluation-dataset)
4. [Running Evaluations](#running-evaluations)
5. [Interpreting Results](#interpreting-results)
6. [Optimization Strategies](#optimization-strategies)
7. [Performance Benchmarks](#performance-benchmarks)

---

## Overview

The Book QA system is evaluated using **RAGAS (Retrieval-Augmented Generation Assessment)** framework, which provides automated metrics to assess the quality of generated answers and retrieved context without requiring manual annotation.

### Evaluation Goals

- **Measure retrieval quality**: Are relevant documents being retrieved?
- **Assess generation quality**: Are answers factually grounded and relevant?
- **Identify bottlenecks**: Which components need improvement?
- **Track improvements**: Monitor system performance over time

### System Under Test

```
Question → Retriever → Retrieved Docs → LLM → Answer
```

---

## Evaluation Metrics

### 1. **Faithfulness** (0.0 - 1.0)

**Definition**: Measures how grounded the generated answer is in the retrieved context. Higher values indicate the answer closely follows information from documents without hallucinating.

**What it measures**:
- Does the answer introduce information NOT in the context?
- Are claims substantiated by retrieved documents?

**Score Interpretation**:
- **0.8+**: Excellent - Answer closely adheres to source material
- **0.6-0.8**: Good - Mostly faithful with minor hallucinations
- **0.4-0.6**: Fair - Some hallucinations present
- **<0.4**: Poor - Significant ungrounded claims

**How to improve**:
- Refine prompt template to emphasize "only use provided context"
- Increase `RETRIEVAL_K` to provide more context
- Reduce LLM temperature (set to 0 for evaluation)

---

### 2. **Answer Relevancy** (0.0 - 1.0)

**Definition**: Measures whether the generated answer directly addresses the question asked.

**What it measures**:
- Does the answer answer what was asked?
- Is irrelevant information included?
- Are key question aspects covered?

**Score Interpretation**:
- **0.8+**: Excellent - Answer fully and directly addresses question
- **0.6-0.8**: Good - Answer mostly relevant with some tangents
- **0.4-0.6**: Fair - Partially addresses question
- **<0.4**: Poor - Misses main question intent

**How to improve**:
- Use clearer, more specific questions in evaluation set
- Improve prompt template to emphasize question focus
- Check if questions are being understood correctly by retriever

---

### 3. **Context Precision** (0.0 - 1.0)

**Definition**: Measures the relevance of retrieved context to the question. High precision means retrieved documents are closely related to what's being asked.

**What it measures**:
- Are retrieved documents topically relevant?
- How much noise (irrelevant content) in results?
- Early precision - are top results most relevant?

**Score Interpretation**:
- **0.8+**: Excellent - Nearly all retrieved docs are relevant
- **0.6-0.8**: Good - Most retrieved docs are relevant
- **0.4-0.6**: Fair - Mixed relevance in results
- **<0.4**: Poor - Retrieved docs largely irrelevant

**How to improve**:
- Tune `CHUNK_SIZE` - larger chunks preserve context, smaller are more specific
- Adjust `CHUNK_OVERLAP` - more overlap helps context continuity
- Test different retrieval K values (fewer = higher precision)
- Review embedding model quality (`nomic-embed-text`)
- Check if queries are clear and specific

---

### 4. **Context Recall** (0.0 - 1.0)

**Definition**: Measures how much relevant information from the documents is actually retrieved. High recall means important context isn't being missed.

**What it measures**:
- Is all relevant information being found?
- Are we retrieving enough context?
- Complete coverage of answer-relevant material?

**Score Interpretation**:
- **0.8+**: Excellent - System finds nearly all relevant information
- **0.6-0.8**: Good - Finds most relevant information
- **0.4-0.6**: Fair - Misses some relevant content
- **<0.4**: Poor - Missing much relevant information

**How to improve**:
- Increase `RETRIEVAL_K` to retrieve more documents
- Reduce `CHUNK_SIZE` to create more granular chunks
- Improve embeddings model
- Better question clarity helps identify relevant content

---

### 5. **Overall Score**

**Calculation**: Average of all four metrics

```
Overall = (Faithfulness + Answer Relevancy + Context Precision + Context Recall) / 4
```

**Interpretation**:
- **0.8+**: Excellent - Production-ready system
- **0.6-0.8**: Good - Acceptable for most use cases
- **0.4-0.6**: Fair - Needs improvement before production
- **<0.4**: Poor - Significant work required

---

## Evaluation Dataset

### Dataset Format

Create `evaluation_data.json` with the following structure:

```json
{
  "questions": [
    {
      "question": "What is the main theme of the book?",
      "ground_truth": "The book explores the impact of artificial intelligence on human society, covering both opportunities and risks."
    },
    {
      "question": "Who is the primary author?",
      "ground_truth": "The book is written by [Author Name]."
    },
    {
      "question": "What are the key chapters?",
      "ground_truth": "The book contains chapters on: Introduction, AI Fundamentals, Societal Impact, Future Implications, and Conclusion."
    }
  ]
}
```

### Dataset Best Practices

1. **Diversity**: Include questions covering:
   - Factual details (names, dates, numbers)
   - Conceptual understanding (themes, relationships)
   - Synthesis (combining information from multiple sections)
   - Edge cases (rare or specific information)

2. **Realism**: Questions should match real user queries:
   - Natural phrasing, not overly technical
   - Vary in length (short to complex)
   - Cover different aspects of the document

3. **Ground Truth Quality**:
   - Comprehensive but concise
   - Factually accurate based on source material
   - Include specific details (page references helpful)

4. **Sample Sizes**:
   - Minimum: 10 questions for baseline evaluation
   - Recommended: 30-50 questions for thorough assessment
   - Comprehensive: 100+ questions for production validation

### Example Dataset (Small Books)

```json
{
  "questions": [
    {
      "question": "What is the main topic?",
      "ground_truth": "Primary subject matter of the book"
    },
    {
      "question": "List key concepts mentioned",
      "ground_truth": "Important ideas discussed in the book"
    },
    {
      "question": "Who wrote this?",
      "ground_truth": "Author information from cover/introduction"
    }
  ]
}
```

---

## Running Evaluations

### Prerequisites

```bash
# Ensure Ollama is running
ollama serve

# In another terminal, pull required models
ollama pull llama3.2:latest
ollama pull nomic-embed-text
```

### Step 1: Prepare Evaluation Data

Create `evaluation_data.json` in the Book QA Project directory:

```bash
# Example structure
cat > evaluation_data.json << 'EOF'
{
  "questions": [
    {
      "question": "What is the book about?",
      "ground_truth": "Description of main topic"
    }
  ]
}
EOF
```

### Step 2: Run RAG Setup (if needed)

```bash
# First time: Setup with your PDF
python book_qa.py setup path/to/your/book.pdf
```

### Step 3: Run Evaluation

```bash
# From Book QA Project directory
python rag_evaluator.py
```

### Expected Output

```
[RAGAS] RAG System Evaluation Tool
----------------------------------------------------------------------
Found 1 indexed document(s)

Loading RAG system...
Generating 3 responses...
  [1/3] Generated response for: What is the book about?...
  [2/3] Generated response for: What are key concepts?...
  [3/3] Generated response for: Who is the author?...

Evaluating 3 responses with RAGAS...
Initializing evaluation models...
Evaluating metrics: faithfulness, answer_relevancy, context_precision, context_recall...

======================================================================
RAGAS EVALUATION REPORT
======================================================================

Metric Scores (0-1, higher is better):
----------------------------------------------------------------------
faithfulness         | ██████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░ | 0.7234
answer_relevancy     | ████████████████████░░░░░░░░░░░░░░░░░░░░░░░ | 0.7891
context_precision    | ███████████████████░░░░░░░░░░░░░░░░░░░░░░░░ | 0.6745
context_recall       | ██████████████████░░░░░░░░░░░░░░░░░░░░░░░░░ | 0.6234
----------------------------------------------------------------------
Overall Score        | ██████████████████░░░░░░░░░░░░░░░░░░░░░░░░░ | 0.7026

Report saved to: evaluation_report.json
```

---

## Interpreting Results

### Quick Assessment

| Overall Score | Status | Action |
|---|---|---|
| 0.8+ | ✅ Production-ready | Monitor and maintain |
| 0.6-0.8 | ⚠️ Good | Optimize weaker metrics |
| 0.4-0.6 | ❌ Fair | Significant improvement needed |
| <0.4 | 🚨 Poor | Debug system components |

### Metric-Specific Issues

#### Low Faithfulness (< 0.6)

**Problem**: Answers contain information not in the retrieved context

**Diagnosis**:
- Check if retrieved context is insufficient
- Review LLM temperature setting
- Verify prompt template clarity

**Solutions**:
1. Increase `RETRIEVAL_K` from 5 → 10-15
2. Lower LLM temperature: `temperature=0` (more deterministic)
3. Update prompt to: "Only use the provided context. Do not add information from your training."

---

#### Low Answer Relevancy (< 0.6)

**Problem**: Answers don't directly address the question

**Diagnosis**:
- Questions may be too complex or ambiguous
- Retriever may not be finding the right context
- LLM may be drifting from the question

**Solutions**:
1. Simplify evaluation questions
2. Increase `CHUNK_SIZE` to preserve question-context relationships
3. Reword prompt: "Answer this specific question: {question}"

---

#### Low Context Precision (< 0.6)

**Problem**: Retrieved documents aren't relevant to the question

**Diagnosis**:
- Chunk size too large (includes unrelated content)
- Embedding model not aligning well with document domain
- `RETRIEVAL_K` too high (retrieving irrelevant results)

**Solutions**:
1. Reduce `CHUNK_SIZE`: 900 → 500-700
2. Try increasing `CHUNK_OVERLAP`: 250 → 400-500
3. Reduce `RETRIEVAL_K`: 15 → 5-8
4. Check if embedding model matches document type

---

#### Low Context Recall (< 0.6)

**Problem**: Missing relevant information in retrieved results

**Diagnosis**:
- Chunks are too small/specific
- Not retrieving enough results
- Important context scattered across document

**Solutions**:
1. Increase `RETRIEVAL_K`: 15 → 25-30
2. Increase `CHUNK_SIZE`: 900 → 1200-1500
3. Verify embeddings are capturing semantic similarity
4. Review document structure and chunking

---

## Optimization Strategies

### Parameter Tuning

Create a tuning experiment by modifying `BOOK_QA_PROJECT/book_qa.py`:

```python
# Conservative (High Precision)
CHUNK_SIZE = 500
CHUNK_OVERLAP = 150
RETRIEVAL_K = 5
LLM_TEMP = 0.3

# Balanced (Recommended)
CHUNK_SIZE = 900
CHUNK_OVERLAP = 250
RETRIEVAL_K = 15
LLM_TEMP = 0.7

# Aggressive (High Recall)
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 400
RETRIEVAL_K = 25
LLM_TEMP = 0.7
```

### Evaluation Workflow

1. **Baseline**: Run with default settings, record scores
2. **Hypothesis**: Identify lowest metric (e.g., "Low Context Precision")
3. **Adjust**: Change one parameter at a time
4. **Test**: Re-run evaluation
5. **Compare**: Note improvement/regression
6. **Iterate**: Return to step 2

### Prompt Optimization

Test different prompt templates in `book_qa.py`:

```python
# Version 1: Strict (High Faithfulness)
prompt_template = """You are an expert document analyst.
IMPORTANT: Only use information from the provided context.
Do NOT add information from your training data.
If the context doesn't contain the answer, say "I cannot find this information in the provided context."

Context:
{context}

Question:
{question}

Answer:"""

# Version 2: Flexible (High Relevancy)
prompt_template = """You are an expert document analyst specializing in answering questions about provided documents.
Use the context to answer the question thoroughly and clearly.
Structure your answer with:
- Direct answer to the question
- Supporting details from the context
- Relevant citations when applicable

Context:
{context}

Question:
{question}

Answer:"""
```

---

## Performance Benchmarks

### Expected Scores by System Quality

#### Small Documents (< 50 pages)
```
Good Baseline System:
- Faithfulness: 0.70-0.80
- Answer Relevancy: 0.75-0.85
- Context Precision: 0.65-0.75
- Context Recall: 0.70-0.80
- Overall: 0.70-0.80
```

#### Medium Documents (50-200 pages)
```
Good Baseline System:
- Faithfulness: 0.65-0.75
- Answer Relevancy: 0.70-0.80
- Context Precision: 0.60-0.70
- Context Recall: 0.60-0.70
- Overall: 0.64-0.74
```

#### Large Documents (200+ pages)
```
Good Baseline System:
- Faithfulness: 0.60-0.70
- Answer Relevancy: 0.65-0.75
- Context Precision: 0.55-0.65
- Context Recall: 0.50-0.60
- Overall: 0.57-0.67
```

---

## Continuous Improvement Checklist

- [ ] Establish baseline with initial evaluation
- [ ] Identify weakest metric
- [ ] Adjust one parameter
- [ ] Re-evaluate and compare
- [ ] Document changes and results
- [ ] Test with diverse question sets
- [ ] Validate improvements hold across datasets
- [ ] Monitor production queries for regression
- [ ] Regular re-evaluation (weekly/monthly)

---

## Troubleshooting

### "No indexed documents found"
```bash
# Ensure vector store exists
python book_qa.py setup path/to/book.pdf
```

### "Ollama connection error"
```bash
# Start Ollama service
ollama serve

# In another terminal
ollama pull llama3.2:latest
ollama pull nomic-embed-text
```

### "evaluation_data.json not found"
```bash
# Create sample evaluation file
python rag_evaluator.py  # Will generate sample
# Edit evaluation_data.json with real Q&A pairs
```

### Evaluation runs very slowly
- Reduce evaluation set size (fewer questions)
- Use faster embedding model if available
- Run evaluation on GPU machine if possible

---

## References

- **RAGAS Framework**: https://docs.ragas.io/
- **Metric Definitions**: https://docs.ragas.io/en/stable/concepts/metrics/
- **LangChain RAG**: https://python.langchain.com/docs/use_cases/question_answering/
- **ChromaDB**: https://docs.trychroma.com/
- **Ollama**: https://ollama.ai/

---

**Last Updated**: 2026-04-20  
**Version**: 1.0
