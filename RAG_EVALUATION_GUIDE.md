# RAG Evaluation with RAGAS

## Overview

RAGAS (Retrieval-Augmented Generation Assessment) is a framework for evaluating RAG systems with four key metrics:

- **Faithfulness**: How grounded the answer is in retrieved context (no hallucinations)
- **Answer Relevancy**: How well the answer addresses the question
- **Context Precision**: How relevant the retrieved context is to the question
- **Context Recall**: How much relevant information is retrieved

---

## Installation

Add RAGAS to your dependencies:

```bash
pip install ragas==0.1.13
```

Or use the updated requirements.txt:

```bash
pip install -r requirements.txt
```

---

## Quick Start

### 1. Prepare Your Data

Edit `evaluation_data.json` with your test questions:

```json
{
  "questions": [
    {
      "question": "What are the key features?",
      "ground_truth": "Feature A, Feature B, and Feature C are the main capabilities."
    },
    {
      "question": "When was it published?",
      "ground_truth": "The document was published in January 2026."
    }
  ]
}
```

### 2. Upload Documents

Upload your PDFs via the web UI or API:

```bash
python web_app.py
# Then upload documents at http://localhost:5000
```

### 3. Run Evaluation

```bash
cd "Book QA Project"
python rag_evaluator.py
```

---

## Metric Scores Explained

Each metric ranges from **0.0 to 1.0**, where higher is better:

### Score Interpretation

| Range | Quality | Meaning |
|-------|---------|---------|
| 0.0 - 0.3 | Poor | Needs significant improvement |
| 0.3 - 0.6 | Fair | Acceptable, but room for improvement |
| 0.6 - 0.8 | Good | Well-functioning system |
| 0.8 - 1.0 | Excellent | High-quality results |

### Individual Metrics

#### Faithfulness (0-1)
Measures if the answer is grounded in retrieved context.

**Low Score Issues:**
- Answer contains information not in the context
- Model hallucinating facts
- Poor prompt template

**Improvements:**
- Add document constraints to prompt
- Increase retrieved context (K parameter)
- Use more specific chunk overlap

#### Answer Relevancy (0-1)
Measures if the answer addresses the question.

**Low Score Issues:**
- Answer diverges from question topic
- Model focusing on irrelevant context
- Poor question understanding

**Improvements:**
- Refine prompt to emphasize question focus
- Add question reformulation step
- Improve retrieval precision

#### Context Precision (0-1)
Measures if retrieved chunks are relevant to the question.

**Low Score Issues:**
- Retrieving off-topic documents
- Poor semantic understanding
- Suboptimal chunk boundaries

**Improvements:**
- Increase retrieval K and filter by relevance
- Adjust chunk size (currently 900 chars)
- Improve embedding model

#### Context Recall (0-1)
Measures if all relevant context is retrieved.

**Low Score Issues:**
- Missing relevant information
- Insufficient retrieval size
- Poor embedding quality

**Improvements:**
- Increase RETRIEVAL_K (currently 15)
- Use hybrid retrieval (keyword + semantic)
- Improve document chunking

---

## Configuration

### Adjustable Parameters

In `multi_doc_rag.py`:

```python
CHUNK_SIZE = 900        # Increase for larger context windows
CHUNK_OVERLAP = 250     # Increase for smoother transitions
RETRIEVAL_K = 15        # Increase to retrieve more context
```

In `rag_evaluator.py`:

```python
OLLAMA_MODEL = "llama3.2:latest"        # Can use other models
EMBEDDING_MODEL = "nomic-embed-text"    # Can try other embeddings
```

---

## Example Results

### Good System (0.7+ overall)

```
Metric Scores:
faithfulness      | ████████████████████░░░░░░░░░░░░░░░░░░░░ | 0.8234
answer_relevancy  | █████████████████████░░░░░░░░░░░░░░░░░░░ | 0.8412
context_precision | ██████████████████░░░░░░░░░░░░░░░░░░░░░░ | 0.7856
context_recall    | ███████████████████░░░░░░░░░░░░░░░░░░░░░ | 0.7945
Overall Score     | ████████████████████░░░░░░░░░░░░░░░░░░░░ | 0.8111
```

### Fair System (0.5-0.7 overall)

```
Metric Scores:
faithfulness      | ██████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ | 0.6234
answer_relevancy  | ███████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ | 0.5412
context_precision | █████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ | 0.4856
context_recall    | ██████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ | 0.5345
Overall Score     | ██████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ | 0.5461
```

---

## Workflow

### Step 1: Prepare Test Dataset

Create `evaluation_data.json` with:
- **questions**: Real questions users might ask
- **ground_truth**: Correct answers you expect

Example:

```json
{
  "questions": [
    {
      "question": "What is machine learning?",
      "ground_truth": "Machine learning is a subset of AI that enables systems to learn from data without explicit programming."
    },
    {
      "question": "Name three machine learning algorithms",
      "ground_truth": "Linear regression, decision trees, and neural networks are common ML algorithms."
    }
  ]
}
```

### Step 2: Upload Documents

1. Start web app: `python web_app.py`
2. Go to http://localhost:5000
3. Upload your PDF documents
4. Wait for processing to complete

### Step 3: Run Evaluation

```bash
cd "Book QA Project"
python rag_evaluator.py
```

### Step 4: Analyze Results

- Review `evaluation_report.json`
- Identify weak metrics
- Apply improvements

### Step 5: Iterate

- Adjust parameters in `multi_doc_rag.py`
- Re-run evaluation
- Compare scores

---

## Optimization Guide

### If Faithfulness is Low

**Problem**: Answers not grounded in context

**Solutions**:
1. Strengthen prompt template (add "Only answer based on context")
2. Increase RETRIEVAL_K (get more context)
3. Reduce CHUNK_OVERLAP (avoid redundancy)
4. Use Claude model with built-in guardrails

### If Answer Relevancy is Low

**Problem**: Answers don't address questions well

**Solutions**:
1. Reformulate questions in retrieval step
2. Add question context to prompt
3. Use multi-turn reasoning
4. Implement query expansion

### If Context Precision is Low

**Problem**: Retrieved chunks are off-topic

**Solutions**:
1. Increase RETRIEVAL_K and filter top-k
2. Add semantic similarity threshold
3. Improve chunk boundaries (adjust chunk size)
4. Implement query routing to specific documents

### If Context Recall is Low

**Problem**: Missing relevant information

**Solutions**:
1. Increase RETRIEVAL_K (currently 15, try 30-50)
2. Implement hybrid search (BM25 + semantic)
3. Increase CHUNK_OVERLAP for better coverage
4. Try better embedding models

---

## Interpretation Examples

### Scenario 1: Balanced Scores (0.75 average)

```
faithfulness=0.75, answer_relevancy=0.78, context_precision=0.72, context_recall=0.76
```

**Interpretation**: System is working well. Minor improvements possible.

**Action**: Monitor performance. Collect user feedback.

---

### Scenario 2: Low Context Recall (0.45)

```
faithfulness=0.80, answer_relevancy=0.82, context_precision=0.75, context_recall=0.45
```

**Interpretation**: Finding relevant chunks but missing some information.

**Action**: Increase RETRIEVAL_K from 15 to 30-50.

---

### Scenario 3: Low Faithfulness (0.35)

```
faithfulness=0.35, answer_relevancy=0.80, context_precision=0.75, context_recall=0.70
```

**Interpretation**: Answers hallucinate or go beyond context.

**Action**: Update prompt to emphasize "answer only from context" constraint.

---

## Advanced Usage

### Custom Metrics

Extend evaluation with custom metrics:

```python
from ragas.metrics import Metric

class CustomMetric(Metric):
    def __call__(self, row):
        # Implement your metric
        return score
```

### Batch Evaluation

Evaluate multiple test sets:

```python
for test_set in test_sets:
    responses = generate_responses(qa_chain, test_set)
    results = evaluate_rag_system(responses)
    print_evaluation_report(results, responses)
```

### Export Results

Results are saved to `evaluation_report.json`:

```json
{
  "scores": {
    "faithfulness": 0.75,
    "answer_relevancy": 0.78,
    "context_precision": 0.72,
    "context_recall": 0.76
  },
  "overall_score": 0.7525,
  "responses_count": 10
}
```

---

## Troubleshooting

### Evaluation Timeout

**Problem**: Evaluation takes too long

**Solution**:
- Reduce number of questions
- Use fewer RETRIEVAL_K chunks
- Increase Ollama CPU threads

### Low Scores on All Metrics

**Problem**: Everything scores poorly

**Solution**:
1. Check if documents are relevant
2. Verify Ollama is running (http://localhost:11434)
3. Test with simple questions first
4. Ensure chunk size (900) suits your documents

### Out of Memory

**Problem**: OOM during evaluation

**Solution**:
- Reduce RETRIEVAL_K
- Use smaller document chunks
- Evaluate fewer question pairs

---

## Best Practices

✅ **DO**
- Start with 5-10 representative questions
- Ensure ground_truth is accurate
- Run evaluation after major changes
- Compare scores before/after improvements
- Document your results

❌ **DON'T**
- Evaluate with biased questions
- Use incomplete ground_truth answers
- Change too many parameters at once
- Ignore low metric scores
- Over-optimize for one metric

---

## Summary

| Task | Command |
|------|---------|
| Install RAGAS | `pip install -r requirements.txt` |
| Prepare data | Edit `evaluation_data.json` |
| Upload docs | Open http://localhost:5000 |
| Run evaluation | `cd "Book QA Project" && python rag_evaluator.py` |
| View report | Open `evaluation_report.json` |

---

**Next Steps**: Run evaluation after each system change to measure improvement!

```bash
python rag_evaluator.py
```
