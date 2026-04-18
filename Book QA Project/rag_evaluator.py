"""
RAGAS Evaluation for RAG System
Evaluates retrieval-augmented generation using RAGAS metrics
"""

import os
import json
from pathlib import Path
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from multi_doc_rag import load_multi_doc_rag, get_indexed_documents

# Configuration
OLLAMA_MODEL = "llama3.2:latest"
EMBEDDING_MODEL = "nomic-embed-text"
EVALUATION_DATA_FILE = "evaluation_data.json"


def prepare_evaluation_dataset(qa_chain):
    """
    Create evaluation dataset from sample Q&A pairs
    Format: {"question": "...", "answer": "...", "contexts": ["..."], "ground_truth": "..."}
    """
    # Load or create sample evaluation data
    if os.path.exists(EVALUATION_DATA_FILE):
        with open(EVALUATION_DATA_FILE, "r") as f:
            eval_data = json.load(f)
            print(f"Loaded {len(eval_data)} evaluation samples")
            return eval_data

    print(f"No {EVALUATION_DATA_FILE} found. Create one with sample Q&A pairs.")
    return []


def generate_responses(qa_chain, questions):
    """Generate answers for evaluation questions"""
    print(f"\nGenerating {len(questions)} responses...")
    responses = []

    for i, item in enumerate(questions, 1):
        question = item.get("question")
        try:
            result = qa_chain.invoke({"query": question})
            answer = result.get("result", "")
            source_docs = result.get("source_documents", [])

            contexts = [
                doc.page_content for doc in source_docs
            ]

            responses.append({
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": item.get("ground_truth", ""),
            })

            print(f"  [{i}/{len(questions)}] Generated response for: {question[:50]}...")

        except Exception as e:
            print(f"  Error generating response: {e}")
            continue

    return responses


def evaluate_rag_system(responses):
    """Evaluate RAG system using RAGAS metrics"""
    if not responses:
        print("No responses to evaluate!")
        return None

    print(f"\nEvaluating {len(responses)} responses with RAGAS...")

    # Create dataset
    eval_dataset = Dataset.from_dict({
        "question": [r["question"] for r in responses],
        "answer": [r["answer"] for r in responses],
        "contexts": [r["contexts"] for r in responses],
        "ground_truth": [r["ground_truth"] for r in responses],
    })

    # Initialize LLM and embeddings for evaluation
    print("Initializing evaluation models...")
    eval_llm = Ollama(model=OLLAMA_MODEL, temperature=0)
    eval_embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    # Configure metrics
    print("Evaluating metrics: faithfulness, answer_relevancy, context_precision, context_recall...")

    try:
        # Run evaluation
        results = evaluate(
            eval_dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
            llm=eval_llm,
            embeddings=eval_embeddings,
        )

        return results

    except Exception as e:
        print(f"Evaluation error: {e}")
        return None


def print_evaluation_report(results, responses):
    """Print evaluation report"""
    print("\n" + "=" * 70)
    print("RAGAS EVALUATION REPORT")
    print("=" * 70)

    if results is None:
        print("Evaluation failed or no results available")
        return

    # Get metric scores
    scores = {
        "faithfulness": results["faithfulness"],
        "answer_relevancy": results["answer_relevancy"],
        "context_precision": results["context_precision"],
        "context_recall": results["context_recall"],
    }

    # Print individual scores
    print("\nMetric Scores (0-1, higher is better):")
    print("-" * 70)
    for metric, score in scores.items():
        bar_length = int(score * 50)
        bar = "█" * bar_length + "░" * (50 - bar_length)
        print(f"{metric:20} | {bar} | {score:.4f}")

    # Calculate overall score
    overall_score = sum(scores.values()) / len(scores)
    print("-" * 70)
    bar_length = int(overall_score * 50)
    bar = "█" * bar_length + "░" * (50 - bar_length)
    print(f"{'Overall Score':20} | {bar} | {overall_score:.4f}")

    # Print interpretation
    print("\nInterpretation:")
    print("-" * 70)
    print("• Faithfulness: How grounded the answer is in the retrieved context")
    print("• Answer Relevancy: How well the answer addresses the question")
    print("• Context Precision: How relevant the retrieved context is to the question")
    print("• Context Recall: How much relevant information is retrieved from documents")

    print("\nScore Ranges:")
    print("  0.0 - 0.3: Poor      (Model needs improvement)")
    print("  0.3 - 0.6: Fair      (Acceptable but can improve)")
    print("  0.6 - 0.8: Good      (Well-functioning system)")
    print("  0.8 - 1.0: Excellent (High-quality results)")

    # Recommendations
    print("\nRecommendations:")
    print("-" * 70)
    for metric, score in scores.items():
        if score < 0.5:
            if metric == "faithfulness":
                print(f"  • {metric}: Answers may hallucinate. Review prompt template.")
            elif metric == "answer_relevancy":
                print(f"  • {metric}: Answers diverge from questions. Improve prompt clarity.")
            elif metric == "context_precision":
                print(f"  • {metric}: Retrieved docs are not relevant. Tune retrieval K or chunk size.")
            elif metric == "context_recall":
                print(f"  • {metric}: Missing relevant context. Increase retrieval K or improve embeddings.")

    # Save results
    report_file = "evaluation_report.json"
    with open(report_file, "w") as f:
        json.dump({
            "scores": scores,
            "overall_score": overall_score,
            "responses_count": len(responses),
        }, f, indent=2)
    print(f"\nReport saved to: {report_file}")

    print("=" * 70)


def main():
    """Main evaluation workflow"""
    print("[RAGAS] RAG System Evaluation Tool")
    print("-" * 70)

    # Check if vector store exists
    docs = get_indexed_documents()
    if not docs:
        print("Error: No indexed documents found.")
        print("Please upload documents first using the web app or CLI.")
        print(f"Create {EVALUATION_DATA_FILE} with format:")
        print("""
{
  "questions": [
    {
      "question": "What is...",
      "ground_truth": "Expected answer..."
    }
  ]
}
        """)
        return

    print(f"Found {len(docs)} indexed document(s)")

    # Load RAG chain
    print("\nLoading RAG system...")
    try:
        qa_chain = load_multi_doc_rag()
    except Exception as e:
        print(f"Error loading RAG: {e}")
        return

    # Prepare evaluation dataset
    eval_data = prepare_evaluation_dataset(qa_chain)
    if not eval_data:
        print(f"\nCreating sample {EVALUATION_DATA_FILE}...")
        sample_data = {
            "questions": [
                {
                    "question": "What is the main topic of this document?",
                    "ground_truth": "The primary subject matter covered in the document.",
                },
                {
                    "question": "Who is the author?",
                    "ground_truth": "The person or entity who wrote the document.",
                },
                {
                    "question": "What are the key findings?",
                    "ground_truth": "Important conclusions or results mentioned in the document.",
                },
            ]
        }
        with open(EVALUATION_DATA_FILE, "w") as f:
            json.dump(sample_data, f, indent=2)
        print(f"Sample created. Edit {EVALUATION_DATA_FILE} with real Q&A pairs and run again.")
        return

    questions = eval_data.get("questions", [])
    if not questions:
        print(f"No questions found in {EVALUATION_DATA_FILE}")
        return

    # Generate responses
    responses = generate_responses(qa_chain, questions)
    if not responses:
        print("Failed to generate responses")
        return

    # Evaluate
    results = evaluate_rag_system(responses)

    # Print report
    print_evaluation_report(results, responses)


if __name__ == "__main__":
    main()
