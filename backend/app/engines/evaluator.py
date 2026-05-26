from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision
)
from datasets import Dataset
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from app.core.config import settings
import time

# RAGAS needs an LLM and embeddings to run evaluations
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    google_api_key=settings.google_api_key,
    temperature=0
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=settings.google_api_key
)

def evaluate_rag_response(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str = None
) -> dict:
    """
    Evaluates a RAG response using RAGAS metrics.
    
    Metrics explained:
    - Faithfulness: Is the answer grounded in the retrieved context?
      Score of 1.0 means every claim in the answer exists in the context.
      Score of 0.0 means the answer hallucinated everything.
      
    - Answer Relevancy: Does the answer actually address the question?
      Score of 1.0 means perfectly on-topic.
      Score of 0.0 means completely off-topic.
      
    - Context Precision: Are the retrieved chunks actually relevant?
      High score means your retrieval found the right chunks.
      
    - Context Recall: Did retrieval find ALL the relevant information?
      Requires ground truth to measure.
    """
    try:
        # RAGAS expects a specific dataset format
        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        }

        # Only add ground truth if provided
        if ground_truth:
            data["ground_truth"] = [ground_truth]

        dataset = Dataset.from_dict(data)

        # Select metrics based on whether ground truth is available
        metrics = [faithfulness, answer_relevancy, context_precision]
        if ground_truth:
            metrics.append(context_recall)

        # Run evaluation
        # Note: This makes additional Gemini API calls
        result = evaluate(
            dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings
        )

        scores = result.to_pandas().to_dict(orient="records")[0]

        # Clean up scores - round to 4 decimal places
        cleaned_scores = {}
        for key, value in scores.items():
            if key not in ["question", "answer", "contexts", "ground_truth"]:
                try:
                    cleaned_scores[key] = round(float(value), 4)
                except:
                    cleaned_scores[key] = value

        # Add interpretation
        faithfulness_score = cleaned_scores.get("faithfulness", 0)
        relevancy_score = cleaned_scores.get("answer_relevancy", 0)

        interpretation = interpret_scores(faithfulness_score, relevancy_score)

        return {
            "status": "success",
            "scores": cleaned_scores,
            "interpretation": interpretation,
            "question": question,
            "answer_preview": answer[:200] + "..." if len(answer) > 200 else answer
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "question": question
        }


def interpret_scores(faithfulness: float, relevancy: float) -> dict:
    """
    Converts raw scores into human-readable interpretation.
    Useful for the dashboard and for understanding what scores mean.
    """
    def score_label(score):
        if score >= 0.8:
            return "Excellent"
        elif score >= 0.6:
            return "Good"
        elif score >= 0.4:
            return "Fair"
        else:
            return "Poor"

    overall = (faithfulness + relevancy) / 2

    return {
        "faithfulness_label": score_label(faithfulness),
        "relevancy_label": score_label(relevancy),
        "overall_score": round(overall, 4),
        "overall_label": score_label(overall),
        "hallucination_risk": "Low" if faithfulness >= 0.7 else "High",
        "recommendation": (
            "Response is reliable and on-topic."
            if overall >= 0.7
            else "Consider re-ingesting documents or rephrasing the question."
        )
    }