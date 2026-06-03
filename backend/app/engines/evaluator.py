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
import math

try:
    from ragas.metrics import LLMContextPrecisionWithoutReference
except Exception:
    LLMContextPrecisionWithoutReference = None

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

def _clean_contexts(contexts: list[str]) -> list[str]:
    """Remove empty or malformed context chunks before sending them to RAGAS."""
    return [
        str(context).strip()
        for context in contexts
        if context and str(context).strip()
    ]

def _clean_answer_for_eval(answer: str) -> str:
    """
    The chat endpoint may append explanation/follow-up text after the direct answer.
    RAGAS should score the factual answer, not the assistant's conversational hook.
    """
    answer = str(answer or "").strip()

    if "\n\n_" in answer:
        answer = answer.split("\n\n_", 1)[0].strip()

    follow_up_markers = [
        "\n\nWant me to ",
        "\n\nWould you like ",
        "\n\nDo you want "
    ]
    for marker in follow_up_markers:
        if marker in answer:
            answer = answer.split(marker, 1)[0].strip()

    return answer

def _safe_score(value):
    """Return rounded score or None for NaN/invalid metric outputs."""
    try:
        numeric = float(value)
        if math.isnan(numeric) or math.isinf(numeric):
            return None
        return round(numeric, 4)
    except Exception:
        return value

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
        clean_contexts = _clean_contexts(contexts)
        clean_answer = _clean_answer_for_eval(answer)

        if not clean_contexts:
            return {
                "status": "error",
                "error": "No valid context chunks were provided for evaluation.",
                "question": question
            }

        if not clean_answer:
            return {
                "status": "error",
                "error": "No valid answer text was provided for evaluation.",
                "question": question
            }

        # RAGAS expects a specific dataset format
        data = {
            "question": [question],
            "answer": [clean_answer],
            "contexts": [clean_contexts],
        }

        # Only add ground truth if provided
        if ground_truth:
            data["ground_truth"] = [ground_truth]
            data["reference"] = [ground_truth]

        dataset = Dataset.from_dict(data)

        # Select metrics based on whether ground truth is available
        metrics = [faithfulness, answer_relevancy]
        if ground_truth:
            metrics.extend([context_precision, context_recall])
        elif LLMContextPrecisionWithoutReference:
            metrics.append(LLMContextPrecisionWithoutReference())

        # Run evaluation
        # Note: This makes additional Gemini API calls
        result = evaluate(
            dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
            raise_exceptions=False,
            show_progress=False
        )

        scores = result.to_pandas().to_dict(orient="records")[0]

        # Clean up scores - round to 4 decimal places
        cleaned_scores = {}
        for key, value in scores.items():
            if key not in ["question", "answer", "contexts", "ground_truth", "reference"]:
                cleaned_scores[key] = _safe_score(value)

        # Add interpretation
        faithfulness_score = cleaned_scores.get("faithfulness")
        relevancy_score = cleaned_scores.get("answer_relevancy")

        interpretation = interpret_scores(faithfulness_score, relevancy_score)

        return {
            "status": "success",
            "scores": cleaned_scores,
            "interpretation": interpretation,
            "question": question,
            "answer_preview": clean_answer[:200] + "..." if len(clean_answer) > 200 else clean_answer,
            "contexts_used": len(clean_contexts),
            "used_ground_truth": bool(ground_truth)
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "question": question
        }


def interpret_scores(faithfulness: float | None, relevancy: float | None) -> dict:
    """
    Converts raw scores into human-readable interpretation.
    Useful for the dashboard and for understanding what scores mean.
    """
    def score_label(score):
        if score is None:
            return "Unavailable"
        if score >= 0.8:
            return "Excellent"
        elif score >= 0.6:
            return "Good"
        elif score >= 0.4:
            return "Fair"
        else:
            return "Poor"

    available_scores = [
        score for score in [faithfulness, relevancy]
        if isinstance(score, (int, float))
    ]
    overall = sum(available_scores) / len(available_scores) if available_scores else None

    return {
        "faithfulness_label": score_label(faithfulness),
        "relevancy_label": score_label(relevancy),
        "overall_score": round(overall, 4) if overall is not None else None,
        "overall_label": score_label(overall),
        "hallucination_risk": (
            "Unknown" if faithfulness is None else
            "Low" if faithfulness >= 0.7 else "High"
        ),
        "recommendation": (
            "Response is reliable and on-topic."
            if overall is not None and overall >= 0.7
            else "Consider re-ingesting documents or rephrasing the question."
        )
    }
