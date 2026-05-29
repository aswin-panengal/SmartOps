from fastapi import APIRouter, UploadFile, File, Form, Request
from typing import Optional
from app.engines.analytical import run_analytical_engine
from app.engines.semantic import ingest_pdf, query_pdf
from app.memory.session import clear_session
from app.agent.graph import smartops_graph
from app.engines.evaluator import evaluate_rag_response

router = APIRouter()

@router.get("/status")
def status():
    return {"engine": "SmartOps ready", "version": "1.0.0"}

@router.post("/analyze/csv")
async def analyze_csv(
    file: UploadFile = File(...),
    question: str = Form(...)
):
    file_bytes = await file.read()
    result = run_analytical_engine(file_bytes, question)
    return result

@router.post("/ingest/pdf")
async def ingest_pdf_route(
    file: UploadFile = File(...)
):
    file_bytes = await file.read()
    result = ingest_pdf(file_bytes, file.filename, session_id="default")
    return result

@router.post("/query/pdf")
async def query_pdf_route(
    question: str = Form(...),
    session_id: str = Form(default="default")
):
    result = query_pdf(question, session_id)
    return result

@router.delete("/session/{session_id}")
def delete_session(session_id: str):
    clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}

@router.post("/ask")
async def ask(
    question: str = Form(...),
    session_id: str = Form(default="default"),
    file: Optional[UploadFile] = File(default=None)
):
    """
    The unified intelligent endpoint.
    Upload a CSV or PDF (optional) and ask any question.
    The agent automatically routes to the right engine.
    """
    clean_msg = question.strip().lower()

    # FAST-ROUTE INTERCEPTOR: Catch greetings instantly to bypass heavy LLM loads
    if clean_msg in ["hey", "hello", "hi", "test", "ping", "good morning", "good afternoon"]:
        return {
            "status": "success",
            "question": question,
            "engine_used": "System Guard",
            "answer": "👋 Hello! I am your SmartOps assistant. To get started, please upload a **CSV** for tabular data analysis or a **PDF** for document-based semantic querying.",
            "sources": [],
            "session_id": session_id,
            "error": None
        }

    file_bytes = None
    filename = None

    if file:
        file_bytes = await file.read()
        filename = file.filename

    # Run through LangGraph for actual queries
    result = smartops_graph.invoke({
        "question": question,
        "session_id": session_id,
        "file_bytes": file_bytes,
        "filename": filename,
        "engine": None,
        "answer": None,
        "sources": None,
        "status": None,
        "error": None
    })

    return {
        "status": result.get("status"),
        "question": question,
        "engine_used": result.get("engine"),
        "answer": result.get("answer"),
        "sources": result.get("sources"),
        "session_id": session_id,
        "error": result.get("error")
    }

# Evaluation store - keeps last 50 evaluations in memory
evaluation_history = []

@router.post("/evaluate")
async def evaluate_response(request: Request):
    """
    Evaluates a RAG response using RAGAS metrics.
    """
    body = await request.json()
    question = body.get("question", "")
    answer = body.get("answer", "")
    contexts = body.get("contexts", [])
    ground_truth = body.get("ground_truth", None)

    if not question or not answer or not contexts:
        return {
            "status": "error",
            "error": "question, answer, and contexts are all required"
        }

    result = evaluate_rag_response(
        question=question,
        answer=answer,
        contexts=contexts,
        ground_truth=ground_truth
    )

    if result["status"] == "success":
        evaluation_history.append(result)
        if len(evaluation_history) > 50:
            evaluation_history.pop(0)

    return result

@router.post("/query-and-evaluate")
async def query_and_evaluate(
    question: str = Form(...),
    session_id: str = Form(default="default"),
    ground_truth: str = Form(default=None)
):
    """
    Single endpoint that queries PDF documents and evaluates the response quality.
    """
    query_result = query_pdf(question, session_id)

    if query_result["status"] == "error":
        return query_result

    eval_result = evaluate_rag_response(
        question=question,
        answer=query_result["answer"],
        contexts=query_result.get("contexts", []),
        ground_truth=ground_truth
    )

    return {
        "status": "success",
        "question": question,
        "answer": query_result["answer"],
        "sources": query_result.get("sources", []),
        "session_id": session_id,
        "evaluation": eval_result.get("scores", {}),
        "interpretation": eval_result.get("interpretation", {}),
        "chunks_used": query_result.get("chunks_used", 0)
    }

@router.get("/evaluations/history")
def get_evaluation_history():
    """
    Returns the last 50 evaluation results.
    """
    if not evaluation_history:
        return {
            "status": "empty",
            "message": "No evaluations yet. Use /query-and-evaluate to start.",
            "evaluations": []
        }

    scores_list = [e["scores"] for e in evaluation_history if "scores" in e]

    avg_faithfulness = round(
        sum(s.get("faithfulness", 0) for s in scores_list) / len(scores_list), 4
    ) if scores_list else 0

    avg_relevancy = round(
        sum(s.get("answer_relevancy", 0) for s in scores_list) / len(scores_list), 4
    ) if scores_list else 0

    return {
        "status": "success",
        "total_evaluations": len(evaluation_history),
        "averages": {
            "faithfulness": avg_faithfulness,
            "answer_relevancy": avg_relevancy,
            "overall": round((avg_faithfulness + avg_relevancy) / 2, 4)
        },
        "evaluations": evaluation_history[-10:]
    }