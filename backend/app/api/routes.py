from fastapi import APIRouter, UploadFile, File, Form, Request
from typing import Optional
from app.engines.analytical import run_analytical_engine
from app.engines.semantic import ingest_pdf, query_pdf
from app.memory.session import clear_session
from app.agent.graph import smartops_graph
# RAGAS evaluation removed — evaluator functionality deprecated

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
    err = result.get("error") or ""
    if result.get("status") == "error" and (
        "429" in str(err) or
        "RESOURCE_EXHAUSTED" in str(err) or
        "quota" in str(err).lower() or
        "rate limit" in str(err).lower() or
        "TooManyRequests" in str(err)
    ):
        return {
            "status": "success",
            "question": question,
            "answer": "I am receiving too many requests right now and hit an API rate limit. Please wait about 60 seconds and try asking again.",
            "sources": []
        }
    return result


@router.post("/ingest/pdf")
async def ingest_pdf_route(
    file: UploadFile = File(...)
):
    file_bytes = await file.read()
    return ingest_pdf(file_bytes, file.filename, session_id="default")


@router.post("/query/pdf")
async def query_pdf_route(
    question: str = Form(...),
    session_id: str = Form(default="default")
):
    result = query_pdf(question, session_id)
    err = result.get("error") or ""
    if result.get("status") == "error" and (
        "429" in str(err) or
        "RESOURCE_EXHAUSTED" in str(err) or
        "quota" in str(err).lower() or
        "rate limit" in str(err).lower() or
        "TooManyRequests" in str(err)
    ):
        return {
            "status": "success",
            "question": question,
            "answer": "I am receiving too many requests right now and hit an API rate limit. Please wait about 60 seconds and try asking again.",
            "sources": []
        }
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
    Unified endpoint. Accepts an optional CSV/PDF plus a user question, then
    routes through the agent graph for clarification, CSV analysis, or PDF RAG.
    """
    file_bytes = None
    filename = None

    if file:
        file_bytes = await file.read()
        filename = file.filename

    result = smartops_graph.invoke({
        "question": question,
        "session_id": session_id,
        "file_bytes": file_bytes,
        "filename": filename,
        "engine": None,
        "answer": None,
        "sources": None,
        "chunks_used": 0,
        "contexts": [],
        "retrieval_scores": [],
        "best_score": None,
        "status": None,
        "error": None
    })

    # If the graph returned an API limit error, surface a friendly chat
    # reply instead of propagating an error. Only trigger on known rate
    # limit indicators; otherwise return the regular result unchanged.
    err = result.get("error") or ""
    if result.get("status") == "error" and (
        "429" in str(err) or
        "RESOURCE_EXHAUSTED" in str(err) or
        "quota" in str(err).lower() or
        "rate limit" in str(err).lower() or
        "TooManyRequests" in str(err)
    ):
        safe_reply = "I am receiving too many requests right now and hit an API rate limit. Please wait for some minutes and try asking again."
        sources = [f"source: {filename}"] if filename else []
        return {
            "status": "success",
            "question": question,
            "answer": safe_reply,
            "sources": sources
        }

    # Only return the source in normal responses. If a file was uploaded,
    # present it as "source: filename"
    sources = [f"source: {filename}"] if filename else []

    return {
        "status": result.get("status"),
        "question": question,
        "answer": result.get("answer"),
        "sources": sources
    }



