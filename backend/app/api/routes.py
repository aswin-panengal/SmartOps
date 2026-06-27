from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from typing import Optional
from app.engines.analytical import run_analytical_engine
from app.engines.semantic import ingest_pdf, query_pdf
from app.memory.session import clear_session
from app.agent.graph import smartops_graph

router = APIRouter()

MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB hard limit

_RATE_LIMIT_MARKERS = ("429", "RESOURCE_EXHAUSTED", "quota", "rate limit", "TooManyRequests")


def _is_rate_limit_error(err: str) -> bool:
    lower = err.lower()
    return any(m.lower() in lower for m in _RATE_LIMIT_MARKERS)


def _rate_limit_reply(question: str, filename: Optional[str] = None) -> dict:
    return {
        "status": "success",
        "question": question,
        "answer": (
            "I am receiving too many requests right now and hit an API rate limit. "
            "Please wait about 60 seconds and try asking again."
        ),
        "sources": ([f"source: {filename}"] if filename else []),
    }


async def _read_upload(file: UploadFile) -> bytes:
    """Buffer upload bytes with a hard size guard before any processing."""
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
        )
    return data


@router.get("/status")
def api_status():
    return {"engine": "SmartOps ready", "version": "1.0.0"}


@router.post("/analyze/csv")
async def analyze_csv(
    file: UploadFile = File(...),
    question: str = Form(...),
):
    file_bytes = await _read_upload(file)
    result = await run_in_threadpool(run_analytical_engine, file_bytes, question)
    err = result.get("error") or ""
    if result.get("status") == "error" and _is_rate_limit_error(err):
        return _rate_limit_reply(question)
    return result


@router.post("/ingest/pdf")
async def ingest_pdf_route(
    file: UploadFile = File(...),
    session_id: str = Form(default="default"),
):
    """
    Legacy ingestion endpoint. Pass session_id explicitly to avoid documents
    from different users sharing the 'default' Qdrant namespace.
    """
    file_bytes = await _read_upload(file)
    return await run_in_threadpool(ingest_pdf, file_bytes, file.filename, session_id)


@router.post("/query/pdf")
async def query_pdf_route(
    question: str = Form(...),
    session_id: str = Form(default="default"),
):
    result = await run_in_threadpool(query_pdf, question, session_id)
    err = result.get("error") or ""
    if result.get("status") == "error" and _is_rate_limit_error(err):
        return _rate_limit_reply(question)
    return result


@router.delete("/session/{session_id}")
def delete_session(session_id: str):
    clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}


@router.post("/ask")
async def ask(
    question: str = Form(...),
    session_id: str = Form(default="default"),
    file: Optional[UploadFile] = File(default=None),
):
    """
    Unified endpoint. Routes through the LangGraph state machine.
    Sync engine work runs in a thread pool so the event loop stays free.
    """
    file_bytes: Optional[bytes] = None
    filename: Optional[str] = None

    if file:
        file_bytes = await _read_upload(file)
        filename = file.filename

    initial_state = {
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
        "error": None,
    }

    result = await run_in_threadpool(smartops_graph.invoke, initial_state)

    err = result.get("error") or ""
    if result.get("status") == "error" and _is_rate_limit_error(err):
        return _rate_limit_reply(question, filename)

    return {
        "status": result.get("status"),
        "question": question,
        "answer": result.get("answer"),
        "sources": result.get("sources") or ([f"source: {filename}"] if filename else []),
    }
