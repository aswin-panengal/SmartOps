from fastapi import APIRouter, UploadFile, File, Form
from typing import Optional 
from app.engines.analytical import run_analytical_engine
from app.engines.semantic import ingest_pdf, query_pdf
from app.memory.session import clear_session
from app.agent.graph import smartops_graph

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
    result = ingest_pdf(file_bytes, file.filename)
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
    file_bytes = None
    filename = None

    if file:
        file_bytes = await file.read()
        filename = file.filename

    # Run through LangGraph
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