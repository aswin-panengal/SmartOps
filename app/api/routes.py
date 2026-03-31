from fastapi import APIRouter, UploadFile, File, Form
from app.engines.analytical import run_analytical_engine
from app.engines.semantic import ingest_pdf, query_pdf
from app.memory.session import clear_session

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