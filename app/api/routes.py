from fastapi import APIRouter, UploadFile, File, Form
from app.engines.analytical import run_analytical_engine

router = APIRouter()

@router.get("/status")
def status():
    return {"engine": "SmartOps ready", "version": "1.0.0"}

@router.post("/analyze/csv")
async def analyze_csv(
    file: UploadFile = File(...),
    question: str = Form(...)
):
    """
    Upload a CSV file and ask a natural language question.
    The engine extracts context, generates pandas code via Gemini,
    executes it, and returns the answer.
    """
    file_bytes = await file.read()
    result = run_analytical_engine(file_bytes, question)
    return result
    