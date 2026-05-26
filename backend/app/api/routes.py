from fastapi import APIRouter, UploadFile, File, Form, Request #, HTTPException
from typing import Optional #, List
from app.engines.analytical import run_analytical_engine
from app.engines.semantic import ingest_pdf, query_pdf
from app.memory.session import clear_session
from app.agent.graph import smartops_graph
# import os 
# from dataset import Dataset
# from pydantic import BaseModel

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
    
# Webhook endpoint for n8n
@router.post("/webhook/ask")
async def webhook_ask(request: Request):
    """
    JSON-based endpoint for n8n and external integrations.
    No file upload — text questions only.
    """
    body = await request.json()
    question = body.get("question", "")
    session_id = body.get("session_id", "default")
    source = body.get("source", "webhook")

    if not question:
        return {"status": "error", "error": "No question provided"}

    result = smartops_graph.invoke({
        "question": question,
        "session_id": session_id,
        "file_bytes": None,
        "filename": None,
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
        "source": source
    }
    
# import base64

# @router.post("/ingest/pdf/base64")
# async def ingest_pdf_base64(request: Request):
#     """
#     Accepts base64 encoded PDF from n8n Google Drive workflow.
#     n8n downloads files as base64, so this endpoint handles that format.
#     """
#     body = await request.json()
#     filename = body.get("filename", "document.pdf")
#     file_base64 = body.get("file_base64", "")
#     source = body.get("source", "n8n")

#     if not file_base64:
#         return {"status": "error", "error": "No file data provided"}

#     try:
#         # 1. CLEAN THE STRING: Remove n8n's "data:application/pdf;base64," prefix if it exists
#         if "," in file_base64:
#             file_base64 = file_base64.split(",")[-1]

#         # 2. Decode the pure base64 back to bytes
#         file_bytes = base64.b64decode(file_base64)
        
#         # 3. Process the PDF
#         result = ingest_pdf(file_bytes, filename)
#         result["source"] = source
#         return result
        
#     except Exception as e:
#         return {"status": "error", "error": str(e)}
    
    
    
    

# # RAGAS and LangChain Imports
# from ragas import evaluate
# from ragas.metrics import faithfulness, answer_relevancy, context_recall
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# router = APIRouter()

# # 1. Define the Pydantic Request Model
# class EvaluationRequest(BaseModel):
#     question: str
#     answer: str
#     contexts: List[str]  # The raw text chunks retrieved from Qdrant
#     ground_truth: str = "" # Optional: What the perfect answer should have been

# @router.post("/api/evaluate")
# async def evaluate_rag_response(req: EvaluationRequest):
#     """
#     Evaluates the quality of the PDF Semantic Engine's response using RAGAS.
#     """
#     try:
#         # 2. Configure Models using your existing Google API Key
#         # Note: Ragas prefers the 'pro' model for evaluation logic, but flash works too.
#         eval_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash") 
#         eval_embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

#         # 3. Format Data for RAGAS (Requires a HuggingFace Dataset)
#         data = {
#             "question": [req.question],
#             "answer": [req.answer],
#             "contexts": [req.contexts],
#             "ground_truth": [req.ground_truth] if req.ground_truth else [""]
#         }
#         dataset = Dataset.from_dict(data)

#         # 4. Run the Evaluation
#         # Faithfulness: Is the answer hallucinated or backed by the contexts?
#         # Answer Relevancy: Does it actually answer the user's question?
#         # Context Recall: Did Qdrant retrieve the right info? (Requires ground_truth)
#         metrics_to_run = [faithfulness, answer_relevancy]
        
#         if req.ground_truth:
#             metrics_to_run.append(context_recall)

#         result = evaluate(
#             dataset=dataset,
#             metrics=metrics_to_run,
#             llm=eval_llm,
#             embeddings=eval_embeddings,
#             raise_exceptions=False 
#         )

#         # 5. Return structured scores
#         return {
#             "status": "success",
#             "scores": result.to_pandas().to_dict(orient="records")[0]
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

from app.engines.evaluator import evaluate_rag_response

# Evaluation store - keeps last 50 evaluations in memory
evaluation_history = []

@router.post("/evaluate")
async def evaluate_response(request: Request):
    """
    Evaluates a RAG response using RAGAS metrics.
    
    Send a question, the answer your system gave, and the
    context chunks that were used. Get back quality scores.
    
    Use this endpoint to measure how good your RAG pipeline is.
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

    # Store in history
    if result["status"] == "success":
        evaluation_history.append(result)
        # Keep only last 50
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
    Single endpoint that:
    1. Queries your PDF documents
    2. Automatically evaluates the response quality
    3. Returns both the answer AND the quality scores
    
    This is the showcase endpoint - shows answer + trustworthiness score together.
    """
    # Step 1: Get the answer
    query_result = query_pdf(question, session_id)

    if query_result["status"] == "error":
        return query_result

    # Step 2: Evaluate the answer
    eval_result = evaluate_rag_response(
        question=question,
        answer=query_result["answer"],
        contexts=query_result.get("contexts", []),
        ground_truth=ground_truth
    )

    # Step 3: Combine results
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
    Used by the dashboard to show quality trends over time.
    """
    if not evaluation_history:
        return {
            "status": "empty",
            "message": "No evaluations yet. Use /query-and-evaluate to start.",
            "evaluations": []
        }

    # Calculate averages
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
        "evaluations": evaluation_history[-10:]  # Return last 10
    }