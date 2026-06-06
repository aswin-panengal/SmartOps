import io
import uuid
from pypdf import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import PointStruct, VectorParams, Distance
from app.core.config import settings

# Use Google embeddings instead of local sentence-transformers
embedder = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=settings.google_api_key,
    output_dimensionality=768
)

# Initialize Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=settings.google_api_key,
    temperature=0
)

# Smart connection
if settings.qdrant_url:
    qdrant = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key
    )
else:
    qdrant = QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port
    )

COLLECTION_NAME = "smartops_documents_v2"
VECTOR_SIZE = 768
SUMMARY_TERMS = {
    "summary", "summarize", "summarise", "overview",
    "key points", "main points", "brief", "gist"
}

def ensure_collection_exists():
    """Create the Qdrant collection and required indices if it doesn't exist yet."""
    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME not in existing:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE
            )
        )
        
        qdrant.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="session_id",
            field_schema=models.PayloadSchemaType.KEYWORD
        )

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Read all text from a PDF file."""
    reader = PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def chunk_text(text: str, chunk_size: int = 150, overlap: int = 30) -> list:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def ingest_pdf(file_bytes: bytes, filename: str, session_id: str) -> dict:
    """Extracts, chunks, embeds, and stores PDF data safely."""
    try:
        ensure_collection_exists()

        text = extract_text_from_pdf(file_bytes)
        if not text.strip():
            return {"status": "error", "error": "Could not extract text from PDF"}

        chunks = chunk_text(text)
        vectors = embedder.embed_documents(chunks)

        points = []
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "text": chunk,
                    "filename": filename,
                    "session_id": session_id,
                    "chunk_index": i
                }
            ))

        qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )

        return {
            "status": "success",
            "filename": filename,
            "chunks_stored": len(chunks),
            "total_characters": len(text)
        }
    except Exception as e:
        # Prevent ingestion failures from crashing the server
        return {"status": "error", "error": f"PDF Ingestion Failed: {str(e)}"}

def query_pdf(question: str, session_id: str = "default") -> dict:
    """Memory-aware RAG query pipeline with robust error catching."""
    try:
        from app.memory.session import (
            build_context_for_prompt,
            add_message
        )

        ensure_collection_exists()

        conversation_context = build_context_for_prompt(session_id)
        question_vector = embedder.embed_query(question)
        lower_question = question.lower()
        is_summary_request = any(term in lower_question for term in SUMMARY_TERMS)
        result_limit = 10 if is_summary_request else 5

        search_response = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=question_vector,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="session_id",
                        match=models.MatchValue(value=session_id)
                    )
                ]
            ),
            limit=result_limit,
            with_payload=True  # CRITICAL FIX: Force Qdrant to return the text chunks
        )

        results = search_response.points

        if not results:
            return {
                "status": "error",
                "error": "I couldn't find any indexed PDF content for this chat. Please upload the PDF again and ask your question."
            }

        # SAFE PARSING: Use .get() to prevent NoneType crashes if payload is ever missing
        context_chunks = [r.payload.get("text", "") for r in results if r.payload]
        context_chunks = [chunk for chunk in context_chunks if chunk.strip()]

        if not context_chunks:
            return {
                "status": "error",
                "error": "I found PDF matches, but their text payload was empty. Please re-upload the PDF."
            }

        context = "\n\n---\n\n".join(context_chunks)
        sources = list(set([r.payload.get("filename", "Unknown") for r in results if r.payload]))
        scores = [
            round(float(r.score), 4)
            for r in results
            if getattr(r, "score", None) is not None
        ]
        best_score = max(scores) if scores else None

        prompt = f"""
        You are a helpful assistant answering questions based on company documents.
        
        {f"Conversation history:{conversation_context}" if conversation_context else ""}
        
        Use ONLY the document context below to answer the question.
        If the answer is not in the context, say "I couldn't find this in the uploaded documents."
        Use conversation history to understand follow-up questions.
        If the user asks for a summary or overview, summarize the available document context clearly.
        
        Document context:
        {context}
        
        Current question: {question}
        
        Give a clear, concise answer.
        """

        response = llm.invoke(prompt)
        answer = response.content

        add_message(session_id, "user", question)
        add_message(session_id, "assistant", answer)

        return {
            "status": "success",
            "question": question,
            "answer": answer,
            "sources": sources,
            "chunks_used": len(results),
            "session_id": session_id,
            "contexts": context_chunks,
            "retrieval_scores": scores,
            "best_score": best_score
        }
    except Exception as e:
        # Safely catch any API limits or database drops and return them cleanly to the UI
        return {
            "status": "error",
            "error": f"Semantic Engine Error: {str(e)}"
        }
