import io
import uuid
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import PointStruct, VectorParams, Distance
from qdrant_client.http.exceptions import UnexpectedResponse
from app.core.config import settings

MAX_PDF_BYTES = 50 * 1024 * 1024  # 50 MB hard limit
EMBED_BATCH_SIZE = 50              # chunks per embedding API call
UPSERT_BATCH_SIZE = 100            # points per Qdrant upsert call

embedder = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=settings.google_api_key,
    output_dimensionality=768,
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=settings.google_api_key,
    temperature=0,
)

if settings.qdrant_url:
    qdrant = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
else:
    qdrant = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)

COLLECTION_NAME = "smartops_documents_v2"
VECTOR_SIZE = 768
SUMMARY_TERMS = {
    "summary", "summarize", "summarise", "overview",
    "key points", "main points", "brief", "gist",
}

# Module-level flag so we only call get_collections() once per process lifetime.
# Reset to False if a collection-level error is detected at runtime.
_collection_ready = False


def ensure_collection_exists() -> None:
    """
    Idempotent collection bootstrap. Uses a module flag to skip the
    get_collections() round-trip on every query after the first check.
    Handles the race where two requests simultaneously try to create it.
    """
    global _collection_ready
    if _collection_ready:
        return

    try:
        existing = {c.name for c in qdrant.get_collections().collections}
        if COLLECTION_NAME not in existing:
            try:
                qdrant.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
                )
                qdrant.create_payload_index(
                    collection_name=COLLECTION_NAME,
                    field_name="session_id",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            except UnexpectedResponse:
                # Another request beat us to creation — that's fine.
                pass
        _collection_ready = True
    except Exception:
        # Leave _collection_ready = False so the next request retries.
        raise


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF. Raises PdfReadError for encrypted/corrupt files."""
    reader = PdfReader(io.BytesIO(file_bytes))
    if reader.is_encrypted:
        raise PdfReadError("PDF is password-protected. Please upload an unlocked file.")
    return "".join(page.extract_text() or "" for page in reader.pages)


def chunk_text(text: str, chunk_size: int = 150, overlap: int = 30) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    start = 0
    while start < len(words):
        chunks.append(" ".join(words[start : start + chunk_size]))
        start += chunk_size - overlap
    return chunks


def _embed_in_batches(chunks: list[str]) -> list[list[float]]:
    """Embed chunks in batches to avoid hitting per-request token limits."""
    vectors: list[list[float]] = []
    for i in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[i : i + EMBED_BATCH_SIZE]
        vectors.extend(embedder.embed_documents(batch))
    return vectors


def ingest_pdf(file_bytes: bytes, filename: str, session_id: str) -> dict:
    """Extract, chunk, embed, and store PDF data with batched upserts."""
    try:
        if len(file_bytes) > MAX_PDF_BYTES:
            return {
                "status": "error",
                "error": f"PDF exceeds the {MAX_PDF_BYTES // (1024*1024)} MB limit.",
            }

        ensure_collection_exists()

        try:
            text = extract_text_from_pdf(file_bytes)
        except PdfReadError as e:
            return {"status": "error", "error": str(e)}

        if not text.strip():
            return {
                "status": "error",
                "error": (
                    "No text could be extracted from this PDF. "
                    "It may be a scanned image. Please use a text-based PDF."
                ),
            }

        chunks = chunk_text(text)
        vectors = _embed_in_batches(chunks)

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "text": chunk,
                    "filename": filename,
                    "session_id": session_id,
                    "chunk_index": i,
                },
            )
            for i, (chunk, vector) in enumerate(zip(chunks, vectors))
        ]

        # Batch upserts to avoid timeout on large documents
        for i in range(0, len(points), UPSERT_BATCH_SIZE):
            qdrant.upsert(
                collection_name=COLLECTION_NAME,
                points=points[i : i + UPSERT_BATCH_SIZE],
            )

        return {
            "status": "success",
            "filename": filename,
            "chunks_stored": len(chunks),
            "total_characters": len(text),
        }
    except Exception as e:
        return {"status": "error", "error": f"PDF Ingestion Failed: {type(e).__name__}: {e}"}


def query_pdf(question: str, session_id: str = "default") -> dict:
    """Memory-aware RAG query pipeline."""
    try:
        from app.memory.session import build_context_for_prompt, add_message

        # Collection must exist; if called after ingest it's already flagged ready.
        ensure_collection_exists()

        conversation_context = build_context_for_prompt(session_id)
        question_vector = embedder.embed_query(question)
        is_summary = any(t in question.lower() for t in SUMMARY_TERMS)
        result_limit = 10 if is_summary else 5

        search_response = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=question_vector,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="session_id",
                        match=models.MatchValue(value=session_id),
                    )
                ]
            ),
            limit=result_limit,
            with_payload=True,
        )

        results = search_response.points

        if not results:
            return {
                "status": "error",
                "error": (
                    "I couldn't find any indexed PDF content for this session. "
                    "Please upload the PDF again and ask your question."
                ),
            }

        context_chunks = [r.payload.get("text", "") for r in results if r.payload]
        context_chunks = [c for c in context_chunks if c.strip()]

        if not context_chunks:
            return {
                "status": "error",
                "error": "PDF matches were found but their text payload was empty. Please re-upload.",
            }

        context = "\n\n---\n\n".join(context_chunks)
        sources = list({r.payload.get("filename", "Unknown") for r in results if r.payload})
        scores = [
            round(float(r.score), 4)
            for r in results
            if getattr(r, "score", None) is not None
        ]
        best_score = max(scores) if scores else None

        prompt = f"""You are a helpful assistant answering questions based on company documents.
{f"Conversation history:{conversation_context}" if conversation_context else ""}

Use ONLY the document context below to answer the question.
If the answer is not in the context, say "I couldn't find this in the uploaded documents."
Use conversation history to understand follow-up questions.

Document context:
{context}

Current question: {question}

Give a clear, concise answer."""

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
            "best_score": best_score,
        }
    except Exception as e:
        return {"status": "error", "error": f"Semantic Engine Error: {type(e).__name__}: {e}"}
