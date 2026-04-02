import io
import uuid
from pypdf import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from app.core.config import settings

# Use Google embeddings instead of local sentence-transformers
# Free, no RAM cost, better quality
embedder = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=settings.google_api_key,
    output_dimensionality=768 # Forces the new 3072 model to fit our 768 Qdrant table
)
# Initialize Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=settings.google_api_key,
    temperature=0
)

# Smart connection -use cloude else falls back to local 
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
VECTOR_SIZE = 768  # Google embedding-001 produces 768-dimensional vectors

def ensure_collection_exists():
    """
    Create the Qdrant collection if it doesn't exist yet.
    A collection is like a table in a regular database.
    """
    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME not in existing:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE
            )
        )

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Read all text from a PDF file."""
    reader = PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def chunk_text(text: str, chunk_size: int = 150, overlap: int = 30) -> list:
    """
    Split text into overlapping chunks.
    Overlap ensures we don't lose context at chunk boundaries.
    
    Example with chunk_size=10, overlap=2:
    "ABCDEFGHIJKLMNOP" becomes:
    ["ABCDEFGHIJ", "IJKLMNOPQR", ...]
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def ingest_pdf(file_bytes: bytes, filename: str) -> dict:
    ensure_collection_exists()

    # Step 1: Extract text
    text = extract_text_from_pdf(file_bytes)
    if not text.strip():
        return {"status": "error", "error": "Could not extract text from PDF"}

    # Step 2: Chunk the text
    chunks = chunk_text(text)

    # Step 3: Google embeddings - embed all chunks at once
    vectors = embedder.embed_documents(chunks)

    # Step 4: Store in Qdrant
    points = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "text": chunk,
                "filename": filename,
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

def query_pdf(question: str, session_id: str = "default") -> dict:
    """
    Memory-aware RAG query pipeline:
    1. Load conversation history for this session
    2. Vectorize the question
    3. Find similar chunks in Qdrant
    4. Build prompt with history + context
    5. Get answer from Gemini
    6. Save exchange to memory
    """
    from app.memory.session import (
        build_context_for_prompt,
        add_message
    )

    ensure_collection_exists()

    # Step 1: Get conversation history
    conversation_context = build_context_for_prompt(session_id)

    # Step 2: Vectorize the question
    question_vector = embedder.embed_query(question)

    # Step 3: Find top 4 most relevant chunks
    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=question_vector,
        limit=4
    ).points

    if not results:
        return {
            "status": "error",
            "error": "No documents found. Please upload a PDF first."
        }

    # Step 4: Build context from retrieved chunks
    context_chunks = [r.payload["text"] for r in results]
    context = "\n\n---\n\n".join(context_chunks)
    sources = list(set([r.payload["filename"] for r in results]))

    # Step 5: Build memory-aware prompt
    prompt = f"""
    You are a helpful assistant answering questions based on company documents.
    
    {f"Conversation history:{conversation_context}" if conversation_context else ""}
    
    Use ONLY the document context below to answer the question.
    If the answer is not in the context, say "I couldn't find this in the uploaded documents."
    Use conversation history to understand follow-up questions.
    
    Document context:
    {context}
    
    Current question: {question}
    
    Give a clear, concise answer.
    """

    response = llm.invoke(prompt)
    answer = response.content

    # Step 6: Save this exchange to memory
    add_message(session_id, "user", question)
    add_message(session_id, "assistant", answer)

    return {
        "status": "success",
        "question": question,
        "answer": answer,
        "sources": sources,
        "chunks_used": len(results),
        "session_id": session_id
    }