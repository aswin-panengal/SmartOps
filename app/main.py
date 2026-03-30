from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.routes import router
from qdrant_client import QdrantClient

app = FastAPI(
    title=settings.app_name,
    description="Dual-engine AI operations platform",
    version="1.0.0"
)

# This allows your future Next.js frontend to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register all API routes
app.include_router(router, prefix="/api")

@app.get("/")
def root():
    return {"message": f"{settings.app_name} is running"}

@app.get("/health")
def health_check():
    try:
        client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port
        )
        client.get_collections()
        return {
            "status": "healthy",
            "qdrant": "connected",
            "app": settings.app_name
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "qdrant": str(e)
        }