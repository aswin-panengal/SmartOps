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

# Secure Production CORS Configuration
# Only requests from these explicit domains will be permitted by the server
origins = [
    "http://localhost:3000",                                        # Local Next.js development server
    "https://smart-7iqexaluz-aswin-panengals-projects.vercel.app",  # Your unique Vercel preview domain
    "https://smart-ops-git-main-aswin-panengals-projects.vercel.app" # Your main Vercel branch deployment tracking link
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,  # Safe to set to True now that wildcards are removed
    allow_methods=["*"],     # Restricts or allows standard methods (POST, GET, OPTIONS, etc.)
    allow_headers=["*"],     # Allows all standard request headers
)

# Register all API routes
app.include_router(router, prefix="/api")

@app.get("/")
def root():
    return {"message": f"{settings.app_name} is running"}

@app.get("/health")
def health_check():
    try:
        # Connect to either cloud or local Qdrant
        if settings.qdrant_url:
            client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key
            )
        else:
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