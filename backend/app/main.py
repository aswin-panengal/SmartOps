from contextlib import asynccontextmanager
from fastapi import FastAPI, Response, status
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.routes import router
from app.engines.semantic import ensure_collection_exists, qdrant


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """
    Bootstrap Qdrant collection at startup so the first ingest/query request
    never races to create it, and so startup failures surface immediately
    rather than on the first user request.
    """
    ensure_collection_exists()
    yield


app = FastAPI(
    title=settings.app_name,
    description="Dual-engine AI operations platform",
    version="1.0.0",
    debug=settings.debug,
    lifespan=lifespan,
    # Disable /docs and /redoc in production to reduce attack surface
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

app.include_router(router, prefix="/api")


@app.get("/")
def root():
    return {"message": f"{settings.app_name} is running"}


@app.get("/health")
def health_check(response: Response):
    try:
        qdrant.get_collections()
        return {"status": "healthy", "qdrant": "connected", "app": settings.app_name}
    except Exception as e:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        # Avoid leaking internal error details in production
        return {
            "status": "unhealthy",
            "qdrant": "unreachable" if not settings.debug else str(e),
        }
