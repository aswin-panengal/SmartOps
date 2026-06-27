# ── Stage 1: builder ─────────────────────────────────────────────────────────
# Install all dependencies (including build-time C extensions) in an isolated
# layer so they don't bloat the final runtime image.
FROM python:3.11-slim AS builder

WORKDIR /build

# System packages needed only to compile Python C extensions (e.g. numpy)
RUN apt-get update && apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy only the installed packages from the builder — no gcc, no apt cache
COPY --from=builder /install /usr/local

# Copy application source (exclude dev/test files via .dockerignore)
COPY app/ app/

# Non-root user reduces blast radius if the container is compromised
RUN useradd --no-create-home --shell /bin/false appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# HEALTHCHECK lets Docker/Kubernetes detect a degraded container automatically.
# Interval: every 30 s; timeout: 10 s; 3 consecutive failures = unhealthy.
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c \
    "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# 2 workers: enough for Render Free Tier (512 MB RAM) to handle concurrent
# requests without blocking the event loop on sync engine calls.
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--timeout-keep-alive", "75"]
