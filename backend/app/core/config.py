from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "SmartOps Agent"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    debug: bool = False           # SAFE default — override with DEBUG=true in dev .env only
    google_api_key: str = ""
    cors_origins: str = (
        "http://localhost:3000,"
        "https://smart-ops-eight.vercel.app,"
        "https://smart-7iqexaluz-aswin-panengals-projects.vercel.app,"
        "https://smart-ops-git-main-aswin-panengals-projects.vercel.app"
    )

    # Env vars pasted into dashboards (Render, Vercel, ...) can pick up a
    # trailing newline or stray whitespace with no visible sign of it — that
    # alone is enough to corrupt an HTTP client's request line/host header.
    # Strip every string field so a stray "\n" can't silently break requests.
    @field_validator(
        "app_name", "qdrant_host", "qdrant_url", "qdrant_api_key",
        "google_api_key", "cors_origins",
        mode="before",
    )
    @classmethod
    def _strip_whitespace(cls, v):
        return v.strip() if isinstance(v, str) else v

    @property
    def allowed_origins(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    model_config = {"env_file": ".env"}


settings = Settings()
