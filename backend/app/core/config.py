from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str = "SmartOps Agent"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    debug: bool = True
    google_api_key: str = ""
    cors_origins: str = (
        "http://localhost:3000,"
        "https://smart-7iqexaluz-aswin-panengals-projects.vercel.app,"
        "https://smart-ops-git-main-aswin-panengals-projects.vercel.app"
    )

    @property
    def allowed_origins(self) -> list[str]:
        return [
            origin.strip()
            for origin in self.cors_origins.split(",")
            if origin.strip()
        ]

    class Config:
        env_file = ".env"

settings = Settings()
