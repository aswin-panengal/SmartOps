from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str = "SmartOps Agent"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    debug: bool = True
    google_api_key: str = ""

    class Config:
        env_file = ".env"

settings = Settings()