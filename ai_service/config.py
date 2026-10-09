from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configuration for the AI Service."""

    # Service settings
    host: str = "127.0.0.1"
    port: int = 8000
    debug: bool = False

    # Model settings
    models_path: str = "./models"
    device: str = "auto"  # auto, cuda, or cpu

    # Inference settings
    enable_inference: bool = True
    max_model_memory: int = 8000  # MB

    # Request settings
    request_timeout: int = 300  # seconds
    max_image_size: int = 16384  # pixels

    class Config:
        env_prefix = "AI_"
        env_file = ".env"
        case_sensitive = False


settings = Settings()
