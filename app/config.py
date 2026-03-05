import os


class Settings:
    """Configuration settings for the application."""
    PROJECT_NAME: str = "Satellite Classification API"
    VERSION: str = "0.1.0"
    MODEL_DIR: str = os.getenv("MODEL_DIR", "models")
    DATA_DIR: str = os.getenv("DATA_DIR", "data")


settings = Settings()
