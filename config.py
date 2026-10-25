import os
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseModel):
    # Server settings
    HOST: str = Field(default=os.getenv("HOST", "0.0.0.0"))
    PORT: int = Field(default=int(os.getenv("PORT", "8000")))
    DEBUG: bool = Field(default=os.getenv("DEBUG", "false").lower() in ("true", "1"))
    
    # Storage settings
    OUTPUT_DIR: str = Field(default=os.getenv("OUTPUT_DIR", "outputs"))
    BASE_URL: str = Field(default=os.getenv("BASE_URL", "http://localhost:8000"))
    
    # Engine & Provider settings
    MUSIC_PROVIDER: str = Field(default=os.getenv("MUSIC_PROVIDER", "procedural"))
    DEFAULT_SAMPLE_RATE: int = Field(default=int(os.getenv("DEFAULT_SAMPLE_RATE", "48000")))
    DEFAULT_TARGET_LUFS: float = Field(default=float(os.getenv("DEFAULT_TARGET_LUFS", "-16.0")))
    DEFAULT_TEMPO_BPM: int = Field(default=int(os.getenv("DEFAULT_TEMPO_BPM", "110")))
    
    # Limits
    MIN_DURATION_S: float = Field(default=float(os.getenv("MIN_DURATION_S", "5.0")))
    MAX_DURATION_S: float = Field(default=float(os.getenv("MAX_DURATION_S", "300.0")))
    MIN_TEMPO_BPM: int = Field(default=int(os.getenv("MIN_TEMPO_BPM", "40")))
    MAX_TEMPO_BPM: int = Field(default=int(os.getenv("MAX_TEMPO_BPM", "220")))

settings = Settings()
