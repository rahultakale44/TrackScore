"""
Centralized configuration for TrackScore application.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional


class Config:
    """Application configuration with environment variable support."""
    
    # Base directories
    BASE_DIR = Path(__file__).parent.parent.parent.parent
    UPLOAD_DIR = BASE_DIR / "uploads"
    OUTPUT_DIR = BASE_DIR / "outputs"
    MODELS_DIR = BASE_DIR / "models"
    
    # API settings
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    
    # Upload limits
    MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_SIZE", str(100 * 1024 * 1024)))  # 100MB
    ALLOWED_VIDEO_EXTENSIONS = [".mp4", ".mov", ".avi"]
    
    # Processing settings
    DEFAULT_MAX_FRAMES: Optional[int] = None
    FRAME_EXTRACTION_INTERVAL = int(os.getenv("FRAME_EXTRACTION_INTERVAL", "1"))
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def ensure_directories(cls) -> None:
        """Ensure all required directories exist."""
        directories = [
            cls.UPLOAD_DIR,
            cls.OUTPUT_DIR,
            cls.OUTPUT_DIR / "jobs",
            cls.MODELS_DIR,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def setup_logging(cls) -> None:
        """Configure application logging."""
        logging.basicConfig(
            level=getattr(logging, cls.LOG_LEVEL),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


# Initialize directories and logging on import
Config.ensure_directories()
Config.setup_logging()


__all__ = ["Config"]
