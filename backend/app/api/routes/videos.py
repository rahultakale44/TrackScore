"""
Video upload endpoints.
"""

import logging
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from backend.app.api.models import ErrorResponse, VideoUploadResponse
from backend.app.core.config import Config

router = APIRouter()
logger = logging.getLogger(__name__)

# Supported video extensions
SUPPORTED_EXTENSIONS = set(Config.ALLOWED_VIDEO_EXTENSIONS)

# Maximum file size
MAX_FILE_SIZE = Config.MAX_UPLOAD_SIZE

# Upload directory
UPLOAD_DIR = Config.UPLOAD_DIR


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent path traversal attacks.
    
    Args:
        filename: Original filename
    
    Returns:
        Sanitized filename
    """
    # Remove any path separators
    filename = filename.replace("/", "_").replace("\\", "_")
    
    # Remove any potentially dangerous characters
    dangerous_chars = ["<", ">", ":", '"', "|", "?", "*"]
    for char in dangerous_chars:
        filename = filename.replace(char, "_")
    
    return filename


def validate_extension(filename: str) -> bool:
    """
    Validate file extension.
    
    Args:
        filename: Filename to validate
    
    Returns:
        True if extension is supported
    """
    extension = Path(filename).suffix.lower()
    return extension in SUPPORTED_EXTENSIONS


def validate_file_size(size: int) -> bool:
    """
    Validate file size.
    
    Args:
        size: File size in bytes
    
    Returns:
        True if size is within limits
    """
    return 0 < size <= MAX_FILE_SIZE


@router.post("/upload", response_model=VideoUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_video(
    file: UploadFile = File(..., description="Video file to upload"),
):
    """
    Upload a video file for analysis.
    
    Validates:
    - File extension (MP4, MOV, AVI)
    - File size (max 100 MB)
    
    Returns:
    - video_id: Unique identifier for the uploaded video
    - filename: Original filename
    - size_bytes: File size
    - upload_path: Storage path
    """
    # Validate filename exists
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required",
        )
    
    logger.info(f"Received upload request for: {file.filename}")
    
    # Validate extension
    if not validate_extension(file.filename):
        logger.warning(f"Unsupported extension for file: {file.filename}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension. Supported: {', '.join(SUPPORTED_EXTENSIONS)}",
        )
    
    # Read file content
    content = await file.read()
    file_size = len(content)
    
    logger.info(f"File size: {file_size} bytes")
    
    # Validate file size
    if not validate_file_size(file_size):
        logger.warning(f"File too large: {file_size} bytes (max: {MAX_FILE_SIZE})")
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds maximum allowed size of {MAX_FILE_SIZE} bytes",
        )
    
    # Generate unique video ID
    video_id = str(uuid.uuid4())
    
    # Sanitize filename
    safe_filename = sanitize_filename(file.filename)
    
    # Get file extension
    extension = Path(safe_filename).suffix
    
    # Create unique filename
    unique_filename = f"{video_id}{extension}"
    
    # Ensure upload directory exists
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save file
    upload_path = UPLOAD_DIR / unique_filename
    
    try:
        with open(upload_path, "wb") as f:
            f.write(content)
        
        logger.info(f"Saved video to: {upload_path}")
        
        return VideoUploadResponse(
            video_id=video_id,
            filename=file.filename,
            size_bytes=file_size,
            upload_path=str(upload_path),
        )
    except Exception as e:
        logger.error(f"Failed to save video: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save video: {str(e)}",
        )


__all__ = ["router"]
