"""
Pydantic models for API request and response validation.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Status of an analysis job."""
    
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")


class VideoUploadResponse(BaseModel):
    """Response after successful video upload."""
    
    video_id: str = Field(..., description="Unique video identifier")
    filename: str = Field(..., description="Original filename")
    size_bytes: int = Field(..., description="File size in bytes")
    upload_path: str = Field(..., description="Storage path")


class AnalysisStartRequest(BaseModel):
    """Request to start video analysis."""
    
    max_frames: Optional[int] = Field(
        None,
        description="Maximum number of frames to process",
        ge=1,
    )


class AnalysisStartResponse(BaseModel):
    """Response after starting analysis."""
    
    job_id: str = Field(..., description="Unique job identifier")
    video_id: str = Field(..., description="Video identifier")
    status: JobStatus = Field(..., description="Initial job status")


class AnalysisStatusResponse(BaseModel):
    """Analysis job status response."""
    
    job_id: str = Field(..., description="Job identifier")
    video_id: str = Field(..., description="Video identifier")
    status: JobStatus = Field(..., description="Current job status")
    progress_percentage: float = Field(
        ...,
        description="Processing progress (0-100)",
        ge=0,
        le=100,
    )
    message: str = Field(..., description="Status message")
    error: Optional[str] = Field(None, description="Error message if failed")


class AnalysisResultResponse(BaseModel):
    """Structured analysis summary response."""
    
    job_id: str = Field(..., description="Job identifier")
    video_id: str = Field(..., description="Video identifier")
    status: JobStatus = Field(..., description="Job status")
    summary: Dict[str, Any] = Field(..., description="Analysis summary")
    warnings: List[str] = Field(
        default_factory=list,
        description="Non-fatal warnings during processing",
    )


class AnalysisMetadataResponse(BaseModel):
    """Full analytics metadata response."""
    
    job_id: str = Field(..., description="Job identifier")
    video_id: str = Field(..., description="Video identifier")
    status: JobStatus = Field(..., description="Job status")
    metadata: Dict[str, Any] = Field(..., description="Complete analytics data")
    warnings: List[str] = Field(
        default_factory=list,
        description="Non-fatal warnings during processing",
    )


class ErrorResponse(BaseModel):
    """Error response."""
    
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")


__all__ = [
    "JobStatus",
    "HealthResponse",
    "VideoUploadResponse",
    "AnalysisStartRequest",
    "AnalysisStartResponse",
    "AnalysisStatusResponse",
    "AnalysisResultResponse",
    "AnalysisMetadataResponse",
    "ErrorResponse",
]
