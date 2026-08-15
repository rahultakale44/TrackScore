"""
Analysis job endpoints.
"""

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import FileResponse

from backend.app.api.models import (
    AnalysisMetadataResponse,
    AnalysisResultResponse,
    AnalysisStartRequest,
    AnalysisStartResponse,
    AnalysisStatusResponse,
    JobStatus,
)

router = APIRouter()

# Upload directory
UPLOAD_DIR = Path("uploads")


@router.post("/start/{video_id}", response_model=AnalysisStartResponse, status_code=status.HTTP_202_ACCEPTED)
async def start_analysis(
    video_id: str,
    request: Request,
    body: Optional[AnalysisStartRequest] = None,
):
    """
    Start video analysis asynchronously.
    
    Args:
        video_id: Unique video identifier from upload
        body: Optional request body with analysis parameters
    
    Returns:
        job_id: Unique job identifier for tracking analysis
        video_id: Video identifier
        status: Initial job status (queued)
    """
    job_manager = request.app.state.job_manager
    
    # Find the video file
    video_path = None
    for extension in [".mp4", ".mov", ".avi"]:
        candidate = UPLOAD_DIR / f"{video_id}{extension}"
        if candidate.exists():
            video_path = candidate
            break
    
    if not video_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video not found: {video_id}",
        )
    
    # Extract parameters
    max_frames = None
    if body:
        max_frames = body.max_frames
    
    # Create job
    job_id = job_manager.create_job(
        video_id=video_id,
        video_path=str(video_path),
        max_frames=max_frames,
    )
    
    # Start processing asynchronously
    job_manager.start_job_async(job_id)
    
    return AnalysisStartResponse(
        job_id=job_id,
        video_id=video_id,
        status=JobStatus.QUEUED,
    )


@router.get("/status/{job_id}", response_model=AnalysisStatusResponse)
async def get_analysis_status(
    job_id: str,
    request: Request,
):
    """
    Get analysis job status.
    
    Args:
        job_id: Job identifier
    
    Returns:
        Job status, progress percentage, and message
    """
    job_manager = request.app.state.job_manager
    
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {job_id}",
        )
    
    return AnalysisStatusResponse(
        job_id=job["job_id"],
        video_id=job["video_id"],
        status=job["status"],
        progress_percentage=job["progress_percentage"],
        message=job["message"],
        error=job.get("error"),
    )


@router.get("/result/{job_id}", response_model=AnalysisResultResponse)
async def get_analysis_result(
    job_id: str,
    request: Request,
):
    """
    Get structured analysis summary.
    
    Args:
        job_id: Job identifier
    
    Returns:
        Structured analysis summary with video metadata and detection counts
    """
    job_manager = request.app.state.job_manager
    
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {job_id}",
        )
    
    if job["status"] != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Analysis not completed. Current status: {job['status']}",
        )
    
    return AnalysisResultResponse(
        job_id=job["job_id"],
        video_id=job["video_id"],
        status=job["status"],
        summary=job["result"] or {},
        warnings=job.get("warnings", []),
    )


@router.get("/metadata/{job_id}", response_model=AnalysisMetadataResponse)
async def get_analysis_metadata(
    job_id: str,
    request: Request,
):
    """
    Get complete analytics metadata.
    
    Args:
        job_id: Job identifier
    
    Returns:
        Full analytics JSON with all detection data
    """
    job_manager = request.app.state.job_manager
    
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {job_id}",
        )
    
    if job["status"] != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Analysis not completed. Current status: {job['status']}",
        )
    
    return AnalysisMetadataResponse(
        job_id=job["job_id"],
        video_id=job["video_id"],
        status=job["status"],
        metadata=job["metadata"] or {},
        warnings=job.get("warnings", []),
    )


@router.get("/video/{job_id}")
async def get_analysis_video(
    job_id: str,
    request: Request,
):
    """
    Download processed/annotated video.
    
    Args:
        job_id: Job identifier
    
    Returns:
        Video file (annotated if available, otherwise original)
    """
    job_manager = request.app.state.job_manager
    
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {job_id}",
        )
    
    if job["status"] != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Analysis not completed. Current status: {job['status']}",
        )
    
    # Try to serve annotated video first
    metadata = job.get("metadata", {})
    annotated_path = metadata.get("annotated_video_path")
    
    if annotated_path and Path(annotated_path).exists():
        return FileResponse(
            path=annotated_path,
            media_type="video/mp4",
            filename=f"annotated_{job_id}.mp4",
        )
    
    # Fall back to original video
    video_path = Path(job["video_path"])
    
    if not video_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video file not found",
        )
    
    return FileResponse(
        path=video_path,
        media_type="video/mp4",
        filename=f"analysis_{job_id}.mp4",
    )


__all__ = ["router"]
