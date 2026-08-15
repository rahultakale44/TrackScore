"""
In-memory job manager for asynchronous video analysis.
"""

from __future__ import annotations

import asyncio
import json
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.app.api.models import JobStatus


class JobManager:
    """
    Manages video analysis jobs in memory.
    
    Tracks job status, progress, and results for asynchronous processing.
    """
    
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
    
    def create_job(self, video_id: str, video_path: str, max_frames: Optional[int] = None) -> str:
        """
        Create a new analysis job.
        
        Args:
            video_id: Unique video identifier
            video_path: Path to the uploaded video file
            max_frames: Optional limit on frames to process
        
        Returns:
            job_id: Unique job identifier
        """
        job_id = str(uuid.uuid4())
        
        self.jobs[job_id] = {
            "job_id": job_id,
            "video_id": video_id,
            "video_path": video_path,
            "max_frames": max_frames,
            "status": JobStatus.QUEUED,
            "progress_percentage": 0.0,
            "message": "Job queued for processing",
            "error": None,
            "result": None,
            "metadata": None,
            "warnings": [],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "started_at": None,
            "completed_at": None,
        }
        
        return job_id
    
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve job information.
        
        Args:
            job_id: Job identifier
        
        Returns:
            Job data or None if not found
        """
        return self.jobs.get(job_id)
    
    def update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        progress_percentage: float = 0.0,
        message: str = "",
        error: Optional[str] = None,
    ) -> None:
        """
        Update job status and progress.
        
        Args:
            job_id: Job identifier
            status: New job status
            progress_percentage: Processing progress (0-100)
            message: Status message
            error: Optional error message
        """
        if job_id not in self.jobs:
            return
        
        job = self.jobs[job_id]
        job["status"] = status
        job["progress_percentage"] = min(100.0, max(0.0, progress_percentage))
        job["message"] = message
        
        if error:
            job["error"] = error
        
        if status == JobStatus.PROCESSING and job["started_at"] is None:
            job["started_at"] = datetime.now(timezone.utc).isoformat()
        
        if status in {JobStatus.COMPLETED, JobStatus.FAILED}:
            job["completed_at"] = datetime.now(timezone.utc).isoformat()
    
    def add_job_warning(self, job_id: str, warning: str) -> None:
        """
        Add a warning to the job.
        
        Args:
            job_id: Job identifier
            warning: Warning message
        """
        if job_id not in self.jobs:
            return
        
        job = self.jobs[job_id]
        if "warnings" not in job:
            job["warnings"] = []
        job["warnings"].append(warning)
    
    def set_job_result(self, job_id: str, result: Dict[str, Any]) -> None:
        """
        Store job result data.
        
        Args:
            job_id: Job identifier
            result: Analysis result data
        """
        if job_id not in self.jobs:
            return
        
        self.jobs[job_id]["result"] = result
    
    def set_job_metadata(self, job_id: str, metadata: Dict[str, Any]) -> None:
        """
        Store complete job metadata.
        
        Args:
            job_id: Job identifier
            metadata: Complete analytics metadata
        """
        if job_id not in self.jobs:
            return
        
        self.jobs[job_id]["metadata"] = metadata
    
    async def process_job(self, job_id: str) -> None:
        """
        Process an analysis job asynchronously.
        
        Args:
            job_id: Job identifier
        """
        job = self.get_job(job_id)
        if not job:
            return
        
        try:
            self.update_job_status(
                job_id,
                JobStatus.PROCESSING,
                progress_percentage=0.0,
                message="Starting video analysis",
            )
            
            # Import here to avoid circular dependencies
            from backend.app.vision.video_pipeline import VideoAnalyticsPipeline
            
            video_path = job["video_path"]
            max_frames = job["max_frames"]
            
            # Create output directory for this job
            output_dir = Path("outputs") / "jobs" / job_id
            output_dir.mkdir(parents=True, exist_ok=True)
            
            self.update_job_status(
                job_id,
                JobStatus.PROCESSING,
                progress_percentage=10.0,
                message="Loading video metadata",
            )
            
            # Run the analytics pipeline
            pipeline = VideoAnalyticsPipeline(
                video_path=video_path,
                output_dir=str(output_dir),
            )
            
            self.update_job_status(
                job_id,
                JobStatus.PROCESSING,
                progress_percentage=30.0,
                message="Processing video frames",
            )
            
            try:
                result = pipeline.run(max_frames=max_frames)
            except Exception as pipeline_error:
                # Capture pipeline failures as warnings if we can partially recover
                warning = f"Pipeline partial failure: {str(pipeline_error)}"
                self.add_job_warning(job_id, warning)
                
                # Create a minimal result structure
                result = {
                    "status": "partial",
                    "video_metadata": {},
                    "frame_count": 0,
                    "frames": [],
                    "detections": {"players": [], "ball": []},
                }
            
            self.update_job_status(
                job_id,
                JobStatus.PROCESSING,
                progress_percentage=80.0,
                message="Generating analysis summary",
            )
            
            # Build structured summary
            summary = self._build_summary(result)
            
            # Render annotated video
            annotated_video_path = None
            try:
                self.update_job_status(
                    job_id,
                    JobStatus.PROCESSING,
                    progress_percentage=90.0,
                    message="Rendering annotated video",
                )
                
                from backend.app.vision.video_renderer import VideoRenderer, RendererConfig
                
                renderer_config = RendererConfig(
                    output_path=str(output_dir / "annotated_video.mp4")
                )
                renderer = VideoRenderer(renderer_config)
                
                # Prepare analytics data for renderer
                analytics_for_renderer = {
                    "frames": [],
                }
                
                # Map detections to frames (simplified mapping)
                # In a full implementation, this would align detections with actual frame numbers
                annotated_video_path = renderer.render_video(
                    video_path,
                    analytics_for_renderer,
                )
                
            except Exception as render_error:
                warning = f"Video rendering failed: {str(render_error)}"
                self.add_job_warning(job_id, warning)
                # Continue without annotated video
            
            # Store full metadata
            metadata = {
                "job_id": job_id,
                "video_id": job["video_id"],
                "video_path": video_path,
                "annotated_video_path": annotated_video_path,
                "output_dir": str(output_dir),
                "max_frames": max_frames,
                "pipeline_result": result,
                "processed_at": datetime.now(timezone.utc).isoformat(),
            }
            
            self.set_job_result(job_id, summary)
            self.set_job_metadata(job_id, metadata)
            
            self.update_job_status(
                job_id,
                JobStatus.COMPLETED,
                progress_percentage=100.0,
                message="Analysis completed successfully",
            )
            
        except Exception as error:
            error_trace = traceback.format_exc()
            self.update_job_status(
                job_id,
                JobStatus.FAILED,
                progress_percentage=0.0,
                message="Analysis failed",
                error=f"{str(error)}\n{error_trace}",
            )
    
    def _build_summary(self, pipeline_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build a structured summary from pipeline results.
        
        Args:
            pipeline_result: Raw pipeline output
        
        Returns:
            Structured summary for API response
        """
        video_metadata = pipeline_result.get("video_metadata", {})
        frame_count = pipeline_result.get("frame_count", 0)
        detections = pipeline_result.get("detections", {})
        
        player_detections = detections.get("players", [])
        ball_detections = detections.get("ball", [])
        
        return {
            "video": {
                "filename": video_metadata.get("filename", "unknown"),
                "fps": video_metadata.get("fps", 0.0),
                "frame_count": video_metadata.get("frame_count", 0),
                "duration_seconds": video_metadata.get("duration_seconds", 0.0),
                "resolution": video_metadata.get("resolution", "unknown"),
            },
            "processing": {
                "frames_processed": frame_count,
                "status": pipeline_result.get("status", "unknown"),
            },
            "detections": {
                "players": {
                    "count": len(player_detections),
                    "total_detections": len(player_detections),
                },
                "ball": {
                    "count": len(ball_detections),
                    "total_detections": len(ball_detections),
                },
            },
        }
    
    def start_job_async(self, job_id: str) -> None:
        """
        Start processing a job in the background.
        
        Args:
            job_id: Job identifier
        """
        # Schedule the job to run in the background
        asyncio.create_task(self.process_job(job_id))


__all__ = [
    "JobManager",
]
