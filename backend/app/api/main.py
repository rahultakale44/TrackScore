"""
FastAPI application for TrackScore video analysis.

Provides endpoints for:
- Video upload
- Asynchronous analysis job management
- Result retrieval
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.job_manager import JobManager
from backend.app.api.routes import analysis, health, videos


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application.
    
    Initializes shared resources like the job manager.
    """
    # Startup
    job_manager = JobManager()
    app.state.job_manager = job_manager
    
    yield
    
    # Shutdown
    # Clean up resources if needed


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    """
    app = FastAPI(
        title="TrackScore API",
        description="Tennis video analysis and scoring API",
        version="1.0.0",
        lifespan=lifespan,
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Register routes
    app.include_router(health.router, prefix="/api", tags=["health"])
    app.include_router(videos.router, prefix="/api/videos", tags=["videos"])
    app.include_router(analysis.router, prefix="/api/analysis", tags=["analysis"])
    
    return app


app = create_app()


__all__ = [
    "app",
    "create_app",
]
