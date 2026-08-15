"""
Health check endpoint.
"""

from fastapi import APIRouter

from backend.app.api.models import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns service status and version.
    """
    return HealthResponse(
        status="healthy",
        version="1.0.0",
    )


__all__ = ["router"]
