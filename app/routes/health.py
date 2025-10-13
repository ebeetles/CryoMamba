from fastapi import APIRouter
from datetime import datetime
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/healthz")
async def health_check():
    """
    Health check endpoint
    Returns OK status for basic health validation
    """
    logger.info("Health check requested")
    return {
        "status": "OK",
        "timestamp": datetime.now().isoformat(),
        "service": "cryomamba-inference-server"
    }
