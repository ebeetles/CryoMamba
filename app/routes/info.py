from fastapi import APIRouter
from datetime import datetime
import logging
import platform
import sys

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/server/info")
async def server_info():
    """
    Server information endpoint
    Returns server details and system information
    """
    logger.info("Server info requested")
    return {
        "service": "cryomamba-inference-server",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "system": {
            "platform": platform.platform(),
            "python_version": sys.version,
            "architecture": platform.architecture()[0]
        },
        "status": "running"
    }
