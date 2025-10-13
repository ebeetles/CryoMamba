from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.middleware import add_error_handling_middleware
import logging
import os

# Configure logging
logging.basicConfig(level=getattr(logging, settings.log_level.upper()))
logger = logging.getLogger(__name__)

# Create FastAPI app instance
app = FastAPI(
    title=settings.app_name,
    description="FastAPI server for cryo-ET segmentation inference",
    version=settings.app_version,
    debug=settings.debug
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add error handling middleware
add_error_handling_middleware(app)

# Include routers
from app.routes import health, info, jobs

app.include_router(health.router, prefix="/v1")
app.include_router(info.router, prefix="/v1")
app.include_router(jobs.router, prefix="/v1")

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "CryoMamba Inference Server", "version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
