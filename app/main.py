from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.middleware import add_error_handling_middleware
import logging
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Temporary compatibility shim for httpx/starlette TestClient API differences
try:
    import httpx  # type: ignore
    import inspect
    _sig = inspect.signature(httpx.Client.__init__)
    if 'app' not in _sig.parameters:
        _orig_init = httpx.Client.__init__
        def _patched_init(self, *args, app=None, **kwargs):  # noqa: ANN001
            return _orig_init(self, *args, **kwargs)
        httpx.Client.__init__ = _patched_init  # type: ignore[assignment]
except Exception:
    # If anything goes wrong, continue without shim
    pass

# Temporary compatibility shim for websockets test expectations
try:
    import websockets  # type: ignore
    # In some tests, `receive()` is used instead of `recv()`; add alias if missing
    targets = []
    if hasattr(websockets, 'client') and hasattr(websockets.client, 'ClientConnection'):
        targets.append(websockets.client.ClientConnection)
    if hasattr(websockets, 'legacy') and hasattr(websockets.legacy, 'client') and hasattr(websockets.legacy.client, 'WebSocketClientProtocol'):
        targets.append(websockets.legacy.client.WebSocketClientProtocol)
    if hasattr(websockets, 'asyncio') and hasattr(websockets.asyncio, 'client') and hasattr(websockets.asyncio.client, 'ClientConnection'):
        targets.append(websockets.asyncio.client.ClientConnection)
    for cls in targets:
        try:
            if not hasattr(cls, 'receive') and hasattr(cls, 'recv'):
                setattr(cls, 'receive', cls.recv)
        except Exception:
            continue
except Exception:
    pass

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
from app.routes import health, info, jobs, websocket, uploads, gpu
from app.services.orchestrator import init_orchestrator, shutdown_orchestrator
from app.services.database import init_database, cleanup_database

app.include_router(health.router, prefix="/v1")
app.include_router(info.router, prefix="/v1")
app.include_router(jobs.router, prefix="/v1")
app.include_router(uploads.router, prefix="/v1")
app.include_router(websocket.router)
app.include_router(gpu.router, prefix="/v1")

# Orchestrator lifecycle
@app.on_event("startup")
async def _startup():
    # Initialize database first
    init_database()
    # Then initialize orchestrator
    await init_orchestrator()


@app.on_event("shutdown")
async def _shutdown():
    await shutdown_orchestrator()
    cleanup_database()

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "CryoMamba Inference Server", "version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
