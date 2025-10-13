# 🧱 Epic 1 — Foundation: Desktop & Server Scaffolds

## Summary
Establish the foundational codebases for the macOS desktop app and FastAPI GPU server, enabling basic `.mrc` file loading, fake job submission, and fake preview streaming.

## Why
Provides the minimal runnable skeleton that all subsequent features (upload, inference, visualization) build on. Directly supports Sections 1–3 of the Architecture Doc.

## Scope
- Desktop app skeleton (Qt + napari) with `.mrc` loader and metadata panel  
- Server skeleton (FastAPI) with health endpoints and dummy job processing  
- WebSocket fake preview streaming for early UI testing  
- Dockerfile & local dev instructions

## Out of Scope
- Real nnU-Net integration  
- Resumable uploads  
- Advanced error handling

## Acceptance Criteria
- `uvicorn app.main:app` runs locally with `/healthz` returning OK  
- Desktop can open `.mrc`, display metadata, connect to WS, and show fake mask overlay at 1 Hz  
- Entire loop runs on CPU with no GPU required  
- One developer can clone repo, run server and desktop, and complete a fake job E2E

## Stories / Tasks
- Scaffold FastAPI app (health route, dummy job route)  
- Scaffold napari app (Qt menus, `.mrc` loader)  
- Fake preview WS (base64 random mask stream)  
- Dockerfile + local compose  
- README setup instructions

## Dependencies
None — this is the first epic.

## Milestone / Time Target
**Day 1** of 5-day sprint.
