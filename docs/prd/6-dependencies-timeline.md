# 6. Dependencies & Timeline

## Dependencies
- **Desktop**: `napari`, `PySide6`, `mrcfile`, `numpy`  
- **Server**: `fastapi`, `uvicorn`, `pydantic`, `torch`, `nnunetv2`, `websockets`  
- CUDA GPU infrastructure, HTTPS termination, optional SLURM/K8s

## 5-Day AI-First MVP Timeline

| Day | Focus | Deliverables |
|-----|-------|-------------|
| 1 | Foundation | Desktop+server scaffolds, dummy endpoints, `.mrc` loader |
| 2 | Upload & Preview | Chunked upload, fake WS previews |
| 3 | Inference | nnU-Net integration, real previews |
| 4 | Visualization & Export | Overlay controls, `.mrc`/PNG export |
| 5 | Integration | End-to-end demo run |
