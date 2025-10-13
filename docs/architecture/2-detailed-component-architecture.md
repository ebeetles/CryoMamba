# 2. Detailed Component Architecture

## 2.1 Desktop Application
- UI: Qt menus, dialogs, status bar
- Visualization: napari (3D volume, labels overlays)
- I/O: `mrcfile` for reading/writing; nibabel/pynrrd for export
- Networking: REST for control/data, WS for streaming previews
- Data Structures: Volume object, JobConfig, Mask overlay layers

## 2.2 Inference Server
- API Layer (FastAPI)
- Job Orchestrator (async queue)
- nnU-Net Runner (sliding window, periodic previews)
- Artifact Manager (mask.mrc, NIfTI, NRRD, stats.json)
- Job Record schema with state, artifacts, error info

## 2.3 Protocols
| Purpose | Method | Why |
|---------|--------|-----|
| Upload | HTTP PUT (chunked) | Reliable large transfer |
| Job control | HTTP POST/GET/DELETE | Simplicity |
| Previews | WebSocket | Low latency streaming |
| Artifacts | Signed GET | Secure & cacheable |

---
