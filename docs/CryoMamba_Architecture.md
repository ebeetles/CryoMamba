# 🧠 CryoMamba — System Architecture Document

## 1. Architecture Overview & Scope

### 1.1 Problem Context
Cryo-ET tomograms are large 3D volumes (hundreds of GB in cohorts; single files up to ~10 GB). Users need interactive visualization on a Mac desktop while offloading heavy segmentation (nnU-Net) to a remote GPU server. The solution must stream previews mid-inference and deliver a final mask that aligns voxel-to-voxel with the source volume.

### 1.2 High-Level Solution
A thin-but-capable **desktop client** (napari + Qt) handles visualization and orchestration, while a **GPU inference server** (FastAPI + nnU-Net v2) handles uploads, job queueing, inference, and preview streaming.

### 1.3 Core Components
- **Desktop App** — visualization, job config, uploads, previews, export  
- **Inference API Server** — REST + WS, nnU-Net runner, artifact manager  
- **Storage** — file-based for MVP (`/data/raw`, `/data/artifacts`)  
- **Auth & Delivery** — HTTPS, bearer tokens, signed URLs

---

## 2. Detailed Component Architecture

### 2.1 Desktop Application
- UI: Qt menus, dialogs, status bar
- Visualization: napari (3D volume, labels overlays)
- I/O: `mrcfile` for reading/writing; nibabel/pynrrd for export
- Networking: REST for control/data, WS for streaming previews
- Data Structures: Volume object, JobConfig, Mask overlay layers

### 2.2 Inference Server
- API Layer (FastAPI)
- Job Orchestrator (async queue)
- nnU-Net Runner (sliding window, periodic previews)
- Artifact Manager (mask.mrc, NIfTI, NRRD, stats.json)
- Job Record schema with state, artifacts, error info

### 2.3 Protocols
| Purpose | Method | Why |
|---------|--------|-----|
| Upload | HTTP PUT (chunked) | Reliable large transfer |
| Job control | HTTP POST/GET/DELETE | Simplicity |
| Previews | WebSocket | Low latency streaming |
| Artifacts | Signed GET | Secure & cacheable |

---

## 3. Data Flow & Interactions

### 3.1 End-to-End Happy Path
1. Load `.mrc` → show metadata  
2. Upload or specify server path  
3. Submit job via REST  
4. Open WS for previews  
5. Server runs inference → emits progress + preview  
6. On completion, download mask and overlay  
7. Export artifacts locally

### 3.2 Upload Flow
- Chunked upload (8–32 MB parts)  
- Resume supported via fingerprint + part list  
- Assemble to temp, then atomic rename

### 3.3 Job Lifecycle
- queued → running → previewing → completed | failed | canceled
- cancellation tokens shared with runner
- crash recovery on boot

### 3.4 Preview Streaming
- Downsampled previews (×4..×8) periodically  
- Latest only; overwrite queue of size 1  
- WS keepalive every 20–30 s

### 3.5 Error & Recovery
- Retry chunks on failure
- Reconnect WS with backoff
- GPU OOM triggers fail with suggestion
- Corrupt MRC rejected early

---

## 4. Data Models & Schemas

### Volume Metadata
Includes filename, shape, voxel_size, intensity stats.

### Upload Session & File Record
Upload tracked by upload_id; assembled into FileRecord.

### Job Record
Tracks job_id, state, params, artifacts, errors.

### Preview Message
JSON with scale, shape, dtype, base64 payload.

### Artifact Stats
JSON of label voxel counts, physical volume, surface area.

### Auth Tokens
Bearer tokens with scopes, expiry.

---

## 5. Infrastructure & Deployment

### Environments
- Local dev (CPU, fake previews)
- GPU staging (1 node, TLS proxy)
- Production (multi-GPU future)

### Topology
Desktop → HTTPS+WS → Reverse Proxy → FastAPI → nnU-Net Runner → /data

### Deployment Units
- Server Container (CUDA base)
- Reverse Proxy (TLS termination, WS pass-through)
- Desktop Packaging (PyInstaller .app)

### Config & Secrets
- Env vars for paths, preview rate, concurrency
- Secret keys for tokens & artifact signing

### Resources
- ≥24 GB VRAM GPU, 64 GB RAM, NVMe 1 TB
- One job/GPU concurrent in MVP

### Observability
- JSON logs, optional Prometheus, desktop debug panel

### Reliability & Backups
- Atomic writes, auto-clean uploads, nightly snapshots

### Security Posture
- TLS, bearer tokens, signed URLs
- Future: tenants, audit, hardening

### CI/CD
- GitHub Actions for CPU tests, GPU build on self-hosted runner
- Desktop notarization pipeline

---

## 6. Runtime Design & Concurrency

### Desktop
- Qt main thread + asyncio (qasync)
- Bounded chunk upload
- WS listener decodes previews async
- Drop old previews if UI lags
- Reconnect WS if dropped

### Server
- Uvicorn workers + orchestrator task
- 1 job/GPU
- sliding-window inference + preview downsample
- preview broadcast channel, drop-if-full
- OOM probing, cancel tokens

### Upload
- Chunk size 8–32 MB, SHA256 optional, resume via fingerprint

### Voxel Semantics
- Arrays ZYX, integer downsampling to preserve label alignment

### Performance
- MIP rendering by default, FP16 inference, pinned memory

---

## 7. API Specifications

### Health & Info
- `/v1/healthz`, `/v1/server/info`

### Uploads
- `POST /uploads/init`, `PUT /uploads/{id}/part/{idx}`, `POST /uploads/{id}/complete`

### Jobs
- `POST /jobs` (create), `GET /jobs/{id}` (status), `DELETE /jobs/{id}` (cancel)

### WebSocket
- `/ws/jobs/{id}`: progress, preview, completed, error messages

### Artifacts
- Signed GET `/files/{job}/{artifact}`

### Error Model
- JSON envelope with code, message, hint, retryable

### Security
- HTTPS, JWT bearer, signed URLs, no raw data in logs

### Rate Limits
- 60 req/min per token; upload concurrency by server policy

---

## 8. Key Design Decisions

- napari for visualization: fast, supports labels & Qt integration
- WS for preview: lower latency than HTTP polling
- File system for artifacts: simple MVP; object storage later
- One GPU job at a time: predictable resource use
- Integer downsampling for previews: exact alignment

---

## 9. Future Extensions

- Multi-GPU or SLURM/K8s orchestration
- SSE or gRPC streams for previews
- Distributed object store for artifacts
- User quotas, per-tenant isolation
- Advanced manual editing on desktop
- Built-in labeling QA workflows

---

© 2025 CryoMamba Architecture — v1.0.0
