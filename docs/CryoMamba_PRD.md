# 🧊 CryoMamba — Product Requirements Document (PRD)

## 1. Background & Goals

Recent advances in Cryo-ET automation and direct electron detectors have dramatically increased the volume and resolution of tomograms, outpacing manual segmentation workflows. Traditional tools like UCSF ChimeraX excel at visualization but are not designed for scalable, automated segmentation.

The **CryoMamba** platform aims to deliver a **macOS desktop application** coupled with a **remote GPU inference server** to enable interactive visualization, automated segmentation using nnU-Net, and real-time feedback through multi-resolution previews. The solution is expected to **reduce segmentation turnaround time from hours to minutes**, **standardize results**, and **support both individual researchers and high-throughput core facilities**.

**Primary Goals:**
- Enable **interactive 3D visualization** of `.mrc` tomograms with real-time overlay of segmentation masks.  
- Provide a **simple interface** to configure and submit nnU-Net inference jobs to a remote GPU server.  
- Offer **multi-resolution previews** during inference for real-time feedback.  
- Support **manual editing** and **artifact filtering** to refine results.  
- Support **export** in standard formats (MRC, NIfTI, NRRD) and generate basic statistics.

**Target Users:**  
- Cryo-ET structural biologists  
- Computational biologists and image analysts  
- Core facility operators managing high-throughput Cryo-ET pipelines

---

## 2. Functional Requirements

### 2.1 Project & Data Management
- Import single or multiple `.mrc` tomograms  
- Display metadata: dimensions, voxel spacing, intensity statistics  
- Cache recent files and visualization states  
- Persist per-file visualization configuration (camera, WL, colormap, overlays)

### 2.2 Inference Configuration & Submission
- Configure nnU-Net inference parameters: patch size, overlap, TTA, FP16, post-processing toggles  
- Choose between uploading the tomogram or referencing a server-side path  
- Support resumable, encrypted file uploads with integrity checks  
- Submit jobs to the remote server; display status, logs, ETA; allow cancel/retry

### 2.3 Visualization (ChimeraX-like)
- Orthogonal slice views (XY, XZ, YZ) + 3D volume rendering  
- Linked crosshair navigation and window/level adjustments  
- Overlay segmentation masks with adjustable transparency and colormaps  
- Multi-resolution preview: downsampled mask arrives during inference, full-res replaces it on completion  
- Optional manual editing: brush/erase, 3D region growing, small artifact filtering  
- Screenshot and short video export (PNG/MP4/GIF with scale bar)

### 2.4 Export & Reporting
- Export final segmentation in `.mrc`, NIfTI, NRRD  
- Export label statistics: voxel counts, volumes, surface area  
- Generate structured report per tomogram (JSON/CSV)

### 2.5 Server Interaction
- Upload files via resumable API  
- Create and manage inference jobs  
- Receive real-time updates via WebSocket for progress and previews  
- Download final artifacts through signed URLs

---

## 3. Non-Functional Requirements

### Performance
- Desktop must maintain **≥30 FPS** during volume navigation and overlays for typical 512³ tomograms.  
- Inference jobs should return **preview masks within 30–90s** and complete full-resolution masks within minutes.  
- Uploads should handle **up to ~10 GB** efficiently via chunked transfers.

### Scalability
- Server must handle multiple concurrent jobs and support future multi-GPU scaling.

### Usability
- Familiar to ChimeraX users; intuitive controls and sensible defaults.

### Reliability
- Robust upload/job recovery; graceful handling of network interruptions.

### Security
- HTTPS for all traffic; bearer tokens; encrypted uploads; signed artifact URLs.

### Portability
- macOS `.app` bundle for desktop; Dockerized server for CUDA hosts.

---

## 4. Success Metrics

### Performance & Productivity
- Segmentation turnaround **≤5 min** for 512³ tomogram.  
- ≥30 FPS visualization with overlays.  
- ≥95% upload success rate.

### Adoption & Satisfaction
- ≥5 labs using CryoMamba in 3 months.  
- ≥80% users report reduced segmentation time.

### Accuracy & Reproducibility
- Dice ≥0.85 vs manual labels on benchmarks.  
- Deterministic outputs across runs.

### Operational Stability
- ≥90% job success rate.  
- 100% encrypted server communication.

---

## 5. Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Large file handling | High | Medium | Chunked uploads, resumable transfers |
| GPU bottlenecks | High | Medium | Job queue, GPU limits, later scaling |
| Preview streaming | Medium | Medium | Start simple, optimize later |
| UI performance | High | Medium | Napari MIP, optional downsampling |
| nnU-Net setup | Medium | Medium | Containerize environment |
| Editing scope creep | Medium | Medium | Limit MVP to basic brush + region grow |
| Data privacy | High | Low–Med | Encryption, signed URLs, on-prem option |
| User adoption | Medium | Medium | Familiar UX, ChimeraX metaphors |

---

## 6. Dependencies & Timeline

### Dependencies
- **Desktop**: `napari`, `PySide6`, `mrcfile`, `numpy`  
- **Server**: `fastapi`, `uvicorn`, `pydantic`, `torch`, `nnunetv2`, `websockets`  
- CUDA GPU infrastructure, HTTPS termination, optional SLURM/K8s

### 5-Day AI-First MVP Timeline

| Day | Focus | Deliverables |
|-----|-------|-------------|
| 1 | Foundation | Desktop+server scaffolds, dummy endpoints, `.mrc` loader |
| 2 | Upload & Preview | Chunked upload, fake WS previews |
| 3 | Inference | nnU-Net integration, real previews |
| 4 | Visualization & Export | Overlay controls, `.mrc`/PNG export |
| 5 | Integration | End-to-end demo run |
