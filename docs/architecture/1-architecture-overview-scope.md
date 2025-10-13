# 1. Architecture Overview & Scope

## 1.1 Problem Context
Cryo-ET tomograms are large 3D volumes (hundreds of GB in cohorts; single files up to ~10 GB). Users need interactive visualization on a Mac desktop while offloading heavy segmentation (nnU-Net) to a remote GPU server. The solution must stream previews mid-inference and deliver a final mask that aligns voxel-to-voxel with the source volume.

## 1.2 High-Level Solution
A thin-but-capable **desktop client** (napari + Qt) handles visualization and orchestration, while a **GPU inference server** (FastAPI + nnU-Net v2) handles uploads, job queueing, inference, and preview streaming.

## 1.3 Core Components
- **Desktop App** — visualization, job config, uploads, previews, export  
- **Inference API Server** — REST + WS, nnU-Net runner, artifact manager  
- **Storage** — file-based for MVP (`/data/raw`, `/data/artifacts`)  
- **Auth & Delivery** — HTTPS, bearer tokens, signed URLs

---
