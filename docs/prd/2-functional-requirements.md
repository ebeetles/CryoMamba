# 2. Functional Requirements

## 2.1 Project & Data Management
- Import single or multiple `.mrc` tomograms  
- Display metadata: dimensions, voxel spacing, intensity statistics  
- Cache recent files and visualization states  
- Persist per-file visualization configuration (camera, WL, colormap, overlays)

## 2.2 Inference Configuration & Submission
- Configure nnU-Net inference parameters: patch size, overlap, TTA, FP16, post-processing toggles  
- Choose between uploading the tomogram or referencing a server-side path  
- Support resumable, encrypted file uploads with integrity checks  
- Submit jobs to the remote server; display status, logs, ETA; allow cancel/retry

## 2.3 Visualization (ChimeraX-like)
- Orthogonal slice views (XY, XZ, YZ) + 3D volume rendering  
- Linked crosshair navigation and window/level adjustments  
- Overlay segmentation masks with adjustable transparency and colormaps  
- Multi-resolution preview: downsampled mask arrives during inference, full-res replaces it on completion  
- Optional manual editing: brush/erase, 3D region growing, small artifact filtering  
- Screenshot and short video export (PNG/MP4/GIF with scale bar)

## 2.4 Export & Reporting
- Export final segmentation in `.mrc`, NIfTI, NRRD  
- Export label statistics: voxel counts, volumes, surface area  
- Generate structured report per tomogram (JSON/CSV)

## 2.5 Server Interaction
- Upload files via resumable API  
- Create and manage inference jobs  
- Receive real-time updates via WebSocket for progress and previews  
- Download final artifacts through signed URLs

---
