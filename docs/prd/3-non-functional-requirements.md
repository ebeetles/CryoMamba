# 3. Non-Functional Requirements

## Performance
- Desktop must maintain **≥30 FPS** during volume navigation and overlays for typical 512³ tomograms.  
- Inference jobs should return **preview masks within 30–90s** and complete full-resolution masks within minutes.  
- Uploads should handle **up to ~10 GB** efficiently via chunked transfers.

## Scalability
- Server must handle multiple concurrent jobs and support future multi-GPU scaling.

## Usability
- Familiar to ChimeraX users; intuitive controls and sensible defaults.

## Reliability
- Robust upload/job recovery; graceful handling of network interruptions.

## Security
- HTTPS for all traffic; bearer tokens; encrypted uploads; signed artifact URLs.

## Portability
- macOS `.app` bundle for desktop; Dockerized server for CUDA hosts.

---
