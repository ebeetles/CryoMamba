# 🧠 Epic 3 — Real nnU-Net Inference & Preview Streaming

## Summary
Integrate nnU-Net v2 into server job runner; enable real segmentation inference and downsampled preview streaming.

## Why
Transforms the fake loop into actual segmentation functionality. Critical to deliver MVP segmentation output.

## Scope
- nnU-Net inference wrapper (sliding window)  
- GPU scheduling (1 job/GPU)  
- Periodic downsampled previews via WS  
- Postprocessing & artifact writing (`mask.mrc`, NIfTI, stats)

## Out of Scope
- Multi-GPU scheduling  
- Advanced error recovery

## Acceptance Criteria
- Real tomogram runs end-to-end on GPU, emits previews every 1–2 s, and returns final mask aligned to source  
- Dice score ≥0.85 on benchmark volumes  
- OOM handled gracefully with error message  
- Artifacts written atomically, accessible via signed URLs

## Stories / Tasks
- nnU-Net wrapper module  
- Preview downsampling & streaming  
- GPU orchestrator (single-GPU)  
- Artifact writing + stats

## Dependencies
- Epics 1–2

## Milestone / Time Target
**Day 3**.
