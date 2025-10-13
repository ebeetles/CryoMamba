# 🌐 Epic 2 — Resumable Uploads & Job Lifecycle

## Summary
Implement chunked upload pipeline, job creation API, and persistent job state machine.

## Why
Large `.mrc` files (5–10 GB) require robust resumable uploads; jobs form the backbone of the inference workflow.

## Scope
- `/uploads/*` chunked endpoints with resume  
- File assembly and integrity check  
- `/jobs` create, get, cancel endpoints  
- In-memory job queue with fake inference

## Out of Scope
- Actual nnU-Net inference  
- Preview downsampling

## Acceptance Criteria
- 10 GB file can upload in chunks; resume after interruption  
- Job records persist correctly; transitions `queued → running → completed`  
- Cancelling a job mid-run works and updates state  
- API responses match Section 7 of architecture doc

## Stories / Tasks
- Implement upload init/part/complete  
- Job state machine & orchestrator stub  
- Cancel flow + error handling  
- Integration test: upload + fake job

## Dependencies
- Epic 1 (server skeleton, fake job)

## Milestone / Time Target
**Day 2**.
