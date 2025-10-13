# 🧪 Epic 5 — Integration, Testing & Packaging

## Summary
End-to-end hardening, error handling, packaging desktop as `.app`, documentation.

## Why
Final polish to ensure MVP is stable, reproducible, and easy to distribute.

## Scope
- End-to-end tests (upload → inference → preview → export)  
- Error handling & recovery flows (upload retry, WS reconnect, cancel)  
- PyInstaller `.app` build, codesigning (if available)  
- Documentation (README, API docs)

## Out of Scope
- Multi-GPU scaling  
- Production auth integrations

## Acceptance Criteria
- “First run” can be completed by new user in <10 min following README  
- Cancels, disconnects, and resumable uploads work reliably  
- `.app` launches without developer tools

## Stories / Tasks
- Integration tests  
- Error flows  
- PyInstaller build  
- Docs polish

## Dependencies
- All previous epics

## Milestone / Time Target
**Day 5**.
