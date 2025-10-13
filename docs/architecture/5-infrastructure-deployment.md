# 5. Infrastructure & Deployment

## Environments
- Local dev (CPU, fake previews)
- GPU staging (1 node, TLS proxy)
- Production (multi-GPU future)

## Topology
Desktop → HTTPS+WS → Reverse Proxy → FastAPI → nnU-Net Runner → /data

## Deployment Units
- Server Container (CUDA base)
- Reverse Proxy (TLS termination, WS pass-through)
- Desktop Packaging (PyInstaller .app)

## Config & Secrets
- Env vars for paths, preview rate, concurrency
- Secret keys for tokens & artifact signing

## Resources
- ≥24 GB VRAM GPU, 64 GB RAM, NVMe 1 TB
- One job/GPU concurrent in MVP

## Observability
- JSON logs, optional Prometheus, desktop debug panel

## Reliability & Backups
- Atomic writes, auto-clean uploads, nightly snapshots

## Security Posture
- TLS, bearer tokens, signed URLs
- Future: tenants, audit, hardening

## CI/CD
- GitHub Actions for CPU tests, GPU build on self-hosted runner
- Desktop notarization pipeline

---
