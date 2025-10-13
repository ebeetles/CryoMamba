# 6. Runtime Design & Concurrency

## Desktop
- Qt main thread + asyncio (qasync)
- Bounded chunk upload
- WS listener decodes previews async
- Drop old previews if UI lags
- Reconnect WS if dropped

## Server
- Uvicorn workers + orchestrator task
- 1 job/GPU
- sliding-window inference + preview downsample
- preview broadcast channel, drop-if-full
- OOM probing, cancel tokens

## Upload
- Chunk size 8–32 MB, SHA256 optional, resume via fingerprint

## Voxel Semantics
- Arrays ZYX, integer downsampling to preserve label alignment

## Performance
- MIP rendering by default, FP16 inference, pinned memory

---
