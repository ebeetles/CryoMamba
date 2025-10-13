# 3. Data Flow & Interactions

## 3.1 End-to-End Happy Path
1. Load `.mrc` → show metadata  
2. Upload or specify server path  
3. Submit job via REST  
4. Open WS for previews  
5. Server runs inference → emits progress + preview  
6. On completion, download mask and overlay  
7. Export artifacts locally

## 3.2 Upload Flow
- Chunked upload (8–32 MB parts)  
- Resume supported via fingerprint + part list  
- Assemble to temp, then atomic rename

## 3.3 Job Lifecycle
- queued → running → previewing → completed | failed | canceled
- cancellation tokens shared with runner
- crash recovery on boot

## 3.4 Preview Streaming
- Downsampled previews (×4..×8) periodically  
- Latest only; overwrite queue of size 1  
- WS keepalive every 20–30 s

## 3.5 Error & Recovery
- Retry chunks on failure
- Reconnect WS with backoff
- GPU OOM triggers fail with suggestion
- Corrupt MRC rejected early

---
