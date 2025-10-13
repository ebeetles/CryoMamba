# 5. Risks & Mitigations

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
