# 8. Key Design Decisions

- napari for visualization: fast, supports labels & Qt integration
- WS for preview: lower latency than HTTP polling
- File system for artifacts: simple MVP; object storage later
- One GPU job at a time: predictable resource use
- Integer downsampling for previews: exact alignment

---
