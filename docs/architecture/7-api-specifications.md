# 7. API Specifications

## Health & Info
- `/v1/healthz`, `/v1/server/info`

## Uploads
- `POST /uploads/init`, `PUT /uploads/{id}/part/{idx}`, `POST /uploads/{id}/complete`

## Jobs
- `POST /jobs` (create), `GET /jobs/{id}` (status), `DELETE /jobs/{id}` (cancel)

## WebSocket
- `/ws/jobs/{id}`: progress, preview, completed, error messages

## Artifacts
- Signed GET `/files/{job}/{artifact}`

## Error Model
- JSON envelope with code, message, hint, retryable

## Security
- HTTPS, JWT bearer, signed URLs, no raw data in logs

## Rate Limits
- 60 req/min per token; upload concurrency by server policy

---
