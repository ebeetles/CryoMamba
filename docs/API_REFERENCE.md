# CryoMamba API Reference

Complete REST and WebSocket API documentation for the CryoMamba GPU inference server.

**Base URL**: `http://localhost:8000` (development) or `https://your-server.com` (production)  
**API Version**: v1  
**Last Updated**: October 2025

---

## Table of Contents

- [Authentication](#authentication)
- [Health & Info Endpoints](#health--info-endpoints)
- [Upload Endpoints](#upload-endpoints)
- [Job Endpoints](#job-endpoints)
- [Artifact Endpoints](#artifact-endpoints)
- [GPU Monitoring Endpoints](#gpu-monitoring-endpoints)
- [WebSocket API](#websocket-api)
- [Data Models](#data-models)
- [Error Handling](#error-handling)
- [Rate Limits](#rate-limits)

---

## Authentication

### Bearer Token Authentication

All endpoints (except health check) require JWT bearer token authentication.

**Request Header**:
```http
Authorization: Bearer <your_jwt_token>
```

**Example**:
```bash
curl -H "Authorization: Bearer eyJhbGc..." http://localhost:8000/v1/server/info
```

### Obtaining Tokens

Contact your system administrator to obtain authentication tokens. Tokens include:
- **Scopes**: `read`, `write`, `admin`
- **Expiry**: Configurable (default: 24 hours)

---

## Health & Info Endpoints

### GET /v1/healthz

Check server health status.

**Authentication**: None required

**Response**: `200 OK`
```json
{
  "status": "healthy",
  "timestamp": "2025-10-17T10:30:00Z",
  "uptime_seconds": 3600
}
```

**Example**:
```bash
curl http://localhost:8000/v1/healthz
```

---

### GET /v1/server/info

Get server information and capabilities.

**Authentication**: Required

**Response**: `200 OK`
```json
{
  "version": "1.0.0",
  "nnunet_version": "2.1.0",
  "cuda_available": true,
  "cuda_version": "12.1",
  "gpu_count": 1,
  "gpu_info": [
    {
      "id": 0,
      "name": "NVIDIA A6000",
      "memory_total": 51539607552,
      "memory_free": 48234123264,
      "compute_capability": "8.6"
    }
  ],
  "max_upload_size": 10737418240,
  "supported_formats": ["mrc", "nifti"],
  "available_models": ["3d_fullres", "3d_lowres"]
}
```

**Example**:
```bash
curl -H "Authorization: Bearer <token>" http://localhost:8000/v1/server/info
```

---

## Upload Endpoints

### POST /v1/uploads/init

Initialize a chunked upload session.

**Authentication**: Required

**Request Body**:
```json
{
  "filename": "volume.mrc",
  "total_size": 5368709120,
  "chunk_size": 5242880,
  "content_type": "application/octet-stream",
  "metadata": {
    "acquisition_date": "2025-01-15",
    "microscope": "Titan Krios"
  }
}
```

**Response**: `201 Created`
```json
{
  "upload_id": "upl_abc123def456",
  "chunk_urls": [
    "/v1/uploads/upl_abc123def456/part/0",
    "/v1/uploads/upl_abc123def456/part/1",
    ...
  ],
  "expires_at": "2025-10-17T12:30:00Z"
}
```

**Example**:
```bash
curl -X POST http://localhost:8000/v1/uploads/init \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "filename": "volume.mrc",
    "total_size": 5368709120,
    "chunk_size": 5242880
  }'
```

**Error Responses**:
- `400 Bad Request`: Invalid parameters (file too large, invalid chunk size)
- `413 Payload Too Large`: File exceeds maximum size
- `507 Insufficient Storage`: Server storage full

---

### PUT /v1/uploads/{upload_id}/part/{part_number}

Upload a file chunk.

**Authentication**: Required

**Path Parameters**:
- `upload_id` (string): Upload session ID from init
- `part_number` (integer): Chunk index (0-based)

**Request Body**: Binary chunk data

**Headers**:
```http
Content-Type: application/octet-stream
Content-Length: 5242880
```

**Response**: `200 OK`
```json
{
  "upload_id": "upl_abc123def456",
  "part_number": 0,
  "received_bytes": 5242880,
  "etag": "a1b2c3d4e5f6"
}
```

**Example**:
```bash
curl -X PUT http://localhost:8000/v1/uploads/upl_abc123def456/part/0 \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @chunk_0.bin
```

**Error Responses**:
- `404 Not Found`: Upload session not found or expired
- `409 Conflict`: Part already uploaded
- `410 Gone`: Upload session expired

---

### POST /v1/uploads/{upload_id}/complete

Complete upload and assemble file.

**Authentication**: Required

**Path Parameters**:
- `upload_id` (string): Upload session ID

**Request Body**:
```json
{
  "parts": [
    {"part_number": 0, "etag": "a1b2c3d4e5f6"},
    {"part_number": 1, "etag": "b2c3d4e5f6g7"},
    ...
  ]
}
```

**Response**: `200 OK`
```json
{
  "file_id": "file_xyz789abc123",
  "filename": "volume.mrc",
  "size": 5368709120,
  "content_type": "application/octet-stream",
  "checksum": "sha256:abc123...",
  "metadata": {
    "shape": [512, 512, 300],
    "voxel_size": [1.0, 1.0, 1.5],
    "dtype": "float32",
    "min_intensity": -150.5,
    "max_intensity": 890.2
  },
  "created_at": "2025-10-17T10:35:00Z"
}
```

**Example**:
```bash
curl -X POST http://localhost:8000/v1/uploads/upl_abc123def456/complete \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "parts": [
      {"part_number": 0, "etag": "a1b2c3d4e5f6"},
      {"part_number": 1, "etag": "b2c3d4e5f6g7"}
    ]
  }'
```

**Error Responses**:
- `400 Bad Request`: Missing parts or invalid ETags
- `404 Not Found`: Upload session not found
- `422 Unprocessable Entity`: File validation failed

---

## Job Endpoints

### POST /v1/jobs

Create a new segmentation job.

**Authentication**: Required

**Request Body**:
```json
{
  "file_id": "file_xyz789abc123",
  "model": "3d_fullres",
  "params": {
    "folds": "all",
    "step_size": 0.5,
    "disable_tta": false,
    "save_probabilities": false
  },
  "preview_config": {
    "enabled": true,
    "interval_seconds": 5,
    "downsample_factor": 4
  }
}
```

**Response**: `201 Created`
```json
{
  "job_id": "job_mno345pqr678",
  "status": "queued",
  "file_id": "file_xyz789abc123",
  "model": "3d_fullres",
  "created_at": "2025-10-17T10:40:00Z",
  "queue_position": 2,
  "estimated_duration_seconds": 1800
}
```

**Example**:
```bash
curl -X POST http://localhost:8000/v1/jobs \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "file_id": "file_xyz789abc123",
    "model": "3d_fullres"
  }'
```

**Error Responses**:
- `400 Bad Request`: Invalid parameters
- `404 Not Found`: File ID not found
- `409 Conflict`: Job already exists for this file
- `503 Service Unavailable`: Server at capacity

---

### GET /v1/jobs/{job_id}

Get job status and details.

**Authentication**: Required

**Path Parameters**:
- `job_id` (string): Job ID

**Response**: `200 OK`
```json
{
  "job_id": "job_mno345pqr678",
  "status": "running",
  "file_id": "file_xyz789abc123",
  "model": "3d_fullres",
  "created_at": "2025-10-17T10:40:00Z",
  "started_at": "2025-10-17T10:42:00Z",
  "updated_at": "2025-10-17T10:45:30Z",
  "progress": {
    "current_step": 3,
    "total_steps": 5,
    "percentage": 60,
    "message": "Processing tile 45/75",
    "eta_seconds": 450
  },
  "artifacts": [],
  "gpu_stats": {
    "gpu_id": 0,
    "memory_used": 18432,
    "memory_total": 49152,
    "utilization": 95
  }
}
```

**Job Status Values**:
- `queued`: Waiting in queue
- `running`: Currently processing
- `completed`: Successfully finished
- `failed`: Encountered error
- `cancelled`: Cancelled by user

**Example**:
```bash
curl -H "Authorization: Bearer <token>" \
  http://localhost:8000/v1/jobs/job_mno345pqr678
```

**Error Responses**:
- `404 Not Found`: Job not found

---

### DELETE /v1/jobs/{job_id}

Cancel a running or queued job.

**Authentication**: Required

**Path Parameters**:
- `job_id` (string): Job ID

**Response**: `200 OK`
```json
{
  "job_id": "job_mno345pqr678",
  "status": "cancelled",
  "message": "Job cancelled successfully"
}
```

**Example**:
```bash
curl -X DELETE -H "Authorization: Bearer <token>" \
  http://localhost:8000/v1/jobs/job_mno345pqr678
```

**Error Responses**:
- `404 Not Found`: Job not found
- `409 Conflict`: Job already completed or failed

---

### GET /v1/jobs

List all jobs (with filtering).

**Authentication**: Required

**Query Parameters**:
- `status` (string, optional): Filter by status (queued, running, completed, failed)
- `limit` (integer, optional): Max results (default: 50, max: 100)
- `offset` (integer, optional): Pagination offset (default: 0)

**Response**: `200 OK`
```json
{
  "jobs": [
    {
      "job_id": "job_mno345pqr678",
      "status": "completed",
      "created_at": "2025-10-17T10:40:00Z",
      "completed_at": "2025-10-17T11:10:00Z"
    },
    ...
  ],
  "total": 150,
  "limit": 50,
  "offset": 0
}
```

**Example**:
```bash
curl -H "Authorization: Bearer <token>" \
  "http://localhost:8000/v1/jobs?status=completed&limit=10"
```

---

## Artifact Endpoints

### GET /v1/files/{job_id}/{artifact_name}

Download job artifact (segmentation mask, statistics, etc.).

**Authentication**: Signed URL (time-limited)

**Path Parameters**:
- `job_id` (string): Job ID
- `artifact_name` (string): Artifact filename (e.g., `segmentation.mrc`)

**Query Parameters**:
- `token` (string): Signed download token
- `expires` (integer): Token expiry timestamp

**Response**: `200 OK`
- **Content-Type**: `application/octet-stream`
- **Content-Disposition**: `attachment; filename="segmentation.mrc"`
- **Body**: Binary file data

**Example**:
```bash
# Get signed URL from job details
curl -H "Authorization: Bearer <token>" \
  http://localhost:8000/v1/jobs/job_mno345pqr678

# Download using signed URL
curl "http://localhost:8000/v1/files/job_mno345pqr678/segmentation.mrc?token=xyz&expires=1697550000" \
  -o segmentation.mrc
```

**Error Responses**:
- `403 Forbidden`: Invalid or expired token
- `404 Not Found`: Artifact not found

---

## GPU Monitoring Endpoints

### GET /v1/gpu/status

Get current GPU status and utilization.

**Authentication**: Required

**Response**: `200 OK`
```json
{
  "gpus": [
    {
      "id": 0,
      "name": "NVIDIA A6000",
      "memory_used": 18432,
      "memory_total": 49152,
      "memory_free": 30720,
      "utilization": 85,
      "temperature": 72,
      "power_usage": 280,
      "power_limit": 300
    }
  ],
  "timestamp": "2025-10-17T10:50:00Z"
}
```

**Example**:
```bash
curl -H "Authorization: Bearer <token>" http://localhost:8000/v1/gpu/status
```

---

## WebSocket API

### Connection: /ws/jobs/{job_id}

Real-time job progress and preview updates via WebSocket.

**Authentication**: Query parameter `token=<jwt_token>`

**Connection URL**:
```
ws://localhost:8000/ws/jobs/{job_id}?token=<jwt_token>
```

### Message Types

#### Server → Client: Progress Update
```json
{
  "type": "progress",
  "job_id": "job_mno345pqr678",
  "timestamp": "2025-10-17T10:45:30Z",
  "progress": {
    "current_step": 3,
    "total_steps": 5,
    "percentage": 60,
    "message": "Processing tile 45/75",
    "eta_seconds": 450
  }
}
```

#### Server → Client: Preview Update
```json
{
  "type": "preview",
  "job_id": "job_mno345pqr678",
  "timestamp": "2025-10-17T10:45:35Z",
  "preview": {
    "scale": 4,
    "shape": [128, 128, 75],
    "dtype": "uint8",
    "slice_idx": 37,
    "encoding": "base64",
    "data": "iVBORw0KGgoAAAANSUhEUgAA..."
  }
}
```

#### Server → Client: Completion
```json
{
  "type": "completed",
  "job_id": "job_mno345pqr678",
  "timestamp": "2025-10-17T11:10:00Z",
  "artifacts": [
    {
      "name": "segmentation.mrc",
      "size": 157286400,
      "url": "/v1/files/job_mno345pqr678/segmentation.mrc?token=..."
    },
    {
      "name": "statistics.json",
      "size": 2048,
      "url": "/v1/files/job_mno345pqr678/statistics.json?token=..."
    }
  ],
  "statistics": {
    "label_counts": {"1": 123456, "2": 78901},
    "physical_volume_nm3": {"1": 1234567.89, "2": 789012.34},
    "surface_area_nm2": {"1": 234567.89, "2": 123456.78}
  }
}
```

#### Server → Client: Error
```json
{
  "type": "error",
  "job_id": "job_mno345pqr678",
  "timestamp": "2025-10-17T10:50:00Z",
  "error": {
    "code": "INFERENCE_ERROR",
    "message": "Out of GPU memory during inference",
    "hint": "Try reducing tile size or disabling TTA",
    "retryable": true
  }
}
```

#### Client → Server: Ping
```json
{
  "type": "ping"
}
```

#### Server → Client: Pong
```json
{
  "type": "pong",
  "timestamp": "2025-10-17T10:45:40Z"
}
```

### Python Client Example
```python
import websocket
import json

def on_message(ws, message):
    data = json.loads(message)
    if data["type"] == "progress":
        print(f"Progress: {data['progress']['percentage']}%")
    elif data["type"] == "preview":
        # Decode and display preview
        import base64
        preview_data = base64.b64decode(data["preview"]["data"])
    elif data["type"] == "completed":
        print("Job completed!")
        print(f"Artifacts: {data['artifacts']}")

def on_error(ws, error):
    print(f"Error: {error}")

def on_close(ws, close_status_code, close_msg):
    print("Connection closed")

ws = websocket.WebSocketApp(
    f"ws://localhost:8000/ws/jobs/{job_id}?token={token}",
    on_message=on_message,
    on_error=on_error,
    on_close=on_close
)
ws.run_forever()
```

### JavaScript Client Example
```javascript
const ws = new WebSocket(`ws://localhost:8000/ws/jobs/${jobId}?token=${token}`);

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  switch(data.type) {
    case 'progress':
      console.log(`Progress: ${data.progress.percentage}%`);
      break;
    case 'preview':
      // Decode and display preview
      const img = new Image();
      img.src = `data:image/png;base64,${data.preview.data}`;
      document.body.appendChild(img);
      break;
    case 'completed':
      console.log('Job completed!', data.artifacts);
      ws.close();
      break;
    case 'error':
      console.error('Job error:', data.error);
      break;
  }
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};

ws.onclose = () => {
  console.log('Connection closed');
};

// Send ping every 30 seconds to keep connection alive
setInterval(() => {
  ws.send(JSON.stringify({type: 'ping'}));
}, 30000);
```

---

## Data Models

### Volume Metadata
```typescript
interface VolumeMetadata {
  filename: string;
  shape: [number, number, number];  // [X, Y, Z]
  voxel_size: [number, number, number];  // [X, Y, Z] in nm
  dtype: string;  // "float32", "uint8", etc.
  min_intensity: number;
  max_intensity: number;
  mean_intensity?: number;
  std_intensity?: number;
}
```

### Upload Session
```typescript
interface UploadSession {
  upload_id: string;
  filename: string;
  total_size: number;
  chunk_size: number;
  content_type: string;
  status: "active" | "completed" | "expired";
  created_at: string;  // ISO 8601
  expires_at: string;  // ISO 8601
}
```

### Job Record
```typescript
interface JobRecord {
  job_id: string;
  status: "queued" | "running" | "completed" | "failed" | "cancelled";
  file_id: string;
  model: string;
  params: Record<string, any>;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  updated_at: string;
  progress?: JobProgress;
  artifacts?: Artifact[];
  error?: ErrorDetail;
}
```

### Job Progress
```typescript
interface JobProgress {
  current_step: number;
  total_steps: number;
  percentage: number;
  message: string;
  eta_seconds?: number;
}
```

### Artifact
```typescript
interface Artifact {
  name: string;
  size: number;
  content_type: string;
  url: string;  // Signed URL
  created_at: string;
}
```

### Artifact Statistics
```typescript
interface ArtifactStatistics {
  label_counts: Record<string, number>;  // Label ID → voxel count
  physical_volume_nm3: Record<string, number>;  // Label ID → volume in nm³
  surface_area_nm2: Record<string, number>;  // Label ID → surface area in nm²
}
```

---

## Error Handling

### Error Response Format

All errors follow a consistent JSON envelope:

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message",
    "hint": "Suggestion for resolving the issue",
    "retryable": true,
    "details": {
      "field": "additional context"
    }
  }
}
```

### Error Codes

| Code | HTTP Status | Description | Retryable |
|------|-------------|-------------|-----------|
| `VALIDATION_ERROR` | 400 | Invalid request parameters | No |
| `AUTHENTICATION_REQUIRED` | 401 | Missing or invalid token | No |
| `FORBIDDEN` | 403 | Insufficient permissions | No |
| `NOT_FOUND` | 404 | Resource not found | No |
| `CONFLICT` | 409 | Resource conflict | No |
| `PAYLOAD_TOO_LARGE` | 413 | File exceeds size limit | No |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests | Yes |
| `INTERNAL_ERROR` | 500 | Server error | Yes |
| `SERVICE_UNAVAILABLE` | 503 | Server overloaded | Yes |
| `INSUFFICIENT_STORAGE` | 507 | Storage full | No |
| `GPU_ERROR` | 500 | GPU-related error | Yes |
| `INFERENCE_ERROR` | 500 | nnU-Net inference error | Maybe |
| `OUT_OF_MEMORY` | 500 | GPU memory exhausted | Maybe |

### Error Handling Best Practices

1. **Check `retryable` field**: Only retry if `true`
2. **Implement exponential backoff**: Wait longer between retries
3. **Display `hint` to users**: Helps resolve issues
4. **Log `details` for debugging**: Contains context
5. **Handle specific codes**: Different UI for different errors

### Example Error Handling (Python)
```python
import requests
import time

def api_call_with_retry(url, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            error_data = e.response.json()
            error = error_data.get("error", {})
            
            if not error.get("retryable"):
                print(f"Non-retryable error: {error['message']}")
                print(f"Hint: {error.get('hint')}")
                raise
            
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
```

---

## Rate Limits

### Default Limits

- **General API**: 60 requests/minute per token
- **Upload endpoints**: 10 concurrent uploads per token
- **WebSocket connections**: 5 concurrent connections per token
- **Job creation**: 20 jobs/hour per token

### Rate Limit Headers

All responses include rate limit information:

```http
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1697550000
```

### Rate Limit Exceeded Response

**Response**: `429 Too Many Requests`
```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded",
    "hint": "Please wait before making more requests",
    "retryable": true,
    "details": {
      "limit": 60,
      "window_seconds": 60,
      "retry_after_seconds": 15
    }
  }
}
```

### Best Practices

1. **Monitor rate limit headers**: Adjust request rate accordingly
2. **Implement client-side throttling**: Stay under limits
3. **Use WebSocket for real-time updates**: More efficient than polling
4. **Batch operations when possible**: Reduce API calls
5. **Cache responses**: Avoid redundant requests

---

## Security Best Practices

### For API Consumers

1. **Use HTTPS in production**: Never send tokens over HTTP
2. **Store tokens securely**: Use environment variables or secure vaults
3. **Validate SSL certificates**: Prevent man-in-the-middle attacks
4. **Implement token rotation**: Refresh tokens regularly
5. **Never log tokens**: Redact sensitive data from logs
6. **Validate file checksums**: Ensure data integrity
7. **Handle errors gracefully**: Don't expose internal details

### Example Secure Configuration (Python)
```python
import os
import requests

# Load token from environment
API_TOKEN = os.getenv("CRYOMAMBA_API_TOKEN")
if not API_TOKEN:
    raise ValueError("API_TOKEN not set")

# Configure session with security headers
session = requests.Session()
session.headers.update({
    "Authorization": f"Bearer {API_TOKEN}",
    "User-Agent": "CryoMamba-Client/1.0"
})

# Verify SSL certificates
session.verify = True

# Make secure request
response = session.get("https://server.com/v1/server/info")
```

---

## Changelog

### Version 1.0.0 (October 2025)
- Initial API release
- REST endpoints for uploads, jobs, artifacts
- WebSocket streaming for progress and previews
- JWT authentication
- Rate limiting

---

## Support

For API questions or issues:
- **Documentation**: [README.md](../README.md)
- **GitHub Issues**: <repository-url>/issues
- **Email**: support@cryomamba.com

---

**API Version**: 1.0.0  
**Last Updated**: October 2025

