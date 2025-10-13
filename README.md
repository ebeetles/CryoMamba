# CryoMamba FastAPI Server

A FastAPI server for cryo-ET segmentation inference, providing health endpoints and dummy job processing capabilities.

## Quick Start

### Prerequisites
- Python 3.11+
- pip or conda

### Local Development

1. **Clone and setup**:
   ```bash
   git clone <repository-url>
   cd CryoMamba
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the server**:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

4. **Test the server**:
   ```bash
   curl http://localhost:8000/v1/healthz
   ```

### Docker Development

1. **Build and run with Docker Compose**:
   ```bash
   docker-compose up --build
   ```

2. **Test the server**:
   ```bash
   curl http://localhost:8000/v1/healthz
   ```

## API Endpoints

### Health & Info
- `GET /v1/healthz` - Health check endpoint
- `GET /v1/server/info` - Server information

### Jobs (Dummy Implementation)
- `POST /v1/jobs` - Create a new job
- `GET /v1/jobs/{job_id}` - Get job status
- `DELETE /v1/jobs/{job_id}` - Cancel job (placeholder)

## Development

### Project Structure
```
app/
├── __init__.py
├── main.py              # FastAPI app instance
├── config.py            # Configuration settings
├── middleware.py        # Error handling middleware
├── models/
│   ├── __init__.py
│   └── job.py          # Job data models
└── routes/
    ├── __init__.py
    ├── health.py       # Health endpoints
    ├── info.py         # Server info endpoints
    └── jobs.py         # Job endpoints
```

### Environment Variables
Create a `.env` file with:
```env
DEBUG=false
LOG_LEVEL=INFO
HOST=0.0.0.0
PORT=8000
CORS_ORIGINS=["http://localhost:3000"]
```

## Testing

### Manual Testing
```bash
# Health check
curl http://localhost:8000/v1/healthz

# Server info
curl http://localhost:8000/v1/server/info

# Create job
curl -X POST http://localhost:8000/v1/jobs

# Get job status
curl http://localhost:8000/v1/jobs/{job_id}
```

## Next Steps

This is the foundation server. Future stories will add:
- Real job processing
- WebSocket preview streaming
- nnU-Net integration
- File upload capabilities
