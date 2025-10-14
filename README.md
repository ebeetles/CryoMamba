# CryoMamba FastAPI Server

A FastAPI server for cryo-ET segmentation inference, providing health endpoints and dummy job processing capabilities.

## Quick Start

### Prerequisites
- Docker and Docker Compose
- Python 3.11+ (for local development)
- pip or conda (for local development)

### Docker Development (Recommended)

1. **Clone and setup**:
   ```bash
   git clone <repository-url>
   cd CryoMamba
   ```

2. **Copy environment configuration**:
   ```bash
   cp .env.example .env
   # Edit .env file as needed for your environment
   ```

3. **Build and run with Docker Compose**:
   ```bash
   docker-compose up --build
   ```

4. **Test the server**:
   ```bash
   curl http://localhost:8000/v1/healthz
   ```

5. **Stop the server**:
   ```bash
   docker-compose down
   ```

### Local Development (Without Docker)

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
Copy `.env.example` to `.env` and modify as needed:
```env
# Application Settings
APP_NAME=CryoMamba Inference Server
APP_VERSION=1.0.0
DEBUG=true

# Server Configuration
HOST=0.0.0.0
PORT=8000

# Logging Configuration
LOG_LEVEL=DEBUG

# CORS Configuration (JSON array format)
CORS_ORIGINS=["http://localhost:3000", "http://localhost:8080", "http://localhost:5173"]
```

### Development Workflow

#### Using Docker Compose (Recommended)
```bash
# Start development server
docker-compose up --build

# View logs
docker-compose logs -f

# Stop server
docker-compose down

# Rebuild after code changes
docker-compose up --build --force-recreate
```

#### Common Development Tasks
```bash
# Check server health
curl http://localhost:8000/v1/healthz

# View server info
curl http://localhost:8000/v1/server/info

# Create a test job
curl -X POST http://localhost:8000/v1/jobs

# Check Docker container status
docker-compose ps

# Access container shell
docker-compose exec cryomamba-server bash
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

## Troubleshooting

### Common Issues

#### Port Already in Use
```bash
# Check what's using port 8000
lsof -i :8000

# Kill the process or use a different port
# Update PORT in .env file and restart
```

#### Docker Build Issues
```bash
# Clean Docker cache
docker system prune -a

# Rebuild without cache
docker-compose build --no-cache
```

#### Container Won't Start
```bash
# Check container logs
docker-compose logs cryomamba-server

# Check if health check is passing
docker-compose exec cryomamba-server curl -f http://localhost:8000/v1/healthz
```

#### CORS Issues
- Ensure `CORS_ORIGINS` in `.env` includes your frontend URL
- Check that the frontend is running on the expected port
- Verify the JSON format is correct (use double quotes)

### Health Check Failures
The Docker container includes a health check that verifies the `/v1/healthz` endpoint. If this fails:
1. Check server logs: `docker-compose logs cryomamba-server`
2. Verify the server is running: `docker-compose ps`
3. Test the endpoint manually: `curl http://localhost:8000/v1/healthz`

## Next Steps

This is the foundation server. Future stories will add:
- Real job processing
- WebSocket preview streaming
- nnU-Net integration
- File upload capabilities
