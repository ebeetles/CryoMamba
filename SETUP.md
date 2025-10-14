# CryoMamba Foundation Setup Guide

This guide provides complete setup instructions for the CryoMamba foundation components, including the FastAPI server and napari desktop application.

## Prerequisites

- Python 3.13+
- pip (Python package manager)
- Git

## Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd CryoMamba
```

### 2. Set Up Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install server dependencies
pip install -r requirements.txt

# Install desktop application dependencies
pip install -r napari_cryomamba/requirements.txt
```

### 4. Run the Server

```bash
# Start the FastAPI server
python app/main.py
```

The server will start on `http://localhost:8000`

### 5. Run the Desktop Application

In a new terminal:

```bash
# Activate virtual environment
source venv/bin/activate

# Start the napari desktop application
python napari_cryomamba/main.py
```

## Detailed Setup

### Server Components

The FastAPI server provides the following endpoints:

- **Health Check**: `GET /v1/healthz`
- **Server Info**: `GET /v1/server/info`
- **Job Management**: 
  - `POST /v1/jobs` - Create a new job
  - `GET /v1/jobs/{job_id}` - Get job status
  - `DELETE /v1/jobs/{job_id}` - Cancel a job
- **WebSocket**: `WS /ws/jobs/{job_id}` - Real-time job updates

### Desktop Application

The napari-based desktop application provides:

- **File Operations**: Open and display `.mrc` files
- **Job Control**: Create, connect to, and monitor jobs
- **Real-time Visualization**: Display fake mask overlays at 1 Hz
- **Error Handling**: Graceful connection failure recovery

### Testing

Run the comprehensive test suite:

```bash
# Run all tests
python run_e2e_tests.py

# Run server-only tests
python run_e2e_tests.py --server-only

# Run individual test files
python -m pytest test_server.py -v
python -m pytest tests/test_e2e_integration.py -v
```

## Usage Workflow

### 1. Start the Server

```bash
python app/main.py
```

You should see output like:
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### 2. Start the Desktop Application

```bash
python napari_cryomamba/main.py
```

This will open the napari viewer with the CryoMamba widget docked.

### 3. Complete End-to-End Workflow

1. **Open a Volume**: Click "Open .mrc File" and select a `.mrc` file
2. **Check Server Connection**: Click "Check Server" to verify connectivity
3. **Create a Job**: Click "Create Job" to start a fake inference job
4. **Connect to Job**: Click "Connect to Job" to receive real-time updates
5. **View Preview**: Watch fake mask overlays update at 1 Hz
6. **Monitor Progress**: Check the info panel for status updates

## Configuration

### Server Configuration

The server can be configured via environment variables:

```bash
# Set log level
export LOG_LEVEL=INFO

# Set CORS origins
export CORS_ORIGINS="http://localhost:3000,http://localhost:8080"

# Enable debug mode
export DEBUG=true
```

### Desktop Configuration

The desktop application connects to the server via WebSocket. Default connection:

- **Server URL**: `ws://localhost:8000`
- **Connection Timeout**: 5 seconds
- **Reconnection Attempts**: 5 with exponential backoff

## Troubleshooting

### Common Issues

#### Server Won't Start

**Problem**: Port 8000 already in use
**Solution**: 
```bash
# Find process using port 8000
lsof -i :8000

# Kill the process
kill -9 <PID>

# Or use a different port
python app/main.py --port 8001
```

#### Desktop Can't Connect to Server

**Problem**: Connection refused error
**Solution**:
1. Verify server is running: `curl http://localhost:8000/v1/healthz`
2. Check server URL in desktop application
3. Ensure no firewall blocking the connection

#### WebSocket Connection Fails

**Problem**: WebSocket connection errors
**Solution**:
1. Check server logs for WebSocket errors
2. Verify job ID exists: `curl http://localhost:8000/v1/jobs/{job_id}`
3. Try manual reconnection using the "Reconnect" button

#### MRC File Loading Issues

**Problem**: MRC file won't load
**Solution**:
1. Verify file format: `file your_file.mrc`
2. Check file permissions
3. Ensure file is not corrupted

### Performance Issues

#### Slow Preview Updates

**Problem**: Preview updates are slow or stuttering
**Solution**:
1. Check system resources (CPU, memory)
2. Reduce preview frequency in server configuration
3. Use smaller test volumes for development

#### High Memory Usage

**Problem**: Application uses too much memory
**Solution**:
1. Close unused napari layers
2. Use smaller test volumes
3. Restart the application periodically

## Development

### Running Tests

```bash
# Run all tests
python run_e2e_tests.py

# Run specific test categories
python -m pytest tests/test_e2e_integration.py::TestEndToEndIntegration -v
python -m pytest tests/test_e2e_integration.py::TestDesktopIntegration -v

# Run with coverage
python -m pytest --cov=app --cov=napari_cryomamba tests/
```

### Code Quality

```bash
# Format code
black app/ napari_cryomamba/

# Lint code
flake8 app/ napari_cryomamba/

# Type checking
mypy app/ napari_cryomamba/
```

### Adding New Features

1. **Server Features**: Add routes in `app/routes/`
2. **Desktop Features**: Extend `napari_cryomamba/napari_cryomamba/widget.py`
3. **WebSocket Features**: Modify `app/routes/websocket.py`
4. **Tests**: Add tests in `tests/` directory

## Architecture Overview

### Server Architecture

```
app/
├── main.py              # FastAPI application entry point
├── config.py            # Configuration management
├── middleware.py        # Error handling middleware
├── models/              # Data models
│   ├── job.py          # Job state management
│   └── preview.py      # Preview message models
└── routes/              # API endpoints
    ├── health.py       # Health check endpoint
    ├── info.py         # Server information
    ├── jobs.py         # Job management
    └── websocket.py    # WebSocket streaming
```

### Desktop Architecture

```
napari_cryomamba/
├── main.py              # Application entry point
└── napari_cryomamba/
    ├── widget.py       # Main UI widget
    └── websocket_client.py  # WebSocket client
```

### Data Flow

1. **Desktop** creates job via REST API
2. **Server** starts fake preview streaming
3. **WebSocket** streams preview data at 1 Hz
4. **Desktop** displays preview overlays in napari

## Support

For issues and questions:

1. Check this documentation
2. Review the test suite for usage examples
3. Check server logs for error details
4. Verify all dependencies are installed correctly

## Performance Benchmarks

### Foundation Components

- **Server Startup**: < 2 seconds
- **Job Creation**: < 100ms
- **WebSocket Connection**: < 500ms
- **Preview Streaming**: 1 Hz (configurable)
- **MRC File Loading**: < 5 seconds for 512³ volumes

### System Requirements

- **Minimum RAM**: 4GB
- **Recommended RAM**: 8GB+
- **CPU**: Any modern multi-core processor
- **Storage**: 1GB free space
- **Network**: Local network (no external dependencies)

## Future Enhancements

The foundation provides the base for future features:

- Real GPU-based inference
- Advanced visualization controls
- Batch processing capabilities
- Cloud deployment options
- Plugin architecture for custom models
