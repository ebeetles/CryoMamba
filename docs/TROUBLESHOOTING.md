# CryoMamba Troubleshooting Guide

Comprehensive guide for diagnosing and resolving common issues with CryoMamba.

**Version**: 1.0.0  
**Last Updated**: October 2025

---

## Table of Contents

- [Quick Diagnostics](#quick-diagnostics)
- [Installation Issues](#installation-issues)
- [Connection Issues](#connection-issues)
- [Upload Issues](#upload-issues)
- [Job Execution Issues](#job-execution-issues)
- [Performance Issues](#performance-issues)
- [Desktop Client Issues](#desktop-client-issues)
- [GPU Server Issues](#gpu-server-issues)
- [Data Issues](#data-issues)
- [Export Issues](#export-issues)
- [Known Issues & Limitations](#known-issues--limitations)
- [FAQ](#faq)
- [Performance Tuning](#performance-tuning)
- [Getting Support](#getting-support)

---

## Quick Diagnostics

### System Health Check

Run this checklist before diving into specific issues:

#### Server Health Check

```bash
# 1. Check server is running
curl http://localhost:8000/v1/healthz

# 2. Check CUDA availability (optional - server works without GPU)
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# 3. Check GPU status (if available)
nvidia-smi  # Skip if running on CPU

# 4. Check disk space
df -h /data

# 5. Check server logs
tail -f logs/server.log

# 6. Check Python version
python --version  # Should be 3.9+

# 7. Check nnU-Net installation
python -c "import nnunetv2; print('nnU-Net OK')"
```

Expected output (CPU mode):
```
✅ Server health: OK
✅ CUDA available: False (CPU mode - this is fine!)
✅ Disk space: 800GB free
✅ Logs: No errors
✅ Python: 3.10.x
✅ nnU-Net: Installed
```

Expected output (GPU mode):
```
✅ Server health: OK
✅ CUDA available: True
✅ GPU detected: NVIDIA A6000
✅ Disk space: 800GB free
✅ Logs: No errors
✅ Python: 3.10.x
✅ nnU-Net: Installed
```

#### Desktop Client Health Check

```bash
# 1. Check Python version
python --version  # Should be 3.8+

# 2. Check napari installation
python -c "import napari; print('napari OK')"

# 3. Check CryoMamba installation
python -c "import napari_cryomamba; print('CryoMamba OK')"

# 4. Check Qt backend
python -c "from qtpy import QtWidgets; print('Qt OK')"

# 5. Test server connectivity
curl http://your-server.com/v1/healthz
```

Expected output:
```
✅ Python: 3.10.x
✅ napari: Installed
✅ CryoMamba: Installed
✅ Qt: Available
✅ Server: Reachable
```

---

## Installation Issues

### Issue: pip install fails with dependency conflicts

**Symptoms**:
```
ERROR: Cannot install package due to conflicting dependencies
```

**Solutions**:

1. **Use clean virtual environment**:
   ```bash
   python -m venv venv_fresh
   source venv_fresh/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **Install with constraints**:
   ```bash
   pip install -c constraints.txt -r requirements.txt
   ```

3. **Install dependencies separately**:
   ```bash
   pip install torch torchvision  # Install PyTorch first
   pip install nnunetv2
   pip install -r requirements.txt
   ```

---

### Issue: CUDA not available after PyTorch installation

**Symptoms**:
```python
>>> import torch
>>> torch.cuda.is_available()
False
```

**Solutions**:

1. **Check NVIDIA drivers**:
   ```bash
   nvidia-smi
   # If command not found, install drivers
   ```

2. **Install CUDA-enabled PyTorch**:
   ```bash
   # Check CUDA version
   nvidia-smi  # Look for CUDA Version: 12.1
   
   # Install matching PyTorch (example for CUDA 12.1)
   pip uninstall torch torchvision
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Verify installation**:
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA version: {torch.version.cuda}")
   print(f"GPU count: {torch.cuda.device_count()}")
   print(f"GPU name: {torch.cuda.get_device_name(0)}")
   ```

---

### Issue: nnU-Net installation fails

**Symptoms**:
```
ERROR: Could not find a version that satisfies the requirement nnunetv2
```

**Solutions**:

1. **Install from source**:
   ```bash
   git clone https://github.com/MIC-DKFZ/nnUNet.git
   cd nnUNet
   pip install -e .
   ```

2. **Check Python version**:
   ```bash
   python --version  # Must be 3.9+
   ```

3. **Install specific version**:
   ```bash
   pip install nnunetv2==2.1.0
   ```

---

### Issue: macOS application won't open

**Symptoms**:
- "CryoMamba.app is damaged and can't be opened"
- Application bounces in dock then closes

**Solutions**:

1. **Remove quarantine attribute**:
   ```bash
   xattr -cr /Applications/CryoMamba.app
   ```

2. **Right-click to open**:
   - Right-click CryoMamba.app
   - Select "Open"
   - Click "Open" in dialog

3. **Check security settings**:
   - System Settings → Security & Privacy
   - Allow applications from "App Store and identified developers"
   - If blocked, click "Open Anyway"

4. **Check logs**:
   ```bash
   # Check system logs
   log show --predicate 'eventMessage contains "CryoMamba"' --info --last 1h
   
   # Check application logs
   cat ~/Library/Logs/CryoMamba/app.log
   ```

---

## Connection Issues

### Issue: Cannot connect to server

**Symptoms**:
- Connection timeout
- "Server unreachable" error
- Red status indicator in client

**Diagnostic Steps**:

1. **Ping server**:
   ```bash
   ping your-server.com
   # Should see replies with <50ms latency
   ```

2. **Test HTTP connectivity**:
   ```bash
   curl http://your-server.com/v1/healthz
   # Should return: {"status": "healthy"}
   ```

3. **Test HTTPS connectivity**:
   ```bash
   curl https://your-server.com/v1/healthz
   # Check for SSL errors
   ```

4. **Check DNS resolution**:
   ```bash
   nslookup your-server.com
   # Should resolve to server IP
   ```

**Solutions**:

1. **Verify server URL**:
   - Check for typos
   - Include protocol: `http://` or `https://`
   - Include port if non-standard: `http://server:8000`

2. **Check firewall**:
   ```bash
   # Server side
   sudo ufw status
   sudo ufw allow 8000/tcp
   
   # Client side
   # Check corporate firewall settings
   ```

3. **Check VPN**:
   - Disconnect VPN and retry
   - If VPN required, check split-tunnel settings
   - Contact network admin for firewall rules

4. **Try alternative network**:
   - Switch from WiFi to Ethernet
   - Try mobile hotspot
   - Bypass proxy if present

5. **Verify server is running**:
   ```bash
   # SSH to server
   ps aux | grep uvicorn
   # Should show running process
   ```

---

### Issue: WebSocket connection drops frequently

**Symptoms**:
- Progress updates stop
- "Disconnected" messages in UI
- Have to reconnect manually

**Solutions**:

1. **Increase timeouts**:
   ```python
   # Client side (widget.py)
   websocket_client = WebSocketClient(
       url=server_url,
       timeout=300,  # Increase from default 60s
       ping_interval=20  # Send keepalive
   )
   ```

2. **Check network stability**:
   ```bash
   # Test packet loss
   ping -c 100 your-server.com
   # Should have 0% packet loss
   ```

3. **Use wired connection**:
   - WiFi may have intermittent drops
   - Ethernet provides stable connection

4. **Check proxy/firewall**:
   - Some proxies drop WebSocket connections
   - May need WebSocket-aware proxy configuration

5. **Server-side configuration**:
   ```python
   # app/routes/websocket.py
   @app.websocket("/ws/jobs/{job_id}")
   async def websocket_endpoint(
       websocket: WebSocket,
       job_id: str,
       ping_interval: int = 20,  # Send ping every 20s
       ping_timeout: int = 60    # Close after 60s no pong
   ):
       # ...
   ```

---

### Issue: Authentication token expired

**Symptoms**:
- "401 Unauthorized" errors
- "Token expired" message
- Worked previously, now fails

**Solutions**:

1. **Request new token**:
   - Contact system administrator
   - Or use token refresh endpoint (if available)

2. **Check token expiry**:
   ```python
   import jwt
   
   token = "your_jwt_token"
   decoded = jwt.decode(token, options={"verify_signature": False})
   print(f"Expires: {decoded['exp']}")
   ```

3. **Configure automatic refresh**:
   ```python
   # Client configuration
   client.configure(
       token=token,
       auto_refresh=True,
       refresh_before_seconds=300  # Refresh 5 min before expiry
   )
   ```

---

## Upload Issues

### Issue: Upload fails with large files

**Symptoms**:
- Timeout during upload
- "Connection reset" error
- Upload progress stops

**Solutions**:

1. **Increase chunk size**:
   ```python
   # Settings → Upload Configuration
   chunk_size = 10 * 1024 * 1024  # 10 MB instead of 5 MB
   ```

2. **Enable upload resume**:
   ```python
   upload_manager.upload(
       filepath,
       resumable=True,
       retry_failed_chunks=True,
       max_retries=3
   )
   ```

3. **Check disk space**:
   ```bash
   # Server side
   df -h /data
   # Ensure sufficient free space
   ```

4. **Use wired connection**:
   - Large uploads sensitive to network stability
   - Avoid WiFi for multi-GB files

5. **Upload during off-peak hours**:
   - Less network congestion
   - Better bandwidth availability

---

### Issue: Upload succeeds but file corrupt

**Symptoms**:
- Upload completes
- Job fails with "invalid file" error
- Cannot open file on server

**Diagnostic Steps**:

1. **Compare checksums**:
   ```bash
   # Local file
   sha256sum volume.mrc
   
   # Server file (via API)
   curl http://server/v1/files/{file_id}/checksum
   ```

2. **Check file size**:
   ```bash
   # Local
   ls -lh volume.mrc
   
   # Server
   curl http://server/v1/files/{file_id} | jq '.size'
   ```

**Solutions**:

1. **Re-upload with validation**:
   ```python
   upload_manager.upload(
       filepath,
       validate_chunks=True,
       verify_checksum=True
   )
   ```

2. **Check network stability**:
   - Use `iperf3` to test bandwidth
   - Check for packet loss with `ping`

3. **Disable network optimizations**:
   - Some "optimization" proxies corrupt data
   - Try direct connection

---

### Issue: "Insufficient storage" error

**Symptoms**:
- Upload fails with 507 error
- "Insufficient storage" message

**Solutions**:

1. **Check server disk space**:
   ```bash
   df -h /data
   ```

2. **Clean up old files**:
   ```bash
   # Delete old uploads
   find /data/uploads -mtime +7 -delete
   
   # Delete completed job artifacts
   find /data/artifacts -mtime +30 -delete
   ```

3. **Increase storage**:
   - Add more disk space
   - Configure external storage
   - Set up automatic cleanup policy

---

## Job Execution Issues

### Issue: Job stuck in "queued" status

**Symptoms**:
- Job remains queued for extended time
- Queue position not changing
- No other jobs running

**Diagnostic Steps**:

1. **Check queue**:
   ```bash
   curl http://server/v1/jobs?status=queued
   ```

2. **Check running jobs**:
   ```bash
   curl http://server/v1/jobs?status=running
   ```

3. **Check GPU availability**:
   ```bash
   nvidia-smi
   # Look for running processes
   ```

**Solutions**:

1. **Restart job scheduler**:
   ```bash
   # Server side
   systemctl restart cryomamba-scheduler
   ```

2. **Clear stuck jobs**:
   ```python
   # Admin script
   from app.services.orchestrator import Orchestrator
   
   orch = Orchestrator()
   orch.clear_stuck_jobs(timeout_minutes=60)
   ```

3. **Check for deadlock**:
   ```bash
   # Check logs for errors
   tail -f logs/scheduler.log | grep ERROR
   ```

---

### Issue: Job fails with "Out of memory"

**Symptoms**:
- Job starts then fails
- Error message: "CUDA out of memory"
- GPU memory full in nvidia-smi

**Solutions**:

1. **Reduce tile size**:
   ```python
   # Job parameters
   params = {
       "tile_step_size": 0.7,  # Increase from 0.5
       "overlap": 0.25  # Reduce from 0.5
   }
   ```

2. **Clear GPU cache**:
   ```python
   # Server side - add to job cleanup
   import torch
   torch.cuda.empty_cache()
   ```

3. **Use smaller batch size**:
   ```python
   params = {
       "batch_size": 1  # Reduce if higher
   }
   ```

4. **Enable gradient checkpointing**:
   ```python
   params = {
       "use_gradient_checkpointing": True
   }
   ```

5. **Kill zombie processes**:
   ```bash
   # Find GPU processes
   nvidia-smi
   
   # Kill specific process
   kill -9 <PID>
   ```

6. **Restart CUDA runtime**:
   ```bash
   # Restart server to reset CUDA
   systemctl restart cryomamba
   ```

---

### Issue: Job fails with segmentation fault

**Symptoms**:
- Job crashes unexpectedly
- "Segmentation fault (core dumped)" in logs
- No detailed error message

**Solutions**:

1. **Check nnU-Net installation**:
   ```bash
   python -c "import nnunetv2; print(nnunetv2.__version__)"
   ```

2. **Update dependencies**:
   ```bash
   pip install --upgrade torch nnunetv2
   ```

3. **Check CUDA/cuDNN compatibility**:
   ```bash
   # Verify versions match
   python -c "import torch; print(torch.version.cuda)"
   nvcc --version
   ```

4. **Run with debug symbols**:
   ```bash
   gdb python
   (gdb) run -m app.services.nnunet_wrapper
   # Wait for segfault
   (gdb) backtrace
   ```

5. **Check system limits**:
   ```bash
   ulimit -a
   # Increase stack size if needed
   ulimit -s unlimited
   ```

---

### Issue: Inference very slow / estimated time very long

**Symptoms**:
- Job running for hours
- Progress crawling (< 1% per minute)
- ETA shows days

**Solutions**:

1. **Disable test-time augmentation**:
   ```python
   params = {
       "disable_tta": True  # 2-4× speedup
   }
   ```

2. **Increase step size**:
   ```python
   params = {
       "step_size": 1.0  # Instead of 0.5
   }
   ```

3. **Check GPU utilization**:
   ```bash
   nvidia-smi
   # GPU should be at 90-100% utilization
   # If low, check for CPU bottleneck
   ```

4. **Reduce workers**:
   ```python
   params = {
       "num_processes": 1  # Instead of 4
   }
   ```

5. **Check thermal throttling**:
   ```bash
   nvidia-smi
   # Check temperature (should be < 80°C)
   # If high, improve cooling
   ```

---

## Performance Issues

### Issue: Preview streaming is slow/laggy

**Symptoms**:
- Previews arrive late
- Choppy updates
- Client becomes unresponsive

**Solutions**:

1. **Reduce preview frequency**:
   ```python
   # Client settings
   preview_config = {
       "interval_seconds": 10,  # Instead of 5
       "downsample_factor": 8    # Instead of 4
   }
   ```

2. **Disable 3D rendering during job**:
   ```python
   # UI setting
   auto_switch_to_2d_during_job = True
   ```

3. **Check network bandwidth**:
   ```bash
   # Test speed to server
   iperf3 -c your-server.com
   # Should have > 10 Mbps
   ```

4. **Close other applications**:
   - Free up RAM
   - Reduce CPU usage
   - Check Activity Monitor

5. **Use preview caching**:
   ```python
   preview_cache_enabled = True
   preview_cache_size_mb = 500
   ```

---

### Issue: High memory usage on client

**Symptoms**:
- Client sluggish
- System swap usage high
- Activity Monitor shows high memory

**Solutions**:

1. **Close unused volumes**:
   ```python
   # File → Close Layer
   # Or close all: Cmd+W
   ```

2. **Reduce cache size**:
   ```python
   # Settings → Performance
   layer_cache_size = 256  # MB, reduce from 1024
   ```

3. **Use 2D view**:
   - 3D rendering uses more memory
   - Switch to 2D for large volumes

4. **Load subvolumes**:
   ```python
   # Load only portion of volume
   volume_loader.load_subvolume(
       filepath,
       z_start=100,
       z_end=200
   )
   ```

5. **Restart application**:
   - Clears memory leaks
   - Resets cache

---

### Issue: Server becomes unresponsive under load

**Symptoms**:
- Server slow to respond
- Multiple jobs queued
- API timeouts

**Solutions**:

1. **Check resource usage**:
   ```bash
   top
   # Look for CPU/memory hogs
   ```

2. **Limit concurrent jobs**:
   ```python
   # app/config.py
   MAX_CONCURRENT_JOBS = 2  # Instead of 4
   ```

3. **Increase worker processes**:
   ```bash
   # Start server with more workers
   uvicorn app.main:app --workers 4
   ```

4. **Enable rate limiting**:
   ```python
   # app/middleware.py
   rate_limit_per_minute = 30  # Reduce from 60
   ```

5. **Scale horizontally**:
   - Add more GPU servers
   - Use load balancer
   - Distribute jobs across servers

---

## Desktop Client Issues

### Issue: Application crashes when loading large .mrc files

**Symptoms**:
- App quits unexpectedly
- "Out of memory" error
- System freeze

**Solutions**:

1. **Increase memory limit**:
   ```bash
   # macOS: Increase app memory limit
   # System Settings → Application → Memory Limit
   ```

2. **Load in chunks**:
   ```python
   # Load only current slice on demand
   lazy_loading = True
   ```

3. **Downsample before loading**:
   ```python
   # Settings → Data Loading
   auto_downsample_threshold_mb = 1000
   downsample_factor = 2
   ```

4. **Close other applications**:
   - Free system RAM
   - Especially memory-hungry apps (browsers, etc.)

---

### Issue: 3D rendering is slow/laggy

**Symptoms**:
- Choppy rotation
- Low frame rate
- Long rendering time

**Solutions**:

1. **Reduce volume resolution**:
   ```python
   # Rendering settings
   rendering_downsampling = 2  # Show every 2nd voxel
   ```

2. **Use volume rendering limits**:
   ```python
   # Limit rendered region
   render_z_range = (100, 200)  # Instead of full volume
   ```

3. **Adjust quality settings**:
   ```python
   # Settings → 3D Rendering
   rendering_quality = "medium"  # Instead of "high"
   rendering_samples = 256       # Instead of 512
   ```

4. **Check GPU availability**:
   ```python
   # napari should use GPU if available
   import OpenGL
   print(OpenGL.GL.glGetString(OpenGL.GL.GL_RENDERER))
   ```

5. **Update graphics drivers**:
   - macOS: Update OS to latest version
   - Check for GPU firmware updates

---

### Issue: Export fails with "Permission denied"

**Symptoms**:
- Export dialog completes
- Error: "Permission denied"
- File not created

**Solutions**:

1. **Check write permissions**:
   ```bash
   # Test write access
   touch /path/to/output/test.txt
   ```

2. **Choose different location**:
   - Try Desktop or Documents
   - Avoid system folders

3. **Check disk space**:
   ```bash
   df -h
   ```

4. **Close file if open**:
   - Ensure output file not open in another app
   - Check for file locks

5. **Grant permissions** (macOS):
   - System Settings → Security & Privacy → Files and Folders
   - Grant CryoMamba access to destination folder

---

## GPU Server Issues

### Issue: Server won't start

**Symptoms**:
- `python dev.py` fails
- Import errors
- Port already in use

**Solutions**:

1. **Check port availability**:
   ```bash
   lsof -i :8000
   # Kill process if occupied
   kill -9 <PID>
   ```

2. **Check Python version**:
   ```bash
   python --version  # Must be 3.9+
   ```

3. **Verify environment**:
   ```bash
   # Check virtual environment active
   which python
   # Should point to venv/bin/python
   ```

4. **Check configuration**:
   ```bash
   # Verify .env file exists
   cat .env
   # Check all required variables set
   ```

5. **View detailed errors**:
   ```bash
   python dev.py --log-level DEBUG
   ```

---

### Issue: Database errors

**Symptoms**:
- "Database locked" error
- "No such table" error
- Job history missing

**Solutions**:

1. **Reset database**:
   ```bash
   # Backup first
   cp cryomamba.db cryomamba.db.backup
   
   # Reinitialize
   python -c "from app.services.database import init_db; init_db()"
   ```

2. **Check file permissions**:
   ```bash
   ls -l cryomamba.db
   # Should be writable
   chmod 644 cryomamba.db
   ```

3. **Check disk space**:
   ```bash
   df -h
   ```

4. **Repair database**:
   ```bash
   sqlite3 cryomamba.db "PRAGMA integrity_check;"
   ```

---

## Data Issues

### Issue: "Invalid MRC format" error

**Symptoms**:
- Upload succeeds
- Job fails with format error
- Cannot read file

**Solutions**:

1. **Validate MRC file**:
   ```python
   import mrcfile
   
   with mrcfile.open('volume.mrc', permissive=True) as mrc:
       print(f"Shape: {mrc.data.shape}")
       print(f"Valid: {mrc.validate()}")
   ```

2. **Fix header issues**:
   ```python
   import mrcfile
   
   # Open in permissive mode and re-save
   with mrcfile.open('volume.mrc', permissive=True) as mrc:
       data = mrc.data
   
   with mrcfile.new('volume_fixed.mrc', overwrite=True) as mrc:
       mrc.set_data(data)
   ```

3. **Convert from other formats**:
   ```python
   import nibabel as nib
   import mrcfile
   
   # From NIFTI
   nii = nib.load('volume.nii.gz')
   data = nii.get_fdata()
   
   with mrcfile.new('volume.mrc', overwrite=True) as mrc:
       mrc.set_data(data.astype('float32'))
   ```

---

### Issue: Voxel size incorrect in results

**Symptoms**:
- Segmentation scale doesn't match EM
- Physical measurements wrong
- Overlay misaligned

**Solutions**:

1. **Check source voxel size**:
   ```python
   import mrcfile
   
   with mrcfile.open('volume.mrc') as mrc:
       print(f"Voxel size: {mrc.voxel_size}")
   ```

2. **Set voxel size explicitly**:
   ```python
   # Before upload
   with mrcfile.open('volume.mrc', mode='r+') as mrc:
       mrc.voxel_size = (1.0, 1.0, 1.5)  # X, Y, Z in nm
   ```

3. **Verify job parameters**:
   ```python
   # Include voxel size in metadata
   metadata = {
       "voxel_size": [1.0, 1.0, 1.5]
   }
   ```

---

## Export Issues

### Issue: Exported file cannot be opened in other software

**Symptoms**:
- Export succeeds
- Cannot open in ImageJ/IMOD/Chimera
- Format error in other software

**Solutions**:

1. **Verify export format**:
   - Use MRC for IMOD/Chimera
   - Use NIFTI for medical imaging software
   - Use TIFF for ImageJ

2. **Check data type**:
   ```python
   # Export with compatible data type
   export_options = {
       "dtype": "uint8",  # Most compatible
       "normalize": True   # Scale to 0-255
   }
   ```

3. **Include metadata**:
   ```python
   export_options = {
       "include_voxel_size": True,
       "include_origin": True
   }
   ```

4. **Validate exported file**:
   ```python
   import mrcfile
   
   with mrcfile.open('exported.mrc') as mrc:
       print(f"Valid: {mrc.validate()}")
       print(f"Shape: {mrc.data.shape}")
   ```

---

## Known Issues & Limitations

### Current Limitations

1. **File Size Limits**
   - Maximum upload: 10 GB (configurable)
   - Maximum dimensions: 2048×2048×2048
   - Larger files may require preprocessing

2. **Supported Formats**
   - Input: MRC only
   - Output: MRC, NIFTI, TIFF
   - Other formats require conversion

3. **Concurrent Processing**
   - Server limits: Based on GPU memory
   - Typical: 2-4 concurrent jobs per GPU
   - Large jobs may monopolize GPU

4. **Network Requirements**
   - WebSocket connections required
   - Proxies may interfere
   - VPN may add latency

5. **macOS Only (Desktop Client)**
   - Windows/Linux support planned
   - Current version: macOS 12+

### Known Bugs

1. **WebSocket reconnection sometimes requires app restart**
   - Workaround: Restart application
   - Fix in progress: Auto-reconnect with exponential backoff

2. **Large exports (>5GB) may timeout**
   - Workaround: Export in chunks or use lower resolution
   - Fix planned: Streaming export

3. **Preview may show artifacts at boundaries**
   - Workaround: Ignore preview, final result correct
   - Cosmetic issue only

4. **First job after server restart is slow**
   - Workaround: Run test job to warm up
   - Due to model loading time

---

## FAQ

### General Questions

**Q: How long does a typical segmentation take?**

A: For a 512×512×300 volume with default settings:
- Fast mode (step_size=1.0, no TTA): 5-8 minutes
- High quality (step_size=0.5, TTA enabled): 15-25 minutes
- Time scales with volume size and GPU performance

**Q: Can I process multiple files simultaneously?**

A: Yes, but limited by GPU memory. Typical limit: 2-4 concurrent jobs per GPU.

**Q: Is my data stored on the server?**

A: Uploaded files stored temporarily. Automatically deleted after:
- Raw uploads: 7 days
- Job artifacts: 30 days (configurable)

**Q: Can I use custom trained models?**

A: Yes, contact administrator to install custom nnU-Net models on server.

**Q: What happens if my connection drops during a job?**

A: Job continues running on server. Reconnect to receive updates.

**Q: How accurate is the segmentation?**

A: Depends on training data. For mitochondria segmentation on similar datasets:
- Dice score: 0.85-0.92
- False positive rate: < 5%
- Manual review recommended for critical applications

### Technical Questions

**Q: What GPU is required?**

A: Minimum 24 GB VRAM (e.g., RTX 3090, A6000). Recommended: 48 GB (A6000, A100).

**Q: Can I run the server on multiple GPUs?**

A: Yes, set `CUDA_VISIBLE_DEVICES=0,1,2,3` to use multiple GPUs.

**Q: Is the system containerized?**

A: Yes, Docker support available. See docker-compose.yml.

**Q: What network bandwidth is needed?**

A: Minimum 10 Mbps for uploads. 1 Gbps recommended for large files.

**Q: Can I use on Windows/Linux?**

A: Server: Yes (Linux recommended). Desktop client: macOS only (for now).

---

## Performance Tuning

### Server Optimization

#### GPU Optimization

```python
# app/config.py

# Memory management
PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:256"
CUDA_LAUNCH_BLOCKING = 0  # Async kernel launches

# Performance
CUDNN_BENCHMARK = True  # Auto-tune convolution algorithms
OMP_NUM_THREADS = 4     # Limit CPU threads

# Precision (if supported by model)
USE_MIXED_PRECISION = True  # FP16 for 2× speedup
```

#### Server Configuration

```python
# Increase worker processes for API handling
uvicorn app.main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker

# Tune concurrent jobs based on GPU memory
MAX_CONCURRENT_JOBS = 2  # For 24 GB VRAM
MAX_CONCURRENT_JOBS = 4  # For 48 GB VRAM

# Enable connection pooling
DB_POOL_SIZE = 10
DB_MAX_OVERFLOW = 20
```

### Client Optimization

```python
# Settings → Performance

# Memory management
layer_cache_size_mb = 512
preview_cache_size_mb = 256
auto_gc_interval_seconds = 300

# Rendering
rendering_downsampling = 2
rendering_quality = "medium"
disable_3d_during_jobs = True

# Network
chunk_size_mb = 10
max_concurrent_uploads = 2
preview_interval_seconds = 10
```

### Network Optimization

```bash
# Increase TCP buffer sizes (Linux server)
sysctl -w net.core.rmem_max=134217728
sysctl -w net.core.wmem_max=134217728

# Enable TCP window scaling
sysctl -w net.ipv4.tcp_window_scaling=1

# Reduce TIME_WAIT timeout
sysctl -w net.ipv4.tcp_fin_timeout=30
```

---

## Getting Support

### Before Requesting Support

1. **Check this troubleshooting guide**
2. **Search existing issues** on GitHub
3. **Collect diagnostic information**:
   ```bash
   # Server info
   python -c "from app import __version__; print(__version__)"
   python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
   nvidia-smi
   
   # Client info
   python -c "import napari_cryomamba; print(napari_cryomamba.__version__)"
   sw_vers  # macOS version
   
   # Logs
   tail -100 logs/server.log > server_logs.txt
   tail -100 ~/.cryomamba/logs/client.log > client_logs.txt
   ```

### Support Channels

1. **GitHub Issues**
   - Bug reports: Use bug template
   - Feature requests: Use feature template
   - Include: Version, OS, error logs, steps to reproduce

2. **GitHub Discussions**
   - General questions
   - Usage help
   - Feature discussions

3. **Email Support** (if applicable)
   - Critical issues
   - Security concerns
   - Email: support@cryomamba.com

### Bug Report Template

```markdown
## Description
Brief description of the issue

## Environment
- CryoMamba version: 1.0.0
- OS: macOS 13.2
- Python version: 3.10.8
- GPU: NVIDIA RTX 3090

## Steps to Reproduce
1. Open application
2. Load file: test_data/volume.mrc
3. Start segmentation
4. Error appears

## Expected Behavior
Should complete segmentation

## Actual Behavior
Fails with "Out of memory" error

## Logs
```
[Paste relevant log excerpt]
```

## Additional Context
- File size: 5 GB
- Previously worked with smaller files
```

---

## Appendix: Log Locations

### Server Logs

```
logs/
  ├── server.log           # Main server log
  ├── orchestrator.log     # Job orchestration
  ├── nnunet.log          # nnU-Net inference
  ├── gpu.log             # GPU monitoring
  └── access.log          # API access log
```

### Client Logs

```
~/.cryomamba/
  └── logs/
      ├── client.log      # Application log
      ├── websocket.log   # WebSocket communication
      └── crash.log       # Crash reports (if any)
```

### Viewing Logs

```bash
# Tail logs in real-time
tail -f logs/server.log

# Search for errors
grep ERROR logs/server.log

# View last 100 lines
tail -100 logs/server.log

# Filter by timestamp
grep "2025-10-17" logs/server.log
```

---

**Troubleshooting Guide Version**: 1.0.0  
**Last Updated**: October 2025  
**Maintained by**: CryoMamba Team

For updates to this guide, check the documentation repository or GitHub wiki.

