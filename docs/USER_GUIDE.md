# CryoMamba User Guide

Complete guide for using CryoMamba to segment cryo-electron tomography volumes.

**Version**: 1.0.0  
**Last Updated**: October 2025

---

## Table of Contents

- [Getting Started](#getting-started)
- [Desktop Application Overview](#desktop-application-overview)
- [Common Workflows](#common-workflows)
- [Advanced Features](#advanced-features)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Tutorial Examples](#tutorial-examples)
- [Tips & Tricks](#tips--tricks)

---

## Getting Started

### Prerequisites

Before using CryoMamba, ensure you have:

1. **Access to GPU Server**: Server URL and authentication token
2. **Desktop Application Installed**: CryoMamba.app or Python environment
3. **Sample Data**: .mrc format cryo-ET volumes
4. **Network Connection**: Stable connection to GPU server

### First Launch

1. **Open CryoMamba**
   - macOS: Double-click CryoMamba.app in Applications
   - Python: Run `python napari_cryomamba/main.py`

2. **Configure Server Connection**
   - Go to **Settings → Server Configuration**
   - Enter server URL (e.g., `https://gpu-server.your-lab.edu`)
   - Enter authentication token (provided by admin)
   - Click **Test Connection**
   - Save settings

3. **Load Sample Data**
   - Click **File → Open** or drag-and-drop a .mrc file
   - Navigate to `test_data/` folder
   - Select `hela_cell_em.mrc`
   - Volume appears in viewer

✅ **You're ready to start segmenting!**

---

## Desktop Application Overview

### Main Window Layout

```
┌────────────────────────────────────────────────────────────┐
│  File  Edit  View  Tools  Export  Help                     │
├────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌───────────────────────────────┐   │
│  │  Control Panel  │  │                               │   │
│  │                 │  │      Viewer Area              │   │
│  │  • Server       │  │                               │   │
│  │  • Load File    │  │      [3D Volume Display]      │   │
│  │  • Segmentation │  │                               │   │
│  │  • Visualization│  │                               │   │
│  │  • Export       │  │                               │   │
│  │                 │  │                               │   │
│  └─────────────────┘  └───────────────────────────────┘   │
├────────────────────────────────────────────────────────────┤
│  Status: Connected | Job: Running (45%) | GPU: 85%        │
└────────────────────────────────────────────────────────────┘
```

### Control Panel Sections

#### 1. Server Section
- **Server Status**: Connection indicator (🟢 Connected / 🔴 Disconnected)
- **Server Info**: URL, GPU availability, queue status
- **Refresh**: Update server status

#### 2. File Section
- **Open File**: Load .mrc volume
- **File Info**: Display volume metadata
- **Recent Files**: Quick access to recent volumes

#### 3. Segmentation Section
- **Model Selection**: Choose segmentation model
- **Parameters**: Configure inference settings
- **Run Segmentation**: Start job
- **Cancel**: Stop running job

#### 4. Visualization Section
- **2D/3D Toggle**: Switch viewing modes
- **Slice Navigation**: Scroll through Z-slices
- **Contrast/Brightness**: Adjust display
- **Colormap**: Change color scheme
- **Overlay Controls**: Blend segmentation with EM

#### 5. Export Section
- **Format Selection**: Choose output format
- **Export Mask**: Save segmentation
- **Export Preview**: Save current view
- **Export Statistics**: Save quantification data

### Viewer Area

#### 2D Slice View
- **Navigation**: Mouse wheel or ↑/↓ arrows to scroll slices
- **Zoom**: Scroll wheel with Shift key
- **Pan**: Click and drag
- **Measure**: Right-click → Measure distance

#### 3D Volume View
- **Rotate**: Click and drag
- **Zoom**: Scroll wheel
- **Pan**: Shift + click and drag
- **Reset View**: Double-click

### Status Bar

Displays real-time information:
- **Server Connection**: Current connection status
- **Job Progress**: Running job percentage
- **GPU Status**: Server GPU utilization
- **Memory Usage**: Client memory usage

---

## Common Workflows

### Workflow 1: Basic Segmentation

**Goal**: Segment mitochondria from a cryo-ET volume

**Steps**:

1. **Load Volume**
   ```
   File → Open → Select hela_cell_em.mrc
   ```

2. **Inspect Volume**
   - View metadata in File Info panel
   - Scroll through slices to assess quality
   - Adjust contrast for better visibility

3. **Configure Segmentation**
   - Model: `3d_fullres` (default)
   - Step Size: `0.5` (higher quality, slower)
   - Test-Time Augmentation: ✅ Enabled (better accuracy)

4. **Start Segmentation**
   - Click **Run Segmentation**
   - Job submits to server
   - Progress bar appears

5. **Monitor Progress**
   - Watch progress updates (5-10 minute typical)
   - Real-time previews appear every 5 seconds
   - Check status messages

6. **View Results**
   - Segmentation mask loads automatically when complete
   - Adjust overlay opacity to compare with EM
   - Toggle between 2D slices and 3D rendering

7. **Export Results**
   ```
   Export → Segmentation Mask → Format: MRC
   Choose location: ~/Desktop/hela_cell_mito.mrc
   ```

**Time**: ~15 minutes for 512×512×300 volume

---

### Workflow 2: Batch Processing

**Goal**: Process multiple volumes from the same dataset

**Steps**:

1. **Prepare File List**
   - Organize volumes in a folder
   - Ensure consistent naming (e.g., `volume_001.mrc`, `volume_002.mrc`)

2. **Process First Volume**
   - Load first volume
   - Find optimal segmentation parameters
   - Note settings for batch

3. **Create Processing Script** (Optional)
   ```python
   # batch_process.py
   from napari_cryomamba import process_volume
   import glob
   
   volumes = glob.glob("data/*.mrc")
   params = {
       "model": "3d_fullres",
       "step_size": 0.5,
       "disable_tta": False
   }
   
   for volume_path in volumes:
       output_path = volume_path.replace("_em.mrc", "_mito.mrc")
       process_volume(volume_path, output_path, params)
   ```

4. **Run Batch**
   ```bash
   python batch_process.py
   ```

5. **Quality Check**
   - Load each result
   - Verify segmentation quality
   - Re-process any failed volumes with adjusted parameters

**Time**: ~20 minutes per volume (automated)

---

### Workflow 3: Comparing Results

**Goal**: Compare segmentations with different parameters

**Steps**:

1. **Load Reference Volume**
   ```
   File → Open → volume_em.mrc
   ```

2. **Run First Segmentation**
   - Parameters Set A: step_size=0.5, TTA enabled
   - Export as `result_A.mrc`

3. **Run Second Segmentation**
   - Parameters Set B: step_size=1.0, TTA disabled
   - Export as `result_B.mrc`

4. **Load Both Results**
   ```
   File → Open Layer → result_A.mrc (Red channel)
   File → Open Layer → result_B.mrc (Blue channel)
   ```

5. **Compare Visually**
   - Areas in purple: Agreement
   - Areas in red: Only in result A
   - Areas in blue: Only in result B

6. **Quantify Differences**
   ```
   Tools → Compare Segmentations
   Select: result_A.mrc vs result_B.mrc
   Metrics: Dice, IoU, Volume difference
   ```

7. **Choose Best Result**
   - Review quantitative metrics
   - Inspect regions of disagreement
   - Select optimal parameters

**Time**: ~30 minutes

---

### Workflow 4: Manual Refinement

**Goal**: Manually correct automatic segmentation

**Steps**:

1. **Load Segmentation Result**
   ```
   File → Open → volume_em.mrc
   File → Open Layer → segmentation.mrc
   ```

2. **Identify Errors**
   - Scroll through slices
   - Mark regions needing correction
   - Common errors: merged objects, missed small objects

3. **Paint Corrections** (Future Feature)
   ```
   Tools → Paint Brush
   Mode: Add (for missed regions)
   Mode: Erase (for false positives)
   Brush Size: Adjust as needed
   ```

4. **3D Refinement**
   - Switch to 3D view
   - Identify disconnected components
   - Use flood fill to correct

5. **Export Corrected Mask**
   ```
   Export → Segmentation Mask → corrected_mask.mrc
   ```

**Time**: Variable (minutes to hours)

---

## Advanced Features

### Custom Model Selection

If your lab has trained custom models:

1. **Configure Model Path**
   ```
   Settings → Advanced → Custom Model Directory
   Browse: /path/to/custom/models/
   ```

2. **Refresh Model List**
   - Segmentation panel shows custom models
   - Select custom model from dropdown

3. **Adjust Parameters**
   - Custom models may need different parameters
   - Consult model documentation

### Preview Configuration

Customize real-time preview behavior:

```
Settings → Preview Configuration

• Enable Previews: ✅
• Update Interval: 5 seconds (reduce for slower networks)
• Downsample Factor: 4 (increase for faster preview)
• Preview Format: PNG (or JPEG for smaller size)
```

### GPU Priority & Queue Management

For shared servers:

```
Settings → Job Configuration

• Job Priority: Normal | High | Low
• Allow Queue Jump: ❌ (admin only)
• Max Concurrent Jobs: 1 (per user)
• Timeout: 4 hours
```

### Metadata Export

Include metadata with exports:

```
Export → Advanced Options

• Include Metadata: ✅
• Voxel Size: Copy from source
• Origin: Copy from source
• Additional Fields:
  - Acquisition Date
  - Microscope
  - Processing Date
  - Model Version
  - Parameters
```

### Keyboard Shortcuts

Customize shortcuts:

```
Settings → Keyboard Shortcuts

Action                  Default         Custom
----------------------------------------
Open File               Cmd+O           
Save Export             Cmd+S           
Toggle 2D/3D            T               
Next Slice              ↓               
Previous Slice          ↑               
Zoom In                 +               
Zoom Out                -               
Reset View              R               
Toggle Mask             M               
Run Segmentation        Cmd+R           
Cancel Job              Cmd+.           
```

---

## Best Practices

### Data Preparation

1. **File Format**
   - Use MRC format for best compatibility
   - Verify file integrity before upload
   - Check header information is correct

2. **Volume Size**
   - Optimal: 256-512 pixels per dimension
   - Large volumes (>1024): Consider cropping or downsampling
   - Small volumes (<128): May need padding

3. **Intensity Normalization**
   - Ensure consistent contrast across dataset
   - Remove extreme outliers
   - Standardize exposure settings during acquisition

### Segmentation Parameters

1. **Model Selection**
   - `3d_fullres`: Best quality, slower (typical choice)
   - `3d_lowres`: Faster, lower quality (for quick tests)
   - Custom models: For specialized organelles

2. **Step Size**
   - `0.5`: High quality, ~2× slower (recommended for final results)
   - `1.0`: Balanced quality/speed (default, good for testing)
   - `1.5`: Faster, lower quality (quick exploration only)

3. **Test-Time Augmentation (TTA)**
   - ✅ Enable for: Final results, difficult samples
   - ❌ Disable for: Testing parameters, time-sensitive work
   - Impact: +10-20% accuracy, 2-4× slower

### Network & Performance

1. **Upload Optimization**
   - Use wired connection for large files
   - Upload during off-peak hours
   - Enable upload resume in settings

2. **Preview Performance**
   - Reduce update frequency on slow connections
   - Increase downsample factor for faster preview
   - Disable 3D rendering during job

3. **Memory Management**
   - Close unused volumes
   - Clear cache regularly: `Settings → Clear Cache`
   - Monitor memory in status bar

### Quality Control

1. **Visual Inspection**
   - Check every 10th slice minimum
   - Look for edge artifacts
   - Verify 3D connectivity

2. **Quantitative Validation**
   - Compare volumes/counts across similar samples
   - Check statistics for outliers
   - Validate against manual annotations (if available)

3. **Reproducibility**
   - Document parameters used
   - Keep metadata with results
   - Version control for custom models

---

## Troubleshooting

### Issue: Connection Failed

**Symptoms**: Cannot connect to server, red status indicator

**Solutions**:
1. Verify server URL is correct
2. Check network connection: `ping server-hostname`
3. Test server: `curl https://server-url/v1/healthz`
4. Verify authentication token hasn't expired
5. Check firewall/VPN settings

---

### Issue: Upload Fails or Stalls

**Symptoms**: Upload progress stuck, timeout errors

**Solutions**:
1. Check file size (max 10 GB by default)
2. Verify disk space on server
3. Resume upload: `File → Resume Upload`
4. Try smaller chunk size: `Settings → Upload Chunk Size`
5. Use direct network connection (avoid WiFi)

---

### Issue: Job Stuck in Queue

**Symptoms**: Job remains "queued" for extended time

**Solutions**:
1. Check queue position: hover over status
2. View server load: `View → Server Stats`
3. Contact admin if queue not moving
4. Cancel and resubmit if position not changing

---

### Issue: Poor Segmentation Quality

**Symptoms**: Missed objects, false positives, artifacts

**Solutions**:
1. Enable Test-Time Augmentation
2. Reduce step size to 0.5
3. Check volume intensity normalization
4. Verify model is appropriate for sample type
5. Try different slice positions for reference
6. Consider manual refinement

---

### Issue: Preview Not Updating

**Symptoms**: Progress bar moves but no visual updates

**Solutions**:
1. Check preview settings are enabled
2. Verify WebSocket connection (see status bar)
3. Refresh view: `View → Refresh`
4. Reconnect: `Server → Reconnect`
5. Check browser console for errors (web version)

---

### Issue: Export Fails

**Symptoms**: Export button grayed out, export errors

**Solutions**:
1. Verify job completed successfully
2. Check disk space on local machine
3. Try different export format
4. Reduce resolution if memory error
5. Export to different location

---

### Issue: Application Crashes

**Symptoms**: Sudden quit, frozen interface

**Solutions**:
1. Check system memory (Activity Monitor)
2. Close other applications
3. Reduce volume size loaded
4. Update to latest version
5. Check crash logs: `~/Library/Logs/CryoMamba/`
6. Report bug with logs

---

## Tutorial Examples

### Tutorial 1: HeLa Cell Mitochondria

**Objective**: Segment mitochondria from HeLa cell cryo-ET volume

**Dataset**: `test_data/hela_cell_em.mrc`

**Expected Time**: 15 minutes

**Steps**:

1. Launch CryoMamba
2. Load `hela_cell_em.mrc`
3. Inspect volume:
   - Shape: 512×512×300
   - Voxel size: 1.0×1.0×1.5 nm
   - Intensity range: -200 to 800
4. Configure segmentation:
   - Model: 3d_fullres
   - Step size: 0.5
   - TTA: Enabled
5. Run segmentation (expected: 8-12 minutes)
6. Review results:
   - Mitochondria count: ~15-20 objects
   - Average volume: ~50,000 nm³
7. Export: `hela_cell_mito.mrc`

**Learning Points**:
- Basic workflow
- Parameter configuration
- Quality assessment

---

### Tutorial 2: C. elegans EM

**Objective**: Segment multiple organelles from C. elegans sample

**Dataset**: `test_data/c_elegans_em.mrc`

**Expected Time**: 25 minutes

**Steps**:

1. Load `c_elegans_em.mrc`
2. First pass - mitochondria:
   - Model: 3d_fullres
   - Run and export: `c_elegans_mito.mrc`
3. Second pass - other organelles:
   - Reload original volume
   - Model: custom_organelle (if available)
   - Export: `c_elegans_organelles.mrc`
4. Compare results:
   - Load both masks in different colors
   - Check for overlap
   - Identify unique vs. shared regions

**Learning Points**:
- Multi-class segmentation
- Result comparison
- Handling complex samples

---

### Tutorial 3: Parameter Optimization

**Objective**: Find optimal parameters for your dataset

**Dataset**: Any .mrc volume

**Expected Time**: 1 hour

**Steps**:

1. **Test 1: Baseline**
   - step_size: 1.0, TTA: disabled
   - Record: time, quality (visual assessment)

2. **Test 2: Higher Quality**
   - step_size: 0.5, TTA: disabled
   - Compare: time difference, quality improvement

3. **Test 3: With TTA**
   - step_size: 0.5, TTA: enabled
   - Compare: additional quality gain vs. time cost

4. **Test 4: Faster**
   - step_size: 1.5, TTA: disabled
   - Compare: acceptable quality threshold

5. **Analysis**:
   - Create table of time vs. quality
   - Choose optimal tradeoff
   - Document recommended settings

**Learning Points**:
- Parameter tuning methodology
- Quality vs. speed tradeoffs
- Systematic evaluation

---

## Tips & Tricks

### 🚀 Speed Tips

1. **Use Lower Quality for Initial Exploration**
   - Test with step_size=1.5 first
   - Switch to 0.5 only for final results

2. **Disable Unnecessary Features**
   - Turn off 3D rendering during processing
   - Reduce preview update frequency
   - Close other applications

3. **Batch During Off-Hours**
   - Submit jobs overnight
   - Use server queue effectively
   - Process multiple volumes sequentially

### 🎯 Quality Tips

1. **Always Inspect Before Export**
   - Check multiple slice positions
   - Verify 3D connectivity
   - Compare with raw EM

2. **Use TTA for Important Results**
   - Final results
   - Difficult samples
   - Publication figures

3. **Calibrate Your Eye**
   - Practice on test data with ground truth
   - Learn what "good" looks like
   - Document your quality standards

### 💡 Workflow Tips

1. **Organize Your Files**
   ```
   project/
     raw/           # Original EM volumes
     segmented/     # Segmentation results
     refined/       # Manually corrected
     exports/       # Final exports
     metadata/      # Processing logs
   ```

2. **Name Files Systematically**
   - Include date, sample ID, parameters
   - Example: `20250117_sample01_fullres_tta.mrc`

3. **Document Your Parameters**
   - Keep log file with all settings
   - Note any issues or observations
   - Track parameter evolution

### 🔧 Troubleshooting Tips

1. **When in Doubt, Test Connection**
   - Many issues are network-related
   - Test with `curl` command
   - Check server logs

2. **Monitor Resource Usage**
   - Watch Activity Monitor (macOS)
   - Check server GPU status
   - Clean up old jobs regularly

3. **Keep Backups**
   - Export important results immediately
   - Keep original data separate
   - Version your results

---

## Keyboard Shortcuts Reference

### File Operations
- `Cmd+O` - Open file
- `Cmd+W` - Close file
- `Cmd+S` - Quick export
- `Cmd+Shift+S` - Export as...

### Navigation
- `↑/↓` - Previous/next slice
- `Page Up/Down` - Jump 10 slices
- `Home/End` - First/last slice
- `Space` - Play slice animation

### View
- `T` - Toggle 2D/3D
- `M` - Toggle mask overlay
- `[/]` - Adjust contrast
- `+/-` - Zoom in/out
- `R` - Reset view
- `F` - Fit to window

### Segmentation
- `Cmd+R` - Run segmentation
- `Cmd+.` - Cancel job
- `Cmd+I` - View job info

### Window
- `Cmd+,` - Preferences
- `Cmd+Q` - Quit application

---

## Glossary

- **Cryo-ET**: Cryo-electron tomography
- **MRC**: Medical Research Council file format
- **nnU-Net**: No-new-Net, self-configuring deep learning framework
- **TTA**: Test-time augmentation
- **Step Size**: Overlapping stride for sliding window inference
- **Voxel**: Volume pixel, 3D equivalent of pixel
- **Downsampling**: Reducing resolution for faster processing
- **Segmentation**: Process of identifying and labeling structures

---

## Additional Resources

- **README**: [README.md](../README.md) - Installation and setup
- **API Reference**: [API_REFERENCE.md](API_REFERENCE.md) - Complete API docs
- **Architecture**: [docs/architecture/](architecture/) - System design
- **Contributing**: [CONTRIBUTING.md](CONTRIBUTING.md) - Development guide

---

## Feedback

Help us improve this guide:
- **Report Issues**: <repository-url>/issues
- **Suggest Improvements**: <repository-url>/discussions
- **Contribute Examples**: Pull requests welcome

---

**User Guide Version**: 1.0.0  
**Last Updated**: October 2025  
**Maintained by**: CryoMamba Team

