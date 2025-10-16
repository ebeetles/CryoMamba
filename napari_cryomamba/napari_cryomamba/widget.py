"""
Main CryoMamba widget for napari integration.
"""

from qtpy.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QTextEdit, QGroupBox, QLineEdit, QSpinBox, QCheckBox, QProgressBar
from qtpy.QtCore import Qt, Signal, QThread, QTimer
import napari
import mrcfile
import numpy as np
from pathlib import Path
import asyncio
import json
import requests
import os
import time
from .websocket_client import WebSocketClient, WebSocketWorker, PreviewDataProcessor


class CryoMambaWidget(QWidget):
    """Main CryoMamba widget for napari."""
    
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.current_volume = None
        self.current_volume_path = None
        self.current_job_id = None
        self.current_upload_id = None
        self.is_inference_running = False
        self._overlay_done = False
        self._job_poll_timer: QTimer = QTimer(self)
        self._job_poll_timer.setInterval(2000)  # 2s
        self._job_poll_timer.timeout.connect(self._poll_job_status)
        
        # WebSocket components
        self.websocket_client = WebSocketClient()
        self.websocket_worker = WebSocketWorker(self.websocket_client)
        self.setup_websocket_connections()
        
        self.setup_ui()
    
    def setup_websocket_connections(self):
        """Set up WebSocket signal connections."""
        self.websocket_client.connected.connect(self.on_websocket_connected)
        self.websocket_client.disconnected.connect(self.on_websocket_disconnected)
        self.websocket_client.preview_received.connect(self.on_preview_received)
        self.websocket_client.progress_received.connect(self.on_progress_received)
        self.websocket_client.error_received.connect(self.on_error_received)
        self.websocket_client.job_completed.connect(self.on_job_completed)
        
        # Start WebSocket worker thread
        self.websocket_worker.start()
        
    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout()
        
        # File operations group
        file_group = QGroupBox("File Operations")
        file_layout = QVBoxLayout()
        
        self.open_button = QPushButton("Open .mrc File")
        self.open_button.clicked.connect(self.open_mrc_file)
        file_layout.addWidget(self.open_button)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # Server configuration group
        server_group = QGroupBox("Server Configuration")
        server_layout = QVBoxLayout()
        
        # Server URL input
        server_url_layout = QHBoxLayout()
        server_url_layout.addWidget(QLabel("Server:"))
        self.server_url_input = QLineEdit("http://localhost:8000")
        server_url_layout.addWidget(self.server_url_input)
        
        # Server status check button
        self.check_server_button = QPushButton("Check Server")
        self.check_server_button.clicked.connect(self.check_server_connection)
        server_url_layout.addWidget(self.check_server_button)
        
        server_layout.addLayout(server_url_layout)
        
        # Server status
        self.server_status = QLabel("Not connected")
        self.server_status.setStyleSheet("color: red;")
        server_layout.addWidget(self.server_status)
        
        server_group.setLayout(server_layout)
        layout.addWidget(server_group)
        
        # Inference configuration group
        inference_group = QGroupBox("Inference Configuration")
        inference_layout = QVBoxLayout()
        
        # Patch size configuration
        patch_layout = QHBoxLayout()
        patch_layout.addWidget(QLabel("Patch Size:"))
        self.patch_size_input = QSpinBox()
        self.patch_size_input.setRange(64, 512)
        self.patch_size_input.setValue(128)
        self.patch_size_input.setSuffix(" px")
        patch_layout.addWidget(self.patch_size_input)
        inference_layout.addLayout(patch_layout)
        
        # Overlap configuration
        overlap_layout = QHBoxLayout()
        overlap_layout.addWidget(QLabel("Overlap:"))
        self.overlap_input = QSpinBox()
        self.overlap_input.setRange(0, 50)
        self.overlap_input.setValue(25)
        self.overlap_input.setSuffix("%")
        overlap_layout.addWidget(self.overlap_input)
        inference_layout.addLayout(overlap_layout)
        
        # Test Time Augmentation checkbox
        self.use_tta_checkbox = QCheckBox("Use Test Time Augmentation (TTA)")
        self.use_tta_checkbox.setChecked(True)
        inference_layout.addWidget(self.use_tta_checkbox)
        
        inference_group.setLayout(inference_layout)
        layout.addWidget(inference_group)
        
        # Main action group
        action_group = QGroupBox("Inference Control")
        action_layout = QVBoxLayout()
        
        # Main inference button
        self.run_inference_button = QPushButton("Upload & Run Inference")
        self.run_inference_button.clicked.connect(self.run_inference)
        self.run_inference_button.setEnabled(False)
        self.run_inference_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px; }")
        action_layout.addWidget(self.run_inference_button)
        
        # Cancel button
        self.cancel_inference_button = QPushButton("Cancel Inference")
        self.cancel_inference_button.clicked.connect(self.cancel_inference)
        self.cancel_inference_button.setEnabled(False)
        self.cancel_inference_button.setStyleSheet("QPushButton { background-color: #f44336; color: white; }")
        action_layout.addWidget(self.cancel_inference_button)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        action_layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("Ready to run inference")
        self.status_label.setStyleSheet("color: green;")
        action_layout.addWidget(self.status_label)
        
        action_group.setLayout(action_layout)
        layout.addWidget(action_group)
        
        # Volume info group
        info_group = QGroupBox("Volume Information")
        info_layout = QVBoxLayout()
        
        self.info_text = QTextEdit()
        self.info_text.setMaximumHeight(150)
        self.info_text.setReadOnly(True)
        info_layout.addWidget(self.info_text)
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # Visualization controls group
        viz_group = QGroupBox("Visualization Controls")
        viz_layout = QVBoxLayout()
        
        self.toggle_3d_button = QPushButton("Switch to 3D")
        self.toggle_3d_button.clicked.connect(self.toggle_3d_view)
        self.toggle_3d_button.setEnabled(False)
        viz_layout.addWidget(self.toggle_3d_button)
        
        self.clear_button = QPushButton("Clear Volume")
        self.clear_button.clicked.connect(self.clear_volume)
        self.clear_button.setEnabled(False)
        viz_layout.addWidget(self.clear_button)
        
        viz_group.setLayout(viz_layout)
        layout.addWidget(viz_group)
        
        self.setLayout(layout)
        
    def open_mrc_file(self):
        """Open and load an .mrc file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Open .mrc File", 
            "", 
            "MRC Files (*.mrc);;All Files (*)"
        )
        
        if file_path:
            try:
                self.load_mrc_file(file_path)
            except Exception as e:
                self.info_text.setText(f"Error loading file: {str(e)}")
                
    def load_mrc_file(self, file_path):
        """Load an .mrc file and display it in napari."""
        try:
            with mrcfile.open(file_path) as mrc:
                data = mrc.data.copy()
                
            # Validate data
            if data.size == 0:
                raise ValueError("MRC file contains no data")
                
            # Display volume metadata
            metadata = self.get_volume_metadata(file_path, data)
            self.display_metadata(metadata)
            
            # Add to napari viewer
            self.viewer.add_image(
                data, 
                name=Path(file_path).stem,
                colormap='gray',
                opacity=0.8
            )
            
            self.current_volume = data
            self.current_volume_path = file_path
            self.toggle_3d_button.setEnabled(True)
            self.clear_button.setEnabled(True)
            self.run_inference_button.setEnabled(True)
            self.status_label.setText("Volume loaded - ready to run inference")
            self.status_label.setStyleSheet("color: green;")
            
        except Exception as e:
            error_msg = f"Failed to load MRC file: {str(e)}"
            self.info_text.setText(error_msg)
            self.current_volume = None
            self.toggle_3d_button.setEnabled(False)
            self.clear_button.setEnabled(False)
            raise
        
    def get_volume_metadata(self, file_path, data):
        """Extract volume metadata."""
        with mrcfile.open(file_path) as mrc:
            header = mrc.header
            
        metadata = {
            'filename': Path(file_path).name,
            'shape': data.shape,
            'dtype': str(data.dtype),
            'voxel_size': (header.cella.x / header.nx, 
                          header.cella.y / header.ny, 
                          header.cella.z / header.nz),
            'min_intensity': float(np.min(data)),
            'max_intensity': float(np.max(data)),
            'mean_intensity': float(np.mean(data)),
            'std_intensity': float(np.std(data))
        }
        
        return metadata
        
    def display_metadata(self, metadata):
        """Display volume metadata in the info panel."""
        info_text = f"""File: {metadata['filename']}
Shape: {metadata['shape']}
Data Type: {metadata['dtype']}
Voxel Size: {metadata['voxel_size']}
Intensity Range: {metadata['min_intensity']:.2f} - {metadata['max_intensity']:.2f}
Mean: {metadata['mean_intensity']:.2f}
Std Dev: {metadata['std_intensity']:.2f}"""
        
        self.info_text.setText(info_text)
        
    def clear_volume(self):
        """Clear the current volume and reset UI state."""
        if self.current_volume is not None:
            # Clear all layers from viewer
            self.viewer.layers.clear()
            
            # Reset UI state
            self.current_volume = None
            self.current_volume_path = None
            self.toggle_3d_button.setEnabled(False)
            self.clear_button.setEnabled(False)
            self.run_inference_button.setEnabled(False)
            self.status_label.setText("No volume loaded")
            self.status_label.setStyleSheet("color: gray;")
            self.info_text.clear()
            
    def toggle_3d_view(self):
        """Toggle between 2D and 3D visualization."""
        if self.current_volume is not None and len(self.viewer.layers) > 0:
            layer = self.viewer.layers[-1]
            current_ndisplay = getattr(self.viewer.dims, 'ndisplay', 2)
            if current_ndisplay == 2:
                self.viewer.dims.ndisplay = 3
                if hasattr(layer, 'rendering'):
                    layer.rendering = 'mip'
                self.toggle_3d_button.setText("Switch to 2D")
                self.info_text.append("Switched to 3D view")
            else:
                self.viewer.dims.ndisplay = 2
                if hasattr(layer, 'rendering'):
                    layer.rendering = 'translucent'
                self.toggle_3d_button.setText("Switch to 3D")
                self.info_text.append("Switched to 2D view")
    
    # WebSocket event handlers
    def on_websocket_connected(self, job_id: str):
        """Handle WebSocket connection."""
        self.server_status.setText(f"Connected to job {job_id}")
        self.server_status.setStyleSheet("color: green;")
        self.status_label.setText(f"Connected to inference job {job_id}")
        self.status_label.setStyleSheet("color: blue;")
        self.info_text.append(f"Connected to job {job_id}")
    
    def on_websocket_disconnected(self, job_id: str):
        """Handle WebSocket disconnection."""
        self.server_status.setText("Disconnected")
        self.server_status.setStyleSheet("color: red;")
        self.status_label.setText("Disconnected from server")
        self.status_label.setStyleSheet("color: red;")
        self.info_text.append(f"Disconnected from job {job_id}")
    
    def on_preview_received(self, job_id: str, preview_data: dict):
        """Handle preview data received from WebSocket."""
        try:
            # Decode preview data
            preview_array = PreviewDataProcessor.decode_preview_data(preview_data)
            
            # Add preview as labels layer
            layer_name = f"Preview_{job_id}"
            
            # Remove existing preview layer if it exists
            for layer in self.viewer.layers:
                if layer.name == layer_name:
                    self.viewer.layers.remove(layer)
                    break
            
            # Add new preview layer with better visibility
            self.viewer.add_labels(
                preview_array,
                name=layer_name,
                opacity=0.8
            )
            
            # Update info with timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.info_text.append(f"[{timestamp}] Preview update: {preview_array.shape}")
            
            # Keep only last 10 messages to avoid UI clutter
            lines = self.info_text.toPlainText().split('\n')
            if len(lines) > 10:
                self.info_text.setText('\n'.join(lines[-10:]))
            
        except Exception as e:
            self.info_text.append(f"Error processing preview: {str(e)}")
    
    def on_progress_received(self, job_id: str, progress_data: dict):
        """Handle progress updates."""
        progress_msg = progress_data.get("message", "Progress update")
        progress_value = float(progress_data.get("progress", 0.0) or 0.0)
        
        # Update progress bar
        if self.progress_bar.isVisible():
            self.progress_bar.setValue(int(progress_value * 100))
        
        self.status_label.setText(f"Inference progress: {int(progress_value * 100)}%")
        self.info_text.append(f"Progress: {int(progress_value * 100)}%")
    
    def on_error_received(self, job_id: str, error_data: dict):
        """Handle error messages."""
        error_msg = error_data.get("message", "Unknown error")
        self.info_text.append(f"Error: {error_msg}")
        
        # Handle specific error types
        if "reconnect" in error_msg.lower():
            # This is a reconnection message, don't treat as critical error
            pass
        elif "failed to reconnect" in error_msg.lower():
            # Max reconnection attempts reached
            self.connection_status.setText("Connection Failed")
            self.connection_status.setStyleSheet("color: red;")
            self.connect_button.setEnabled(True)
            self.disconnect_button.setEnabled(False)
            self.reconnect_button.setEnabled(True)
        elif "invalid server url" in error_msg.lower():
            self.info_text.append("Please check the server URL format (should start with ws:// or wss://)")
            self.reconnect_button.setEnabled(True)
    
    def on_job_completed(self, job_id: str, completion_data: dict):
        """Handle job completion."""
        self.info_text.append(f"Job {job_id} completed")
        self.status_label.setText("Inference completed successfully!")
        self.status_label.setStyleSheet("color: green;")
        
        # Hide progress bar and re-enable controls
        self.progress_bar.setVisible(False)
        self.run_inference_button.setEnabled(True)
        self.cancel_inference_button.setEnabled(False)
        self.is_inference_running = False
        
        # Stop poller and download/overlay if not done yet
        if self._job_poll_timer.isActive():
            self._job_poll_timer.stop()
        if not self._overlay_done:
            self.download_and_overlay_results(job_id)
    
    # Production workflow methods
    def check_server_connection(self):
        """Check server connection status."""
        try:
            server_url = self.server_url_input.text()
            
            # Check health endpoint
            health_response = requests.get(f"{server_url}/v1/healthz", timeout=5)
            health_response.raise_for_status()
            
            # Check server info endpoint
            info_response = requests.get(f"{server_url}/v1/server/info", timeout=5)
            info_response.raise_for_status()
            
            info_data = info_response.json()
            self.server_status.setText(f"Connected: {info_data.get('service', 'Unknown')} v{info_data.get('version', 'Unknown')}")
            self.server_status.setStyleSheet("color: green;")
            self.info_text.append(f"Server connected: {info_data.get('service', 'Unknown')} v{info_data.get('version', 'Unknown')}")
            
        except requests.exceptions.ConnectionError:
            self.server_status.setText("Connection failed")
            self.server_status.setStyleSheet("color: red;")
            self.info_text.append("Server connection failed: Cannot reach server")
        except requests.exceptions.Timeout:
            self.server_status.setText("Connection timeout")
            self.server_status.setStyleSheet("color: red;")
            self.info_text.append("Server connection failed: Request timed out")
        except requests.exceptions.HTTPError as e:
            self.server_status.setText(f"HTTP Error {e.response.status_code}")
            self.server_status.setStyleSheet("color: red;")
            self.info_text.append(f"Server connection failed: HTTP {e.response.status_code}")
        except Exception as e:
            self.server_status.setText("Connection error")
            self.server_status.setStyleSheet("color: red;")
            self.info_text.append(f"Server connection failed: {str(e)}")
    
    def run_inference(self):
        """Main production workflow: Upload file and run inference."""
        if not self.current_volume_path:
            self.info_text.append("No volume loaded. Please load an .mrc file first.")
            return
        
        if self.is_inference_running:
            self.info_text.append("Inference is already running.")
            return
        
        try:
            self.is_inference_running = True
            self.run_inference_button.setEnabled(False)
            self.cancel_inference_button.setEnabled(True)
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.status_label.setText("Starting inference...")
            self.status_label.setStyleSheet("color: blue;")
            
            # Step 1: Upload file
            self.status_label.setText("Uploading volume...")
            upload_id = self.upload_volume()
            if not upload_id:
                raise Exception("Failed to upload volume")
            
            # Step 2: Create job with upload reference
            self.status_label.setText("Creating inference job...")
            job_id = self.create_inference_job(upload_id)
            if not job_id:
                raise Exception("Failed to create inference job")
            
            # Step 3: Connect to job via WebSocket
            self.status_label.setText("Connecting to inference job...")
            self.connect_to_inference_job(job_id)
            # Start poller as fallback to WS
            self._overlay_done = False
            self._job_poll_timer.start()
            
            # Step 4: Start inference
            self.status_label.setText("Starting inference...")
            self.start_inference_processing(job_id)
            
        except Exception as e:
            self.info_text.append(f"Error running inference: {str(e)}")
            self.status_label.setText("Inference failed")
            self.status_label.setStyleSheet("color: red;")
            self.is_inference_running = False
            self.run_inference_button.setEnabled(True)
            self.cancel_inference_button.setEnabled(False)
            self.progress_bar.setVisible(False)
    
    def upload_volume(self):
        """Upload the current volume to the server."""
        try:
            server_url = self.server_url_input.text()
            
            # Initialize upload
            chunk_size = 8 * 1024 * 1024  # 8MB chunks
            init_response = requests.post(f"{server_url}/v1/uploads/init", json={
                "filename": Path(self.current_volume_path).name,
                "total_size_bytes": self.current_volume.nbytes,
                "chunk_size_bytes": chunk_size
            }, timeout=10)
            init_response.raise_for_status()
            
            upload_data = init_response.json()
            upload_id = upload_data["upload_id"]
            self.current_upload_id = upload_id
            
            # Upload file in chunks
            total_chunks = (self.current_volume.nbytes + chunk_size - 1) // chunk_size
            
            with open(self.current_volume_path, 'rb') as f:
                for chunk_idx in range(total_chunks):
                    chunk_data = f.read(chunk_size)
                    
                    # Upload chunk
                    files = {'content': (f'chunk_{chunk_idx}', chunk_data, 'application/octet-stream')}
                    upload_response = requests.put(f"{server_url}/v1/uploads/{upload_id}/part/{chunk_idx}", files=files, timeout=30)
                    upload_response.raise_for_status()
                    
                    # Update progress
                    upload_progress = (chunk_idx + 1) / total_chunks * 0.3  # Upload is 30% of total progress
                    self.progress_bar.setValue(int(upload_progress * 100))
                    self.status_label.setText(f"Uploading... {int(upload_progress * 100)}%")
            
            # Complete the upload
            self.status_label.setText("Completing upload...")
            complete_response = requests.post(f"{server_url}/v1/uploads/{upload_id}/complete", json={}, timeout=30)
            complete_response.raise_for_status()
            
            self.info_text.append(f"Volume uploaded successfully: {upload_id}")
            return upload_id
            
        except Exception as e:
            self.info_text.append(f"Upload failed: {str(e)}")
            return None
    
    def create_inference_job(self, upload_id):
        """Create an inference job with the uploaded volume."""
        try:
            server_url = self.server_url_input.text()
            
            # Get inference parameters from UI
            job_params = {
                "volume_shape": list(self.current_volume.shape),
                "volume_dtype": str(self.current_volume.dtype),
                "has_volume": True,
                "params": {
                    "upload_id": upload_id,
                    "patch_size": self.patch_size_input.value(),
                    "overlap_percent": self.overlap_input.value(),
                    "use_tta": self.use_tta_checkbox.isChecked(),
                    "dummy_inference": False  # Use real nnU-Net
                }
            }
            
            # Create job
            response = requests.post(f"{server_url}/v1/jobs", json=job_params, timeout=10)
            response.raise_for_status()
            
            job_data = response.json()
            job_id = job_data.get("job_id")
            self.current_job_id = job_id
            
            self.info_text.append(f"Created inference job: {job_id}")
            return job_id
            
        except Exception as e:
            self.info_text.append(f"Failed to create job: {str(e)}")
            return None
    
    def connect_to_inference_job(self, job_id):
        """Connect to the inference job via WebSocket."""
        try:
            # Update WebSocket client URL
            ws_url = self.server_url_input.text().replace("http://", "ws://").replace("https://", "wss://")
            self.websocket_client.server_url = ws_url
            
            # Connect to job
            asyncio.run_coroutine_threadsafe(
                self.websocket_client.connect_to_job(job_id),
                self.websocket_worker.loop
            )
            
        except Exception as e:
            self.info_text.append(f"Failed to connect to job: {str(e)}")
            raise
    
    def start_inference_processing(self, job_id):
        """Start the actual inference processing.
        If the request times out, assume start succeeded and rely on WebSocket/polling.
        """
        try:
            server_url = self.server_url_input.text()
            
            self.info_text.append("Starting inference...")
            try:
                response = requests.put(
                    f"{server_url}/v1/jobs/{job_id}",
                    json={
                        "start_inference": True,
                        "dummy_inference": False
                    },
                    timeout=5,  # keep UI responsive; server should respond immediately
                )
                response.raise_for_status()
                self.info_text.append(f"Started inference for job {job_id}")
            except requests.exceptions.Timeout:
                # Treat as non-fatal: server may still have started the job
                self.info_text.append("Start request timed out; assuming started. Monitoring via WebSocket...")
            except requests.exceptions.ConnectionError as e:
                self.info_text.append(f"Connection error when starting inference: {str(e)}")
                raise
        except Exception as e:
            self.info_text.append(f"Failed to start inference: {str(e)}")
            raise
    
    def cancel_inference(self):
        """Cancel the current inference job."""
        if not self.current_job_id:
            return
        
        try:
            server_url = self.server_url_input.text()
            
            # Cancel job
            response = requests.delete(f"{server_url}/v1/jobs/{self.current_job_id}", timeout=10)
            response.raise_for_status()
            
            # Reset UI state
            self.is_inference_running = False
            self.run_inference_button.setEnabled(True)
            self.cancel_inference_button.setEnabled(False)
            self.progress_bar.setVisible(False)
            self.status_label.setText("Inference cancelled")
            self.status_label.setStyleSheet("color: orange;")
            
            # Disconnect WebSocket
            asyncio.run_coroutine_threadsafe(
                self.websocket_client.disconnect(),
                self.websocket_worker.loop
            )
            
            self.info_text.append(f"Cancelled inference job {self.current_job_id}")
            
        except Exception as e:
            self.info_text.append(f"Failed to cancel inference: {str(e)}")
    
    def download_and_overlay_results(self, job_id):
        """Download and overlay the inference results."""
        try:
            server_url = self.server_url_input.text()
            
            # Get job details to find artifacts
            response = requests.get(f"{server_url}/v1/jobs/{job_id}", timeout=10)
            response.raise_for_status()
            
            job_data = response.json()
            artifacts = job_data.get("artifacts", {})
            
            # Download mask if available
            if "mask_nifti" in artifacts:
                # Download the artifact from the server
                artifact_url = f"{server_url}/v1/jobs/{job_id}/artifacts/mask_nifti"
                mask_response = requests.get(artifact_url, timeout=30)
                mask_response.raise_for_status()
                
                # Save mask temporarily
                mask_path = f"/tmp/cryomamba_mask_{job_id}.nii.gz"
                with open(mask_path, 'wb') as f:
                    f.write(mask_response.content)
                
                # Load and overlay mask
                import nibabel as nib
                mask_img = nib.load(mask_path)
                mask_data = mask_img.get_fdata()
                
                # Add mask as labels layer
                self.viewer.add_labels(
                    mask_data.astype(np.uint8),
                    name=f"Segmentation_{job_id}",
                    opacity=0.7
                )
                
                self.info_text.append(f"Downloaded and overlaid segmentation mask")
                self._overlay_done = True
                
                # Clean up temp file
                os.remove(mask_path)
            
        except Exception as e:
            self.info_text.append(f"Failed to download results: {str(e)}")
    
    def closeEvent(self, event):
        """Handle widget close event."""
        # Clean up WebSocket connections
        if self.websocket_worker.isRunning():
            self.websocket_worker.stop()
            self.websocket_worker.wait()
        
        event.accept()

    def _poll_job_status(self):
        """Fallback polling to fetch job status and artifacts if WS is missed."""
        try:
            if not self.current_job_id:
                return
            server_url = self.server_url_input.text()
            r = requests.get(f"{server_url}/v1/jobs/{self.current_job_id}", timeout=5)
            if r.status_code != 200:
                return
            data = r.json()
            # Update progress
            progress = float(data.get("progress", 0.0) or 0.0)
            if self.progress_bar.isVisible():
                self.progress_bar.setValue(int(progress * 100))
            self.status_label.setText(f"Inference progress: {int(progress * 100)}%")
            # If completed and not yet overlaid, download
            state = data.get("state")
            if state == "completed" and not self._overlay_done:
                self.download_and_overlay_results(self.current_job_id)
                self._job_poll_timer.stop()
        except Exception:
            # Silent fallback; WS remains primary
            pass
