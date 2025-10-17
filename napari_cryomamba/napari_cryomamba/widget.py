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


class PollingWorker(QThread):
    """Worker thread for polling job status without blocking UI."""
    
    # Signals
    status_updated = Signal(dict)  # job status data
    error_occurred = Signal(str)  # error message
    
    def __init__(self, server_url: str, job_id: str, poll_interval: float = 2.0):
        super().__init__()
        self.server_url = server_url
        self.job_id = job_id
        self.poll_interval = poll_interval
        self._should_stop = False
    
    def stop(self):
        """Request the worker to stop."""
        self._should_stop = True
    
    def run(self):
        """Poll job status in background thread."""
        import time
        
        while not self._should_stop:
            try:
                # Make HTTP request in background thread
                r = requests.get(
                    f"{self.server_url}/v1/jobs/{self.job_id}", 
                    timeout=5
                )
                if r.status_code == 200:
                    data = r.json()
                    self.status_updated.emit(data)
                    
                    # Stop polling if job completed or failed
                    state = data.get("state", "")
                    if state in ["completed", "failed", "cancelled"]:
                        break
                        
            except Exception as e:
                # Don't emit errors for connection issues during polling
                pass
            
            # Sleep between polls
            for _ in range(int(self.poll_interval * 10)):
                if self._should_stop:
                    break
                time.sleep(0.1)


class InferenceWorker(QThread):
    """Worker thread for running inference operations without blocking UI."""
    
    # Signals for communicating with main thread
    job_created = Signal(str)  # job_id
    inference_started = Signal(str)  # job_id
    error_occurred = Signal(str)  # error_message
    finished = Signal()
    
    def __init__(self, server_url: str, volume_path: str, volume: np.ndarray, 
                 patch_size: int, overlap_percent: int, use_tta: bool):
        super().__init__()
        self.server_url = server_url
        self.volume_path = volume_path
        self.volume = volume
        self.patch_size = patch_size
        self.overlap_percent = overlap_percent
        self.use_tta = use_tta
        self.upload_id = None
        self.job_id = None
        self._should_stop = False
    
    def stop(self):
        """Request the worker to stop."""
        self._should_stop = True
    
    def run(self):
        """Run the inference workflow in background thread."""
        try:
            # Step 1: Upload volume
            if self._should_stop:
                return
            upload_id = self._upload_volume()
            if not upload_id:
                self.error_occurred.emit("Failed to upload volume")
                return
            self.upload_id = upload_id
            
            # Step 2: Create job
            if self._should_stop:
                return
            job_id = self._create_inference_job(upload_id)
            if not job_id:
                self.error_occurred.emit("Failed to create inference job")
                return
            self.job_id = job_id
            self.job_created.emit(job_id)
            
            # Step 3: Start inference (server will handle async execution)
            if self._should_stop:
                return
            if not self._start_inference(job_id):
                self.error_occurred.emit("Failed to start inference")
                return
            
            self.inference_started.emit(job_id)
            self.finished.emit()
            
        except Exception as e:
            self.error_occurred.emit(f"Inference workflow error: {str(e)}")
    
    def _upload_volume(self):
        """Upload the volume to the server."""
        try:
            # Initialize upload
            chunk_size = 8 * 1024 * 1024  # 8MB chunks
            init_response = requests.post(f"{self.server_url}/v1/uploads/init", json={
                "filename": Path(self.volume_path).name,
                "total_size_bytes": self.volume.nbytes,
                "chunk_size_bytes": chunk_size
            }, timeout=10)
            init_response.raise_for_status()
            
            upload_data = init_response.json()
            upload_id = upload_data["upload_id"]
            
            # Upload file in chunks
            total_chunks = (self.volume.nbytes + chunk_size - 1) // chunk_size
            
            with open(self.volume_path, 'rb') as f:
                for chunk_idx in range(total_chunks):
                    if self._should_stop:
                        return None
                    
                    chunk_data = f.read(chunk_size)
                    
                    # Upload chunk
                    files = {'content': (f'chunk_{chunk_idx}', chunk_data, 'application/octet-stream')}
                    upload_response = requests.put(
                        f"{self.server_url}/v1/uploads/{upload_id}/part/{chunk_idx}", 
                        files=files, 
                        timeout=30
                    )
                    upload_response.raise_for_status()
            
            # Complete the upload
            if self._should_stop:
                return None
            complete_response = requests.post(
                f"{self.server_url}/v1/uploads/{upload_id}/complete", 
                json={}, 
                timeout=30
            )
            complete_response.raise_for_status()
            
            return upload_id
            
        except Exception as e:
            self.error_occurred.emit(f"Upload failed: {str(e)}")
            return None
    
    def _create_inference_job(self, upload_id):
        """Create an inference job with the uploaded volume."""
        try:
            job_params = {
                "volume_shape": list(self.volume.shape),
                "volume_dtype": str(self.volume.dtype),
                "has_volume": True,
                "params": {
                    "upload_id": upload_id,
                    "patch_size": self.patch_size,
                    "overlap_percent": self.overlap_percent,
                    "use_tta": self.use_tta,
                    "dummy_inference": False
                }
            }
            
            response = requests.post(
                f"{self.server_url}/v1/jobs", 
                json=job_params, 
                timeout=10
            )
            response.raise_for_status()
            
            job_data = response.json()
            return job_data.get("job_id")
            
        except Exception as e:
            self.error_occurred.emit(f"Failed to create job: {str(e)}")
            return None
    
    def _start_inference(self, job_id):
        """Start the inference processing."""
        try:
            response = requests.put(
                f"{self.server_url}/v1/jobs/{job_id}",
                json={
                    "start_inference": True,
                    "dummy_inference": False
                },
                timeout=5
            )
            response.raise_for_status()
            return True
        except requests.exceptions.Timeout:
            # Treat as non-fatal: server may still have started the job
            return True
        except Exception as e:
            self.error_occurred.emit(f"Failed to start inference: {str(e)}")
            return False


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
        
        # WebSocket components
        self.websocket_client = WebSocketClient()
        self.websocket_worker = WebSocketWorker(self.websocket_client)
        self.setup_websocket_connections()
        
        # Worker threads for async operations
        self.inference_worker = None
        self.polling_worker = None
        
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
        self.open_button.setStyleSheet("""
            QPushButton { 
                background-color: #e3f2fd; 
                color: #1976d2; 
                border: 1px solid #bbdefb;
                border-radius: 3px;
                padding: 6px 12px;
                font-weight: normal;
            }
            QPushButton:hover { 
                background-color: #bbdefb; 
                border-color: #90caf9;
            }
            QPushButton:pressed { 
                background-color: #90caf9; 
                border-color: #64b5f6;
            }
        """)
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
        self.check_server_button.setStyleSheet("""
            QPushButton { 
                background-color: #f3e5f5; 
                color: #7b1fa2; 
                border: 1px solid #e1bee7;
                border-radius: 3px;
                padding: 6px 12px;
            }
            QPushButton:hover { 
                background-color: #e1bee7; 
                border-color: #ce93d8;
            }
            QPushButton:pressed { 
                background-color: #ce93d8; 
                border-color: #ba68c8;
            }
        """)
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
        self.run_inference_button = QPushButton("Run Inference")
        self.run_inference_button.clicked.connect(self.run_inference)
        self.run_inference_button.setEnabled(False)
        self.run_inference_button.setStyleSheet("""
            QPushButton { 
                background-color: #4CAF50; 
                color: white; 
                border: none;
                border-radius: 3px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover { 
                background-color: #45a049; 
            }
            QPushButton:pressed { 
                background-color: #3d8b40; 
            }
            QPushButton:disabled { 
                background-color: #cccccc; 
                color: #666666; 
            }
        """)
        action_layout.addWidget(self.run_inference_button)
        
        # Cancel button
        self.cancel_inference_button = QPushButton("Cancel Inference")
        self.cancel_inference_button.clicked.connect(self.cancel_inference)
        self.cancel_inference_button.setEnabled(False)
        self.cancel_inference_button.setStyleSheet("""
            QPushButton { 
                background-color: #f44336; 
                color: white; 
                border: none;
                border-radius: 3px;
                padding: 8px 16px;
            }
            QPushButton:hover { 
                background-color: #da190b; 
            }
            QPushButton:pressed { 
                background-color: #c1170a; 
            }
            QPushButton:disabled { 
                background-color: #cccccc; 
                color: #666666; 
            }
        """)
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
        self.toggle_3d_button.setStyleSheet("""
            QPushButton { 
                background-color: #e8f5e8; 
                color: #2e7d32; 
                border: 1px solid #c8e6c9;
                border-radius: 3px;
                padding: 6px 12px;
            }
            QPushButton:hover { 
                background-color: #c8e6c9; 
                border-color: #a5d6a7;
            }
            QPushButton:pressed { 
                background-color: #a5d6a7; 
                border-color: #81c784;
            }
            QPushButton:disabled { 
                background-color: #cccccc; 
                color: #666666; 
            }
        """)
        viz_layout.addWidget(self.toggle_3d_button)
        
        self.clear_button = QPushButton("Clear Volume")
        self.clear_button.clicked.connect(self.clear_volume)
        self.clear_button.setEnabled(False)
        self.clear_button.setStyleSheet("""
            QPushButton { 
                background-color: #fff3e0; 
                color: #ef6c00; 
                border: 1px solid #ffcc02;
                border-radius: 3px;
                padding: 6px 12px;
            }
            QPushButton:hover { 
                background-color: #ffcc02; 
                border-color: #ffb300;
            }
            QPushButton:pressed { 
                background-color: #ffb300; 
                border-color: #ff9800;
            }
            QPushButton:disabled { 
                background-color: #cccccc; 
                color: #666666; 
            }
        """)
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
        """Main production workflow: Upload file and run inference asynchronously."""
        if not self.current_volume_path:
            self.info_text.append("No volume loaded. Please load an .mrc file first.")
            return
        
        if self.is_inference_running:
            self.info_text.append("Inference is already running.")
            return
        
        try:
            # Set UI state
            self.is_inference_running = True
            self.run_inference_button.setEnabled(False)
            self.cancel_inference_button.setEnabled(True)
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.status_label.setText("Starting inference...")
            self.status_label.setStyleSheet("color: blue;")
            self.info_text.append("Starting asynchronous inference workflow...")
            
            # Force UI update before starting worker
            from qtpy.QtWidgets import QApplication
            QApplication.processEvents()
            
            # Create and configure worker thread
            # Note: We only pass the volume path, not the volume data itself to avoid copying
            self.info_text.append("Creating worker thread...")
            self.inference_worker = InferenceWorker(
                server_url=self.server_url_input.text(),
                volume_path=self.current_volume_path,
                volume=self.current_volume,  # Only used for metadata (shape, dtype)
                patch_size=self.patch_size_input.value(),
                overlap_percent=self.overlap_input.value(),
                use_tta=self.use_tta_checkbox.isChecked()
            )
            
            # Connect worker signals
            self.info_text.append("Connecting signals...")
            self.inference_worker.job_created.connect(self.on_worker_job_created)
            self.inference_worker.inference_started.connect(self.on_worker_inference_started)
            self.inference_worker.error_occurred.connect(self.on_worker_error)
            self.inference_worker.finished.connect(self.on_worker_finished)
            
            # Start worker - this should return immediately
            self.info_text.append("Starting worker thread... UI should remain responsive!")
            self.inference_worker.start()
            self.info_text.append("Worker thread started! You can now interact with the UI.")
            
            # Force another UI update
            QApplication.processEvents()
            
        except Exception as e:
            self.info_text.append(f"Error starting inference: {str(e)}")
            self.status_label.setText("Failed to start inference")
            self.status_label.setStyleSheet("color: red;")
            self.is_inference_running = False
            self.run_inference_button.setEnabled(True)
            self.cancel_inference_button.setEnabled(False)
            self.progress_bar.setVisible(False)
    
    
    def on_worker_job_created(self, job_id: str):
        """Handle job creation notification from worker."""
        self.current_job_id = job_id
        self.info_text.append(f"Created inference job: {job_id}")
        
        # Connect to job via WebSocket
        try:
            ws_url = self.server_url_input.text().replace("http://", "ws://").replace("https://", "wss://")
            self.websocket_client.server_url = ws_url
            
            # Ensure WebSocket worker is running
            if not self.websocket_worker.isRunning():
                self.info_text.append("Starting WebSocket worker...")
                self.websocket_worker.start()
                # Give it a moment to start
                import time
                time.sleep(0.1)
            
            # Connect to the job
            asyncio.run_coroutine_threadsafe(
                self.websocket_client.connect_to_job(job_id),
                self.websocket_worker.loop
            )
        except Exception as e:
            self.info_text.append(f"Failed to connect to job WebSocket: {str(e)}")
    
    def on_worker_inference_started(self, job_id: str):
        """Handle inference start notification from worker."""
        self.info_text.append(f"Started inference for job {job_id}")
        self.status_label.setText("Inference running...")
        self.status_label.setStyleSheet("color: blue;")
        
        # Start polling worker as fallback (non-blocking)
        self._overlay_done = False
        self._start_polling(job_id)
    
    def on_worker_error(self, error_message: str):
        """Handle error from worker."""
        self.info_text.append(f"Error: {error_message}")
        self.status_label.setText("Inference failed")
        self.status_label.setStyleSheet("color: red;")
        
        # Reset UI state
        self.is_inference_running = False
        self.run_inference_button.setEnabled(True)
        self.cancel_inference_button.setEnabled(False)
        self.progress_bar.setVisible(False)
    
    def on_worker_finished(self):
        """Handle worker completion."""
        self.info_text.append("Inference workflow completed, monitoring progress...")
        # Don't reset UI yet - wait for job completion via WebSocket/polling
    
    def cancel_inference(self):
        """Cancel the current inference job."""
        # Stop the worker threads if they're running
        if self.inference_worker and self.inference_worker.isRunning():
            self.inference_worker.stop()
            self.inference_worker.wait(3000)  # Wait up to 3 seconds
        
        # Stop the polling worker
        self._stop_polling()
        
        if not self.current_job_id:
            # Reset UI even if no job ID
            self.is_inference_running = False
            self.run_inference_button.setEnabled(True)
            self.cancel_inference_button.setEnabled(False)
            self.progress_bar.setVisible(False)
            self.status_label.setText("Cancelled")
            self.status_label.setStyleSheet("color: orange;")
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
        # Clean up inference worker
        if self.inference_worker and self.inference_worker.isRunning():
            self.inference_worker.stop()
            self.inference_worker.wait(5000)  # Wait up to 5 seconds
        
        # Clean up polling worker
        self._stop_polling()
        
        # Clean up WebSocket connections
        if self.websocket_worker.isRunning():
            self.websocket_worker.stop()
            self.websocket_worker.wait()
        
        event.accept()

    def _start_polling(self, job_id: str):
        """Start background polling worker for job status."""
        # Stop any existing polling worker
        self._stop_polling()
        
        # Create and start new polling worker
        server_url = self.server_url_input.text()
        self.polling_worker = PollingWorker(server_url, job_id, poll_interval=2.0)
        self.polling_worker.status_updated.connect(self._on_poll_status_update)
        self.polling_worker.start()
        self.info_text.append("Started background status polling (non-blocking)")
    
    def _stop_polling(self):
        """Stop the background polling worker."""
        if self.polling_worker and self.polling_worker.isRunning():
            self.polling_worker.stop()
            self.polling_worker.wait(3000)  # Wait up to 3 seconds
            self.polling_worker = None
    
    def _on_poll_status_update(self, data: dict):
        """Handle status update from polling worker (runs on main thread via signal)."""
        try:
            # Update progress
            progress = float(data.get("progress", 0.0) or 0.0)
            if self.progress_bar.isVisible():
                self.progress_bar.setValue(int(progress * 100))
            self.status_label.setText(f"Inference progress: {int(progress * 100)}%")
            
            # If completed and not yet overlaid, download
            state = data.get("state")
            if state == "completed" and not self._overlay_done:
                self.download_and_overlay_results(self.current_job_id)
                self._stop_polling()
        except Exception as e:
            # Silent fallback; WS remains primary
            pass
