"""
Main CryoMamba widget for napari integration.
"""

from qtpy.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QTextEdit, QGroupBox, QLineEdit, QSpinBox
from qtpy.QtCore import Qt, Signal, QThread
import napari
import mrcfile
import numpy as np
from pathlib import Path
import asyncio
import json
from .websocket_client import WebSocketClient, WebSocketWorker, PreviewDataProcessor


class CryoMambaWidget(QWidget):
    """Main CryoMamba widget for napari."""
    
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.current_volume = None
        self.current_job_id = None
        
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
        
        # Job control group
        job_group = QGroupBox("Job Control")
        job_layout = QVBoxLayout()
        
        # Server URL input
        server_layout = QHBoxLayout()
        server_layout.addWidget(QLabel("Server:"))
        self.server_url_input = QLineEdit("ws://localhost:8000")
        server_layout.addWidget(self.server_url_input)
        job_layout.addLayout(server_layout)
        
        # Job ID input
        job_id_layout = QHBoxLayout()
        job_id_layout.addWidget(QLabel("Job ID:"))
        self.job_id_input = QLineEdit()
        self.job_id_input.setPlaceholderText("Enter job ID to connect")
        job_id_layout.addWidget(self.job_id_input)
        job_layout.addLayout(job_id_layout)
        
        # Job control buttons
        button_layout = QHBoxLayout()
        
        self.create_job_button = QPushButton("Create Job")
        self.create_job_button.clicked.connect(self.create_job)
        button_layout.addWidget(self.create_job_button)
        
        self.connect_button = QPushButton("Connect to Job")
        self.connect_button.clicked.connect(self.connect_to_job)
        self.connect_button.setEnabled(False)
        button_layout.addWidget(self.connect_button)
        
        self.disconnect_button = QPushButton("Disconnect")
        self.disconnect_button.clicked.connect(self.disconnect_from_job)
        self.disconnect_button.setEnabled(False)
        button_layout.addWidget(self.disconnect_button)
        
        job_layout.addLayout(button_layout)
        
        # Connection status
        self.connection_status = QLabel("Disconnected")
        self.connection_status.setStyleSheet("color: red;")
        job_layout.addWidget(self.connection_status)
        
        job_group.setLayout(job_layout)
        layout.addWidget(job_group)
        
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
            self.toggle_3d_button.setEnabled(True)
            self.clear_button.setEnabled(True)
            
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
            self.toggle_3d_button.setEnabled(False)
            self.clear_button.setEnabled(False)
            self.info_text.clear()
            
    def toggle_3d_view(self):
        """Toggle between 2D and 3D visualization."""
        if self.current_volume is not None and len(self.viewer.layers) > 0:
            # Get the current layer
            layer = self.viewer.layers[-1]
            
            # Check current view mode by looking at viewer dimensions
            current_ndisplay = self.viewer.dims.ndisplay
            
            if current_ndisplay == 2:
                # Switch to 3D view
                self.viewer.dims.ndisplay = 3
                layer.rendering = 'mip'  # Set 3D rendering mode
                self.toggle_3d_button.setText("Switch to 2D")
                self.info_text.append("Switched to 3D view")
            else:
                # Switch to 2D view
                self.viewer.dims.ndisplay = 2
                layer.rendering = 'translucent'  # Better for 2D
                self.toggle_3d_button.setText("Switch to 3D")
                self.info_text.append("Switched to 2D view")
    
    # WebSocket event handlers
    def on_websocket_connected(self, job_id: str):
        """Handle WebSocket connection."""
        self.connection_status.setText(f"Connected to job {job_id}")
        self.connection_status.setStyleSheet("color: green;")
        self.connect_button.setEnabled(False)
        self.disconnect_button.setEnabled(True)
        self.info_text.append(f"Connected to job {job_id}")
    
    def on_websocket_disconnected(self, job_id: str):
        """Handle WebSocket disconnection."""
        self.connection_status.setText("Disconnected")
        self.connection_status.setStyleSheet("color: red;")
        self.connect_button.setEnabled(True)
        self.disconnect_button.setEnabled(False)
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
            
            # Add new preview layer
            self.viewer.add_labels(
                preview_array,
                name=layer_name,
                opacity=0.7
            )
            
            self.info_text.append(f"Received preview data: {preview_array.shape}")
            
        except Exception as e:
            self.info_text.append(f"Error processing preview: {str(e)}")
    
    def on_progress_received(self, job_id: str, progress_data: dict):
        """Handle progress updates."""
        progress_msg = progress_data.get("message", "Progress update")
        self.info_text.append(f"Progress: {progress_msg}")
    
    def on_error_received(self, job_id: str, error_data: dict):
        """Handle error messages."""
        error_msg = error_data.get("message", "Unknown error")
        self.info_text.append(f"Error: {error_msg}")
    
    def on_job_completed(self, job_id: str, completion_data: dict):
        """Handle job completion."""
        self.info_text.append(f"Job {job_id} completed")
        self.disconnect_from_job()
    
    # Job control methods
    def create_job(self):
        """Create a new job on the server."""
        import requests
        
        try:
            # Extract server URL from WebSocket URL
            server_url = self.server_url_input.text().replace("ws://", "http://").replace("wss://", "https://")
            
            response = requests.post(f"{server_url}/v1/jobs")
            response.raise_for_status()
            
            job_data = response.json()
            job_id = job_data.get("job_id")
            
            if job_id:
                self.job_id_input.setText(job_id)
                self.connect_button.setEnabled(True)
                self.info_text.append(f"Created job: {job_id}")
            else:
                self.info_text.append("Failed to create job: No job ID returned")
                
        except Exception as e:
            self.info_text.append(f"Error creating job: {str(e)}")
    
    def connect_to_job(self):
        """Connect to a job via WebSocket."""
        job_id = self.job_id_input.text().strip()
        if not job_id:
            self.info_text.append("Please enter a job ID")
            return
        
        self.current_job_id = job_id
        
        # Update WebSocket client URL
        self.websocket_client.server_url = self.server_url_input.text()
        
        # Connect to job (this will run in the worker thread)
        asyncio.run_coroutine_threadsafe(
            self.websocket_client.connect_to_job(job_id),
            self.websocket_worker.loop
        )
    
    def disconnect_from_job(self):
        """Disconnect from current job."""
        if self.current_job_id:
            asyncio.run_coroutine_threadsafe(
                self.websocket_client.disconnect(),
                self.websocket_worker.loop
            )
            self.current_job_id = None
    
    def closeEvent(self, event):
        """Handle widget close event."""
        # Clean up WebSocket connections
        if self.websocket_worker.isRunning():
            self.websocket_worker.stop()
            self.websocket_worker.wait()
        
        event.accept()
