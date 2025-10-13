"""
Main CryoMamba widget for napari integration.
"""

from qtpy.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QTextEdit, QGroupBox
from qtpy.QtCore import Qt, Signal
import napari
import mrcfile
import numpy as np
from pathlib import Path


class CryoMambaWidget(QWidget):
    """Main CryoMamba widget for napari."""
    
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.current_volume = None
        self.setup_ui()
        
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
        
        self.toggle_3d_button = QPushButton("Toggle 3D View")
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
            
            # Toggle 3D rendering
            if hasattr(layer, 'rendering'):
                if layer.rendering == 'mip':
                    layer.rendering = 'iso'
                    self.toggle_3d_button.setText("Switch to 2D")
                else:
                    layer.rendering = 'mip'
                    self.toggle_3d_button.setText("Switch to 3D")
            else:
                # For napari versions without rendering attribute
                layer.rendering = 'mip'
                self.toggle_3d_button.setText("Switch to 3D")
