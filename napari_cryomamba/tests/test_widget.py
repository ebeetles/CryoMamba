"""
Tests for CryoMamba napari plugin.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from qtpy.QtWidgets import QApplication
from napari_cryomamba.widget import CryoMambaWidget


class TestCryoMambaWidget:
    """Test cases for CryoMambaWidget."""
    
    @pytest.fixture(autouse=True)
    def setup_qt_app(self):
        """Set up Qt application for testing."""
        if not QApplication.instance():
            self.app = QApplication([])
        else:
            self.app = QApplication.instance()
        yield
        # Cleanup handled by pytest-qt
    
    def test_widget_initialization(self):
        """Test widget initializes correctly."""
        mock_viewer = Mock()
        widget = CryoMambaWidget(mock_viewer)
        
        assert widget.viewer == mock_viewer
        assert widget.current_volume is None
        assert widget.open_button is not None
        assert widget.toggle_3d_button is not None
        assert not widget.toggle_3d_button.isEnabled()
        
    def test_metadata_extraction(self):
        """Test volume metadata extraction."""
        mock_viewer = Mock()
        widget = CryoMambaWidget(mock_viewer)
        
        # Create test data
        test_data = np.random.rand(10, 10, 10).astype(np.float32)
        test_path = "/test/path/test.mrc"
        
        # Mock mrcfile.open with context manager support
        mock_mrc = Mock()
        mock_mrc.header.cella.x = 100.0
        mock_mrc.header.cella.y = 100.0
        mock_mrc.header.cella.z = 100.0
        mock_mrc.header.nx = 10
        mock_mrc.header.ny = 10
        mock_mrc.header.nz = 10
        mock_mrc.__enter__ = Mock(return_value=mock_mrc)
        mock_mrc.__exit__ = Mock(return_value=None)
        
        with patch('mrcfile.open', return_value=mock_mrc):
            metadata = widget.get_volume_metadata(test_path, test_data)
            
        assert metadata['filename'] == 'test.mrc'
        assert metadata['shape'] == (10, 10, 10)
        assert metadata['dtype'] == 'float32'
        assert metadata['voxel_size'] == (10.0, 10.0, 10.0)
        assert 'min_intensity' in metadata
        assert 'max_intensity' in metadata
        assert 'mean_intensity' in metadata
        assert 'std_intensity' in metadata
        
    def test_metadata_display(self):
        """Test metadata display formatting."""
        mock_viewer = Mock()
        widget = CryoMambaWidget(mock_viewer)
        
        metadata = {
            'filename': 'test.mrc',
            'shape': (10, 10, 10),
            'dtype': 'float32',
            'voxel_size': (1.0, 1.0, 1.0),
            'min_intensity': 0.0,
            'max_intensity': 1.0,
            'mean_intensity': 0.5,
            'std_intensity': 0.1
        }
        
        widget.display_metadata(metadata)
        
        info_text = widget.info_text.toPlainText()
        assert 'test.mrc' in info_text
        assert '(10, 10, 10)' in info_text
        assert 'float32' in info_text
        assert '(1.0, 1.0, 1.0)' in info_text


if __name__ == "__main__":
    pytest.main([__file__])