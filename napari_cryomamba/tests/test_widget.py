"""
Tests for CryoMamba napari plugin.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from qtpy.QtWidgets import QApplication
from napari_cryomamba.widget import CryoMambaWidget, InferenceWorker


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
        assert widget.toggle_3d_button.text() == "Switch to 3D"
        
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
        
    def test_2d_3d_toggle(self):
        """Test 2D/3D view toggle functionality."""
        mock_viewer = Mock()
        mock_viewer.dims.ndisplay = 2  # Start in 2D mode
        mock_viewer.layers = [Mock()]  # Mock layer
        mock_layer = mock_viewer.layers[0]
        mock_layer.rendering = 'translucent'
        
        widget = CryoMambaWidget(mock_viewer)
        widget.current_volume = np.random.rand(10, 10, 10)  # Mock volume data
        
        # Test switching from 2D to 3D
        widget.toggle_3d_view()
        assert mock_viewer.dims.ndisplay == 3
        assert mock_layer.rendering == 'mip'
        assert widget.toggle_3d_button.text() == "Switch to 2D"
        
        # Test switching from 3D to 2D
        widget.toggle_3d_view()
        assert mock_viewer.dims.ndisplay == 2
        assert mock_layer.rendering == 'translucent'
        assert widget.toggle_3d_button.text() == "Switch to 3D"
    
    def test_async_inference_workflow(self):
        """Test that inference uses async worker thread instead of blocking."""
        from qtpy.QtWidgets import QApplication
        
        mock_viewer = Mock()
        widget = CryoMambaWidget(mock_viewer)
        
        # Set up test data
        widget.current_volume = np.random.rand(10, 10, 10).astype(np.float32)
        widget.current_volume_path = "/tmp/test.mrc"
        widget.server_url_input.setText("http://localhost:8000")
        
        # Create a mock worker with proper signal support
        mock_worker = MagicMock()
        
        # Mock the signals - MagicMock will handle .connect() calls
        mock_worker.upload_progress.connect = Mock()
        mock_worker.job_created.connect = Mock()
        mock_worker.inference_started.connect = Mock()
        mock_worker.error_occurred.connect = Mock()
        mock_worker.finished.connect = Mock()
        mock_worker.start = Mock()
        
        # Patch InferenceWorker to return our mock
        with patch('napari_cryomamba.widget.InferenceWorker', return_value=mock_worker):
            # Trigger inference
            widget.run_inference()
            
            # Process Qt events to update UI
            QApplication.processEvents()
            
            # Verify worker was assigned - THIS IS THE KEY TEST
            # The worker thread handles the blocking operations
            assert widget.inference_worker is mock_worker
            
            # Verify worker was started in a separate thread
            mock_worker.start.assert_called_once()
            
            # Verify all signals were connected so UI stays responsive
            mock_worker.upload_progress.connect.assert_called_once()
            mock_worker.job_created.connect.assert_called_once()
            mock_worker.inference_started.connect.assert_called_once()
            mock_worker.error_occurred.connect.assert_called_once()
            mock_worker.finished.connect.assert_called_once()
            
            # Verify UI state changes indicating async operation started
            assert widget.is_inference_running
            assert not widget.run_inference_button.isEnabled()
            assert widget.cancel_inference_button.isEnabled()
            # Note: progress_bar.isVisible() may be False in test environment
            # without a real window, but the important thing is the worker thread is used
    
    def test_worker_signal_handlers(self):
        """Test that worker signal handlers update UI correctly."""
        mock_viewer = Mock()
        widget = CryoMambaWidget(mock_viewer)
        widget.websocket_worker.loop = Mock()  # Mock asyncio loop
        
        # Test upload progress handler
        widget.on_worker_upload_progress(50, "Uploading... 50%")
        assert widget.progress_bar.value() == 50
        assert "Uploading" in widget.status_label.text()
        
        # Test job created handler
        with patch('asyncio.run_coroutine_threadsafe'):
            widget.on_worker_job_created("test-job-123")
            assert widget.current_job_id == "test-job-123"
        
        # Test error handler
        widget.on_worker_error("Test error")
        assert not widget.is_inference_running
        assert widget.run_inference_button.isEnabled()
        assert not widget.cancel_inference_button.isEnabled()
        assert not widget.progress_bar.isVisible()


if __name__ == "__main__":
    pytest.main([__file__])