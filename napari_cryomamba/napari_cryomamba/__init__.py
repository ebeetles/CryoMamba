"""
CryoMamba napari plugin for cryo-EM volume visualization and analysis.
"""

from napari_plugin_engine import napari_hook_implementation
from .widget import CryoMambaWidget


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    """Provide the CryoMamba dock widget."""
    return CryoMambaWidget


__version__ = "0.1.0"
