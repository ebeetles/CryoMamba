#!/usr/bin/env python3
"""
CryoMamba Desktop Application - PyInstaller Entry Point
Main entry point for the PyInstaller-built desktop application.
"""

import sys
import os
from pathlib import Path

# Add the napari_cryomamba package to the path
if getattr(sys, 'frozen', False):
    # Running as PyInstaller bundle
    bundle_dir = Path(sys._MEIPASS)
    napari_cryomamba_path = bundle_dir / "napari_cryomamba"
else:
    # Running as script
    napari_cryomamba_path = Path(__file__).parent / "napari_cryomamba"

sys.path.insert(0, str(napari_cryomamba_path))

try:
    import napari
    from napari_cryomamba.widget import CryoMambaWidget
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure all dependencies are installed.")
    sys.exit(1)


def main():
    """Main entry point for the CryoMamba desktop application."""
    try:
        # Create napari viewer
        viewer = napari.Viewer(title="CryoMamba")
        
        # Add CryoMamba widget
        widget = CryoMambaWidget(viewer)
        viewer.window.add_dock_widget(widget, name="CryoMamba")
        
        # Run the application
        napari.run()
        
    except Exception as e:
        print(f"Error starting CryoMamba: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()