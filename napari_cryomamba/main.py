#!/usr/bin/env python3
"""
CryoMamba Desktop Application
Main entry point for the napari-based desktop application.
"""

import sys
import napari
from napari_cryomamba.widget import CryoMambaWidget


def main():
    """Main entry point for the CryoMamba desktop application."""
    # Create napari viewer
    viewer = napari.Viewer(title="CryoMamba")
    
    # Add CryoMamba widget
    widget = CryoMambaWidget(viewer)
    viewer.window.add_dock_widget(widget, name="CryoMamba")
    
    # Run the application
    napari.run()


if __name__ == "__main__":
    main()