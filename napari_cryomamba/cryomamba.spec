# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from pathlib import Path

# Get the directory containing this spec file
spec_dir = Path(__file__).parent.absolute()

# Define paths
napari_cryomamba_dir = spec_dir / "napari_cryomamba"
main_script = spec_dir / "main.py"

# Collect all data files and dependencies
datas = []
binaries = []
hiddenimports = []

# Add napari data files
try:
    import napari
    napari_path = Path(napari.__file__).parent
    datas.append((str(napari_path / "resources"), "napari/resources"))
except ImportError:
    pass

# Add qtpy data files
try:
    import qtpy
    qtpy_path = Path(qtpy.__file__).parent
    datas.append((str(qtpy_path), "qtpy"))
except ImportError:
    pass

# Add mrcfile data files
try:
    import mrcfile
    mrcfile_path = Path(mrcfile.__file__).parent
    datas.append((str(mrcfile_path), "mrcfile"))
except ImportError:
    pass

# Hidden imports for napari and Qt
hiddenimports.extend([
    'napari',
    'napari.viewer',
    'napari.qt',
    'qtpy',
    'qtpy.QtCore',
    'qtpy.QtWidgets',
    'qtpy.QtGui',
    'mrcfile',
    'numpy',
    'aiohttp',
    'asyncio',
    'json',
    'requests',
    'pathlib',
    'time',
    'os',
    'sys',
])

# macOS specific configuration
block_cipher = None

a = Analysis(
    [str(main_script)],
    pathex=[str(spec_dir)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'matplotlib',
        'IPython',
        'jupyter',
        'notebook',
        'pandas',
        'scipy.sparse.csgraph',
        'scipy.spatial.distance',
        'scipy.ndimage.filters',
        'scipy.ndimage.interpolation',
        'scipy.ndimage.measurements',
        'scipy.ndimage.morphology',
        'scipy.ndimage.segmentation',
        'scipy.ndimage.watershed',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='CryoMamba',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='CryoMamba',
)

# Create macOS app bundle
app = BUNDLE(
    coll,
    name='CryoMamba.app',
    icon=None,  # Add icon path if available
    bundle_identifier='com.cryomamba.desktop',
    info_plist=str(spec_dir / 'Info.plist'),
)
