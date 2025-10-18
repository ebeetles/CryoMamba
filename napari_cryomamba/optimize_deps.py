#!/usr/bin/env python3
"""
CryoMamba Dependency Optimizer
Optimize dependencies for PyInstaller builds to reduce size and improve performance.
"""

import os
import sys
import subprocess
import json
from pathlib import Path
import argparse


def run_command(cmd, check=True, capture_output=True):
    """Run a command and return the result."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=capture_output, text=True)
    if check and result.returncode != 0:
        print(f"Command failed: {cmd}")
        if capture_output:
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
        sys.exit(1)
    return result


def analyze_dependencies():
    """Analyze current dependencies and their sizes."""
    print("Analyzing dependencies...")
    
    # Get pip list with sizes
    result = run_command("pip list --format=json")
    packages = json.loads(result.stdout)
    
    print(f"Found {len(packages)} installed packages")
    
    # Calculate approximate sizes (this is a rough estimate)
    total_size = 0
    large_packages = []
    
    for package in packages:
        name = package['name']
        version = package['version']
        
        # Try to get package size
        try:
            import pkg_resources
            dist = pkg_resources.get_distribution(name)
            size = sum(os.path.getsize(os.path.join(dirpath, filename))
                      for dirpath, dirnames, filenames in os.walk(dist.location)
                      for filename in filenames)
            
            size_mb = size / (1024 * 1024)
            total_size += size_mb
            
            if size_mb > 10:  # Packages larger than 10MB
                large_packages.append((name, version, size_mb))
                
        except Exception:
            pass
    
    print(f"Total estimated size: {total_size:.1f} MB")
    print(f"Large packages (>10MB):")
    for name, version, size in sorted(large_packages, key=lambda x: x[2], reverse=True):
        print(f"  {name} {version}: {size:.1f} MB")
    
    return packages, large_packages


def create_optimized_spec():
    """Create an optimized PyInstaller spec file with better exclusions."""
    print("Creating optimized PyInstaller spec...")
    
    spec_content = '''# -*- mode: python ; coding: utf-8 -*-

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

# Essential hidden imports
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
        # GUI frameworks we don't use
        'tkinter',
        'matplotlib',
        'IPython',
        'jupyter',
        'notebook',
        'pandas',
        
        # Large scipy modules we don't need
        'scipy.sparse.csgraph',
        'scipy.spatial.distance',
        'scipy.ndimage.filters',
        'scipy.ndimage.interpolation',
        'scipy.ndimage.measurements',
        'scipy.ndimage.morphology',
        'scipy.ndimage.segmentation',
        'scipy.ndimage.watershed',
        'scipy.sparse.linalg',
        'scipy.sparse.linalg.dsolve',
        'scipy.sparse.linalg.eigen',
        'scipy.sparse.linalg.isolve',
        'scipy.sparse.linalg.dsolve.linsolve',
        'scipy.sparse.linalg.eigen.arpack',
        'scipy.sparse.linalg.eigen.lobpcg',
        'scipy.sparse.linalg.isolve.iterative',
        'scipy.sparse.linalg.isolve.minres',
        'scipy.sparse.linalg.isolve.lsqr',
        'scipy.sparse.linalg.isolve.lsmr',
        'scipy.sparse.linalg.isolve.lgmres',
        'scipy.sparse.linalg.isolve.bicg',
        'scipy.sparse.linalg.isolve.bicgstab',
        'scipy.sparse.linalg.isolve.cg',
        'scipy.sparse.linalg.isolve.cgs',
        'scipy.sparse.linalg.isolve.gmres',
        'scipy.sparse.linalg.isolve.qmr',
        'scipy.sparse.linalg.isolve.tfqmr',
        
        # Development tools
        'pytest',
        'pytest_qt',
        'black',
        'flake8',
        'isort',
        'mypy',
        
        # Documentation tools
        'sphinx',
        'docutils',
        'jinja2',
        
        # Testing frameworks
        'unittest',
        'doctest',
        
        # Large optional dependencies
        'sklearn',
        'tensorflow',
        'torch',
        'keras',
        'theano',
        'caffe',
        'mxnet',
        
        # Image processing we don't use
        'PIL.ImageTk',
        'PIL.ImageQt',
        'PIL.ImageMath',
        'PIL.ImageOps',
        'PIL.ImageSequence',
        'PIL.ImageStat',
        'PIL.ImageTransform',
        'PIL.ImageWin',
        
        # Network libraries we don't use
        'urllib3',
        'certifi',
        'charset_normalizer',
        'idna',
        
        # Compression libraries we don't use
        'bz2',
        'lzma',
        'zlib',
        
        # Database libraries we don't use
        'sqlite3',
        'pymongo',
        'psycopg2',
        'mysql',
        
        # Web frameworks we don't use
        'flask',
        'django',
        'fastapi',
        'tornado',
        'bottle',
        
        # Scientific computing we don't use
        'sympy',
        'statsmodels',
        'networkx',
        'bokeh',
        'plotly',
        'seaborn',
        
        # Machine learning we don't use
        'xgboost',
        'lightgbm',
        'catboost',
        'optuna',
        
        # Cloud services we don't use
        'boto3',
        'azure',
        'google',
        
        # Other large packages
        'cv2',
        'opencv',
        'ffmpeg',
        'gstreamer',
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
'''
    
    with open('cryomamba_optimized.spec', 'w') as f:
        f.write(spec_content)
    
    print("✓ Optimized spec file created: cryomamba_optimized.spec")


def create_minimal_requirements():
    """Create minimal requirements file with only essential dependencies."""
    print("Creating minimal requirements file...")
    
    minimal_requirements = """# CryoMamba Minimal Requirements
# Only essential dependencies for PyInstaller build

# Core application
napari>=0.4.0
mrcfile>=1.4.0
numpy>=1.20.0
qtpy>=2.0.0
aiohttp>=3.8.0

# Build tools
pyinstaller>=5.0.0
setuptools>=45.0.0
wheel>=0.36.0

# macOS specific
pyobjc-framework-Cocoa>=8.0.0
pyobjc-framework-Quartz>=8.0.0
pyobjc-framework-AppKit>=8.0.0
"""
    
    with open('requirements_minimal.txt', 'w') as f:
        f.write(minimal_requirements)
    
    print("✓ Minimal requirements file created: requirements_minimal.txt")


def optimize_build():
    """Run build optimization process."""
    print("Running build optimization...")
    
    # Create optimized spec
    create_optimized_spec()
    
    # Create minimal requirements
    create_minimal_requirements()
    
    print("\nOptimization complete!")
    print("Files created:")
    print("  - cryomamba_optimized.spec (optimized PyInstaller spec)")
    print("  - requirements_minimal.txt (minimal dependencies)")
    print("\nTo use the optimized build:")
    print("  pyinstaller cryomamba_optimized.spec --clean --noconfirm")


def main():
    """Main optimization process."""
    parser = argparse.ArgumentParser(description="Optimize CryoMamba dependencies")
    parser.add_argument("--analyze", action="store_true", help="Analyze current dependencies")
    parser.add_argument("--optimize", action="store_true", help="Create optimized build files")
    parser.add_argument("--all", action="store_true", help="Run all optimization steps")
    
    args = parser.parse_args()
    
    print("CryoMamba Dependency Optimizer")
    print("=" * 50)
    
    if args.analyze or args.all:
        analyze_dependencies()
    
    if args.optimize or args.all:
        optimize_build()
    
    if not any([args.analyze, args.optimize, args.all]):
        print("No action specified. Use --help for options.")


if __name__ == "__main__":
    main()
