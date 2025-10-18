#!/usr/bin/env python3
"""
CryoMamba Build Script
Automated build script for creating PyInstaller packages.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import argparse
import json


def run_command(cmd, cwd=None, check=True):
    """Run a command and return the result."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Command failed: {cmd}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        sys.exit(1)
    return result


def check_dependencies():
    """Check if all required dependencies are installed."""
    print("Checking dependencies...")
    
    required_packages = [
        'pyinstaller',
        'napari',
        'mrcfile',
        'numpy',
        'qtpy',
        'aiohttp'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package}")
    
    if missing_packages:
        print(f"\nMissing packages: {missing_packages}")
        print("Please install them with: pip install " + " ".join(missing_packages))
        sys.exit(1)
    
    print("All dependencies found!")


def clean_build():
    """Clean previous build artifacts."""
    print("Cleaning previous build artifacts...")
    
    build_dirs = ['build', 'dist', '__pycache__']
    for dir_name in build_dirs:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"Removed {dir_name}")
    
    # Clean .pyc files
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.pyc'):
                os.remove(os.path.join(root, file))


def build_app():
    """Build the CryoMamba app using PyInstaller."""
    print("Building CryoMamba app...")
    
    # Run PyInstaller
    cmd = "pyinstaller cryomamba.spec --clean --noconfirm"
    result = run_command(cmd)
    
    if result.returncode == 0:
        print("✓ Build completed successfully!")
        return True
    else:
        print("✗ Build failed!")
        return False


def validate_build():
    """Validate the built application."""
    print("Validating build...")
    
    app_path = Path("dist/CryoMamba.app")
    if not app_path.exists():
        print("✗ App bundle not found!")
        return False
    
    executable_path = app_path / "Contents/MacOS/CryoMamba"
    if not executable_path.exists():
        print("✗ Executable not found!")
        return False
    
    print("✓ App bundle structure is valid!")
    return True


def create_dmg():
    """Create a DMG installer for distribution."""
    print("Creating DMG installer...")
    
    app_path = Path("dist/CryoMamba.app")
    dmg_path = Path("dist/CryoMamba.dmg")
    
    if not app_path.exists():
        print("✗ App bundle not found for DMG creation!")
        return False
    
    # Use the installer creation script
    cmd = f"python create_installers.py '{app_path}' --dmg '{dmg_path}'"
    result = run_command(cmd, check=False)
    
    if result.returncode == 0:
        print(f"✓ DMG created: {dmg_path}")
        return True
    else:
        print("✗ DMG creation failed!")
        return False


def create_installers():
    """Create all installer types."""
    print("Creating installers...")
    
    app_path = Path("dist/CryoMamba.app")
    
    if not app_path.exists():
        print("✗ App bundle not found for installer creation!")
        return False
    
    # Use the installer creation script
    cmd = f"python create_installers.py '{app_path}' --all --output-dir dist/installers"
    result = run_command(cmd, check=False)
    
    if result.returncode == 0:
        print("✓ All installers created successfully!")
        return True
    else:
        print("✗ Installer creation failed!")
        return False


def sign_app():
    """Sign the application (if certificates are available)."""
    print("Signing application...")
    
    app_path = Path("dist/CryoMamba.app")
    if not app_path.exists():
        print("✗ App bundle not found for signing!")
        return False
    
    # Check if we have a signing identity
    result = run_command("security find-identity -v -p codesigning", check=False)
    if result.returncode != 0 or "0 valid identities found" in result.stdout:
        print("⚠ No code signing certificates found. Skipping signing.")
        return True
    
    # Sign the app
    cmd = f"codesign --force --deep --sign - '{app_path}'"
    result = run_command(cmd, check=False)
    
    if result.returncode == 0:
        print("✓ Application signed successfully!")
        return True
    else:
        print("⚠ Code signing failed, but continuing...")
        return True


def main():
    """Main build process."""
    parser = argparse.ArgumentParser(description="Build CryoMamba desktop application")
    parser.add_argument("--clean", action="store_true", help="Clean build artifacts")
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency check")
    parser.add_argument("--skip-sign", action="store_true", help="Skip code signing")
    parser.add_argument("--create-dmg", action="store_true", help="Create DMG installer")
    parser.add_argument("--create-installers", action="store_true", help="Create all installer types")
    parser.add_argument("--validate-only", action="store_true", help="Only validate existing build")
    
    args = parser.parse_args()
    
    print("CryoMamba Build Script")
    print("=" * 50)
    
    # Change to the napari_cryomamba directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    if args.validate_only:
        if validate_build():
            print("✓ Build validation passed!")
            sys.exit(0)
        else:
            print("✗ Build validation failed!")
            sys.exit(1)
    
    if args.clean:
        clean_build()
    
    if not args.skip_deps:
        check_dependencies()
    
    if build_app():
        if validate_build():
            if not args.skip_sign:
                sign_app()
            
            if args.create_dmg:
                create_dmg()
            
            if args.create_installers:
                create_installers()
            
            print("\n" + "=" * 50)
            print("✓ Build completed successfully!")
            print(f"App bundle: {Path('dist/CryoMamba.app').absolute()}")
            if args.create_dmg:
                print(f"DMG installer: {Path('dist/CryoMamba.dmg').absolute()}")
        else:
            print("✗ Build validation failed!")
            sys.exit(1)
    else:
        print("✗ Build failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
