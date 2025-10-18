#!/usr/bin/env python3
"""
CryoMamba Installer Creator
Create DMG and PKG installers for macOS distribution.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import argparse
import tempfile
import json


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


def create_dmg_installer(app_path, output_path=None, volume_name="CryoMamba"):
    """Create a DMG installer."""
    print(f"Creating DMG installer for: {app_path}")
    
    app_path = Path(app_path)
    if not app_path.exists():
        print(f"✗ App bundle not found: {app_path}")
        return False
    
    if output_path is None:
        output_path = app_path.parent / f"{volume_name}.dmg"
    else:
        output_path = Path(output_path)
    
    # Create temporary directory for DMG contents
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Copy app to temp directory
        temp_app_path = temp_path / app_path.name
        shutil.copytree(app_path, temp_app_path)
        
        # Create Applications symlink
        apps_link = temp_path / "Applications"
        os.symlink("/Applications", apps_link)
        
        # Create README file
        readme_path = temp_path / "README.txt"
        readme_content = """CryoMamba Desktop Application

Installation Instructions:
1. Drag CryoMamba.app to the Applications folder
2. Launch CryoMamba from Applications or Spotlight

System Requirements:
- macOS 10.15 or later
- 8GB RAM minimum
- 2GB free disk space

For support, visit: https://github.com/cryomamba/napari-cryomamba

© 2024 CryoMamba Team
"""
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        # Create DMG using hdiutil
        cmd = f"hdiutil create -volname '{volume_name}' -srcfolder '{temp_path}' -ov -format UDZO '{output_path}'"
        result = run_command(cmd, check=False)
        
        if result.returncode == 0:
            print(f"✓ DMG created: {output_path}")
            return True
        else:
            print(f"✗ DMG creation failed: {result.stderr}")
            return False


def create_pkg_installer(app_path, output_path=None, package_name="CryoMamba"):
    """Create a PKG installer."""
    print(f"Creating PKG installer for: {app_path}")
    
    app_path = Path(app_path)
    if not app_path.exists():
        print(f"✗ App bundle not found: {app_path}")
        return False
    
    if output_path is None:
        output_path = app_path.parent / f"{package_name}.pkg"
    else:
        output_path = Path(output_path)
    
    # Create temporary directory for package contents
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create package structure
        apps_dir = temp_path / "Applications"
        apps_dir.mkdir()
        
        # Copy app to Applications folder
        temp_app_path = apps_dir / app_path.name
        shutil.copytree(app_path, temp_app_path)
        
        # Create package info
        package_info = {
            "identifier": "com.cryomamba.desktop",
            "version": "0.1.0",
            "install_location": "/Applications",
            "description": "CryoMamba Desktop Application",
            "minimum_os_version": "10.15"
        }
        
        info_path = temp_path / "package_info.json"
        with open(info_path, 'w') as f:
            json.dump(package_info, f, indent=2)
        
        # Create PKG using pkgbuild
        cmd = f"pkgbuild --root '{temp_path}' --identifier com.cryomamba.desktop --version 0.1.0 --install-location / '{output_path}'"
        result = run_command(cmd, check=False)
        
        if result.returncode == 0:
            print(f"✓ PKG created: {output_path}")
            return True
        else:
            print(f"✗ PKG creation failed: {result.stderr}")
            return False


def create_installer_script(app_path, output_path=None):
    """Create a shell script installer."""
    print(f"Creating installer script for: {app_path}")
    
    app_path = Path(app_path)
    if not app_path.exists():
        print(f"✗ App bundle not found: {app_path}")
        return False
    
    if output_path is None:
        output_path = app_path.parent / "install_cryomamba.sh"
    else:
        output_path = Path(output_path)
    
    script_content = f"""#!/bin/bash
# CryoMamba Installer Script

set -e

APP_NAME="CryoMamba.app"
INSTALL_DIR="/Applications"
SOURCE_DIR="$(dirname "$0")"

echo "CryoMamba Desktop Application Installer"
echo "========================================"

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo "Please do not run this installer as root."
    echo "The installer will prompt for your password when needed."
    exit 1
fi

# Check if app exists
if [ ! -d "$SOURCE_DIR/$APP_NAME" ]; then
    echo "Error: CryoMamba.app not found in $SOURCE_DIR"
    exit 1
fi

# Check if app is already installed
if [ -d "$INSTALL_DIR/$APP_NAME" ]; then
    echo "CryoMamba is already installed in $INSTALL_DIR"
    read -p "Do you want to replace it? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Installation cancelled."
        exit 0
    fi
    
    echo "Removing existing installation..."
    sudo rm -rf "$INSTALL_DIR/$APP_NAME"
fi

# Install the app
echo "Installing CryoMamba to $INSTALL_DIR..."
sudo cp -R "$SOURCE_DIR/$APP_NAME" "$INSTALL_DIR/"

# Set proper permissions
sudo chown -R root:admin "$INSTALL_DIR/$APP_NAME"
sudo chmod -R 755 "$INSTALL_DIR/$APP_NAME"

echo "✓ CryoMamba installed successfully!"
echo ""
echo "You can now launch CryoMamba from:"
echo "  - Applications folder"
echo "  - Spotlight (Cmd+Space, then type 'CryoMamba')"
echo "  - Launchpad"
echo ""
echo "For support, visit: https://github.com/cryomamba/napari-cryomamba"
"""
    
    with open(output_path, 'w') as f:
        f.write(script_content)
    
    # Make script executable
    os.chmod(output_path, 0o755)
    
    print(f"✓ Installer script created: {output_path}")
    return True


def create_uninstaller(app_path, output_path=None):
    """Create an uninstaller script."""
    print(f"Creating uninstaller for: {app_path}")
    
    app_path = Path(app_path)
    if not app_path.exists():
        print(f"✗ App bundle not found: {app_path}")
        return False
    
    if output_path is None:
        output_path = app_path.parent / "uninstall_cryomamba.sh"
    else:
        output_path = Path(output_path)
    
    script_content = """#!/bin/bash
# CryoMamba Uninstaller Script

set -e

APP_NAME="CryoMamba.app"
INSTALL_DIR="/Applications"

echo "CryoMamba Desktop Application Uninstaller"
echo "=========================================="

# Check if app is installed
if [ ! -d "$INSTALL_DIR/$APP_NAME" ]; then
    echo "CryoMamba is not installed in $INSTALL_DIR"
    exit 0
fi

# Confirm uninstallation
echo "CryoMamba is installed in $INSTALL_DIR"
read -p "Are you sure you want to uninstall CryoMamba? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Uninstallation cancelled."
    exit 0
fi

# Remove the app
echo "Removing CryoMamba..."
sudo rm -rf "$INSTALL_DIR/$APP_NAME"

echo "✓ CryoMamba uninstalled successfully!"
echo ""
echo "Note: This uninstaller only removes the application."
echo "It does not remove user preferences or data files."
echo "To completely remove CryoMamba, you may also want to delete:"
echo "  ~/Library/Preferences/com.cryomamba.desktop.plist"
echo "  ~/Library/Application Support/CryoMamba/"
"""
    
    with open(output_path, 'w') as f:
        f.write(script_content)
    
    # Make script executable
    os.chmod(output_path, 0o755)
    
    print(f"✓ Uninstaller created: {output_path}")
    return True


def create_distribution_package(app_path, output_dir=None):
    """Create a complete distribution package."""
    print(f"Creating distribution package for: {app_path}")
    
    app_path = Path(app_path)
    if not app_path.exists():
        print(f"✗ App bundle not found: {app_path}")
        return False
    
    if output_dir is None:
        output_dir = app_path.parent / "distribution"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True)
    
    # Create all installer types
    success = True
    
    # DMG installer
    dmg_path = output_dir / "CryoMamba.dmg"
    if not create_dmg_installer(app_path, dmg_path):
        success = False
    
    # PKG installer
    pkg_path = output_dir / "CryoMamba.pkg"
    if not create_pkg_installer(app_path, pkg_path):
        success = False
    
    # Shell script installer
    script_path = output_dir / "install_cryomamba.sh"
    if not create_installer_script(app_path, script_path):
        success = False
    
    # Uninstaller
    uninstaller_path = output_dir / "uninstall_cryomamba.sh"
    if not create_uninstaller(app_path, uninstaller_path):
        success = False
    
    # Create README for distribution
    readme_path = output_dir / "README.txt"
    readme_content = """CryoMamba Desktop Application Distribution

This package contains multiple installation options for CryoMamba:

1. CryoMamba.dmg - Drag-and-drop installer (recommended)
   - Double-click to mount
   - Drag CryoMamba.app to Applications folder
   - Eject the DMG when done

2. CryoMamba.pkg - Standard macOS package installer
   - Double-click to run the installer
   - Follow the installation wizard

3. install_cryomamba.sh - Command-line installer
   - Open Terminal
   - Run: ./install_cryomamba.sh
   - Enter your password when prompted

4. uninstall_cryomamba.sh - Uninstaller script
   - Run: ./uninstall_cryomamba.sh
   - Follow the prompts

System Requirements:
- macOS 10.15 or later
- 8GB RAM minimum
- 2GB free disk space

For support and updates, visit:
https://github.com/cryomamba/napari-cryomamba

© 2024 CryoMamba Team
"""
    
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"✓ Distribution package created in: {output_dir}")
    print("Contents:")
    for item in output_dir.iterdir():
        print(f"  - {item.name}")
    
    return success


def main():
    """Main installer creation process."""
    parser = argparse.ArgumentParser(description="Create CryoMamba installers")
    parser.add_argument("app_path", help="Path to the CryoMamba.app bundle")
    parser.add_argument("--dmg", help="Create DMG installer")
    parser.add_argument("--pkg", help="Create PKG installer")
    parser.add_argument("--script", help="Create shell script installer")
    parser.add_argument("--uninstaller", help="Create uninstaller script")
    parser.add_argument("--all", action="store_true", help="Create all installer types")
    parser.add_argument("--output-dir", help="Output directory for distribution package")
    
    args = parser.parse_args()
    
    app_path = Path(args.app_path)
    if not app_path.exists():
        print(f"✗ App bundle not found: {app_path}")
        sys.exit(1)
    
    print("CryoMamba Installer Creator")
    print("=" * 50)
    
    success = True
    
    if args.all:
        # Create complete distribution package
        if not create_distribution_package(app_path, args.output_dir):
            success = False
    else:
        # Create specific installer types
        if args.dmg:
            if not create_dmg_installer(app_path, args.dmg):
                success = False
        
        if args.pkg:
            if not create_pkg_installer(app_path, args.pkg):
                success = False
        
        if args.script:
            if not create_installer_script(app_path, args.script):
                success = False
        
        if args.uninstaller:
            if not create_uninstaller(app_path, args.uninstaller):
                success = False
        
        if not any([args.dmg, args.pkg, args.script, args.uninstaller]):
            print("No installer type specified. Use --help for options.")
            sys.exit(1)
    
    if success:
        print("\n✓ Installer creation completed successfully!")
        sys.exit(0)
    else:
        print("\n✗ Some installer creation steps failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
