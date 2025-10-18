#!/usr/bin/env python3
"""
CryoMamba Code Signing Script
Handle code signing and notarization for macOS distribution.
"""

import os
import sys
import subprocess
import time
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


def check_signing_identity():
    """Check if code signing identity is available."""
    print("Checking code signing identity...")
    
    result = run_command("security find-identity -v -p codesigning", check=False)
    
    if result.returncode != 0:
        print("✗ No code signing identities found")
        return None
    
    lines = result.stdout.strip().split('\n')
    identities = []
    
    for line in lines:
        if 'valid identities found' in line:
            continue
        if line.strip():
            identities.append(line.strip())
    
    if not identities:
        print("✗ No valid code signing identities found")
        return None
    
    print(f"✓ Found {len(identities)} code signing identity(ies)")
    for i, identity in enumerate(identities):
        print(f"  {i+1}. {identity}")
    
    return identities[0] if identities else None


def sign_app(app_path, identity=None):
    """Sign the CryoMamba application."""
    print(f"Signing application: {app_path}")
    
    if not Path(app_path).exists():
        print(f"✗ App bundle not found: {app_path}")
        return False
    
    if not identity:
        identity = check_signing_identity()
        if not identity:
            print("⚠ No signing identity available. Skipping signing.")
            return True
    
    # Extract identity from the security output
    if ")" in identity:
        identity = identity.split(")")[1].strip()
    
    # Sign the app
    cmd = f"codesign --force --deep --sign '{identity}' '{app_path}'"
    result = run_command(cmd, check=False)
    
    if result.returncode == 0:
        print("✓ Application signed successfully!")
        
        # Verify the signature
        verify_cmd = f"codesign --verify --verbose '{app_path}'"
        verify_result = run_command(verify_cmd, check=False)
        
        if verify_result.returncode == 0:
            print("✓ Signature verification passed!")
            return True
        else:
            print("⚠ Signature verification failed!")
            return False
    else:
        print("✗ Code signing failed!")
        print(f"Error: {result.stderr}")
        return False


def create_entitlements():
    """Create entitlements file for the application."""
    print("Creating entitlements file...")
    
    entitlements_content = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>com.apple.security.cs.allow-jit</key>
    <true/>
    <key>com.apple.security.cs.allow-unsigned-executable-memory</key>
    <true/>
    <key>com.apple.security.cs.allow-dyld-environment-variables</key>
    <true/>
    <key>com.apple.security.cs.disable-library-validation</key>
    <true/>
    <key>com.apple.security.network.client</key>
    <true/>
    <key>com.apple.security.network.server</key>
    <true/>
</dict>
</plist>"""
    
    entitlements_path = Path("entitlements.plist")
    with open(entitlements_path, 'w') as f:
        f.write(entitlements_content)
    
    print(f"✓ Entitlements file created: {entitlements_path}")
    return str(entitlements_path)


def notarize_app(app_path, apple_id=None, password=None, team_id=None):
    """Notarize the application for macOS distribution."""
    print(f"Notarizing application: {app_path}")
    
    if not Path(app_path).exists():
        print(f"✗ App bundle not found: {app_path}")
        return False
    
    # Check if we have the required credentials
    if not apple_id or not password:
        print("⚠ Apple ID credentials not provided. Skipping notarization.")
        print("To enable notarization, set APPLE_ID and APPLE_PASSWORD environment variables.")
        return True
    
    # Create zip file for notarization
    zip_path = Path(app_path).with_suffix('.zip')
    cmd = f"ditto -c -k --keepParent '{app_path}' '{zip_path}'"
    run_command(cmd)
    
    print(f"✓ Created zip file: {zip_path}")
    
    # Submit for notarization
    notarize_cmd = f"xcrun notarytool submit '{zip_path}' --apple-id '{apple_id}' --password '{password}'"
    if team_id:
        notarize_cmd += f" --team-id '{team_id}'"
    notarize_cmd += " --wait"
    
    result = run_command(notarize_cmd, check=False)
    
    if result.returncode == 0:
        print("✓ Notarization completed successfully!")
        
        # Staple the notarization
        staple_cmd = f"xcrun stapler staple '{app_path}'"
        staple_result = run_command(staple_cmd, check=False)
        
        if staple_result.returncode == 0:
            print("✓ Notarization stapled successfully!")
            return True
        else:
            print("⚠ Stapling failed, but notarization was successful")
            return True
    else:
        print("✗ Notarization failed!")
        print(f"Error: {result.stderr}")
        return False


def main():
    """Main signing and notarization process."""
    parser = argparse.ArgumentParser(description="Sign and notarize CryoMamba app")
    parser.add_argument("--app-path", default="dist/CryoMamba.app", help="Path to the app bundle")
    parser.add_argument("--identity", help="Code signing identity")
    parser.add_argument("--apple-id", help="Apple ID for notarization")
    parser.add_argument("--password", help="App-specific password for notarization")
    parser.add_argument("--team-id", help="Apple Developer Team ID")
    parser.add_argument("--skip-sign", action="store_true", help="Skip code signing")
    parser.add_argument("--skip-notarize", action="store_true", help="Skip notarization")
    
    args = parser.parse_args()
    
    print("CryoMamba Code Signing & Notarization")
    print("=" * 50)
    
    app_path = Path(args.app_path)
    if not app_path.exists():
        print(f"✗ App bundle not found: {app_path}")
        print("Please build the application first using: python build.py")
        sys.exit(1)
    
    success = True
    
    # Code signing
    if not args.skip_sign:
        if not sign_app(str(app_path), args.identity):
            success = False
    
    # Notarization
    if not args.skip_notarize and success:
        apple_id = args.apple_id or os.getenv('APPLE_ID')
        password = args.password or os.getenv('APPLE_PASSWORD')
        team_id = args.team_id or os.getenv('TEAM_ID')
        
        if not notarize_app(str(app_path), apple_id, password, team_id):
            success = False
    
    if success:
        print("\n" + "=" * 50)
        print("✓ Code signing and notarization completed successfully!")
        print(f"Signed app: {app_path.absolute()}")
    else:
        print("\n" + "=" * 50)
        print("✗ Code signing or notarization failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
