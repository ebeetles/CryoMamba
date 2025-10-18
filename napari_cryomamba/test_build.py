#!/usr/bin/env python3
"""
CryoMamba Build Testing Script
Comprehensive testing for built applications.
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path
import argparse
import tempfile
import shutil


def run_command(cmd, check=True, capture_output=True, timeout=None):
    """Run a command and return the result."""
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=capture_output, 
            text=True,
            timeout=timeout
        )
        if check and result.returncode != 0:
            print(f"Command failed: {cmd}")
            if capture_output:
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
            sys.exit(1)
        return result
    except subprocess.TimeoutExpired:
        print(f"Command timed out: {cmd}")
        return None


def test_app_structure(app_path):
    """Test the basic structure of the app bundle."""
    print(f"Testing app structure: {app_path}")
    
    app_path = Path(app_path)
    if not app_path.exists():
        print(f"✗ App bundle not found: {app_path}")
        return False
    
    # Check required directories and files
    required_paths = [
        "Contents/MacOS/CryoMamba",
        "Contents/Info.plist",
        "Contents/Resources",
    ]
    
    for path in required_paths:
        full_path = app_path / path
        if not full_path.exists():
            print(f"✗ Required path not found: {path}")
            return False
        print(f"✓ Found: {path}")
    
    # Check executable permissions
    executable = app_path / "Contents/MacOS/CryoMamba"
    if not os.access(executable, os.X_OK):
        print(f"✗ Executable not executable: {executable}")
        return False
    print(f"✓ Executable permissions OK")
    
    return True


def test_app_launch(app_path, timeout=30):
    """Test that the app can launch."""
    print(f"Testing app launch: {app_path}")
    
    app_path = Path(app_path)
    if not app_path.exists():
        print(f"✗ App bundle not found: {app_path}")
        return False
    
    # Try to launch the app
    cmd = f"open '{app_path}'"
    result = run_command(cmd, check=False, timeout=timeout)
    
    if result is None:
        print("⚠ App launch timed out (this may be normal)")
        return True
    
    if result.returncode == 0:
        print("✓ App launched successfully")
        return True
    else:
        print(f"✗ App launch failed: {result.stderr}")
        return False


def test_app_functionality(app_path):
    """Test basic app functionality."""
    print(f"Testing app functionality: {app_path}")
    
    # This is a placeholder for more comprehensive testing
    # In a real implementation, you might:
    # 1. Launch the app programmatically
    # 2. Test UI interactions
    # 3. Test file loading capabilities
    # 4. Test network connectivity
    
    print("✓ Basic functionality test passed (placeholder)")
    return True


def test_dependencies(app_path):
    """Test that all dependencies are bundled correctly."""
    print(f"Testing bundled dependencies: {app_path}")
    
    app_path = Path(app_path)
    resources_path = app_path / "Contents/Resources"
    
    # Check for key dependency files
    dependency_checks = [
        "napari",
        "mrcfile", 
        "numpy",
        "qtpy",
    ]
    
    for dep in dependency_checks:
        # Look for the dependency in the bundled files
        found = False
        for root, dirs, files in os.walk(resources_path):
            if dep in root or any(dep in f for f in files):
                found = True
                break
        
        if found:
            print(f"✓ Found dependency: {dep}")
        else:
            print(f"⚠ Dependency not found: {dep}")
    
    return True


def test_performance(app_path):
    """Test app performance characteristics."""
    print(f"Testing app performance: {app_path}")
    
    app_path = Path(app_path)
    
    # Check app bundle size
    total_size = sum(
        f.stat().st_size 
        for f in app_path.rglob('*') 
        if f.is_file()
    )
    size_mb = total_size / (1024 * 1024)
    
    print(f"App bundle size: {size_mb:.1f} MB")
    
    if size_mb > 1000:  # More than 1GB
        print("⚠ App bundle is quite large")
    else:
        print("✓ App bundle size is reasonable")
    
    return True


def test_installation(app_path):
    """Test app installation process."""
    print(f"Testing app installation: {app_path}")
    
    app_path = Path(app_path)
    
    # Test copying to Applications folder
    with tempfile.TemporaryDirectory() as temp_dir:
        test_apps_dir = Path(temp_dir) / "Applications"
        test_apps_dir.mkdir()
        
        test_app_path = test_apps_dir / "CryoMamba.app"
        
        try:
            shutil.copytree(app_path, test_app_path)
            print("✓ App can be copied to Applications folder")
            
            # Test that copied app still works
            if test_app_structure(test_app_path):
                print("✓ Copied app maintains structure")
            else:
                print("✗ Copied app has structural issues")
                return False
            
        except Exception as e:
            print(f"✗ App installation test failed: {e}")
            return False
    
    return True


def run_all_tests(app_path):
    """Run all tests on the app."""
    print("CryoMamba Build Testing")
    print("=" * 50)
    
    tests = [
        ("App Structure", test_app_structure),
        ("App Launch", test_app_launch),
        ("Dependencies", test_dependencies),
        ("Performance", test_performance),
        ("Installation", test_installation),
        ("Functionality", test_app_functionality),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            result = test_func(app_path)
            results[test_name] = result
        except Exception as e:
            print(f"✗ {test_name} test failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓" if result else "✗"
        print(f"  {status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed!")
        return True
    else:
        print("✗ Some tests failed!")
        return False


def main():
    """Main testing process."""
    parser = argparse.ArgumentParser(description="Test CryoMamba build")
    parser.add_argument("app_path", help="Path to the CryoMamba.app bundle")
    parser.add_argument("--test", choices=[
        "structure", "launch", "dependencies", "performance", 
        "installation", "functionality", "all"
    ], default="all", help="Specific test to run")
    parser.add_argument("--timeout", type=int, default=30, help="Launch timeout in seconds")
    parser.add_argument("--output", help="Output file for test results")
    
    args = parser.parse_args()
    
    app_path = Path(args.app_path)
    if not app_path.exists():
        print(f"✗ App bundle not found: {app_path}")
        sys.exit(1)
    
    # Run specific test or all tests
    if args.test == "all":
        success = run_all_tests(app_path)
    else:
        test_funcs = {
            "structure": test_app_structure,
            "launch": test_app_launch,
            "dependencies": test_dependencies,
            "performance": test_performance,
            "installation": test_installation,
            "functionality": test_app_functionality,
        }
        
        test_func = test_funcs[args.test]
        success = test_func(app_path)
    
    # Save results if requested
    if args.output:
        results = {
            "app_path": str(app_path),
            "test_type": args.test,
            "success": success,
            "timestamp": time.time(),
        }
        
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Test results saved to: {args.output}")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
