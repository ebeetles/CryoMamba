#!/usr/bin/env python3
"""
Test runner for CryoMamba end-to-end integration tests.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_tests():
    """Run the end-to-end integration tests."""
    print("Running CryoMamba End-to-End Integration Tests")
    print("=" * 50)
    
    # Change to project root
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Run pytest with the integration tests
    test_file = "tests/test_e2e_integration.py"
    
    cmd = [
        sys.executable, "-m", "pytest",
        test_file,
        "-v",
        "--tb=short",
        "--color=yes"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n" + "=" * 50)
        print("✅ All tests passed!")
        return True
    except subprocess.CalledProcessError as e:
        print("\n" + "=" * 50)
        print("❌ Some tests failed!")
        print(f"Exit code: {e.returncode}")
        return False
    except FileNotFoundError:
        print("❌ pytest not found. Please install pytest:")
        print("pip install pytest")
        return False


def run_server_tests_only():
    """Run only server-side tests (no WebSocket tests)."""
    print("Running Server-Only Tests")
    print("=" * 30)
    
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/test_e2e_integration.py::TestEndToEndIntegration::test_server_health_endpoints",
        "tests/test_e2e_integration.py::TestEndToEndIntegration::test_job_lifecycle",
        "tests/test_e2e_integration.py::TestEndToEndIntegration::test_job_with_volume_parameters",
        "tests/test_e2e_integration.py::TestEndToEndIntegration::test_preview_data_encoding_decoding",
        "tests/test_e2e_integration.py::TestEndToEndIntegration::test_mrc_file_loading",
        "tests/test_e2e_integration.py::TestEndToEndIntegration::test_error_scenarios",
        "tests/test_e2e_integration.py::TestDesktopIntegration",
        "-v",
        "--tb=short",
        "--color=yes"
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n" + "=" * 30)
        print("✅ Server tests passed!")
        return True
    except subprocess.CalledProcessError as e:
        print("\n" + "=" * 30)
        print("❌ Some server tests failed!")
        return False


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--server-only":
        success = run_server_tests_only()
    else:
        success = run_tests()
    
    sys.exit(0 if success else 1)
