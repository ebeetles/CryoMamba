#!/usr/bin/env python3
"""
CryoMamba Runtime Dependency Validator
Validate that all required dependencies are available at runtime.
"""

import sys
import os
import importlib
from pathlib import Path
import json


class DependencyValidator:
    """Validates runtime dependencies for CryoMamba."""
    
    def __init__(self):
        self.required_modules = [
            'napari',
            'mrcfile',
            'numpy',
            'qtpy',
            'aiohttp',
            'asyncio',
            'json',
            'requests',
            'pathlib',
            'time',
            'os',
            'sys',
        ]
        
        self.optional_modules = [
            'scipy',
            'scikit-image',
            'PIL',
        ]
        
        self.validation_results = {}
    
    def validate_module(self, module_name):
        """Validate a single module."""
        try:
            module = importlib.import_module(module_name)
            
            # Check if module has expected attributes/functions
            if module_name == 'napari':
                if not hasattr(module, 'Viewer'):
                    return False, "napari.Viewer not found"
            
            elif module_name == 'mrcfile':
                if not hasattr(module, 'open'):
                    return False, "mrcfile.open not found"
            
            elif module_name == 'numpy':
                if not hasattr(module, 'array'):
                    return False, "numpy.array not found"
            
            elif module_name == 'qtpy':
                if not hasattr(module, 'QtCore'):
                    return False, "qtpy.QtCore not found"
            
            elif module_name == 'aiohttp':
                if not hasattr(module, 'ClientSession'):
                    return False, "aiohttp.ClientSession not found"
            
            return True, "OK"
            
        except ImportError as e:
            return False, f"Import error: {e}"
        except Exception as e:
            return False, f"Validation error: {e}"
    
    def validate_all(self):
        """Validate all required and optional modules."""
        print("Validating CryoMamba dependencies...")
        print("=" * 50)
        
        # Validate required modules
        print("Required modules:")
        required_failed = []
        
        for module in self.required_modules:
            success, message = self.validate_module(module)
            status = "✓" if success else "✗"
            print(f"  {status} {module}: {message}")
            
            self.validation_results[module] = {
                'required': True,
                'success': success,
                'message': message
            }
            
            if not success:
                required_failed.append(module)
        
        # Validate optional modules
        print("\nOptional modules:")
        optional_failed = []
        
        for module in self.optional_modules:
            success, message = self.validate_module(module)
            status = "✓" if success else "⚠"
            print(f"  {status} {module}: {message}")
            
            self.validation_results[module] = {
                'required': False,
                'success': success,
                'message': message
            }
            
            if not success:
                optional_failed.append(module)
        
        # Summary
        print("\n" + "=" * 50)
        print("Validation Summary:")
        print(f"Required modules: {len(self.required_modules) - len(required_failed)}/{len(self.required_modules)} passed")
        print(f"Optional modules: {len(self.optional_modules) - len(optional_failed)}/{len(self.optional_modules)} passed")
        
        if required_failed:
            print(f"\n✗ Failed required modules: {', '.join(required_failed)}")
            return False
        
        if optional_failed:
            print(f"\n⚠ Missing optional modules: {', '.join(optional_failed)}")
            print("These modules are optional but may provide additional functionality.")
        
        print("\n✓ All required dependencies are available!")
        return True
    
    def save_report(self, filename="dependency_report.json"):
        """Save validation report to file."""
        report = {
            'timestamp': str(Path.cwd()),
            'python_version': sys.version,
            'platform': sys.platform,
            'results': self.validation_results
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Validation report saved to: {filename}")
    
    def check_pyinstaller_bundle(self):
        """Check if running as PyInstaller bundle."""
        if getattr(sys, 'frozen', False):
            print("Running as PyInstaller bundle")
            print(f"Bundle directory: {sys._MEIPASS}")
            return True
        else:
            print("Running as Python script")
            return False


def main():
    """Main validation process."""
    validator = DependencyValidator()
    
    # Check if running as bundle
    validator.check_pyinstaller_bundle()
    
    # Validate dependencies
    success = validator.validate_all()
    
    # Save report
    validator.save_report()
    
    if not success:
        print("\n✗ Dependency validation failed!")
        print("Please install missing dependencies:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    else:
        print("\n✓ Dependency validation passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
