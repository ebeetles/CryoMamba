# CryoMamba Build Makefile
# Makefile for building CryoMamba desktop application

.PHONY: help clean install-deps build validate sign dmg installers all

# Default target
help:
	@echo "CryoMamba Build Targets:"
	@echo "  install-deps  - Install build dependencies"
	@echo "  clean        - Clean build artifacts"
	@echo "  build        - Build the application"
	@echo "  validate     - Validate the build"
	@echo "  sign         - Sign the application (if certificates available)"
	@echo "  dmg          - Create DMG installer"
	@echo "  installers   - Create all installer types (DMG, PKG, scripts)"
	@echo "  all          - Complete build process"
	@echo "  test-build   - Test the built application"

# Install build dependencies
install-deps:
	@echo "Installing build dependencies..."
	pip install -r build-requirements.txt

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	python build.py --clean

# Build the application
build:
	@echo "Building CryoMamba application..."
	python build.py

# Validate the build
validate:
	@echo "Validating build..."
	python build.py --validate-only

# Sign the application
sign:
	@echo "Signing application..."
	python build.py --skip-deps --validate-only
	codesign --force --deep --sign - dist/CryoMamba.app

# Create DMG installer
dmg:
	@echo "Creating DMG installer..."
	python build.py --skip-deps --create-dmg

# Create all installers
installers:
	@echo "Creating all installers..."
	python build.py --skip-deps --create-installers

# Complete build process
all: clean install-deps build validate dmg
	@echo "Complete build process finished!"

# Test the built application
test-build:
	@echo "Testing built application..."
	@if [ -d "dist/CryoMamba.app" ]; then \
		echo "Opening CryoMamba.app for testing..."; \
		open dist/CryoMamba.app; \
	else \
		echo "No app bundle found. Run 'make build' first."; \
		exit 1; \
	fi

# Development build (faster, no signing/DMG)
dev-build: clean build validate
	@echo "Development build completed!"

# Production build (full process with signing)
prod-build: clean install-deps build validate sign dmg
	@echo "Production build completed!"
