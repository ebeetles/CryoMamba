# Contributing to CryoMamba

Thank you for your interest in contributing to CryoMamba! This document provides guidelines and instructions for contributing to the project.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Submitting Changes](#submitting-changes)
- [Architecture Overview](#architecture-overview)
- [Development Workflow](#development-workflow)
- [Deployment Procedures](#deployment-procedures)

---

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors.

### Expected Behavior

- Be respectful and considerate
- Welcome newcomers and help them get started
- Focus on constructive feedback
- Accept responsibility for mistakes

### Unacceptable Behavior

- Harassment or discrimination
- Trolling or insulting comments
- Publishing private information
- Unprofessional conduct

### Enforcement

Violations can be reported to the project maintainers. All complaints will be reviewed and investigated.

---

## Getting Started

### Prerequisites

- Python 3.9+ (server) or Python 3.8+ (desktop)
- Git for version control
- GPU with CUDA support (for server development)
- macOS (for desktop development)

### Fork and Clone

1. **Fork the repository** on GitHub
2. **Clone your fork**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/CryoMamba.git
   cd CryoMamba
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/original/CryoMamba.git
   ```

### Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-number-description
```

Branch naming conventions:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `test/` - Test additions or modifications

---

## Development Setup

### Server Development Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install server dependencies
pip install -r requirements.txt

# Install nnU-Net (required for inference)
pip install nnunetv2

# Install development dependencies
pip install pytest pytest-cov black isort flake8 mypy

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# Initialize database (if needed)
python -c "from app.services.database import init_db; init_db()"

# Run development server
python dev.py
```

### Desktop Client Development Setup

```bash
# Install desktop client dependencies (from project root)
pip install -r napari_cryomamba/requirements.txt

# Navigate to desktop client directory
cd napari_cryomamba

# Install the package (simple and reliable)
pip install .

# Run application
python main.py
```

**For development (editable install):**
If you want to edit code and see changes immediately:
```bash
# Upgrade pip (required for pyproject.toml editable installs)
pip install --upgrade pip

# Install in editable mode
pip install -e .
```

### Pre-commit Hooks (Optional but Recommended)

```bash
# Install pre-commit
pip install pre-commit

# Set up git hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

---

## Coding Standards

### Python Style Guide

We follow **PEP 8** with some modifications:

- **Line Length**: 100 characters (not 79)
- **Quotes**: Double quotes for strings
- **Imports**: Organized by stdlib, third-party, local
- **Type Hints**: Required for public functions

### Code Formatting

We use **Black** for code formatting:

```bash
# Format all Python files
black app/ napari_cryomamba/ tests/

# Check without modifying
black --check app/ napari_cryomamba/ tests/
```

### Import Sorting

We use **isort** for import organization:

```bash
# Sort imports
isort app/ napari_cryomamba/ tests/

# Check without modifying
isort --check-only app/ napari_cryomamba/ tests/
```

### Linting

We use **flake8** for linting:

```bash
# Lint code
flake8 app/ napari_cryomamba/ tests/

# Configuration in .flake8 or setup.cfg
```

### Type Checking

We use **mypy** for static type checking:

```bash
# Type check
mypy app/ napari_cryomamba/

# Configuration in mypy.ini or setup.cfg
```

### Naming Conventions

```python
# Classes: PascalCase
class JobOrchestrator:
    pass

# Functions/Methods: snake_case
def process_upload():
    pass

# Constants: UPPER_SNAKE_CASE
MAX_UPLOAD_SIZE = 10 * 1024 * 1024 * 1024

# Private: prefix with underscore
def _internal_helper():
    pass

# Type variables: PascalCase with T prefix
TModel = TypeVar("TModel")
```

### Documentation

All public functions/classes must have docstrings:

```python
def create_job(file_id: str, model: str, params: dict) -> Job:
    """
    Create a new segmentation job.
    
    Args:
        file_id: Unique identifier for uploaded file
        model: Model name (e.g., "3d_fullres")
        params: Model parameters and configuration
        
    Returns:
        Created Job instance with assigned job_id
        
    Raises:
        FileNotFoundError: If file_id doesn't exist
        ValueError: If model is not supported
        
    Example:
        >>> job = create_job("file_123", "3d_fullres", {"step_size": 0.5})
        >>> print(job.job_id)
        'job_abc123'
    """
    pass
```

### Error Handling

Use explicit exception handling with descriptive messages:

```python
# Good
try:
    volume = load_mrc_file(filepath)
except FileNotFoundError:
    raise FileNotFoundError(f"MRC file not found: {filepath}")
except mrcfile.MrcError as e:
    raise ValueError(f"Invalid MRC format: {e}")

# Avoid bare except
try:
    risky_operation()
except:  # Bad - too broad
    pass
```

### Logging

Use structured logging:

```python
import logging

logger = logging.getLogger(__name__)

# Use appropriate levels
logger.debug("Detailed debug info")
logger.info("General information")
logger.warning("Warning about potential issue")
logger.error("Error occurred", exc_info=True)

# Include context
logger.info("Job started", extra={"job_id": job_id, "user": user_id})
```

---

## Testing Guidelines

### Test Structure

```
tests/
  ├── test_unit_*.py         # Unit tests
  ├── test_integration_*.py  # Integration tests
  ├── test_e2e_*.py          # End-to-end tests
  └── conftest.py            # Pytest fixtures
```

### Writing Tests

```python
import pytest
from app.services.orchestrator import JobOrchestrator

class TestJobOrchestrator:
    """Test suite for JobOrchestrator."""
    
    @pytest.fixture
    def orchestrator(self):
        """Create test orchestrator instance."""
        return JobOrchestrator()
    
    def test_create_job_success(self, orchestrator):
        """Test successful job creation."""
        job = orchestrator.create_job(
            file_id="test_file",
            model="3d_fullres",
            params={}
        )
        assert job.job_id is not None
        assert job.status == "queued"
    
    def test_create_job_invalid_file(self, orchestrator):
        """Test job creation with invalid file ID."""
        with pytest.raises(FileNotFoundError):
            orchestrator.create_job(
                file_id="nonexistent",
                model="3d_fullres",
                params={}
            )
    
    @pytest.mark.slow
    def test_full_inference_workflow(self, orchestrator):
        """Test complete inference workflow (slow test)."""
        # This test might take minutes
        pass
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_orchestrator.py

# Run specific test
pytest tests/test_orchestrator.py::TestJobOrchestrator::test_create_job_success

# Run with coverage
pytest --cov=app --cov-report=html

# Run only fast tests (skip @pytest.mark.slow)
pytest -m "not slow"

# Run with verbose output
pytest -v

# Stop on first failure
pytest -x
```

### Test Coverage

Aim for:
- **Unit tests**: >80% coverage
- **Integration tests**: Critical paths covered
- **E2E tests**: Main workflows covered

Check coverage:

```bash
pytest --cov=app --cov-report=term-missing
```

### Mocking and Fixtures

Use pytest fixtures for common test setup:

```python
# conftest.py
import pytest
from app.services.database import Database

@pytest.fixture
def db():
    """Provide test database."""
    db = Database(":memory:")  # SQLite in-memory
    yield db
    db.close()

@pytest.fixture
def sample_mrc_file(tmp_path):
    """Create sample MRC file for testing."""
    filepath = tmp_path / "test.mrc"
    # Create test file
    return str(filepath)
```

Mock external dependencies:

```python
from unittest.mock import Mock, patch

def test_job_with_mocked_gpu():
    """Test job execution with mocked GPU."""
    with patch("app.services.gpu_monitor.get_gpu_status") as mock_gpu:
        mock_gpu.return_value = {"memory_free": 50000}
        # Test code here
```

---

## Submitting Changes

### Before Submitting

1. **Run tests**: `pytest`
2. **Check coverage**: `pytest --cov`
3. **Format code**: `black app/ napari_cryomamba/ tests/`
4. **Sort imports**: `isort app/ napari_cryomamba/ tests/`
5. **Lint code**: `flake8 app/ napari_cryomamba/ tests/`
6. **Type check**: `mypy app/ napari_cryomamba/`
7. **Update documentation**: If adding features

### Commit Messages

Follow conventional commits format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Test additions or modifications
- `chore`: Maintenance tasks

Examples:

```
feat(orchestrator): add job priority queue

Implement priority-based job scheduling to allow high-priority
jobs to jump the queue. Adds new JobPriority enum and updates
queue management logic.

Closes #123
```

```
fix(upload): handle connection timeout during chunked upload

Add retry logic with exponential backoff for chunk uploads.
Fixes issue where large files would fail on slow connections.

Fixes #456
```

### Pull Request Process

1. **Update your branch** with latest upstream:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create Pull Request** on GitHub:
   - Use descriptive title
   - Fill out PR template
   - Link related issues
   - Add screenshots/demos if UI changes

4. **Address review feedback**:
   - Make requested changes
   - Respond to comments
   - Push additional commits

5. **Merge**:
   - Maintainer will merge when approved
   - PR will be squashed into single commit

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review performed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] No new warnings
- [ ] Tests pass locally

## Related Issues
Closes #123

## Screenshots (if applicable)
```

---

## Architecture Overview

### System Architecture

```
CryoMamba/
├── app/                    # GPU Server (FastAPI)
│   ├── main.py            # Application entry point
│   ├── config.py          # Configuration management
│   ├── middleware.py      # Request/response middleware
│   ├── models/            # Data models (Pydantic)
│   ├── routes/            # API endpoints
│   └── services/          # Business logic
│       ├── orchestrator.py       # Job orchestration
│       ├── nnunet_wrapper.py     # nnU-Net interface
│       ├── preview_streamer.py   # Preview generation
│       ├── gpu_monitor.py        # GPU monitoring
│       └── database.py           # Data persistence
├── napari_cryomamba/      # Desktop Client
│   ├── main.py            # Application entry point
│   ├── widget.py          # Main UI widget
│   └── napari_cryomamba/  # Package modules
│       ├── websocket_client.py   # Server communication
│       └── export_service.py     # Export functionality
├── tests/                 # Test suite
├── docs/                  # Documentation
└── test_data/             # Sample data for testing
```

### Key Components

#### GPU Server Components

1. **Orchestrator** (`app/services/orchestrator.py`)
   - Job queue management
   - GPU scheduling
   - Lifecycle management

2. **nnU-Net Wrapper** (`app/services/nnunet_wrapper.py`)
   - nnU-Net inference execution
   - Preview generation hooks
   - Error handling

3. **Preview Streamer** (`app/services/preview_streamer.py`)
   - Real-time preview downsampling
   - WebSocket message formatting
   - Performance optimization

4. **GPU Monitor** (`app/services/gpu_monitor.py`)
   - GPU utilization tracking
   - Memory management
   - Performance metrics

#### Desktop Client Components

1. **Main Widget** (`napari_cryomamba/widget.py`)
   - UI layout and controls
   - User interaction handling
   - Napari integration

2. **WebSocket Client** (`napari_cryomamba/websocket_client.py`)
   - Server connection management
   - Message handling
   - Reconnection logic

3. **Export Service** (`napari_cryomamba/export_service.py`)
   - Multiple format support
   - Metadata preservation
   - File validation

### Design Decisions

See [docs/architecture/8-key-design-decisions.md](architecture/8-key-design-decisions.md) for detailed design rationale.

Key decisions:
- **Chunked uploads**: Handle large files efficiently
- **WebSocket streaming**: Real-time preview updates
- **File-based storage**: Simplicity for MVP
- **nnU-Net v2**: State-of-the-art segmentation
- **napari framework**: Rich 3D visualization

---

## Development Workflow

### Feature Development

1. **Plan**
   - Review requirements
   - Check existing issues
   - Discuss approach with maintainers

2. **Implement**
   - Write failing tests first (TDD)
   - Implement feature
   - Pass all tests

3. **Document**
   - Update docstrings
   - Update user documentation
   - Add architecture notes if needed

4. **Review**
   - Self-review code
   - Run all checks
   - Submit PR

### Bug Fixes

1. **Reproduce**
   - Write test that reproduces bug
   - Confirm test fails

2. **Fix**
   - Implement fix
   - Verify test passes
   - Check for side effects

3. **Verify**
   - Run full test suite
   - Manual testing
   - Update documentation if needed

### Refactoring

1. **Ensure tests exist**
   - Add tests if missing
   - Verify current behavior

2. **Refactor incrementally**
   - Small changes
   - Run tests after each change
   - Commit working states

3. **Verify**
   - All tests pass
   - Performance not degraded
   - Documentation still accurate

---

## Deployment Procedures

### Server Deployment

#### Development Deployment

```bash
# Pull latest code
git pull origin main

# Update dependencies
pip install -r requirements.txt

# Run migrations (if any)
# python migrate.py

# Restart development server
python dev.py
```

#### Production Deployment

```bash
# Pull latest code
git pull origin main

# Backup database
cp cryomamba.db cryomamba.db.backup

# Update dependencies
pip install -r requirements.txt

# Run migrations
# python migrate.py

# Run tests
pytest

# Restart production server
systemctl restart cryomamba

# Verify
curl https://your-server.com/v1/healthz
```

### Desktop Client Deployment

#### Development Build

```bash
cd napari_cryomamba

# Update version in pyproject.toml
# version = "1.1.0"

# Build application
python build.py

# Test build
python test_build.py
```

#### Production Build

```bash
# Build for distribution
python build.py --production

# Sign application (macOS)
python sign.py

# Create installer
python create_installers.py

# Test installer
# Install on clean machine
# Run smoke tests

# Upload to release
# Tag release on GitHub
```

### Release Process

1. **Version Bump**
   ```bash
   # Update version in:
   # - app/config.py
   # - napari_cryomamba/pyproject.toml
   # - docs/README.md
   ```

2. **Changelog**
   ```bash
   # Update CHANGELOG.md with changes
   # Group by: Added, Changed, Fixed, Removed
   ```

3. **Tag Release**
   ```bash
   git tag -a v1.1.0 -m "Release version 1.1.0"
   git push origin v1.1.0
   ```

4. **Build Artifacts**
   - Server Docker image
   - Desktop macOS .dmg

5. **Publish Release**
   - Create GitHub release
   - Upload artifacts
   - Publish release notes

### Rollback Procedure

If deployment fails:

```bash
# Revert to previous version
git revert HEAD
git push origin main

# Or checkout previous tag
git checkout v1.0.0

# Restart services
systemctl restart cryomamba

# Verify
curl https://your-server.com/v1/healthz
```

---

## Additional Resources

- **Architecture Documentation**: [docs/architecture/](architecture/)
- **API Reference**: [docs/API_REFERENCE.md](API_REFERENCE.md)
- **User Guide**: [docs/USER_GUIDE.md](USER_GUIDE.md)
- **Project Board**: GitHub Projects
- **Discussions**: GitHub Discussions

---

## Getting Help

- **Questions**: Open a Discussion on GitHub
- **Bugs**: Open an Issue with reproduction steps
- **Security**: Email security@cryomamba.com (if applicable)

---

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

---

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Credited in publications (for significant contributions)

Thank you for contributing to CryoMamba! 🚀

---

**Last Updated**: October 2025

