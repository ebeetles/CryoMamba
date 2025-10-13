# CryoMamba Desktop Application

A napari-based desktop application for cryo-EM volume visualization and analysis.

## Features

- Load and visualize .mrc files
- Display volume metadata
- Toggle between 2D and 3D visualization
- Qt-based user interface
- macOS compatible

## Installation

### Development Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd CryoMamba
```

2. Install the napari plugin in development mode:
```bash
cd napari_cryomamba
pip install -e .
```

3. Install additional dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Application

```bash
python napari_cryomamba/main.py
```

### Using as a napari Plugin

1. Install napari:
```bash
pip install napari[all]
```

2. Install the CryoMamba plugin:
```bash
pip install -e .
```

3. Launch napari:
```bash
napari
```

4. The CryoMamba widget will be available in the Plugins menu.

### Loading .mrc Files

1. Click "Open .mrc File" button
2. Select your .mrc file
3. The volume will be loaded and displayed
4. Use "Toggle 3D View" to switch between 2D and 3D visualization

## Development

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black napari_cryomamba/
isort napari_cryomamba/
```

## Requirements

- Python 3.8+
- macOS (for desktop application)
- napari 0.4.0+
- mrcfile 1.4.0+