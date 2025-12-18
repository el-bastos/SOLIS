# SOLIS - Singlet Oxygen Luminescence Investigation System

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![PyQt6](https://img.shields.io/badge/GUI-PyQt6-green.svg)](https://www.riverbankcomputing.com/software/pyqt/)

A comprehensive PyQt6-based GUI application for analyzing singlet oxygen (¹O₂) decay kinetics data from time-resolved near-infrared luminescence measurements.

<!-- Screenshot coming soon -->
<!-- ![SOLIS Interface](docs/screenshots/main_interface.png) -->

## Features

- **Modern GUI**: Origin-style split-panel interface with integrated file browser and plot viewer
- **Data Browser**: Load and manage multiple compound datasets with automatic file detection
- **Homogeneous Analysis**: Single exponential and biexponential decay fitting with advanced spike detection
- **Heterogeneous Analysis**:
  - Surplus method for partition analysis
  - Vesicle diffusion model with Numba-accelerated Monte Carlo simulations
  - Single-step grid search with presets (fast/medium/slow)
- **Interactive Plotting**: Publication-quality plots with Matplotlib
  - Embedded plot viewer with zoom, pan, and export tools
  - PDF and CSV export for all plots
- **Session Management**: Save and load complete analysis sessions (.solis.json format)
- **Export**: CSV export for all fitted parameters and plots
- **Performance**:
  - Numba JIT compilation for fast simulations
  - Optimized algorithms (40-60% faster than previous versions)
  - LRU caching for file operations

## Quick Start

### Prerequisites

- Python 3.13 (Numba requires <3.14)
- Windows, macOS, or Linux

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/el-bastos/SOLIS.git
cd SOLIS
```

2. **Set up virtual environment and install dependencies**:

**Windows (PowerShell)**:
```powershell
.\setup_venv.ps1
```

**Windows (CMD)**:
```cmd
setup_venv.bat
```

**Linux/macOS**:
```bash
python3.13 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Running SOLIS

**Windows (PowerShell)**:
```powershell
.\launch_solis.ps1
```

**Windows (CMD)**:
```cmd
launch_solis.bat
```

**Linux/macOS**:
```bash
source venv/bin/activate
python show_splash_then_load.py
```

## Using Pre-built Binaries

Pre-compiled standalone executables are available for:
- Windows (`.exe`)
- macOS (`.app`)
- Linux (`.AppImage`)

Download from the [Releases](https://github.com/el-bastos/SOLIS/releases) page.

## Documentation

- [Building from Source](BUILD.md) - Detailed build instructions for all platforms
- [Setup Instructions](SETUP_INSTRUCTIONS.md) - Development environment setup
- [Contributing Guidelines](CONTRIBUTING.md) - How to contribute to the project
- **Technical Documentation:**
  - [Backend Reference](docs/BACKEND_REFERENCE.md) - Core API and data structures
  - [Codebase Analysis](docs/CODEBASE_ANALYSIS_2025-11-02.md) - Complete technical overview
  - [Development Memory](docs/MEMORY.md) - Complete development history
- **Session History:** [docs/sessions/](docs/sessions/) - Development session summaries
- **Archive:** [docs/archive/](docs/archive/) - Historical debugging and migration notes

## Project Structure

```
SOLIS/
├── show_splash_then_load.py  # Main application entry point
├── solis_gui.py              # Main GUI window
├── core/                     # Analysis engine (kinetics fitting, SNR, statistics)
├── gui/                      # GUI components (browser, dialogs, plot viewer)
├── plotting/                 # Visualization modules (matplotlib-based)
├── heterogeneous/            # Heterogeneous analysis (vesicle diffusion model)
├── surplus/                  # Surplus method for partition analysis
├── data/                     # Data parsing and validation
├── utils/                    # Utilities (session manager, logger, CSV export)
├── test/                     # Test scripts and validation
├── examples/                 # Example datasets
├── docs/                     # Documentation
│   ├── sessions/             # Development session history
│   └── archive/              # Historical debugging notes
├── logo/                     # Application icons and branding
├── build_scripts/            # Build automation scripts
└── requirements.txt          # Python dependencies
```

## Performance Expectations

| Preset | Time | Purpose |
|--------|------|---------|
| Fast   | 3-4 min | Testing and quick analysis |
| Medium | 15-20 min | Publication quality (DEFAULT) |
| Slow   | 25-90 min | Highest quality fits |

When you run heterogeneous analysis, verify Numba is working by checking the console for:
```
INFO: DiffusionSimulatorNumba initialized (Numba JIT enabled)
```

## Citation

If you use SOLIS in your research, please cite:

```bibtex
@software{bastos2025solis,
  author = {Bastos, Erick Leite},
  title = {SOLIS: Singlet Oxygen Luminescence Investigation System},
  year = {2025},
  version = {1.0.0},
  url = {https://github.com/el-bastos/SOLIS}
}
```

## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0). See the [LICENSE](LICENSE) file for details.

**You are free to**:
- Share and adapt the material for non-commercial purposes
- Proper attribution is required

**You cannot**:
- Use the material for commercial purposes

## Author

**Erick Leite Bastos**
Instituto de Química, Universidade de São Paulo
Email: elbastos@iq.usp.br

## Acknowledgments

Developed at the Institute of Chemistry, University of São Paulo (IQ-USP).

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Recent Updates

### December 2025
- **PyInstaller Fixes**: Added missing hiddenimports for PDF export
- **Cross-Platform Verification**: Confirmed Linux and macOS compatibility
- **Documentation**: Complete setup instructions for all platforms

### November 2025
- **GUI Modernization**: Origin-style split-panel browser interface
- **Performance**: 40-60% faster analysis with optimized algorithms
- **Session Management**: Complete save/load functionality (.solis.json format)
- **Plotting**: Migration to matplotlib for better Qt integration
- **Branding**: New splash screen and application icons

### October 2025
- **Heterogeneous Analysis**: Single-step grid search algorithm
- **Preview System**: Real-time spike mask adjustment
- **Documentation**: Comprehensive codebase analysis and reorganization

## Version History

- **1.0.0-beta** (December 2025): Beta release
  - Complete homogeneous and heterogeneous analysis
  - Modern GUI with integrated browser
  - Session save/load functionality
  - Numba-accelerated simulations (all platforms including Apple Silicon)
  - Cross-platform support (Windows, macOS Intel/ARM, Linux x86_64/ARM64)
  - Performance optimizations
