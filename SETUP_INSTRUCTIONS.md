# SOLIS Setup Instructions

**Last Updated:** 2025-12-02 (Session 55)

---

## System Requirements

### Supported Platforms

| Platform | Status | Notes |
|----------|--------|-------|
| **Windows 10/11** | ✅ Fully tested | Primary development platform |
| **macOS (Intel)** | ✅ Supported | Native menu bar integration |
| **macOS (Apple Silicon)** | ✅ Supported | M1/M2/M3 chips supported |
| **Linux (x86_64)** | ✅ Supported | Ubuntu 20.04+ recommended |
| **Linux (ARM64)** | ✅ Supported | Raspberry Pi 4+, ARM servers |

### Python Requirements

- **Python 3.10 - 3.13** (recommended: 3.13)
- **Python 3.14+**: NOT supported (Numba incompatible)

---

## Installation

### Option 1: Quick Setup (Recommended)

#### Windows (PowerShell)
```powershell
git clone https://github.com/el-bastos/SOLIS.git
cd SOLIS
.\setup_venv.ps1
```

#### Windows (CMD)
```cmd
git clone https://github.com/el-bastos/SOLIS.git
cd SOLIS
setup_venv.bat
```

#### Linux / macOS
```bash
git clone https://github.com/el-bastos/SOLIS.git
cd SOLIS
python3.13 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Option 2: Manual Setup

1. **Create virtual environment:**
   ```bash
   python -m venv venv
   ```

2. **Activate virtual environment:**
   - Windows: `venv\Scripts\activate`
   - Linux/macOS: `source venv/bin/activate`

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Running SOLIS

### Windows (PowerShell)
```powershell
.\launch_solis.ps1
```

### Windows (CMD)
```cmd
launch_solis.bat
```

### Linux / macOS
```bash
source venv/bin/activate
python show_splash_then_load.py
```

---

## Verifying Installation

### Check Numba (Critical for Heterogeneous Analysis)

```bash
# Activate venv first
python -c "import numba; print(f'Numba {numba.__version__} installed')"
```

When SOLIS starts, check the console for:
```
INFO: DiffusionSimulatorNumba initialized (Numba JIT enabled)
```

If you see this instead, Numba is not working:
```
WARNING: Numba not available, using pure Python fallback
```

### Check All Dependencies

```bash
python -c "
import PyQt6; print(f'PyQt6: {PyQt6.QtCore.PYQT_VERSION_STR}')
import numpy; print(f'NumPy: {numpy.__version__}')
import scipy; print(f'SciPy: {scipy.__version__}')
import pandas; print(f'Pandas: {pandas.__version__}')
import matplotlib; print(f'Matplotlib: {matplotlib.__version__}')
import numba; print(f'Numba: {numba.__version__}')
"
```

---

## Platform-Specific Notes

### macOS Apple Silicon (M1/M2/M3)

Numba is fully supported on Apple Silicon since version 0.55. If you encounter issues:

```bash
# Use conda for easier installation
conda create -n solis python=3.13
conda activate solis
conda install -c conda-forge numba pyqt numpy scipy pandas matplotlib
```

### Linux

Some distributions require additional system libraries for PyQt6:

**Ubuntu/Debian:**
```bash
sudo apt-get install -y libxcb-xinerama0 libxcb-cursor0 libxkbcommon-x11-0
```

**Fedora:**
```bash
sudo dnf install libxcb libxkbcommon-x11
```

---

## Dependencies

### Core Dependencies (requirements.txt)

| Package | Minimum Version | Purpose |
|---------|-----------------|---------|
| PyQt6 | 6.6.0 | GUI framework |
| numpy | 1.24.0 | Numerical computing |
| scipy | 1.11.0 | Scientific computing |
| pandas | 2.0.0 | Data analysis |
| matplotlib | 3.7.0 | Plotting |
| numba | 0.58.0 | JIT acceleration |

### Development Dependencies (optional)

```bash
pip install pyinstaller  # For building executables
```

---

## Troubleshooting

### "No module named 'numba'"

```bash
pip install numba
```

If that fails on Apple Silicon:
```bash
conda install -c conda-forge numba llvmlite
```

### "PyQt6 not found" or "Qt platform plugin" errors

**Linux:**
```bash
sudo apt-get install -y libxcb-xinerama0 libxcb-cursor0 libxkbcommon-x11-0 libgl1-mesa-glx
```

**macOS:**
```bash
pip uninstall PyQt6 PyQt6-Qt6 PyQt6-sip
pip install PyQt6
```

### Heterogeneous analysis is slow

Numba is not working. Check:
1. Python version < 3.14
2. Numba is installed: `pip show numba`
3. Console shows "Numba JIT enabled" on startup

### "Permission denied" on Linux/macOS

```bash
chmod +x launch_solis.sh
./launch_solis.sh
```

---

## Development Setup

For contributing to SOLIS:

```bash
git clone https://github.com/el-bastos/SOLIS.git
cd SOLIS
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
pip install pyinstaller  # For building

# Run tests
python -m pytest test/

# Run SOLIS
python show_splash_then_load.py
```

---

## Next Steps

After installation:

1. **Load example data:** File → Open Folder → select `examples/homogeneous/`
2. **Run analysis:** Select compounds → Analysis → Homogeneous Analysis
3. **View plots:** Click items in Plots section of browser
4. **Export results:** Save CSV/PDF from plot tabs

See [README.md](README.md) for full usage instructions.

---

**Questions?** Open an issue on [GitHub](https://github.com/el-bastos/SOLIS/issues)
