# 🛠️ Installation Guide - Complete Setup untuk Data Science

Panduan lengkap instalasi Python dan tools yang dibutuhkan untuk course ini.

## 📋 Table of Contents

- [Windows Installation](#windows-installation)
- [macOS Installation](#macos-installation)
- [Linux Installation](#linux-installation)
- [Verifikasi Installation](#verifikasi-installation)
- [Troubleshooting](#troubleshooting)

---

## 🪟 Windows Installation

### Step 1: Download Python

1. **Buka browser** dan kunjungi: https://www.python.org/downloads/
2. **Download Python 3.10+** (recommended: latest stable version)
3. **Run installer** (.exe file yang sudah di-download)

### Step 2: Install Python

**PENTING!** Saat install:
- ✅ **CHECK** "Add Python to PATH" (sangat penting!)
- ✅ **CHECK** "Install pip"
- Click "Install Now"

![Python Installation](https://docs.python.org/3/_images/win_installer.png)

Tunggu hingga instalasi selesai.

### Step 3: Verify Python Installation

1. **Buka Command Prompt**:
   - Tekan `Win + R`
   - Ketik `cmd` 
   - Enter

2. **Check Python version**:
   ```cmd
   python --version
   ```
   Output: `Python 3.10.x` atau lebih baru

3. **Check pip**:
   ```cmd
   pip --version
   ```
   Output: `pip 23.x.x from ...`

✅ Jika keduanya berhasil, Python sudah terinstall!

### Step 4: Install VS Code (Code Editor)

1. Download VS Code: https://code.visualstudio.com/
2. Run installer
3. Install dengan default settings
4. Buka VS Code
5. Install Python extension:
   - Click Extensions icon (atau Ctrl+Shift+X)
   - Search "Python"
   - Install "Python" by Microsoft

### Step 5: Install Jupyter

```cmd
pip install jupyter jupyterlab notebook
```

Test Jupyter:
```cmd
jupyter lab
```

Ini akan membuka browser dengan Jupyter Lab interface.

### Step 6: Setup Virtual Environment (Recommended)

**Di project directory:**

```cmd
cd path\to\aaw-data-science-course
python -m venv venv
venv\Scripts\activate
```

Setelah activate, prompt akan berubah: `(venv) C:\...>`

**Install dependencies:**
```cmd
pip install -r requirements.txt
```

---

## 🍎 macOS Installation

### Step 1: Install Homebrew (jika belum punya)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### Step 2: Install Python

```bash
brew install python@3.10
```

### Step 3: Verify Installation

```bash
python3 --version
pip3 --version
```

### Step 4: Install VS Code

1. Download: https://code.visualstudio.com/
2. Drag to Applications folder
3. Open VS Code
4. Install Python extension (search "Python" in Extensions)

### Step 5: Install Jupyter

```bash
pip3 install jupyter jupyterlab notebook
```

Test:
```bash
jupyter lab
```

### Step 6: Setup Virtual Environment

```bash
cd /path/to/aaw-data-science-course
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 🐧 Linux Installation

### Step 1: Install Python (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install python3.10 python3-pip python3-venv
```

**For Fedora/CentOS:**
```bash
sudo dnf install python3.10 python3-pip
```

### Step 2: Verify Installation

```bash
python3 --version
pip3 --version
```

### Step 3: Install VS Code

**Ubuntu/Debian:**
```bash
sudo snap install code --classic
```

**Or download .deb from**: https://code.visualstudio.com/

Install Python extension di VS Code.

### Step 4: Install Jupyter

```bash
pip3 install jupyter jupyterlab notebook
```

### Step 5: Setup Virtual Environment

```bash
cd /path/to/aaw-data-science-course
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## ✅ Verifikasi Installation

Setelah semua terinstall, jalankan verification script:

```bash
python verify_installation.py
```

Output yang diharapkan:
```
======================================================================
Data Science Zero to Hero - Installation Verification
======================================================================

1. Python Version:
   ✅ Python 3.10.x

2. Core Data Science Libraries:
   ✅ NumPy (1.24.x)
   ✅ Pandas (2.0.x)
   ✅ Matplotlib (3.7.x)
   ✅ Seaborn (0.12.x)
   ✅ Scikit-learn (1.3.x)

3. Machine Learning Libraries:
   ✅ XGBoost (2.0.x)
   ✅ LightGBM (4.0.x)
   ✅ CatBoost (1.2.x)

...

======================================================================
✅ All checks passed! You're ready to start learning.
======================================================================
```

Jika ada yang ❌ (failed), install package yang missing:
```bash
pip install package-name
```

---

## 🔧 Troubleshooting

### Problem 1: "Python is not recognized" (Windows)

**Cause**: Python tidak ada di PATH

**Solution**:
1. Search "Environment Variables" di Windows
2. Click "Environment Variables"
3. Di "System variables", find "Path"
4. Click "Edit"
5. Add Python path (biasanya `C:\Users\YourName\AppData\Local\Programs\Python\Python310\`)
6. Restart Command Prompt

### Problem 2: Permission Denied saat pip install (macOS/Linux)

**Solution**:
```bash
pip3 install --user package-name
```

Or use virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate
pip install package-name
```

### Problem 3: Jupyter tidak bisa dibuka

**Solution**:
```bash
# Reinstall Jupyter
pip uninstall jupyter
pip install jupyter

# Atau coba dengan --user flag
pip install --user jupyter
```

### Problem 4: Import Error - Module not found

**Cause**: Package belum terinstall atau virtual environment belum diaktivasi

**Solution**:
```bash
# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install package
pip install package-name
```

### Problem 5: pip is slow atau timeout

**Solution**: Gunakan mirror yang lebih cepat
```bash
# Untuk instalasi single package
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple package-name

# Atau set permanently
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### Problem 6: TensorFlow GPU not working

**Solution**:
1. Check NVIDIA GPU:
   ```bash
   nvidia-smi
   ```
2. Install CUDA toolkit
3. Install cuDNN
4. Install TensorFlow GPU:
   ```bash
   pip install tensorflow[and-cuda]
   ```

Refer to: https://www.tensorflow.org/install/gpu

---

## 📝 Post-Installation Checklist

Sebelum mulai course, pastikan:
- [ ] Python 3.10+ terinstall
- [ ] pip berfungsi
- [ ] Virtual environment dibuat dan diaktivasi
- [ ] requirements.txt sudah di-install
- [ ] Jupyter Lab bisa dibuka
- [ ] VS Code terinstall dengan Python extension
- [ ] verify_installation.py shows all ✅

---

## 🎓 Next Steps

Setelah installation selesai:
1. ✅ Buka `00_python_crash_course.ipynb` di Jupyter Lab
2. 🎯 Mulai belajar Python basics!

---

## 💡 Tips

### Menggunakan Virtual Environment

**Kenapa penting?**
- Isolate dependencies per project
- Avoid conflicts between different projects
- Easy to recreate environment

**Best Practice:**
- Create venv untuk setiap project
- Always activate sebelum coding
- Install packages with pip setelah activate

### Updating Packages

Check outdated packages:
```bash
pip list --outdated
```

Update specific package:
```bash
pip install --upgrade package-name
```

Update all packages (use with caution):
```bash
pip install --upgrade -r requirements.txt
```

---

**Installation complete! Ready to learn Data Science! 🚀**

**Questions?** Check [Troubleshooting](#troubleshooting) atau create GitHub issue!
