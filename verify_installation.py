#!/usr/bin/env python3
"""
Verification script untuk Data Science Zero to Hero Course
Script ini akan verify bahwa semua dependencies terinstall dengan benar.
"""

import sys
from typing import List, Tuple

def check_python_version() -> Tuple[bool, str]:
    """Check Python version (requires 3.10+)"""
    version = sys.version_info
    required = (3, 10)
    
    if version >= required:
        return True, f"✅ Python {version.major}.{version.minor}.{version.micro}"
    else:
        return False, f"❌ Python {version.major}.{version.minor} (Required: 3.10+)"

def check_package(package_name: str, import_name: str = None) -> Tuple[bool, str]:
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown version')
        return True, f"✅ {package_name} ({version})"
    except ImportError:
        return False, f"❌ {package_name} (NOT INSTALLED)"

def main():
    """Run all verification checks"""
    print("=" * 70)
    print("Data Science Zero to Hero - Installation Verification")
    print("=" * 70)
    print()
    
    # Check Python version
    print("1. Python Version:")
    success, message = check_python_version()
    print(f"   {message}")
    if not success:
        print("\n⚠️  Please upgrade Python to 3.10 or higher")
        sys.exit(1)
    print()
    
    # Core packages to check
    core_packages = [
        ('NumPy', 'numpy'),
        ('Pandas', 'pandas'),
        ('Matplotlib', 'matplotlib'),
        ('Seaborn', 'seaborn'),
        ('Scikit-learn', 'sklearn'),
    ]
    
    ml_packages = [
        ('XGBoost', 'xgboost'),
        ('LightGBM', 'lightgbm'),
        ('CatBoost', 'catboost'),
    ]
    
    dl_packages = [
        ('TensorFlow', 'tensorflow'),
        ('Keras', 'keras'),
    ]
    
    stats_packages = [
        ('SciPy', 'scipy'),
        ('Statsmodels', 'statsmodels'),
    ]
    
    mlops_packages = [
        ('MLflow', 'mlflow'),
        ('FastAPI', 'fastapi'),
        ('SHAP', 'shap'),
    ]
    
    jupyter_packages = [
        ('Jupyter', 'jupyter'),
        ('JupyterLab', 'jupyterlab'),
    ]
    
    # Run checks
    all_passed = True
    
    print("2. Core Data Science Libraries:")
    for name, import_name in core_packages:
        success, message = check_package(name, import_name)
        print(f"   {message}")
        if not success:
            all_passed = False
    print()
    
    print("3. Machine Learning Libraries:")
    for name, import_name in ml_packages:
        success, message = check_package(name, import_name)
        print(f"   {message}")
        if not success:
            all_passed = False
    print()
    
    print("4. Deep Learning Libraries:")
    for name, import_name in dl_packages:
        success, message = check_package(name, import_name)
        print(f"   {message}")
        if not success:
            all_passed = False
    print()
    
    print("5. Statistics Libraries:")
    for name, import_name in stats_packages:
        success, message = check_package(name, import_name)
        print(f"   {message}")
        if not success:
            all_passed = False
    print()
    
    print("6. MLOps & Deployment:")
    for name, import_name in mlops_packages:
        success, message = check_package(name, import_name)
        print(f"   {message}")
        if not success:
            all_passed = False
    print()
    
    print("7. Jupyter Environment:")
    for name, import_name in jupyter_packages:
        success, message = check_package(name, import_name)
        print(f"   {message}")
        if not success:
            all_passed = False
    print()
    
    print("=" * 70)
    if all_passed:
        print("✅ All checks passed! You're ready to start learning.")
        print("\nNext steps:")
        print("  1. Open Jupyter Lab: jupyter lab")
        print("  2. Navigate to 00_Pre_Course/ or 01_Foundations/")
        print("  3. Start your Data Science journey! 🚀")
    else:
        print("❌ Some packages are missing.")
        print("\nTo install missing packages:")
        print("  Using conda: conda env create -f environment.yml")
        print("  Using pip: pip install -r requirements.txt")
    print("=" * 70)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
