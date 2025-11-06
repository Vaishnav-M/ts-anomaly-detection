"""
Environment Setup Verification Script
Run this to verify all dependencies are installed correctly
"""

import sys
import subprocess

def check_python_version():
    """Check if Python 3.10 is being used"""
    version = sys.version_info
    print(f"✓ Python Version: {version.major}.{version.minor}.{version.micro}")
    if version.major == 3 and version.minor == 10:
        print("  ✓ Python 3.10 detected - Good for compatibility!")
    else:
        print(f"  ⚠ Warning: Expected Python 3.10, got {version.major}.{version.minor}")
    print()

def check_gpu_availability():
    """Check if GPU is available for TensorFlow"""
    try:
        import tensorflow as tf
        print("✓ TensorFlow imported successfully")
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"  ✓ GPU Available: {len(gpus)} GPU(s) detected")
            for gpu in gpus:
                print(f"    - {gpu.name}")
                # Get GPU memory info
                try:
                    gpu_details = tf.config.experimental.get_memory_info(gpu.name)
                    print(f"      Memory: {gpu_details.get('current', 0) / 1024**3:.2f} GB")
                except:
                    pass
        else:
            print("  ⚠ No GPU detected - will use CPU (slower for deep learning)")
        print()
    except ImportError:
        print("✗ TensorFlow not installed")
        print()
    except Exception as e:
        print(f"✗ Error checking GPU: {e}")
        print()

def check_required_packages():
    """Check if all required packages are installed"""
    packages = [
        'numpy',
        'pandas', 
        'scipy',
        'matplotlib',
        'seaborn',
        'plotly',
        'sklearn',
        'tensorflow',
        'statsmodels',
        'jupyter',
        'notebook',
        'ipykernel',
        'tqdm',
        'joblib'
    ]
    
    print("Checking Required Packages:")
    print("-" * 50)
    
    all_installed = True
    for package in packages:
        try:
            if package == 'sklearn':
                __import__('sklearn')
                import sklearn
                version = sklearn.__version__
            else:
                module = __import__(package)
                version = getattr(module, '__version__', 'unknown')
            print(f"✓ {package:20s} - version {version}")
        except ImportError:
            print(f"✗ {package:20s} - NOT INSTALLED")
            all_installed = False
    
    print("-" * 50)
    if all_installed:
        print("✓ All packages installed successfully!\n")
    else:
        print("✗ Some packages are missing. Please run: pip install -r requirements.txt\n")
    
    return all_installed

def check_directory_structure():
    """Verify project directory structure"""
    import os
    
    print("Checking Project Structure:")
    print("-" * 50)
    
    required_dirs = [
        'data/raw',
        'data/processed',
        'notebooks',
        'src/models',
        'outputs/plots',
        'outputs/models',
        'outputs/results'
    ]
    
    all_exist = True
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✓ {directory}")
        else:
            print(f"✗ {directory} - MISSING")
            all_exist = False
    
    print("-" * 50)
    if all_exist:
        print("✓ All directories present!\n")
    else:
        print("✗ Some directories are missing.\n")
    
    return all_exist

def print_system_info():
    """Print system information"""
    import platform
    import psutil
    
    print("System Information:")
    print("-" * 50)
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Processor: {platform.processor()}")
    print(f"RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    print(f"Available RAM: {psutil.virtual_memory().available / (1024**3):.2f} GB")
    print("-" * 50)
    print()

def main():
    """Main verification function"""
    print("=" * 50)
    print("TIME SERIES ANOMALY DETECTION - SETUP VERIFICATION")
    print("=" * 50)
    print()
    
    # Check Python version
    check_python_version()
    
    # Check system info
    try:
        print_system_info()
    except ImportError:
        print("⚠ psutil not installed, skipping system info\n")
    
    # Check directory structure
    check_directory_structure()
    
    # Check packages
    packages_ok = check_required_packages()
    
    # Check GPU
    check_gpu_availability()
    
    # Final summary
    print("=" * 50)
    print("SETUP VERIFICATION COMPLETE")
    print("=" * 50)
    print()
    
    if packages_ok:
        print("✓ Your environment is ready!")
        print("\nNext Steps:")
        print("1. Download a dataset (NASA Bearing or AWS Metrics)")
        print("2. Place it in the data/raw/ directory")
        print("3. Start Jupyter: jupyter notebook")
        print("4. Open notebooks/01_data_exploration.ipynb")
    else:
        print("⚠ Please fix the issues above before proceeding.")

if __name__ == "__main__":
    main()
