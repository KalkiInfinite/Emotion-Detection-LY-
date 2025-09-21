#!/usr/bin/env python3
"""
Setup script for the Emotion Detection project.
This script will install all required dependencies and verify the installation.
"""

import subprocess
import sys
import os
import platform


def run_command(command, description):
    """Run a shell command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in {description}:")
        print(f"   Command: {command}")
        print(f"   Error: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    python_version = sys.version_info
    
    if python_version.major == 3 and python_version.minor >= 7:
        print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro} is compatible!")
        return True
    else:
        print(f"❌ Python {python_version.major}.{python_version.minor} is not compatible!")
        print("   This project requires Python 3.7 or higher.")
        return False


def check_pip():
    """Check if pip is available."""
    print("📦 Checking pip availability...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], check=True, capture_output=True)
        print("✅ pip is available!")
        return True
    except subprocess.CalledProcessError:
        print("❌ pip is not available!")
        print("   Please install pip first.")
        return False


def install_requirements():
    """Install required packages from requirements.txt."""
    requirements_file = "requirements.txt"
    
    if not os.path.exists(requirements_file):
        print(f"❌ {requirements_file} not found!")
        return False
    
    print("📚 Installing required packages...")
    print("   This may take a few minutes...")
    
    # Upgrade pip first
    if not run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install requirements
    if not run_command(f"{sys.executable} -m pip install -r {requirements_file}", "Installing requirements"):
        return False
    
    return True


def verify_installation():
    """Verify that all required packages are installed correctly."""
    print("\n🔍 Verifying installation...")
    
    required_packages = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'datasets': 'Datasets',
        'sklearn': 'Scikit-learn',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'tqdm': 'TQDM'
    }
    
    failed_imports = []
    
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✅ {name} imported successfully!")
        except ImportError as e:
            print(f"❌ Failed to import {name}: {str(e)}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n❌ Installation verification failed for: {', '.join(failed_imports)}")
        return False
    else:
        print("\n🎉 All packages verified successfully!")
        return True


def check_gpu_support():
    """Check if CUDA/GPU support is available."""
    print("\n🖥️  Checking GPU support...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ CUDA is available!")
            print(f"   GPU Count: {gpu_count}")
            print(f"   GPU Name: {gpu_name}")
            print(f"   CUDA Version: {torch.version.cuda}")
        else:
            print("⚠️  CUDA is not available. The model will run on CPU.")
            print("   This is normal if you don't have a CUDA-compatible GPU.")
    except ImportError:
        print("❌ Cannot check GPU support (PyTorch not installed)")


def create_project_structure():
    """Create necessary directories for the project."""
    print("\n📁 Creating project structure...")
    
    directories = [
        "results",
        "models", 
        "plots",
        "saved_emotion_model"
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ Created directory: {directory}")
        except Exception as e:
            print(f"❌ Failed to create directory {directory}: {str(e)}")


def display_system_info():
    """Display system information."""
    print("\n💻 System Information:")
    print(f"   OS: {platform.system()} {platform.release()}")
    print(f"   Architecture: {platform.machine()}")
    print(f"   Python: {platform.python_version()}")
    print(f"   Python Executable: {sys.executable}")


def display_usage_instructions():
    """Display usage instructions after successful setup."""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    print("\n📖 How to use the Emotion Detection system:")
    print("\n1. Train a new model:")
    print("   python emotion_detection.py")
    
    print("\n2. Run examples with different usage patterns:")
    print("   python examples.py")
    
    print("\n3. Use individual components:")
    print("   from emotion_detection import EmotionClassifier")
    print("   classifier = EmotionClassifier()")
    print("   # ... your code here")
    
    print("\n📚 Key files in this project:")
    print("   - emotion_detection.py: Main training and inference script")
    print("   - examples.py: Usage examples and demonstrations")
    print("   - config.py: Configuration settings")
    print("   - utils.py: Utility functions")
    print("   - requirements.txt: Required Python packages")
    
    print("\n🎭 Supported emotions:")
    print("   sadness, joy, love, anger, fear, surprise")
    
    print("\n🤖 Supported models:")
    print("   - bert-base-uncased (default)")
    print("   - roberta-base")
    print("   - distilbert-base-uncased")
    
    print("\n💡 Tips:")
    print("   - First run will download the dataset and model (requires internet)")
    print("   - Training typically takes 15-30 minutes depending on your hardware")
    print("   - GPU significantly speeds up training if available")
    print("   - Trained models are saved automatically for reuse")


def main():
    """Main setup function."""
    print("🎭 EMOTION DETECTION PROJECT SETUP")
    print("="*60)
    
    # Display system info
    display_system_info()
    
    # Check prerequisites
    if not check_python_version():
        sys.exit(1)
    
    if not check_pip():
        sys.exit(1)
    
    # Install packages
    print("\n" + "="*60)
    print("📦 INSTALLING DEPENDENCIES")
    print("="*60)
    
    if not install_requirements():
        print("\n❌ Installation failed!")
        print("💡 Try running: pip install --upgrade pip")
        print("💡 Or try: pip install -r requirements.txt --user")
        sys.exit(1)
    
    # Verify installation
    if not verify_installation():
        print("\n❌ Installation verification failed!")
        sys.exit(1)
    
    # Check GPU support
    check_gpu_support()
    
    # Create project structure
    create_project_structure()
    
    # Display usage instructions
    display_usage_instructions()


if __name__ == "__main__":
    main()
