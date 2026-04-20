#!/usr/bin/env python3
"""
HGVCS Setup and Installation Script

This script sets up the HGVCS environment, creates necessary directories,
and performs initial configuration.

Usage:
    python scripts/setup.py
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
from datetime import datetime

class Colors:
    """Terminal colors for pretty output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    """Print a formatted header."""
    print(f"\n{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{text.center(60)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}\n")

def print_success(text):
    """Print a success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.ENDC}")

def print_warning(text):
    """Print a warning message."""
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")

def print_error(text):
    """Print an error message."""
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")

def print_info(text):
    """Print an info message."""
    print(f"{Colors.CYAN}ℹ {text}{Colors.ENDC}")

def check_python_version():
    """Check if Python version is compatible."""
    print_info("Checking Python version...")
    
    version = sys.version_info
    if version < (3, 8):
        print_error(f"Python {version.major}.{version.minor} is not supported")
        print_info("Please install Python 3.8 or higher")
        return False
    
    print_success(f"Python {version.major}.{version.minor}.{version.micro} ✓")
    return True

def create_directories():
    """Create necessary directories."""
    print_info("Creating directories...")
    
    directories = [
        'data',
        'data/training',
        'data/training/gestures',
        'data/training/voice',
        'data/calibration',
        'data/cache',
        'logs',
        'models',
        'certs',
        'src/models',
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print_success(f"Created: {directory}")

def install_dependencies():
    """Install Python dependencies."""
    print_info("Installing dependencies...")
    
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
        ])
        print_success("Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install dependencies: {e}")
        return False

def download_models():
    """Download required AI models."""
    print_info("Downloading AI models...")
    
    # Download spaCy model
    try:
        print_info("Downloading spaCy English model...")
        subprocess.check_call([
            sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'
        ])
        print_success("spaCy model downloaded")
    except subprocess.CalledProcessError:
        print_warning("Failed to download spaCy model (will retry on first run)")
    
    # Whisper model is downloaded automatically on first use
    print_info("Whisper model will be downloaded on first use")

def check_camera():
    """Check if camera is available."""
    print_info("Checking camera...")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                print_success("Camera is available ✓")
                return True
        
        print_warning("Camera not accessible (may need permissions)")
        return False
    except Exception as e:
        print_warning(f"Camera check failed: {e}")
        return False

def check_microphone():
    """Check if microphone is available."""
    print_info("Checking microphone...")
    
    try:
        import pyaudio
        audio = pyaudio.PyAudio()
        
        # List available devices
        info = audio.get_host_api_info_by_index(0)
        num_devices = info.get('deviceCount')
        
        has_input = False
        for i in range(num_devices):
            device_info = audio.get_device_info_by_host_api_device_index(0, i)
            if device_info.get('maxInputChannels') > 0:
                has_input = True
                break
        
        audio.terminate()
        
        if has_input:
            print_success("Microphone is available ✓")
            return True
        else:
            print_warning("No microphone found")
            return False
    except Exception as e:
        print_warning(f"Microphone check failed: {e}")
        return False

def check_gpu():
    """Check if GPU is available for acceleration."""
    print_info("Checking GPU availability...")
    
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            print_success(f"GPU detected: {len(gpus)} device(s) ✓")
            for gpu in gpus:
                print_info(f"  - {gpu}")
            return True
        else:
            print_warning("No GPU detected (CPU mode will be used)")
            return False
    except Exception as e:
        print_warning(f"GPU check failed: {e}")
        return False

def create_default_config():
    """Create default configuration file if it doesn't exist."""
    config_path = Path('config/user_config.yaml')
    
    if config_path.exists():
        print_info("User config already exists")
        return
    
    print_info("Creating default user configuration...")
    
    default_config = f"""# HGVCS User Configuration
# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

# This is your personal configuration file.
# Edit this to customize HGVCS for your needs.

user:
  name: ""
  email: ""
  
preferences:
  theme: "dark"
  language: "en"
  notifications: true
  sound_effects: true
  
gesture:
  # Adjust these based on your environment
  confidence_threshold: 0.75
  cooldown_period: 0.5
  
voice:
  wake_word: "Hey System"
  confidence_threshold: 0.7
  
network:
  device_name: "{platform.node()}"
  default_download_path: "~/Downloads/HGVCS"
"""
    
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        f.write(default_config)
    
    print_success(f"Created: {config_path}")

def print_next_steps():
    """Print next steps for the user."""
    print_header("Setup Complete!")
    
    print(f"{Colors.BOLD}Next Steps:{Colors.ENDC}\n")
    
    print("1. Run the application:")
    print(f"   {Colors.CYAN}python main.py{Colors.ENDC}\n")
    
    print("2. Run with calibration:")
    print(f"   {Colors.CYAN}python main.py --calibrate{Colors.ENDC}\n")
    
    print("3. Train custom gesture model:")
    print(f"   {Colors.CYAN}python scripts/collect_gesture_data.py --gesture all --samples 100{Colors.ENDC}")
    print(f"   {Colors.CYAN}python scripts/train_gesture_model.py{Colors.ENDC}\n")
    
    print("4. View all options:")
    print(f"   {Colors.CYAN}python main.py --help{Colors.ENDC}\n")
    
    print(f"{Colors.BOLD}Documentation:{Colors.ENDC}")
    print(f"  - README.md - Project overview")
    print(f"  - docs/GESTURE_GUIDE.md - Complete gesture reference")
    print(f"  - docs/VOICE_COMMANDS.md - Voice command reference\n")
    
    print(f"{Colors.GREEN}Enjoy using HGVCS! 🚀{Colors.ENDC}\n")

def main():
    """Main setup function."""
    print_header("HGVCS Setup")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_dependencies():
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Download models
    download_models()
    
    # Check hardware
    print_header("Hardware Check")
    check_camera()
    check_microphone()
    check_gpu()
    
    # Create config
    create_default_config()
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Setup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
