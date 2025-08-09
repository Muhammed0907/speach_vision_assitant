#!/usr/bin/env python3
"""
Dependency checker for emotion detection project
Checks and helps install missing dependencies for both Windows and Linux
"""

import subprocess
import sys
import platform
import importlib
import os

def check_python_package(package_name, install_name=None):
    """Check if a Python package is installed"""
    if install_name is None:
        install_name = package_name
    
    try:
        importlib.import_module(package_name)
        print(f"✓ {package_name} is installed")
        return True
    except ImportError:
        print(f"✗ {package_name} is NOT installed")
        print(f"  Install with: pip install {install_name}")
        return False

def check_ffmpeg():
    """Check if ffmpeg is installed and accessible"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✓ ffmpeg is installed")
            return True
    except FileNotFoundError:
        pass
    except subprocess.TimeoutExpired:
        pass
    
    print("✗ ffmpeg is NOT installed")
    
    if platform.system() == 'Windows':
        print("  Windows install options:")
        print("  1. Download from: https://www.gyan.dev/ffmpeg/builds/")
        print("  2. Extract and add ffmpeg.exe to your PATH")
        print("  3. Or use chocolatey: choco install ffmpeg")
        print("  4. Or use winget: winget install FFmpeg")
    else:
        print("  Linux install options:")
        print("  Ubuntu/Debian: sudo apt update && sudo apt install ffmpeg")
        print("  CentOS/RHEL: sudo yum install ffmpeg or sudo dnf install ffmpeg")
        print("  Arch: sudo pacman -S ffmpeg")
    
    return False

def check_audio_system():
    """Check audio system availability"""
    is_windows = platform.system() == 'Windows'
    
    if is_windows:
        # Windows: Check if audio service is running
        try:
            result = subprocess.run(['sc', 'query', 'AudioSrv'], 
                                  capture_output=True, text=True, timeout=5)
            if 'RUNNING' in result.stdout:
                print("✓ Windows Audio Service is running")
                return True
            else:
                print("✗ Windows Audio Service is not running")
                print("  Try: net start AudioSrv")
                return False
        except Exception as e:
            print(f"? Could not check Windows Audio Service: {e}")
            return False
    else:
        # Linux: Check ALSA/PulseAudio
        alsa_ok = False
        pulse_ok = False
        
        try:
            subprocess.run(['aplay', '--version'], capture_output=True, timeout=3)
            print("✓ ALSA is available")
            alsa_ok = True
        except FileNotFoundError:
            print("✗ ALSA is not available")
            print("  Ubuntu/Debian: sudo apt install alsa-utils")
        
        try:
            subprocess.run(['pulseaudio', '--version'], capture_output=True, timeout=3)
            print("✓ PulseAudio is available")
            pulse_ok = True
        except FileNotFoundError:
            print("? PulseAudio not found (optional)")
        
        return alsa_ok or pulse_ok

def main():
    """Main dependency check function"""
    print("Checking dependencies for emotion detection project...")
    print("=" * 60)
    
    missing_deps = []
    
    # Check Python packages
    required_packages = [
        ('cv2', 'opencv-python'),
        ('numpy', 'numpy'),
        ('pyaudio', 'pyaudio'),
        ('dashscope', 'dashscope'),
        ('openai', 'openai'),
        ('insightface', 'insightface'),
        ('multiprocessing', None),  # Built-in
        ('threading', None),        # Built-in
        ('queue', None),           # Built-in
        ('dotenv', 'python-dotenv')
    ]
    
    print("\nPython packages:")
    for package, install_name in required_packages:
        if install_name is None:
            print(f"✓ {package} (built-in)")
        else:
            if not check_python_package(package, install_name):
                missing_deps.append(install_name)
    
    # Check system dependencies
    print("\nSystem dependencies:")
    if not check_ffmpeg():
        missing_deps.append('ffmpeg (system)')
    
    if not check_audio_system():
        missing_deps.append('audio system')
    
    # Summary
    print("\n" + "=" * 60)
    if missing_deps:
        print(f"❌ {len(missing_deps)} dependencies are missing:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("\nInstall missing Python packages with:")
        python_deps = [dep for dep in missing_deps if dep != 'ffmpeg (system)' and dep != 'audio system']
        if python_deps:
            print(f"pip install {' '.join(python_deps)}")
    else:
        print("✅ All dependencies are satisfied!")
    
    print("\nSpecial notes:")
    print("- Make sure you have a .env file with MODEL_API_KEY or DASHSCOPEAPIKEY")
    print("- Ensure your audio drivers are properly installed")
    if platform.system() == 'Windows':
        print("- On Windows, PyAudio may require Microsoft Visual C++ Build Tools")

if __name__ == "__main__":
    main()