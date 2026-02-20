#!/usr/bin/env python3
"""
Face Recognition App Auto-Start Script (Python version)
This script automatically checks and installs all dependencies before running.
Works on Windows, Linux, and macOS.
"""

import os
import sys
import subprocess
import platform
import shutil
import venv
from pathlib import Path

################################################################################
# Configuration
################################################################################

SCRIPT_DIR = Path(__file__).resolve().parent
VENV_DIR = SCRIPT_DIR / "venv"
REQUIREMENTS_FILE = SCRIPT_DIR / "requirements.txt"
APP_FILE = SCRIPT_DIR / "face_recognition_opencv.py"

IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"
IS_MAC = platform.system() == "Darwin"

# Paths inside the venv
if IS_WINDOWS:
    VENV_PYTHON = VENV_DIR / "Scripts" / "python.exe"
    VENV_PIP = VENV_DIR / "Scripts" / "pip.exe"
else:
    VENV_PYTHON = VENV_DIR / "bin" / "python3"
    VENV_PIP = VENV_DIR / "bin" / "pip"

################################################################################
# Color output helpers
################################################################################

# ANSI colors (supported on Windows 10+ terminals and all Unix terminals)
RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
BLUE = "\033[0;34m"
NC = "\033[0m"  # No Color


def enable_windows_ansi():
    """Enable ANSI escape codes on Windows."""
    if IS_WINDOWS:
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:
            pass


def print_header(msg):
    print(f"\n{BLUE}{'=' * 50}{NC}")
    print(f"{BLUE}{msg}{NC}")
    print(f"{BLUE}{'=' * 50}{NC}")


def print_success(msg):
    print(f"  {GREEN}✓ {msg}{NC}")


def print_warning(msg):
    print(f"  {YELLOW}⚠ {msg}{NC}")


def print_error(msg):
    print(f"  {RED}✗ {msg}{NC}")


def print_info(msg):
    print(f"  {BLUE}ℹ {msg}{NC}")


################################################################################
# Run commands inside the venv
################################################################################

def run_venv_command(args, check=True, capture=False):
    """Run a command using the venv Python/pip."""
    try:
        result = subprocess.run(
            args,
            check=check,
            capture_output=capture,
            text=True,
            cwd=str(SCRIPT_DIR),
        )
        return result
    except subprocess.CalledProcessError as e:
        if capture:
            return e
        raise


def pip_install(*args):
    """Run pip install with given arguments inside the venv."""
    cmd = [str(VENV_PIP), "install"] + list(args)
    return run_venv_command(cmd, check=True)


def pip_show(package):
    """Check if a package is installed in the venv."""
    result = run_venv_command(
        [str(VENV_PIP), "show", package], check=False, capture=True
    )
    return result.returncode == 0


################################################################################
# Check System Prerequisites
################################################################################

def check_system_dependencies():
    print_header("Checking System Dependencies")

    # Check Python version
    py_version = platform.python_version()
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 8):
        print_error(f"Python 3.8+ is required. Found: {py_version}")
        sys.exit(1)
    print_success(f"Python {py_version} detected")

    # Check pip availability
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            check=True, capture_output=True, text=True,
        )
        print_success("pip is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print_error("pip is not available. Please install pip first.")
        sys.exit(1)

    # Check cmake (needed for dlib)
    cmake_path = shutil.which("cmake")
    if cmake_path:
        print_success(f"cmake found: {cmake_path}")
    else:
        print_warning("cmake not found — dlib may fail to build")
        if IS_WINDOWS:
            print_info("Install CMake from https://cmake.org/download/")
            print_info("Or run: winget install Kitware.CMake")
        elif IS_LINUX:
            print_info("Install with: sudo apt-get install cmake")
        elif IS_MAC:
            print_info("Install with: brew install cmake")

    # Check Visual C++ Build Tools on Windows (needed for dlib)
    if IS_WINDOWS:
        vs_path = shutil.which("cl")
        if vs_path:
            print_success("Visual C++ compiler found")
        else:
            print_warning(
                "Visual C++ Build Tools not detected in PATH — "
                "dlib may fail to build"
            )
            print_info(
                "Install 'Desktop development with C++' workload from "
                "https://visualstudio.microsoft.com/visual-cpp-build-tools/"
            )

    # Linux-specific checks
    if IS_LINUX:
        missing = []
        for pkg in [
            "python3-venv", "python3-tk", "build-essential",
            "libopencv-dev", "libboost-all-dev", "libgtk-3-dev", "libx11-dev",
        ]:
            result = subprocess.run(
                ["dpkg", "-l", pkg],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                missing.append(pkg)

        if missing:
            print_warning(f"Missing system packages: {', '.join(missing)}")
            print_info("Installing with apt-get (requires sudo)...")
            subprocess.run(["sudo", "apt-get", "update"], check=True)
            subprocess.run(
                ["sudo", "apt-get", "install", "-y"] + missing, check=True
            )
            print_success("System packages installed")
        else:
            print_success("All system packages present")


################################################################################
# Virtual Environment
################################################################################

def setup_virtual_environment():
    print_header("Setting Up Python Virtual Environment")

    if not VENV_DIR.exists():
        print_info("Creating virtual environment...")
        venv.create(str(VENV_DIR), with_pip=True)
        print_success(f"Virtual environment created at {VENV_DIR}")
    else:
        print_success("Virtual environment already exists")

    # Verify venv python exists
    if not VENV_PYTHON.exists():
        print_error(f"Venv Python not found at {VENV_PYTHON}")
        print_info("Recreating virtual environment...")
        shutil.rmtree(VENV_DIR, ignore_errors=True)
        venv.create(str(VENV_DIR), with_pip=True)
        if not VENV_PYTHON.exists():
            print_error("Failed to create virtual environment")
            sys.exit(1)
        print_success("Virtual environment recreated")

    # Upgrade pip, setuptools, wheel
    print_info("Upgrading pip, setuptools, wheel...")
    run_venv_command(
        [str(VENV_PYTHON), "-m", "pip", "install", "--upgrade",
         "pip", "setuptools", "wheel"],
        check=True,
    )
    print_success("pip upgraded")


################################################################################
# Install Python Dependencies
################################################################################

def install_python_dependencies():
    print_header("Installing Python Dependencies")

    # Install base requirements
    if REQUIREMENTS_FILE.exists():
        print_info("Installing base requirements from requirements.txt...")
        pip_install("-r", str(REQUIREMENTS_FILE))
        print_success("Base requirements installed")
    else:
        print_warning("requirements.txt not found, installing core packages manually...")
        pip_install(
            "opencv-contrib-python>=4.5.0",
            "numpy>=1.21.0",
            "Pillow>=8.0.0",
            "customtkinter>=5.0.0",
        )
        print_success("Core packages installed manually")

    # Enhanced / optional dependencies
    optional_packages = [
        ("dlib", "dlib>=19.24.0", "dlib (this may take several minutes)"),
        ("face-recognition", "face-recognition>=1.3.0", "face-recognition"),
        ("mediapipe", "mediapipe>=0.10.0", "mediapipe"),
    ]

    print_info("Installing enhanced face detection libraries...")
    for pip_name, spec, label in optional_packages:
        if pip_show(pip_name):
            print_success(f"{label} already installed")
        else:
            print_info(f"Installing {label}...")
            try:
                pip_install(spec)
                print_success(f"{label} installed successfully")
            except subprocess.CalledProcessError:
                print_warning(f"{label} installation failed, continuing without it")


################################################################################
# Verify Installation
################################################################################

def verify_installation():
    print_header("Verifying Installation")

    verify_script = r'''
import sys, warnings
warnings.filterwarnings("ignore")

modules = [
    ("cv2",              "OpenCV"),
    ("numpy",            "NumPy"),
    ("PIL",              "Pillow"),
    ("customtkinter",    "CustomTkinter"),
    ("dlib",             "dlib (optional)"),
    ("face_recognition", "face_recognition (optional)"),
    ("mediapipe",        "mediapipe (optional)"),
]

ok = True
for mod, name in modules:
    try:
        __import__(mod)
        print(f"  \033[0;32m✓ {name} — OK\033[0m")
    except ImportError as e:
        if "optional" in name.lower():
            print(f"  \033[1;33m⚠ {name} — Not installed (optional)\033[0m")
        else:
            print(f"  \033[0;31m✗ {name} — MISSING: {e}\033[0m")
            ok = False
    except Exception:
        print(f"  \033[0;32m✓ {name} — OK (with warnings)\033[0m")

sys.exit(0 if ok else 1)
'''

    result = run_venv_command(
        [str(VENV_PYTHON), "-c", verify_script], check=False,
    )

    if result.returncode == 0:
        print_success("All required modules verified")
    else:
        print_error("Some required modules are missing!")
        sys.exit(1)


################################################################################
# Run Application
################################################################################

def run_application():
    print_header("Starting Face Recognition Application")

    if not APP_FILE.exists():
        print_error(f"face_recognition_opencv.py not found in {SCRIPT_DIR}")
        sys.exit(1)

    print_info("Launching application...\n")

    # Replace current process with the app running inside the venv
    if IS_WINDOWS:
        # os.execv doesn't work well on Windows; use subprocess instead
        result = subprocess.run(
            [str(VENV_PYTHON), str(APP_FILE)],
            cwd=str(SCRIPT_DIR),
        )
        sys.exit(result.returncode)
    else:
        os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), str(APP_FILE)])


################################################################################
# Main
################################################################################

def main():
    enable_windows_ansi()

    print_header("Face Recognition App — Auto Setup & Start")

    os_name = platform.system()
    print_info(f"Detected OS: {os_name} ({platform.platform()})")

    if not IS_LINUX:
        print_warning(
            "This script is optimized for Linux. "
            "Some system-level checks may be skipped on other platforms."
        )
    print()

    # Step 1
    check_system_dependencies()

    # Step 2
    setup_virtual_environment()

    # Step 3
    install_python_dependencies()

    # Step 4
    verify_installation()

    # Step 5
    run_application()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{RED}Interrupted by user.{NC}")
        sys.exit(130)
    except Exception as e:
        print(f"\n{RED}✗ An error occurred: {e}{NC}")
        sys.exit(1)
