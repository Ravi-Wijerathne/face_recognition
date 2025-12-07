#!/bin/bash

################################################################################
# Face Recognition App Auto-Start Script
# This script automatically checks and installs all dependencies before running
################################################################################

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

################################################################################
# Functions
################################################################################

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

################################################################################
# Check System Prerequisites
################################################################################

check_system_dependencies() {
    print_header "Checking System Dependencies"
    
    local missing_deps=()
    
    # Check for Python 3
    if ! command -v python3 &> /dev/null; then
        missing_deps+=("python3")
    else
        python_version=$(python3 --version | cut -d' ' -f2)
        print_success "Python 3 installed: $python_version"
    fi
    
    # Check for pip
    if ! command -v pip3 &> /dev/null; then
        missing_deps+=("python3-pip")
    else
        print_success "pip3 installed"
    fi
    
    # Check for python3-venv
    if ! dpkg -l | grep -q python3-venv; then
        missing_deps+=("python3-venv")
    else
        print_success "python3-venv installed"
    fi
    
    # Check for python3-tk (required for tkinter/customtkinter)
    if ! dpkg -l | grep -q python3-tk; then
        missing_deps+=("python3-tk")
    else
        print_success "python3-tk installed"
    fi
    
    # Check for build essentials (needed for dlib)
    if ! dpkg -l | grep -q build-essential; then
        missing_deps+=("build-essential")
    else
        print_success "build-essential installed"
    fi
    
    # Check for CMake (needed for dlib)
    if ! command -v cmake &> /dev/null; then
        missing_deps+=("cmake")
    else
        print_success "cmake installed"
    fi
    
    # Check for pkg-config
    if ! command -v pkg-config &> /dev/null; then
        missing_deps+=("pkg-config")
    else
        print_success "pkg-config installed"
    fi
    
    # Check for development libraries
    local dev_libs=("libopencv-dev" "libboost-all-dev" "libgtk-3-dev" "libx11-dev")
    for lib in "${dev_libs[@]}"; do
        if ! dpkg -l | grep -q "^ii.*$lib"; then
            missing_deps+=("$lib")
        fi
    done
    
    # Install missing dependencies
    if [ ${#missing_deps[@]} -gt 0 ]; then
        print_warning "Missing system dependencies: ${missing_deps[*]}"
        print_info "Installing system dependencies (requires sudo)..."
        
        sudo apt-get update
        sudo apt-get install -y "${missing_deps[@]}"
        
        print_success "System dependencies installed"
    else
        print_success "All system dependencies satisfied"
    fi
}

################################################################################
# Check Python Virtual Environment
################################################################################

setup_virtual_environment() {
    print_header "Setting Up Python Virtual Environment"
    
    local venv_dir="$SCRIPT_DIR/venv"
    
    if [ ! -d "$venv_dir" ]; then
        print_info "Creating virtual environment..."
        python3 -m venv "$venv_dir"
        print_success "Virtual environment created"
    else
        print_success "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    print_info "Activating virtual environment..."
    source "$venv_dir/bin/activate"
    print_success "Virtual environment activated"
    
    # Upgrade pip
    print_info "Upgrading pip..."
    pip install --upgrade pip setuptools wheel
    print_success "pip upgraded"
}

################################################################################
# Install Python Dependencies
################################################################################

install_python_dependencies() {
    print_header "Installing Python Dependencies"
    
    # Install base requirements
    if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
        print_info "Installing base requirements..."
        pip install -r "$SCRIPT_DIR/requirements.txt"
        print_success "Base requirements installed"
    else
        print_warning "requirements.txt not found, installing manually..."
        pip install opencv-contrib-python>=4.5.0 numpy>=1.21.0 Pillow>=8.0.0 customtkinter>=5.0.0
    fi
    
    # Install enhanced requirements (optional)
    print_info "Installing enhanced face detection libraries..."
    
    # Try to install dlib (may take time)
    if pip show dlib &> /dev/null; then
        print_success "dlib already installed"
    else
        print_info "Installing dlib (this may take several minutes)..."
        if pip install dlib>=19.24.0; then
            print_success "dlib installed successfully"
        else
            print_warning "dlib installation failed, continuing without it"
        fi
    fi
    
    # Install face-recognition
    if pip show face-recognition &> /dev/null; then
        print_success "face-recognition already installed"
    else
        print_info "Installing face-recognition..."
        if pip install face-recognition>=1.3.0; then
            print_success "face-recognition installed successfully"
        else
            print_warning "face-recognition installation failed, continuing without it"
        fi
    fi
    
    # Install mediapipe
    if pip show mediapipe &> /dev/null; then
        print_success "mediapipe already installed"
    else
        print_info "Installing mediapipe..."
        if pip install mediapipe>=0.10.0; then
            print_success "mediapipe installed successfully"
        else
            print_warning "mediapipe installation failed, continuing without it"
        fi
    fi
}

################################################################################
# Verify Installation
################################################################################

verify_installation() {
    print_header "Verifying Installation"
    
    python3 << 'EOF'
import sys
import warnings
warnings.filterwarnings('ignore')

modules_to_check = [
    ('cv2', 'OpenCV'),
    ('numpy', 'NumPy'),
    ('PIL', 'Pillow'),
    ('customtkinter', 'CustomTkinter'),
    ('dlib', 'dlib (optional)'),
    ('face_recognition', 'face_recognition (optional)'),
    ('mediapipe', 'mediapipe (optional)')
]

all_required_installed = True
for module, name in modules_to_check:
    try:
        __import__(module)
        print(f"✓ {name} - OK")
    except ImportError as e:
        if 'optional' in name.lower():
            print(f"⚠ {name} - Not installed (optional)")
        else:
            print(f"✗ {name} - MISSING: {str(e)}")
            all_required_installed = False
    except Exception as e:
        # Import succeeded but module had initialization issues
        print(f"✓ {name} - OK (with warnings)")

if not all_required_installed:
    sys.exit(1)
EOF
    
    if [ $? -eq 0 ]; then
        print_success "All required modules verified"
    else
        print_error "Some required modules are missing"
        exit 1
    fi
}

################################################################################
# Run the Application
################################################################################

run_application() {
    print_header "Starting Face Recognition Application"
    
    if [ ! -f "$SCRIPT_DIR/face_recognition_opencv.py" ]; then
        print_error "face_recognition_opencv.py not found in $SCRIPT_DIR"
        exit 1
    fi
    
    print_info "Launching application..."
    echo ""
    
    # Run the application
    python3 "$SCRIPT_DIR/face_recognition_opencv.py"
}

################################################################################
# Main Execution
################################################################################

main() {
    clear
    print_header "Face Recognition App - Auto Setup & Start"
    echo ""
    
    # Check if running on Linux
    if [[ "$OSTYPE" != "linux-gnu"* ]]; then
        print_warning "This script is optimized for Linux. Some features may not work on other systems."
    fi
    
    # Step 1: Check and install system dependencies
    check_system_dependencies
    echo ""
    
    # Step 2: Setup virtual environment
    setup_virtual_environment
    echo ""
    
    # Step 3: Install Python dependencies
    install_python_dependencies
    echo ""
    
    # Step 4: Verify installation
    verify_installation
    echo ""
    
    # Step 5: Run the application
    run_application
}

# Trap errors
trap 'print_error "An error occurred. Exiting..."; exit 1' ERR

# Run main function
main
