# Advanced Face Recognition System

A modern face recognition application built with Python, OpenCV, and CustomTkinter GUI.

## Features
- Real-time face detection and recognition
- Multiple detection methods (Haar, dlib, face_recognition, MediaPipe)
- Modern CustomTkinter GUI with dark/light themes
- Face registration and management
- High accuracy recognition
- Automated setup script for hassle-free installation

## Quick Start

### 🚀 Automated Setup (Recommended - Linux)

The easiest way to run the application on Linux systems:

```bash
git clone https://github.com/Ravi-Wijerathne/face_recognition.git
cd face_recognition
chmod +x start_app.sh
./start_app.sh
```

The `start_app.sh` script will automatically:
- ✅ Check and install all system dependencies (Python, CMake, build tools, etc.)
- ✅ Create and configure a virtual environment
- ✅ Install all required Python packages (OpenCV, dlib, face-recognition, mediapipe)
- ✅ Verify the installation
- ✅ Launch the application

**First-time setup may take 5-10 minutes as it compiles dlib and downloads models.**

### 📋 Manual Setup (All Platforms)

#### 1. Clone Repository
```bash
git clone https://github.com/Ravi-Wijerathne/face_recognition.git
cd face_recognition
```

#### 2. Install System Dependencies (Linux/Ubuntu)
```bash
sudo apt-get update
sudo apt-get install python3 python3-pip python3-venv python3-tk
sudo apt-get install build-essential cmake pkg-config
sudo apt-get install libopencv-dev libboost-all-dev
```

#### 3. Setup Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### 4. Install Python Dependencies
```bash
# Base requirements
pip install -r requirements.txt

# Optional: Enhanced face detection libraries
pip install dlib face-recognition mediapipe
```

#### 5. Run Application
```bash
python face_recognition_opencv.py
```

## Usage
1. Launch the application using `./start_app.sh` or `python face_recognition_opencv.py`
2. Start camera by clicking the camera button
3. Add new faces by clicking "Add New Face"
4. Click "Recognize Faces" to start identification
5. Select detection method from dropdown (Haar, dlib, face_recognition, MediaPipe)

## Files
- `face_recognition_opencv.py` - Main application file
- `start_app.sh` - Automated setup and launch script (Linux)
- `requirements.txt` - Python dependencies

## Requirements
- **Python 3.7+**
- **Webcam**
- **Linux** (recommended for automated script) or Windows/macOS (manual setup)

### System Requirements (Linux)
The automated script will install these if missing:
- CMake (for dlib compilation)
- build-essential
- python3-tk (for GUI)
- OpenCV development libraries

## Detection Methods
- **Haar Cascades** - Fast, lightweight detection
- **dlib** - High accuracy with facial landmarks
- **face_recognition** - Best overall accuracy (recommended)
- **MediaPipe** - Real-time performance with high accuracy

## Troubleshooting

### Script Issues
- **Permission denied**: Run `chmod +x start_app.sh`
- **Python not found**: Install Python 3.7+ first
- **dlib compilation fails**: Ensure CMake and build-essential are installed

### Application Issues
- **Camera not working**: Check camera permissions
- **Import errors**: Activate virtual environment: `source venv/bin/activate`
- **Slow performance**: Try different detection methods (Haar is fastest)

## License
See LICENSE file for details.

Made with ❤️ using Python and Computer Vision
