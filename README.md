# Advanced Face Recognition System

A modern face recognition application built with Python, OpenCV, and CustomTkinter GUI.

## Features
- Real-time face detection and recognition
- Multiple detection methods (Haar, dlib, face_recognition)
- Modern CustomTkinter GUI with dark/light themes
- Face registration and management
- High accuracy recognition

## Quick Start

### 1. Clone Repository
`
git clone https://github.com/Ravi-Wijerathne/face_recognition.git
cd face_recognition
`

### 2. Setup Virtual Environment
`
python -m venv .venv
.venv\Scripts\activate
`

### 3. Install Dependencies
`
pip install opencv-contrib-python numpy Pillow customtkinter
pip install dlib face-recognition setuptools
`

### 4. Run Application
`
python face_recognition_opencv.py
`

## Usage
1. Start camera
2. Add new faces by clicking "Add New Face"
3. Click "Recognize Faces" to start identification
4. Select detection method from dropdown

## Requirements
- Python 3.7+
- Webcam
- CMake (for dlib installation)

## Detection Methods
- **Haar Cascades** - Fast, lightweight
- **dlib** - High accuracy with landmarks
- **face_recognition** - Best accuracy (recommended)

Made with ❤️ using Python and Computer Vision
