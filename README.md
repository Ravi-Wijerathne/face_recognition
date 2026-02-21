# Face Recognition System

Real-time face detection and recognition using Python, OpenCV, and CustomTkinter. Supports multiple detection methods: Haar Cascades, dlib, face_recognition, and MediaPipe.

## Requirements

- Python 3.7+
- Webcam

## Setup

### Automated (Cross-Platform — Python Script)

```bash
git clone https://github.com/Ravi-Wijerathne/face_recognition.git
cd face_recognition
python start_app.py
```

Works on Windows, Linux, and macOS. The script checks and installs all dependencies, sets up a virtual environment, and launches the app.

### Automated (Linux — Shell Script)

```bash
git clone https://github.com/Ravi-Wijerathne/face_recognition.git
cd face_recognition
chmod +x start_app.sh
./start_app.sh
```

Same as above but as a Bash script for Linux. First run may take 5-10 minutes.

### Manual Setup (All Platforms)

**1. Clone and enter the repository**
```bash
git clone https://github.com/Ravi-Wijerathne/face_recognition.git
cd face_recognition
```

**2. Install system dependencies (Linux/Ubuntu only)**
```bash
sudo apt-get update
sudo apt-get install python3 python3-pip python3-venv python3-tk
sudo apt-get install build-essential cmake pkg-config
sudo apt-get install libopencv-dev libboost-all-dev
```

**3. Create and activate a virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**4. Install Python dependencies**
```bash
pip install -r requirements.txt

# Optional: enhanced detection libraries
pip install dlib face-recognition mediapipe
```

**5. Run the application**
```bash
python face_recognition_opencv.py
```

## Usage

1. Start the camera using the camera button
2. Click **Add New Face** to register faces
3. Click **Recognize Faces** to begin identification
4. Switch detection methods from the dropdown

## License

See [LICENSE](LICENSE) for details.
