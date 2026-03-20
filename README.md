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
python scripts/start_app.py
```

Works on Windows, Linux, and macOS. The script checks and installs all dependencies, sets up a virtual environment, and launches the app.

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

## Testing

This project includes comprehensive unit and integration tests using pytest.

### Test Structure

The `tests/` directory contains:

- **test_camera.py** — Integration tests for camera functionality (requires camera hardware)
- **test_data_management.py** — Unit tests for data management (save, load, export, import)
- **test_face_detection.py** — Unit tests for face detection methods (Haar Cascades, dlib, face_recognition, MediaPipe)
- **test_gui.py** — Unit tests for GUI components and user interface
- **test_recognition.py** — Unit tests for face recognition logic (confidence, labeling, processing)
- **test_start_script.py** — Unit tests for the start_app.py setup script

### Running Tests

**Run all tests:**
```bash
pytest
```

**Run specific test file:**
```bash
pytest tests/test_face_detection.py
```

**Run tests by marker:**
```bash
pytest -m unit          # Run only unit tests
pytest -m integration   # Run only integration tests
pytest -m camera        # Run only camera tests (requires hardware)
```

**Run with verbose output:**
```bash
pytest -v
```

**Run with coverage report:**
```bash
pip install pytest-cov
pytest --cov=. tests/
```

### Test Configuration

See `pytest.ini` for test configuration including markers, test discovery patterns, and warning filters.

## License

See [LICENSE](LICENSE) for details.
