# Face Recognition System

A real-time face recognition application built with OpenCV and Python. This system allows you to register faces and recognize them in real-time using your webcam.

![Face Recognition System](https://img.shields.io/badge/Python-3.7+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)

## 🚀 Features

- **Real-time Face Recognition** - Identify registered faces from webcam feed
- **Face Registration** - Add new people with multiple training samples
- **High Accuracy** - Uses OpenCV's LBPH (Local Binary Pattern Histogram) algorithm
- **User-friendly GUI** - Easy-to-use graphical interface
- **Data Persistence** - Saves registered faces between sessions
- **Confidence Scoring** - Shows recognition accuracy percentage
- **Face Management** - View, delete, and manage registered faces

## 📋 Requirements

### System Requirements
- **Python 3.7 or higher**
- **Webcam** (built-in or external)
- **Windows, macOS, or Linux**

### Python Packages
- `opencv-contrib-python` - Computer vision and face recognition
- `numpy` - Numerical computing
- `Pillow` - Image processing
- `tkinter` - GUI framework (usually included with Python)

## 🔧 Installation

### 1. Clone or Download
```bash
git clone https://github.com/Ravi-Wijerathne/face_recognition.git
cd face_recognition
```

### 2. Install Required Packages
```bash
pip install opencv-contrib-python numpy Pillow
```

### 3. Run the Application

**Option 1: Easy Launch (Recommended)**
- Double-click `run_face_recognition.bat`

**Option 2: Command Line**
```bash
python face_recognition_opencv.py
```

## 📖 How to Use

### 🎥 Starting the Camera
1. Launch the application
2. Click **"Start Camera"** button
3. Your webcam feed will appear in the camera view area
4. Click **"Stop Camera"** to turn off the camera

### 👤 Adding New Faces
1. Make sure the camera is running
2. Click **"Add New Face"** button
3. Enter the person's name in the dialog box
4. Click **"Capture"**
5. Look at the camera and move slightly during capture
6. The system will capture 20 samples automatically
7. Wait for "Successfully added" confirmation

### 🔍 Recognizing Faces
1. Ensure you have registered at least one face
2. Start the camera if not already running
3. Click **"Recognize Faces"** button
4. Registered faces will be highlighted with:
   - **Green box** - Recognized person with confidence percentage
   - **Red box** - Unknown/unrecognized face
5. Click **"Stop Recognition"** to stop the recognition mode

### 📊 Managing Registered Faces
- **View Faces**: Check the "Registered Faces" list to see all saved people
- **Delete Individual**: Select a person and click "Delete Selected"
- **Clear All**: Click "Clear All" to remove all registered faces

## 🎯 Usage Tips

### For Best Results
- **Good Lighting**: Ensure adequate lighting on your face
- **Face Position**: Look directly at the camera during registration
- **Multiple Angles**: Move slightly during capture to get varied samples
- **Clear View**: Avoid obstructions like glasses or hats if possible
- **Distance**: Stay 2-3 feet away from the camera

### Registration Best Practices
- **Multiple Sessions**: Register the same person in different lighting conditions
- **Expression Variety**: Capture samples with different expressions
- **Update Regularly**: Add more samples over time to improve accuracy

## 📁 File Structure

```
face_recognition/
├── .gitignore                      # Git ignore file
├── face_recognition_opencv.py      # Main application file
├── face_data_opencv.json          # Face metadata (auto-generated)
├── face_data_opencv.npy           # Face training data (auto-generated)
├── face_labels_opencv.npy         # Face labels (auto-generated)
├── LICENSE                        # MIT License file
├── README.md                      # This file
└── run_face_recognition.bat       # Easy launcher for Windows
```

## ⚙️ Configuration

### Recognition Sensitivity
You can adjust recognition sensitivity by modifying the confidence threshold in the code:

```python
# In the process_recognition function, line ~162
if confidence < 100:  # Lower = more strict, Higher = more lenient
```

### Sample Count
Change the number of training samples per person:

```python
# In the capture_face_samples function, line ~240
target_samples = 20  # Increase for better accuracy
```

## 🛠️ Troubleshooting

### Common Issues

#### "Cannot access camera"
- **Solution**: Check if another application is using the camera
- Close other video applications (Skype, Teams, etc.)
- Try restarting the application

#### "No faces registered yet"
- **Solution**: Add at least one face before trying recognition
- Follow the "Adding New Faces" steps above

#### Application won't start
- **Solution**: Check if all packages are installed
```bash
python -c "import cv2; import numpy; from PIL import Image; print('All packages OK!')"
```

#### Poor recognition accuracy
- **Solutions**:
  - Add more training samples for the person
  - Ensure good lighting during registration
  - Re-register faces in current lighting conditions
  - Check camera quality and positioning

### Package Installation Issues

#### Windows: "pip is not recognized"
```bash
python -m pip install opencv-contrib-python numpy Pillow
```

#### Permission issues
```bash
pip install --user opencv-contrib-python numpy Pillow
```

## 🔒 Privacy & Data

- **Local Storage**: All face data is stored locally on your computer
- **No Cloud**: No data is sent to external servers
- **File Location**: Face data is saved in the same folder as the application
- **Git Protection**: Face data files are automatically ignored by Git (see `.gitignore`)
- **Data Removal**: Simply delete the `.npy` and `.json` files to remove all data

## 🎛️ Advanced Usage

### Command Line Options
```bash
# Easy launch (Windows)
run_face_recognition.bat

# Run with Python command
python face_recognition_opencv.py

# Run with specific Python version
python3 face_recognition_opencv.py

# Run with full path (if Python not in PATH)
"C:/path/to/python.exe" face_recognition_opencv.py
```

### Backup Your Data
```bash
# Backup face data
copy face_data_opencv.* backup_folder/

# Restore face data
copy backup_folder/face_data_opencv.* ./
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙋‍♂️ Support

If you encounter any issues or have questions:

1. Check the troubleshooting section above
2. Create an issue on GitHub
3. Make sure to include:
   - Your operating system
   - Python version
   - Error messages (if any)
   - Steps to reproduce the issue

## 🔄 Version History

- **v1.0** - Initial release with OpenCV-based recognition
- **v1.1** - Fixed data loading issues and improved stability
- **v1.2** - Enhanced GUI and added better error handling

---

**Made with ❤️ using Python and OpenCV**

*Last updated: September 22, 2025*