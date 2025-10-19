# Advanced Face Recognition System# Face Recognition System



A modern, real-time face recognition application built with Python, OpenCV, and enhanced with multiple state-of-the-art detection algorithms. Features a professional CustomTkinter GUI with dark/light theme support and high-accuracy face recognition capabilities.A real-time face recognition application built with OpenCV and Python. This system allows you to register faces and recognize them in real-time using your webcam.



![Python](https://img.shields.io/badge/Python-3.13-blue.svg)![Face Recognition System](https://img.shields.io/badge/Python-3.7+-blue.svg)

![OpenCV](https://img.shields.io/badge/OpenCV-4.12-green.svg)![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)

![CustomTkinter](https://img.shields.io/badge/CustomTkinter-5.2-orange.svg)![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)

![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)

## 🌟 Features

### Dark Mode Interface

- **🎥 Real-time Face Detection** - Live webcam face detection and tracking

- **👤 Face Registration** - Add and train new faces with multiple samplesThe application features a modern, clean interface with support for both dark and light themes.

- **🔍 Face Recognition** - Identify registered faces with confidence scores

- **💾 Data Persistence** - Automatic saving and loading of face data## 🚀 Features

- **📊 Face Management** - View, delete, and manage registered faces

- **Real-time Face Recognition** - Identify registered faces from webcam feed

### Advanced Detection Methods- **Multiple Detection Methods** - Choose between Haar Cascades, dlib, face_recognition library, or MediaPipe

- **🔧 Haar Cascades** - Fast, lightweight detection (OpenCV built-in)  - **Haar Cascades** - Fast, lightweight (always available)

- **🎯 dlib** - High-accuracy detection with facial landmarks  - **dlib** - High accuracy HOG-based detection

- **🧠 face_recognition** - Deep learning-based recognition (state-of-the-art)  - **face_recognition** - Optimized detection with excellent accuracy (recommended)

- **⚙️ Switchable Algorithms** - Real-time switching between detection methods  - **MediaPipe** - Real-time performance with GPU acceleration support

- **Face Registration** - Add new people with multiple training samples

### Modern Interface- **High Accuracy** - Uses OpenCV's LBPH (Local Binary Pattern Histogram) algorithm

- **🎨 CustomTkinter GUI** - Modern, professional interface- **Modern GUI** - Beautiful, modern interface powered by CustomTkinter

- **🌙 Dark/Light Themes** - Automatic theme detection and switching- **Dark/Light Theme** - Built-in dark and light mode support

- **📱 Responsive Design** - Clean, intuitive user experience- **Data Persistence** - Saves registered faces between sessions

- **📈 Real-time Feedback** - Live confidence scores and status updates- **Confidence Scoring** - Shows recognition accuracy percentage

- **Face Management** - View, delete, and manage registered faces

## 📋 System Requirements

## 📋 Requirements

### Software Requirements

- **Python 3.7+** (Tested with Python 3.13)### System Requirements

- **Webcam** (Built-in or external camera)- **Python 3.7 or higher**

- **Operating System**: Windows, macOS, or Linux- **Webcam** (built-in or external)

- **Windows, macOS, or Linux**

### Hardware Requirements

- **RAM**: 4GB minimum, 8GB recommended### Python Packages

- **CPU**: Multi-core processor recommended for real-time processing- `opencv-contrib-python` - Computer vision and face recognition

- **Storage**: 500MB free space (for dependencies and face data)- `numpy` - Numerical computing

- `Pillow` - Image processing

## 🚀 Quick Start- `customtkinter` - Modern GUI framework with dark/light theme support

- `tkinter` - GUI framework (usually included with Python, required by CustomTkinter)

### 1. Clone the Repository

```bash### Optional Packages (for Enhanced Detection)

git clone https://github.com/Ravi-Wijerathne/face_recognition.gitThe application works with Haar Cascades by default, but you can install these packages for better accuracy and performance:

cd face_recognition- `face-recognition` - **Recommended** - Provides better face detection accuracy

```- `mediapipe` - Real-time face detection with excellent performance

- `dlib` - Advanced face detection (included via face-recognition)

### 2. Set Up Virtual Environment

```bash## 🔧 Installation

# Create virtual environment

python -m venv .venv### 1. Clone or Download

```bash

# Activate virtual environmentgit clone https://github.com/Ravi-Wijerathne/face_recognition.git

# Windows:cd face_recognition

.venv\Scripts\activate```

# macOS/Linux:

source .venv/bin/activate### 2. Set Up Virtual Environment (Recommended)

``````bash

# Create virtual environment

### 3. Install Dependenciespython -m venv .venv



**Basic Installation (Haar Cascades only):**# Activate virtual environment

```bash# Windows:

pip install opencv-contrib-python numpy Pillow customtkinter.venv\Scripts\activate

```# macOS/Linux:

source .venv/bin/activate

**Full Installation (All detection methods):**```

```bash

# Install basic requirements### 3. Install Required Packages

pip install -r requirements.txt```bash

# Install from requirements file (recommended)

# Install enhanced detection librariespip install -r requirements.txt

pip install dlib face-recognition setuptools

# OR install manually

# Note: dlib requires CMake to be installed on your systempip install opencv-contrib-python numpy Pillow customtkinter

``````



### 4. Run the Application### 3a. Install Enhanced Detection (Optional but Recommended)

```bashFor better face detection accuracy and performance:

# Make sure virtual environment is activated```bash

(.venv) $ python face_recognition_opencv.py# Install enhanced detection packages

```pip install -r requirements-enhanced.txt



## 📖 User Guide# OR install manually

pip install face-recognition mediapipe

### Getting Started```

1. **Launch Application** - Run `python face_recognition_opencv.py`

2. **Start Camera** - Click "Start Camera" to begin webcam feed**Note:** The `face-recognition` package requires CMake and dlib to be compiled. On some systems, you may need to install additional dependencies first:

3. **Choose Detection Method** - Select from dropdown (Haar, dlib, face_recognition)

**Windows:**

### Registering New Faces- Install Visual Studio Build Tools or Visual Studio with C++ development tools

1. **Start Camera** - Ensure camera is active- Or use pre-built wheels: `pip install face-recognition`

2. **Add Face** - Click "Add New Face" button

3. **Enter Name** - Type the person's name in the dialog**Linux (Ubuntu/Debian):**

4. **Position Face** - Look directly at camera, ensure good lighting```bash

5. **Capture Samples** - System automatically captures 20 training samplessudo apt-get update

6. **Training** - Wait for "Successfully added" confirmationsudo apt-get install cmake build-essential

pip install face-recognition mediapipe

### Face Recognition```

1. **Start Recognition** - Click "Recognize Faces" button

2. **View Results** - Recognized faces show green boxes with names and confidence**macOS:**

3. **Unknown Faces** - Unregistered faces show red boxes marked "Unknown"```bash

4. **Stop Recognition** - Click "Stop Recognition" to end detection modebrew install cmake

pip install face-recognition mediapipe

### Managing Faces```

- **View List** - All registered faces appear in the "Registered Faces" list

- **Delete Individual** - Select a person and click "Delete Selected"If you encounter issues installing `face-recognition`, the application will still work with the default Haar Cascades detector.

- **Clear All Data** - Use "Clear All" to remove all registered faces

**Note for Linux users:** If you encounter a `ModuleNotFoundError: No module named 'tkinter'` error, you need to install the system tkinter package:

## ⚙️ Detection Methods```bash

# Ubuntu/Debian

### Haar Cascades (Default)sudo apt-get install python3-tk

- **Speed**: ⚡ Very Fast

- **Accuracy**: 📊 Good# Fedora

- **Requirements**: None (built into OpenCV)sudo dnf install python3-tkinter

- **Best For**: Real-time applications, low-resource systems

# Arch Linux

### dlib Detectionsudo pacman -S tk

- **Speed**: ⚡ Fast```

- **Accuracy**: 📊 High

- **Requirements**: CMake, dlib library### 4. Run the Application

- **Best For**: Accurate face detection, facial landmarks

**Make sure your virtual environment is activated first:**

### face_recognition (Recommended)```bash

- **Speed**: ⚡ Medium# Windows:

- **Accuracy**: 📊 Excellent(.venv) PS> python face_recognition_opencv.py

- **Requirements**: dlib, face_recognition library

- **Best For**: High-accuracy recognition, production use# macOS/Linux:

(.venv) $ python face_recognition_opencv.py

## 🛠️ Installation Troubleshooting```



### CMake Installation (Required for dlib)**If virtual environment is not activated, activate it first:**

**Windows:**```bash

1. Download CMake from [cmake.org](https://cmake.org/download/)# Windows:

2. Run installer and select "Add CMake to system PATH".venv\Scripts\activate

3. Restart terminal and verify: `cmake --version`python face_recognition_opencv.py



**macOS:**# macOS/Linux:

```bashsource .venv/bin/activate

brew install cmakepython face_recognition_opencv.py

``````



**Ubuntu/Debian:****Alternative (run directly without activation):**

```bash```bash

sudo apt-get install cmake# Windows:

```C:/path/to/your/project/.venv/Scripts/python.exe face_recognition_opencv.py



### Common Issues# macOS/Linux:

./venv/bin/python face_recognition_opencv.py

#### "No module named 'cv2'"```

```bash

pip install opencv-contrib-python## 📖 How to Use

```

### 🎥 Starting the Camera

#### "Cannot access camera"1. Launch the application

- Close other applications using the camera (Skype, Teams, etc.)2. Click **"Start Camera"** button

- Check camera permissions in system settings3. Your webcam feed will appear in the camera view area

- Try a different camera index if multiple cameras are available4. Click **"Stop Camera"** to turn off the camera



#### "dlib compilation failed"### 🎯 Selecting Detection Method

- Ensure CMake is installed and in PATH1. Use the **"Detection Method"** dropdown to choose your preferred face detection algorithm:

- On Windows, install Visual Studio Build Tools   - **haar** - Fast and lightweight, always available (default if enhanced packages not installed)

- Try pre-compiled wheels: `pip install dlib`   - **dlib** - High accuracy, requires dlib package

   - **face_recognition** - Best balance of speed and accuracy (recommended, requires face-recognition package)

#### Virtual Environment Issues   - **mediapipe** - Excellent real-time performance (requires mediapipe package)

```bash2. The dropdown will only show methods that have their required packages installed

# Recreate virtual environment3. Change the method anytime during operation

rm -rf .venv  # or rmdir /s .venv on Windows

python -m venv .venv### 👤 Adding New Faces

.venv\Scripts\activate  # Windows1. Make sure the camera is running

pip install -r requirements.txt2. Click **"Add New Face"** button

```3. Enter the person's name in the dialog box

4. Click **"Capture"**

## 📁 Project Structure5. Look at the camera and move slightly during capture

6. The system will capture 20 samples automatically

```7. Wait for "Successfully added" confirmation

face_recognition/

├── 📄 face_recognition_opencv.py    # Main application file### 🔍 Recognizing Faces

├── 📄 requirements.txt              # Basic dependencies1. Ensure you have registered at least one face

├── 📄 requirements-enhanced.txt     # All dependencies (including optional)2. Start the camera if not already running

├── 📄 README.md                     # This file3. Click **"Recognize Faces"** button

├── 📄 LICENSE                       # MIT License4. Registered faces will be highlighted with:

├── 📄 .gitignore                    # Git ignore rules   - **Green box** - Recognized person with confidence percentage

├── 📄 CHANGES.md                    # Version history   - **Red box** - Unknown/unrecognized face

├── 📂 .venv/                        # Virtual environment (local)5. Click **"Stop Recognition"** to stop the recognition mode

├── 📊 face_data_opencv.json         # Face metadata (auto-generated)

├── 📊 face_data_opencv.npy          # Face training data (auto-generated)### 📊 Managing Registered Faces

└── 📊 face_labels_opencv.npy        # Face labels (auto-generated)- **View Faces**: Check the "Registered Faces" list to see all saved people

```- **Delete Individual**: Select a person and click "Delete Selected"

- **Clear All**: Click "Clear All" to remove all registered faces

## ⚙️ Configuration

## 🎯 Usage Tips

### Detection Settings

You can modify detection sensitivity in the code:### For Best Results

- **Good Lighting**: Ensure adequate lighting on your face

```python- **Face Position**: Look directly at the camera during registration

# In face_recognition_opencv.py- **Multiple Angles**: Move slightly during capture to get varied samples

# Adjust recognition confidence threshold (line ~300)- **Clear View**: Avoid obstructions like glasses or hats if possible

if confidence < 100:  # Lower = more strict, Higher = more lenient- **Distance**: Stay 2-3 feet away from the camera

    # Face recognized- **Detection Method**: Use 'face_recognition' or 'mediapipe' for best accuracy

```

### Detection Method Comparison

### Training Parameters| Method | Speed | Accuracy | Requirements |

```python|--------|-------|----------|--------------|

# Number of training samples per person (line ~250)| Haar Cascades | ⚡⚡⚡ Fast | ⭐⭐ Moderate | Always available |

target_samples = 20  # Increase for better accuracy| dlib | ⚡⚡ Moderate | ⭐⭐⭐⭐ High | Requires dlib |

| face_recognition | ⚡⚡ Moderate | ⭐⭐⭐⭐⭐ Excellent | Requires face-recognition (recommended) |

# Face detection parameters (line ~180)| MediaPipe | ⚡⚡⚡ Very Fast | ⭐⭐⭐⭐ High | Requires mediapipe |

faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

# Adjust scale factor and min neighbors for different sensitivity### Registration Best Practices

```- **Multiple Sessions**: Register the same person in different lighting conditions

- **Expression Variety**: Capture samples with different expressions

## 🔒 Privacy & Security- **Update Regularly**: Add more samples over time to improve accuracy



### Data Storage## 📁 File Structure

- **Local Only** - All face data stored locally on your computer

- **No Cloud** - No data transmitted to external servers```

- **Encryption Ready** - Face data can be encrypted (see advanced configuration)face_recognition/

├── .gitignore                      # Git ignore file

### Data Files├── face_recognition_opencv.py      # Main application file

- `face_data_opencv.json` - Face metadata and names├── face_data_opencv.json          # Face metadata (auto-generated)

- `face_data_opencv.npy` - Numerical face training data├── face_data_opencv.npy           # Face training data (auto-generated)

- `face_labels_opencv.npy` - Face label mappings├── face_labels_opencv.npy         # Face labels (auto-generated)

├── LICENSE                        # MIT License file

### Git Protection├── README.md                      # This file

Face data files are automatically excluded from version control via `.gitignore`.├── requirements.txt               # Core Python package dependencies

└── requirements-enhanced.txt      # Optional enhanced detection packages

## 🚀 Advanced Usage```



### Command Line Options## ⚙️ Configuration

```bash

# Run with activated virtual environment (recommended)### Theme Customization

(.venv) $ python face_recognition_opencv.pyThe application uses CustomTkinter which supports dark and light themes. The default theme is set to dark mode, but you can change it by modifying the code:



# Run with specific Python version```python

python3.13 face_recognition_opencv.py# In face_recognition_opencv.py, in the __init__ method:

ctk.set_appearance_mode("dark")   # Options: "dark", "light", "system"

# Run with full virtual environment pathctk.set_default_color_theme("blue")  # Options: "blue", "green", "dark-blue"

./venv/Scripts/python.exe face_recognition_opencv.py  # Windows```

./venv/bin/python face_recognition_opencv.py          # macOS/Linux

```The "system" option will automatically match your operating system's theme preference.



### Backup and Restore### Recognition Sensitivity

```bashYou can adjust recognition sensitivity by modifying the confidence threshold in the code:

# Backup face data

cp face_data_opencv.* backup_folder/```python

# In the process_recognition function, line ~162

# Restore face dataif confidence < 100:  # Lower = more strict, Higher = more lenient

cp backup_folder/face_data_opencv.* ./```

```

### Sample Count

### Environment VariablesChange the number of training samples per person:

```bash

# Set custom data directory (optional)```python

export FACE_DATA_DIR="/path/to/custom/directory"# In the capture_face_samples function, line ~240

```target_samples = 20  # Increase for better accuracy

```

## 🧪 Development

## 🛠️ Troubleshooting

### Running in Development Mode

```bash### Common Issues

# Install development dependencies

pip install -r requirements-enhanced.txt#### "Cannot access camera"

- **Solution**: Check if another application is using the camera

# Run with debug output- Close other video applications (Skype, Teams, etc.)

python face_recognition_opencv.py --debug- Try restarting the application

```

#### "No faces registered yet"

### Code Structure- **Solution**: Add at least one face before trying recognition

- **Main Class**: `FaceRecognitionApp` - Core application logic- Follow the "Adding New Faces" steps above

- **Detection Methods**: Modular detection algorithm implementations

- **GUI Components**: CustomTkinter interface elements#### Application won't start

- **Data Management**: JSON and NumPy data persistence- **Solution**: Check if all packages are installed

```bash

## 🤝 Contributingpython -c "import cv2; import numpy; from PIL import Image; print('All packages OK!')"

```

1. **Fork** the repository

2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)#### Detection method not showing in dropdown

3. **Commit** your changes (`git commit -m 'Add amazing feature'`)- **Solution**: The required package for that method is not installed

4. **Push** to the branch (`git push origin feature/amazing-feature`)- Install the package from requirements-enhanced.txt

5. **Open** a Pull Request- Restart the application to see the new method



### Development Guidelines#### Poor recognition accuracy

- Follow PEP 8 style guidelines- **Solutions**:

- Add comments for complex algorithms  - Try using 'face_recognition' or 'mediapipe' detection method for better accuracy

- Test with multiple detection methods  - Add more training samples for the person

- Ensure compatibility with Python 3.7+  - Ensure good lighting during registration

  - Re-register faces in current lighting conditions

## 📄 License  - Check camera quality and positioning



This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.### Package Installation Issues



## 🙏 Acknowledgments#### Enhanced Detection Packages



- **OpenCV** - Computer vision library**face-recognition installation fails:**

- **dlib** - Machine learning library for facial recognition```bash

- **face_recognition** - Simple facial recognition library# Try installing with pre-built wheels

- **CustomTkinter** - Modern GUI frameworkpip install --upgrade pip

- **Python Community** - Amazing ecosystem and supportpip install face-recognition



## 📞 Support# If still fails on Windows, you may need Visual Studio Build Tools

# Or download pre-built dlib wheel for your Python version

### Getting Help```

- **📖 Documentation**: Check this README and inline code comments

- **🐛 Issues**: Report bugs via GitHub Issues**mediapipe installation fails:**

- **💡 Features**: Request features via GitHub Issues```bash

- **❓ Questions**: Use GitHub Discussions# Update pip and try again

pip install --upgrade pip setuptools wheel

### System Information for Bug Reportspip install mediapipe

When reporting issues, please include:```

- Operating System and version

- Python version (`python --version`)**Note:** If enhanced packages fail to install, the application will still work with Haar Cascades detection.

- Package versions (`pip list`)

- Error messages and stack traces#### Basic Package Installation Issues

- Steps to reproduce the issue

#### Windows: "pip is not recognized"

## 🔄 Version History```bash

python -m pip install -r requirements.txt

### v2.0.0 (Current)```

- ✨ Added CustomTkinter modern GUI

- ✨ Multiple detection algorithms (Haar, dlib, face_recognition)#### Permission issues

- ✨ Enhanced face recognition accuracy```bash

- ✨ Improved user interface and experiencepip install --user -r requirements.txt

- 🐛 Fixed UI layout inconsistencies```

- 📚 Comprehensive documentation

#### Virtual Environment Issues

### v1.0.0```bash

- 🎉 Initial release with basic OpenCV functionality# If venv activation fails, try:

- ✨ Haar Cascade face detectionpython -m venv .venv --clear

- ✨ LBPH face recognition

- ✨ Basic Tkinter interface# Then activate and install:

.venv\Scripts\activate

## 🌟 Showcasepip install -r requirements.txt

```

This face recognition system demonstrates:

- **Modern Python Development** - Best practices and clean code## 🔒 Privacy & Data

- **Computer Vision Expertise** - Multiple detection algorithms

- **User Experience Design** - Professional, intuitive interface- **Local Storage**: All face data is stored locally on your computer

- **Production Readiness** - Error handling, data persistence, documentation- **No Cloud**: No data is sent to external servers

- **File Location**: Face data is saved in the same folder as the application

Perfect for portfolios, educational projects, or as a foundation for commercial applications!- **Git Protection**: Face data files are automatically ignored by Git (see `.gitignore`)

- **Data Removal**: Simply delete the `.npy` and `.json` files to remove all data

---

## 🎛️ Advanced Usage

**Made with ❤️ using Python, OpenCV, and Computer Vision**

### Command Line Options

*Last updated: October 20, 2025*```bash
# Run with virtual environment activated (Recommended)
(.venv) PS> python face_recognition_opencv.py
(.venv) $ python face_recognition_opencv.py

# Run with Python command (if venv not activated)
python face_recognition_opencv.py

# Run with specific Python version
python3 face_recognition_opencv.py

# Run with full virtual environment path
# Windows:
C:/path/to/your/project/.venv/Scripts/python.exe face_recognition_opencv.py
# macOS/Linux:
./venv/bin/python face_recognition_opencv.py

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
- **v2.0** - Migrated from Tkinter to CustomTkinter with modern dark/light theme support
- **v2.1** - Added support for multiple face detection methods (dlib, face_recognition, MediaPipe)

---

**Made with ❤️ using Python and OpenCV**

*Last updated: September 22, 2025*