import cv2
import numpy as np
import customtkinter as ctk
from tkinter import messagebox, filedialog
import json
import os
from PIL import Image, ImageTk
import threading

# Optional imports for enhanced face detection
try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False
    print("Warning: dlib not installed. Haar and dlib detection methods will not be available.")

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("Warning: face_recognition not installed. face_recognition detection method will not be available.")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: mediapipe not installed. MediaPipe detection method will not be available.")

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition System - OpenCV Version")
        self.root.geometry("800x600")
        
        # Set appearance mode and color theme
        ctk.set_appearance_mode("dark")  # Options: "dark", "light", "system"
        ctk.set_default_color_theme("blue")  # Options: "blue", "green", "dark-blue"
        
        # Determine available detection methods
        self.available_methods = ['haar']  # Haar is always available with OpenCV
        
        if DLIB_AVAILABLE:
            self.available_methods.append('dlib')
        
        if FACE_RECOGNITION_AVAILABLE:
            self.available_methods.append('face_recognition')
            
        if MEDIAPIPE_AVAILABLE:
            self.available_methods.append('mediapipe')
        
        # Set default detection method (prefer face_recognition if available, otherwise haar)
        if FACE_RECOGNITION_AVAILABLE:
            self.detection_method = 'face_recognition'
        else:
            self.detection_method = 'haar'
        
        # Initialize OpenCV face detector (Haar Cascades - always available)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize dlib face detector if available
        if DLIB_AVAILABLE:
            self.dlib_detector = dlib.get_frontal_face_detector()
        else:
            self.dlib_detector = None
        
        # Initialize MediaPipe Face Detection if available
        if MEDIAPIPE_AVAILABLE:
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_face_detector = self.mp_face_detection.FaceDetection(
                model_selection=0,  # 0 for short-range detection (within 2 meters)
                min_detection_confidence=0.5
            )
        else:
            self.mp_face_detector = None
        
        # Face recognizer
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # Data storage
        self.face_data = []
        self.face_labels = []
        self.name_to_id = {}
        self.id_to_name = {}
        self.data_file = "face_data_opencv.json"
        
        # Camera
        self.cap = None
        self.is_camera_on = False
        
        # Load existing data
        self.load_data()
        
        # Setup GUI
        self.setup_gui()
        
    def setup_gui(self):
        # Main frame
        main_frame = ctk.CTkFrame(self.root)
        main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Title
        title_label = ctk.CTkLabel(main_frame, text="Face Recognition System", 
                                    font=("Arial", 20, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Camera frame
        camera_frame = ctk.CTkFrame(main_frame)
        camera_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(0, 10))
        
        self.camera_label = ctk.CTkLabel(camera_frame, text="Camera Off", 
                                         fg_color="black", text_color="white",
                                         font=("Arial", 12), width=640, height=480)
        self.camera_label.grid(row=0, column=0, padx=10, pady=10)
        
        # Control buttons
        control_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        control_frame.grid(row=2, column=0, columnspan=3, pady=10)
        
        self.camera_btn = ctk.CTkButton(control_frame, text="Start Camera", 
                                        command=self.toggle_camera, width=150)
        self.camera_btn.grid(row=0, column=0, padx=5)
        
        self.capture_btn = ctk.CTkButton(control_frame, text="Add New Face", 
                                         command=self.add_face_dialog, width=150)
        self.capture_btn.grid(row=0, column=1, padx=5)
        
        self.recognize_btn = ctk.CTkButton(control_frame, text="Recognize Faces", 
                                           command=self.toggle_recognition, width=150)
        self.recognize_btn.grid(row=0, column=2, padx=5)
        
        # Detection method selector
        method_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        method_frame.grid(row=2, column=0, columnspan=3, pady=(50, 10))
        
        method_label = ctk.CTkLabel(method_frame, text="Detection Method:", 
                                    font=("Arial", 12))
        method_label.grid(row=0, column=0, padx=5)
        
        self.method_var = ctk.StringVar(value=self.detection_method)
        self.method_selector = ctk.CTkOptionMenu(
            method_frame,
            values=self.available_methods,
            variable=self.method_var,
            command=self.change_detection_method,
            width=150
        )
        self.method_selector.grid(row=0, column=1, padx=5)
        
        # Face list
        list_frame = ctk.CTkFrame(main_frame)
        list_frame.grid(row=3, column=0, columnspan=3, sticky="nsew", pady=10)
        
        list_label = ctk.CTkLabel(list_frame, text="Registered Faces", 
                                  font=("Arial", 14, "bold"))
        list_label.grid(row=0, column=0, columnspan=2, pady=(5, 10))
        
        # Create a frame for the listbox with scrollbar
        listbox_frame = ctk.CTkFrame(list_frame)
        listbox_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=10, pady=(0, 10))
        
        self.face_listbox = ctk.CTkTextbox(listbox_frame, height=120, width=500)
        self.face_listbox.grid(row=0, column=0, sticky="nsew")
        
        # List control buttons
        list_control_frame = ctk.CTkFrame(list_frame, fg_color="transparent")
        list_control_frame.grid(row=2, column=0, columnspan=2, pady=(10, 10))
        
        ctk.CTkButton(list_control_frame, text="Delete Selected", 
                      command=self.delete_selected_face, width=150).grid(row=0, column=0, padx=5)
        ctk.CTkButton(list_control_frame, text="Clear All", 
                      command=self.clear_all_faces, width=150).grid(row=0, column=1, padx=5)
        
        # Status bar
        self.status_var = ctk.StringVar()
        self.status_var.set("Ready")
        status_bar = ctk.CTkLabel(main_frame, textvariable=self.status_var, 
                                  anchor="w", height=30)
        status_bar.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        
        # Configure grid weights
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(3, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)
        list_frame.grid_rowconfigure(1, weight=1)
        list_frame.grid_columnconfigure(0, weight=1)
        listbox_frame.grid_rowconfigure(0, weight=1)
        listbox_frame.grid_columnconfigure(0, weight=1)
        
        # Update face list
        self.update_face_list()
        
        # Variables for recognition
        self.recognition_active = False
        
    def toggle_camera(self):
        if not self.is_camera_on:
            self.start_camera()
        else:
            self.stop_camera()
    
    def change_detection_method(self, method):
        """Change the face detection method"""
        self.detection_method = method
        self.status_var.set(f"Detection method changed to: {method}")
        print(f"Detection method changed to: {method}")
            
    def start_camera(self):
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Cannot access camera")
                return
                
            self.is_camera_on = True
            self.camera_btn.configure(text="Stop Camera")
            self.status_var.set("Camera started")
            
            # Start video thread
            self.video_thread = threading.Thread(target=self.update_video)
            self.video_thread.daemon = True
            self.video_thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start camera: {str(e)}")
            
    def stop_camera(self):
        self.is_camera_on = False
        self.recognition_active = False
        
        if self.cap:
            self.cap.release()
            
        self.camera_btn.configure(text="Start Camera")
        self.recognize_btn.configure(text="Recognize Faces")
        self.camera_label.configure(text="Camera Off", image="")
        self.status_var.set("Camera stopped")
        
    def update_video(self):
        while self.is_camera_on:
            ret, frame = self.cap.read()
            if ret:
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Process frame for recognition if active
                if self.recognition_active and len(self.face_data) > 0:
                    frame = self.process_recognition(frame)
                
                # Convert frame to displayable format
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                frame_pil = frame_pil.resize((640, 480), Image.Resampling.LANCZOS)
                frame_tk = ImageTk.PhotoImage(frame_pil)
                
                # Update camera label
                self.camera_label.configure(image=frame_tk, text="")
                self.camera_label.image = frame_tk
                
            # Small delay to prevent excessive CPU usage
            self.root.after(30)
    
    def detect_faces(self, frame, gray):
        """Detect faces using the selected method
        
        Args:
            frame: BGR color frame
            gray: Grayscale version of frame
            
        Returns:
            List of tuples (x, y, w, h) representing face bounding boxes
        """
        faces = []
        
        if self.detection_method == 'haar':
            # Haar Cascades (legacy method, always available)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            faces = [(x, y, w, h) for (x, y, w, h) in faces]
            
        elif self.detection_method == 'dlib' and DLIB_AVAILABLE:
            # dlib HOG-based detector
            dlib_faces = self.dlib_detector(gray, 1)
            faces = [(face.left(), face.top(), 
                     face.right() - face.left(), 
                     face.bottom() - face.top()) 
                    for face in dlib_faces]
            
        elif self.detection_method == 'face_recognition' and FACE_RECOGNITION_AVAILABLE:
            # face_recognition library (uses dlib internally but with better defaults)
            # Convert BGR to RGB for face_recognition
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame, model='hog')
            # Convert from (top, right, bottom, left) to (x, y, w, h)
            faces = [(left, top, right - left, bottom - top) 
                    for (top, right, bottom, left) in face_locations]
            
        elif self.detection_method == 'mediapipe' and MEDIAPIPE_AVAILABLE:
            # MediaPipe Face Detection (fast and accurate for real-time)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_face_detector.process(rgb_frame)
            
            if results.detections:
                h, w, _ = frame.shape
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    x = int(bboxC.xmin * w)
                    y = int(bboxC.ymin * h)
                    width = int(bboxC.width * w)
                    height = int(bboxC.height * h)
                    faces.append((x, y, width, height))
        else:
            # Fallback to Haar if selected method is not available
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            faces = [(x, y, w, h) for (x, y, w, h) in faces]
        
        return faces
            
    def process_recognition(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detect_faces(frame, gray)
        
        for (x, y, w, h) in faces:
            face_region = gray[y:y+h, x:x+w]
            face_region = cv2.resize(face_region, (100, 100))
            
            # Predict using trained recognizer
            if len(self.face_data) > 0:
                label, confidence = self.face_recognizer.predict(face_region)
                
                # Lower confidence means better match
                if confidence < 100:  # Threshold for recognition
                    name = self.id_to_name.get(label, "Unknown")
                    confidence_text = f"{name} ({100-confidence:.1f}%)"
                    color = (0, 255, 0)  # Green for recognized
                else:
                    confidence_text = "Unknown"
                    color = (0, 0, 255)  # Red for unknown
                    
                # Draw rectangle and text
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, confidence_text, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            else:
                # No trained data available
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, "No training data", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                
        return frame
        
    def add_face_dialog(self):
        if not self.is_camera_on:
            messagebox.showwarning("Warning", "Please start the camera first")
            return
            
        # Create dialog for name input
        dialog = ctk.CTkToplevel(self.root)
        dialog.title("Add New Face")
        dialog.geometry("300x180")
        dialog.transient(self.root)
        dialog.grab_set()
        
        ctk.CTkLabel(dialog, text="Enter person's name:", 
                     font=("Arial", 14)).pack(pady=15)
        
        name_var = ctk.StringVar()
        name_entry = ctk.CTkEntry(dialog, textvariable=name_var, width=250)
        name_entry.pack(pady=10)
        name_entry.focus()
        
        button_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        button_frame.pack(pady=20)
        
        def capture_face():
            name = name_var.get().strip()
            if not name:
                messagebox.showwarning("Warning", "Please enter a name")
                return
                
            if name in self.name_to_id:
                if not messagebox.askyesno("Confirm", f"Person '{name}' already exists. Add more samples?"):
                    return
                    
            dialog.destroy()
            self.capture_face_samples(name)
            
        ctk.CTkButton(button_frame, text="Capture", command=capture_face, width=100).pack(side="left", padx=5)
        ctk.CTkButton(button_frame, text="Cancel", command=dialog.destroy, width=100).pack(side="left", padx=5)
        
        # Bind Enter key
        dialog.bind('<Return>', lambda e: capture_face())
        
    def capture_face_samples(self, name):
        if not self.cap:
            return
            
        samples_captured = 0
        target_samples = 20  # Number of samples to capture
        
        self.status_var.set(f"Capturing samples for {name}... Look at camera and move slightly")
        
        captured_faces = []
        
        while samples_captured < target_samples and self.is_camera_on:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.detect_faces(frame, gray)
                
                for (x, y, w, h) in faces:
                    # Extract and resize face
                    face_region = gray[y:y+h, x:x+w]
                    face_region = cv2.resize(face_region, (100, 100))
                    captured_faces.append(face_region)
                    
                    samples_captured += 1
                    
                    # Draw rectangle and progress
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, f"Sample {samples_captured}/{target_samples}", 
                               (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    break  # Only capture one face per frame
                
                # Display frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                frame_pil = frame_pil.resize((640, 480), Image.Resampling.LANCZOS)
                frame_tk = ImageTk.PhotoImage(frame_pil)
                
                self.camera_label.configure(image=frame_tk, text="")
                self.camera_label.image = frame_tk
                
                self.root.update()
                
            if samples_captured >= target_samples:
                break
                
        if captured_faces:
            # Get or create ID for this person
            if name not in self.name_to_id:
                person_id = len(self.name_to_id)
                self.name_to_id[name] = person_id
                self.id_to_name[person_id] = name
            else:
                person_id = self.name_to_id[name]
                
            # Add captured faces to training data
            for face in captured_faces:
                self.face_data.append(face)
                self.face_labels.append(person_id)
                
            # Train the recognizer
            self.train_recognizer()
            
            # Save data
            self.save_data()
            
            # Update GUI
            self.update_face_list()
            
            self.status_var.set(f"Successfully added {len(captured_faces)} samples for {name}")
            messagebox.showinfo("Success", f"Added {len(captured_faces)} face samples for {name}")
        else:
            self.status_var.set("No face detected during capture")
            messagebox.showwarning("Warning", "No face was detected. Please try again.")
            
    def train_recognizer(self):
        if len(self.face_data) > 0:
            self.face_recognizer.train(self.face_data, np.array(self.face_labels))
            
    def toggle_recognition(self):
        if not self.is_camera_on:
            messagebox.showwarning("Warning", "Please start the camera first")
            return
            
        if len(self.face_data) == 0:
            messagebox.showwarning("Warning", "No faces registered yet. Please add faces first.")
            return
            
        self.recognition_active = not self.recognition_active
        
        if self.recognition_active:
            self.recognize_btn.configure(text="Stop Recognition")
            self.status_var.set("Face recognition active")
        else:
            self.recognize_btn.configure(text="Recognize Faces")
            self.status_var.set("Face recognition stopped")
            
    def update_face_list(self):
        self.face_listbox.delete("1.0", "end")
        
        # Count samples per person
        person_counts = {}
        for label in self.face_labels:
            # Convert numpy types to regular int
            label_int = int(label)
            if label_int in self.id_to_name:
                name = self.id_to_name[label_int]
                person_counts[name] = person_counts.get(name, 0) + 1
            
        for name, count in person_counts.items():
            self.face_listbox.insert("end", f"{name} ({count} samples)\n")
            
    def delete_selected_face(self):
        # Get current selection from textbox
        try:
            selected_text = self.face_listbox.get("sel.first", "sel.last").strip()
        except:
            messagebox.showwarning("Warning", "Please select a person to delete")
            return
            
        if not selected_text:
            messagebox.showwarning("Warning", "Please select a person to delete")
            return
            
        # Extract name from selection
        name = selected_text.split(" (")[0]
        
        if messagebox.askyesno("Confirm", f"Delete all data for '{name}'?"):
            # Remove data for this person
            person_id = self.name_to_id[name]
            
            # Remove from face_data and face_labels
            new_face_data = []
            new_face_labels = []
            
            for i, label in enumerate(self.face_labels):
                # Convert numpy types to regular int for comparison
                if int(label) != person_id:
                    new_face_data.append(self.face_data[i])
                    new_face_labels.append(label)
                    
            self.face_data = new_face_data
            self.face_labels = new_face_labels
            
            # Remove from mappings
            del self.name_to_id[name]
            del self.id_to_name[person_id]
            
            # Retrain if data exists
            if len(self.face_data) > 0:
                self.train_recognizer()
                
            # Save and update GUI
            self.save_data()
            self.update_face_list()
            
            self.status_var.set(f"Deleted data for {name}")
            
    def clear_all_faces(self):
        if messagebox.askyesno("Confirm", "Delete all face data?"):
            self.face_data = []
            self.face_labels = []
            self.name_to_id = {}
            self.id_to_name = {}
            
            self.save_data()
            self.update_face_list()
            
            self.status_var.set("All face data cleared")
            
    def save_data(self):
        try:
            data = {
                'name_to_id': self.name_to_id,
                'id_to_name': {str(k): v for k, v in self.id_to_name.items()},
                'face_count': len(self.face_data)
            }
            
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            # Save face data as numpy arrays
            if len(self.face_data) > 0:
                np.save('face_data_opencv.npy', np.array(self.face_data))
                np.save('face_labels_opencv.npy', np.array(self.face_labels))
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save data: {str(e)}")
            
    def load_data(self):
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    
                self.name_to_id = data.get('name_to_id', {})
                self.id_to_name = {int(k): v for k, v in data.get('id_to_name', {}).items()}
                
                # Load face data arrays
                if os.path.exists('face_data_opencv.npy') and os.path.exists('face_labels_opencv.npy'):
                    face_data_loaded = np.load('face_data_opencv.npy')
                    face_labels_loaded = np.load('face_labels_opencv.npy')
                    
                    # Convert to lists and ensure data types are consistent
                    self.face_data = [face for face in face_data_loaded]
                    self.face_labels = [int(label) for label in face_labels_loaded]
                    
                    # Train recognizer with loaded data
                    if len(self.face_data) > 0:
                        self.train_recognizer()
                        
        except Exception as e:
            print(f"Failed to load data: {str(e)}")
            # Reset to empty state if loading fails
            self.face_data = []
            self.face_labels = []
            self.name_to_id = {}
            self.id_to_name = {}
            
    def on_closing(self):
        self.stop_camera()
        # Clean up MediaPipe resources
        if self.mp_face_detector:
            self.mp_face_detector.close()
        self.root.destroy()

def main():
    root = ctk.CTk()
    app = FaceRecognitionApp(root)
    
    # Handle window closing
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # Center window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()

if __name__ == "__main__":
    main()