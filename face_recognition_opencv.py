import cv2
import numpy as np
import customtkinter as ctk
from tkinter import messagebox, filedialog
import json
import os
from PIL import Image, ImageTk
import threading
import time

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
    _ = mp.solutions
    MEDIAPIPE_AVAILABLE = True
except (ImportError, AttributeError):
    MEDIAPIPE_AVAILABLE = False
    print("Warning: mediapipe not fully installed. MediaPipe detection method will not be available.")


class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition System")
        self.root.geometry("1000x750")
        self.root.minsize(900, 650)
        
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("green")
        
        self.available_methods = ['haar']
        
        if DLIB_AVAILABLE:
            self.available_methods.append('dlib')
        
        if FACE_RECOGNITION_AVAILABLE:
            self.available_methods.append('face_recognition')
            
        if MEDIAPIPE_AVAILABLE:
            self.available_methods.append('mediapipe')
        
        if FACE_RECOGNITION_AVAILABLE:
            self.detection_method = 'face_recognition'
        else:
            self.detection_method = 'haar'
        
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        if DLIB_AVAILABLE:
            self.dlib_detector = dlib.get_frontal_face_detector()
        else:
            self.dlib_detector = None
        
        if MEDIAPIPE_AVAILABLE:
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_face_detector = self.mp_face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=0.5
            )
        else:
            self.mp_face_detector = None
        
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        self.face_data = []
        self.face_labels = []
        self.name_to_id = {}
        self.id_to_name = {}
        self.data_file = "face_data_opencv.json"
        
        self.cap = None
        self.is_camera_on = False
        self.recognition_active = False
        self.capture_in_progress = False
        
        self.load_data()
        self.setup_gui()
        self.bind_shortcuts()
        
    def bind_shortcuts(self):
        self.root.bind('<space>', lambda e: self.toggle_camera())
        self.root.bind('<Control-a>', lambda e: self.add_face_dialog() if self.is_camera_on else self.status_var.set("Start camera first"))
        self.root.bind('<Control-r>', lambda e: self.toggle_recognition())
        self.root.bind('<Escape>', lambda e: self.on_closing())
        
    def setup_gui(self):
        main_container = ctk.CTkScrollableFrame(self.root, fg_color=("#1a1a1a", "#0d0d0d"))
        main_container.pack(fill="both", expand=True, padx=0, pady=0)
        
        content_frame = ctk.CTkFrame(main_container, fg_color="transparent")
        content_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        self.create_header(content_frame)
        self.create_camera_section(content_frame)
        self.create_control_panel(content_frame)
        self.create_detection_section(content_frame)
        self.create_face_list_section(content_frame)
        self.create_status_bar(content_frame)
        
    def create_header(self, parent):
        header_frame = ctk.CTkFrame(parent, fg_color="transparent")
        header_frame.pack(fill="x", pady=(0, 15))
        
        title_label = ctk.CTkLabel(
            header_frame, 
            text="Face Recognition System",
            font=("Segoe UI", 28, "bold"),
            text_color=("#00b894", "#55efc4")
        )
        title_label.pack(side="left")
        
        theme_btn = ctk.CTkButton(
            header_frame,
            text="",
            width=40,
            height=40,
            fg_color="transparent",
            hover_color=("#2d2d2d", "#1a1a1a"),
            command=self.toggle_theme
        )
        theme_btn.pack(side="right", padx=5)
        self.theme_btn = theme_btn
        self.update_theme_icon()
        
        info_btn = ctk.CTkButton(
            header_frame,
            text="i",
            width=40,
            height=40,
            font=("Segoe UI", 18, "bold"),
            text_color=("#ffffff", "#000000"),
            fg_color=("#0984e3", "#74b9ff"),
            hover_color=("#74b9ff", "#0984e3"),
            corner_radius=20,
            command=self.show_info
        )
        info_btn.pack(side="right", padx=5)
        
    def update_theme_icon(self):
        mode = ctk.get_appearance_mode()
        if mode == "Dark":
            self.theme_btn.configure(text="☀")
        else:
            self.theme_btn.configure(text="🌙")
    
    def toggle_theme(self):
        current = ctk.get_appearance_mode()
        new_mode = "Light" if current == "Dark" else "Dark"
        ctk.set_appearance_mode(new_mode)
        self.update_theme_icon()
        
    def show_info(self):
        info_dialog = ctk.CTkToplevel(self.root)
        info_dialog.title("Information")
        info_dialog.geometry("450x400")
        info_dialog.transient(self.root)
        info_dialog.grab_set()
        
        info_text = """
Keyboard Shortcuts:

• Space        - Toggle Camera
• Ctrl+A       - Add New Face
• Ctrl+R       - Toggle Recognition
• Escape       - Exit Application

Detection Methods:

• haar         - OpenCV Haar Cascades
• dlib         - dlib HOG-based detector
• face_recognition - dlib with better defaults
• mediapipe    - Fast MediaPipe detector

Tips:
• Ensure good lighting for better detection
• Look directly at the camera when capturing
• Move slightly between captures for variety
• Use 'face_recognition' method for best results
        """
        
        info_label = ctk.CTkLabel(
            info_dialog,
            text=info_text.strip(),
            font=("Consolas", 11),
            justify="left",
            anchor="nw"
        )
        info_label.pack(fill="both", expand=True, padx=20, pady=20)
        
        ctk.CTkButton(
            info_dialog,
            text="Close",
            command=info_dialog.destroy,
            width=100
        ).pack(pady=(0, 20))
        
    def create_camera_section(self, parent):
        camera_container = ctk.CTkFrame(parent, fg_color=("#2d2d2d", "#1a1a1a"), corner_radius=15)
        camera_container.pack(fill="both", expand=True, pady=(0, 15))
        
        camera_header = ctk.CTkFrame(camera_container, fg_color="transparent")
        camera_header.pack(fill="x", padx=15, pady=(15, 5))
        
        ctk.CTkLabel(
            camera_header,
            text="📷 Camera Feed",
            font=("Segoe UI", 16, "bold")
        ).pack(side="left")
        
        self.camera_status_indicator = ctk.CTkLabel(
            camera_header,
            text="● OFF",
            text_color="#d63031",
            font=("Segoe UI", 12, "bold")
        )
        self.camera_status_indicator.pack(side="right")
        
        camera_frame = ctk.CTkFrame(camera_container, fg_color="#000000", corner_radius=10)
        camera_frame.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        
        self.camera_label = ctk.CTkLabel(
            camera_frame, 
            text="Camera Off\n\nClick 'Start Camera' to begin",
            fg_color="#000000",
            text_color="#636e72",
            font=("Segoe UI", 14),
            height=400
        )
        self.camera_label.pack(fill="both", expand=True, padx=5, pady=5)
        
    def create_control_panel(self, parent):
        control_container = ctk.CTkFrame(parent, fg_color=("#2d2d2d", "#1a1a1a"), corner_radius=15)
        control_container.pack(fill="x", pady=(0, 15))
        
        control_header = ctk.CTkFrame(control_container, fg_color="transparent")
        control_header.pack(fill="x", padx=15, pady=(15, 10))
        
        ctk.CTkLabel(
            control_header,
            text="⚡ Control Panel",
            font=("Segoe UI", 16, "bold")
        ).pack(side="left")
        
        control_frame = ctk.CTkFrame(control_container, fg_color="transparent")
        control_frame.pack(fill="x", padx=15, pady=(0, 15))
        
        self.camera_btn = ctk.CTkButton(
            control_frame,
            text="▶ Start Camera",
            command=self.toggle_camera,
            width=180,
            height=45,
            font=("Segoe UI", 14, "bold"),
            text_color=("#ffffff", "#000000"),
            fg_color=("#00b894", "#55efc4"),
            hover_color=("#00a381", "#00b894"),
            corner_radius=10
        )
        self.camera_btn.grid(row=0, column=0, padx=8, pady=5)
        
        self.capture_btn = ctk.CTkButton(
            control_frame,
            text="➕ Add New Face",
            command=self.add_face_dialog,
            width=180,
            height=45,
            font=("Segoe UI", 14, "bold"),
            text_color=("#ffffff", "#000000"),
            fg_color=("#0984e3", "#74b9ff"),
            hover_color=("#0074c4", "#0984e3"),
            corner_radius=10,
            state="disabled"
        )
        self.capture_btn.grid(row=0, column=1, padx=8, pady=5)
        
        self.recognize_btn = ctk.CTkButton(
            control_frame,
            text="🎯 Recognize Faces",
            command=self.toggle_recognition,
            width=180,
            height=45,
            font=("Segoe UI", 14, "bold"),
            text_color=("#ffffff", "#000000"),
            fg_color=("#6c5ce7", "#a29bfe"),
            hover_color=("#5b4cc4", "#6c5ce7"),
            corner_radius=10,
            state="disabled"
        )
        self.recognize_btn.grid(row=0, column=2, padx=8, pady=5)
        
        self.capture_progress_label = ctk.CTkLabel(
            control_frame,
            text="",
            font=("Segoe UI", 11),
            text_color="#00b894"
        )
        self.capture_progress_label.grid(row=1, column=0, columnspan=3, pady=5)
        
        for i in range(3):
            control_frame.grid_columnconfigure(i, weight=1)
            
    def create_detection_section(self, parent):
        detection_container = ctk.CTkFrame(parent, fg_color=("#2d2d2d", "#1a1a1a"), corner_radius=15)
        detection_container.pack(fill="x", pady=(0, 15))
        
        detection_header = ctk.CTkFrame(detection_container, fg_color="transparent")
        detection_header.pack(fill="x", padx=15, pady=(15, 10))
        
        ctk.CTkLabel(
            detection_header,
            text="🔍 Detection Settings",
            font=("Segoe UI", 16, "bold")
        ).pack(side="left")
        
        detection_frame = ctk.CTkFrame(detection_container, fg_color="transparent")
        detection_frame.pack(fill="x", padx=15, pady=(0, 15))
        
        ctk.CTkLabel(
            detection_frame,
            text="Detection Method:",
            font=("Segoe UI", 12)
        ).grid(row=0, column=0, padx=(0, 10), pady=10, sticky="w")
        
        self.method_var = ctk.StringVar(value=self.detection_method)
        self.method_selector = ctk.CTkOptionMenu(
            detection_frame,
            values=self.available_methods,
            variable=self.method_var,
            command=self.change_detection_method,
            width=200,
            height=35,
            font=("Segoe UI", 11),
            dropdown_font=("Segoe UI", 11)
        )
        self.method_selector.grid(row=0, column=1, padx=(0, 15), pady=10, sticky="w")
        
        method_info = {
            'haar': 'Basic OpenCV Haar Cascades',
            'dlib': 'HOG-based dlib detector',
            'face_recognition': 'Advanced dlib-based recognition',
            'mediapipe': 'Fast MediaPipe detection'
        }
        
        self.method_info_label = ctk.CTkLabel(
            detection_frame,
            text=method_info.get(self.detection_method, ''),
            font=("Segoe UI", 10),
            text_color="#636e72"
        )
        self.method_info_label.grid(row=0, column=2, pady=10, sticky="w")
        
    def change_detection_method(self, method):
        self.detection_method = method
        method_info = {
            'haar': 'Basic OpenCV Haar Cascades',
            'dlib': 'HOG-based dlib detector',
            'face_recognition': 'Advanced dlib-based recognition',
            'mediapipe': 'Fast MediaPipe detection'
        }
        self.method_info_label.configure(text=method_info.get(method, ''))
        self.status_var.set(f"Detection method: {method}")
        
    def create_face_list_section(self, parent):
        list_container = ctk.CTkFrame(parent, fg_color=("#2d2d2d", "#1a1a1a"), corner_radius=15)
        list_container.pack(fill="both", expand=True, pady=(0, 15))
        
        list_header = ctk.CTkFrame(list_container, fg_color="transparent")
        list_header.pack(fill="x", padx=15, pady=(15, 5))
        
        ctk.CTkLabel(
            list_header,
            text="👥 Registered Faces",
            font=("Segoe UI", 16, "bold")
        ).pack(side="left")
        
        self.face_count_label = ctk.CTkLabel(
            list_header,
            text="0 faces",
            font=("Segoe UI", 12),
            text_color="#636e72"
        )
        self.face_count_label.pack(side="right")
        
        listbox_container = ctk.CTkFrame(list_container, fg_color="#1a1a1a", corner_radius=10)
        listbox_container.pack(fill="both", expand=True, padx=15, pady=(0, 10))
        
        scrollbar_y = ctk.CTkScrollbar(listbox_container, orientation="vertical")
        scrollbar_y.pack(side="right", fill="y", padx=(0, 5), pady=5)
        
        scrollbar_x = ctk.CTkScrollbar(listbox_container, orientation="horizontal")
        scrollbar_x.pack(side="bottom", fill="x", padx=5, pady=(0, 5))
        
        self.face_listbox = ctk.CTkTextbox(
            listbox_container,
            height=150,
            font=("Consolas", 12),
            fg_color="#1a1a1a",
            text_color="#dfe6e9",
            border_width=0,
            scrollbar_button_color=("#00b894", "#55efc4"),
            scrollbar_button_hover_color=("#00a381", "#00b894"),
            yscrollcommand=scrollbar_y.set,
            xscrollcommand=scrollbar_x.set
        )
        self.face_listbox.pack(fill="both", expand=True, padx=5, pady=5, side="left")
        
        scrollbar_y.configure(command=self.face_listbox.yview)
        scrollbar_x.configure(command=self.face_listbox.xview)
        
        list_control_frame = ctk.CTkFrame(list_container, fg_color="transparent")
        list_control_frame.pack(fill="x", padx=15, pady=(0, 15))
        
        ctk.CTkButton(
            list_control_frame,
            text="🗑 Delete Selected",
            command=self.delete_selected_face,
            width=150,
            height=35,
            font=("Segoe UI", 12, "bold"),
            text_color="white",
            fg_color=("#d63031", "#e74c3c"),
            hover_color=("#c0392b", "#d63031"),
            corner_radius=8
        ).grid(row=0, column=0, padx=8, pady=5)
        
        ctk.CTkButton(
            list_control_frame,
            text="🧹 Clear All",
            command=self.clear_all_faces,
            width=150,
            height=35,
            font=("Segoe UI", 12, "bold"),
            text_color="white",
            fg_color=("#636e72", "#74788d"),
            hover_color=("#535b66", "#636e72"),
            corner_radius=8
        ).grid(row=0, column=1, padx=8, pady=5)
        
        export_btn = ctk.CTkButton(
            list_control_frame,
            text="💾 Export Data",
            command=self.export_data,
            width=150,
            height=35,
            font=("Segoe UI", 12, "bold"),
            text_color=("#ffffff", "#000000"),
            fg_color=("#00b894", "#55efc4"),
            hover_color=("#00a381", "#00b894"),
            corner_radius=8
        )
        export_btn.grid(row=0, column=2, padx=8, pady=5)
        
        import_btn = ctk.CTkButton(
            list_control_frame,
            text="📂 Import Data",
            command=self.import_data,
            width=150,
            height=35,
            font=("Segoe UI", 12, "bold"),
            text_color=("#ffffff", "#000000"),
            fg_color=("#0984e3", "#74b9ff"),
            hover_color=("#0074c4", "#0984e3"),
            corner_radius=8
        )
        import_btn.grid(row=0, column=3, padx=8, pady=5)
        
        for i in range(4):
            list_control_frame.grid_columnconfigure(i, weight=1)
            
    def create_status_bar(self, parent):
        status_container = ctk.CTkFrame(parent, fg_color=("#2d2d2d", "#1a1a1a"), corner_radius=10)
        status_container.pack(fill="x", pady=(0, 5))
        
        self.status_var = ctk.StringVar()
        self.status_var.set("Ready - Press 'Start Camera' to begin")
        
        self.status_icon = ctk.CTkLabel(
            status_container,
            text="●",
            text_color="#00b894",
            font=("Segoe UI", 12, "bold")
        )
        self.status_icon.pack(side="left", padx=(15, 5), pady=10)
        
        self.status_bar = ctk.CTkLabel(
            status_container,
            textvariable=self.status_var,
            anchor="w",
            font=("Segoe UI", 11)
        )
        self.status_bar.pack(side="left", fill="x", expand=True, padx=(0, 15), pady=10)
        
    def update_status(self, message, success=True):
        self.status_var.set(message)
        self.status_icon.configure(text_color="#00b894" if success else "#d63031")
        
    def toggle_camera(self):
        if not self.is_camera_on:
            self.start_camera()
        else:
            self.stop_camera()
    
    def start_camera(self):
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Cannot access camera")
                return
                
            self.is_camera_on = True
            self.camera_btn.configure(text="■ Stop Camera")
            self.capture_btn.configure(state="normal")
            self.camera_status_indicator.configure(text="● ON", text_color="#00b894")
            self.update_status("Camera started - You can now add faces or recognize", True)
            
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
            
        self.camera_btn.configure(text="▶ Start Camera")
        self.capture_btn.configure(state="disabled")
        self.recognize_btn.configure(text="🎯 Recognize Faces", state="disabled" if len(self.face_data) == 0 else "normal")
        self.camera_label.configure(text="Camera Off\n\nClick 'Start Camera' to begin", image="")
        self.camera_status_indicator.configure(text="● OFF", text_color="#d63031")
        self.update_status("Camera stopped", False)
        
    def resize_with_aspect_ratio(self, frame, max_width, max_height):
        h, w = frame.shape[:2]
        aspect_ratio = w / h
        
        if w > h:
            new_width = min(max_width, w)
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = min(max_height, h)
            new_width = int(new_height * aspect_ratio)
        
        pil_image = Image.fromarray(frame)
        pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        canvas = Image.new('RGB', (max_width, max_height), (0, 0, 0))
        paste_x = (max_width - new_width) // 2
        paste_y = (max_height - new_height) // 2
        canvas.paste(pil_image, (paste_x, paste_y))
        
        return canvas
    
    def update_video(self):
        while self.is_camera_on:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                
                if self.recognition_active and len(self.face_data) > 0:
                    frame = self.process_recognition(frame)
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                label_w = max(640, self.camera_label.winfo_width() - 20)
                label_h = max(400, self.camera_label.winfo_height() - 20)
                
                frame_pil = self.resize_with_aspect_ratio(frame_rgb, label_w, label_h)
                frame_tk = ImageTk.PhotoImage(frame_pil)
                
                self.camera_label.configure(image=frame_tk, text="")
                self.camera_label.image = frame_tk
                
            self.root.after(30)
    
    def detect_faces(self, frame, gray):
        faces = []
        
        if self.detection_method == 'haar':
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            faces = [(x, y, w, h) for (x, y, w, h) in faces]
            
        elif self.detection_method == 'dlib' and DLIB_AVAILABLE:
            dlib_faces = self.dlib_detector(gray, 1)
            faces = [(face.left(), face.top(), 
                     face.right() - face.left(), 
                     face.bottom() - face.top()) 
                    for face in dlib_faces]
            
        elif self.detection_method == 'face_recognition' and FACE_RECOGNITION_AVAILABLE:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame, model='hog')
            faces = [(left, top, right - left, bottom - top) 
                    for (top, right, bottom, left) in face_locations]
            
        elif self.detection_method == 'mediapipe' and MEDIAPIPE_AVAILABLE:
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
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            faces = [(x, y, w, h) for (x, y, w, h) in faces]
        
        return faces
            
    def process_recognition(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detect_faces(frame, gray)
        
        for (x, y, w, h) in faces:
            face_region = gray[y:y+h, x:x+w]
            face_region = cv2.resize(face_region, (100, 100))
            
            if len(self.face_data) > 0:
                label, confidence = self.face_recognizer.predict(face_region)
                
                if confidence < 100:
                    name = self.id_to_name.get(label, "Unknown")
                    confidence_text = f"{name} ({100-confidence:.1f}%)"
                    color = (0, 255, 0)
                else:
                    confidence_text = "Unknown"
                    color = (0, 0, 255)
                    
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, confidence_text, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            else:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, "No training data", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                
        return frame
        
    def add_face_dialog(self):
        if not self.is_camera_on:
            messagebox.showwarning("Warning", "Please start the camera first")
            return
            
        dialog = ctk.CTkToplevel(self.root)
        dialog.title("Add New Face")
        dialog.geometry("400x220")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(False, False)
        
        header_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        header_frame.pack(fill="x", padx=20, pady=(20, 10))
        
        ctk.CTkLabel(
            header_frame,
            text="👤 Add New Face",
            font=("Segoe UI", 16, "bold")
        ).pack(side="left")
        
        ctk.CTkLabel(
            dialog,
            text="Enter the person's name to register:",
            font=("Segoe UI", 11),
            text_color="#b2bec3"
        ).pack(anchor="w", padx=20, pady=(10, 5))
        
        name_var = ctk.StringVar()
        name_entry = ctk.CTkEntry(
            dialog,
            textvariable=name_var,
            width=350,
            height=40,
            font=("Segoe UI", 14),
            placeholder_text="Enter name here..."
        )
        name_entry.pack(pady=10)
        name_entry.focus()
        
        button_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        button_frame.pack(pady=(10, 20))
        
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
            
        ctk.CTkButton(
            button_frame,
            text="Start Capture",
            command=capture_face,
            width=130,
            height=40,
            font=("Segoe UI", 13, "bold"),
            text_color=("#ffffff", "#000000"),
            fg_color=("#00b894", "#55efc4"),
            hover_color=("#00a381", "#00b894"),
            corner_radius=8
        ).pack(side="left", padx=8)
        
        ctk.CTkButton(
            button_frame,
            text="Cancel",
            command=dialog.destroy,
            width=130,
            height=40,
            font=("Segoe UI", 13, "bold"),
            text_color="white",
            fg_color=("#636e72", "#74788d"),
            hover_color=("#535b66", "#636e72"),
            corner_radius=8
        ).pack(side="left", padx=8)
        
        dialog.bind('<Return>', lambda e: capture_face())
        dialog.bind('<Escape>', lambda e: dialog.destroy())
        
    def capture_face_samples(self, name):
        if not self.cap:
            return
            
        samples_captured = 0
        target_samples = 20
        
        self.capture_in_progress = True
        self.capture_btn.configure(state="disabled", text="⏳ Capturing...")
        self.update_status(f"Capturing samples for {name}... Look at camera", True)
        
        captured_faces = []
        start_time = time.time()
        
        while samples_captured < target_samples and self.is_camera_on and self.capture_in_progress:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.detect_faces(frame, gray)
                
                for (x, y, w, h) in faces:
                    face_region = gray[y:y+h, x:x+w]
                    face_region = cv2.resize(face_region, (100, 100))
                    captured_faces.append(face_region)
                    
                    samples_captured += 1
                    elapsed = int(time.time() - start_time)
                    self.capture_progress_label.configure(
                        text=f"Capturing: {samples_captured}/{target_samples} | Time: {elapsed}s"
                    )
                    
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, f"Sample {samples_captured}/{target_samples}", 
                               (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    break
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = self.resize_with_aspect_ratio(frame_rgb, 640, 400)
                frame_tk = ImageTk.PhotoImage(frame_pil)
                
                self.camera_label.configure(image=frame_tk, text="")
                self.camera_label.image = frame_tk
                
                self.root.update()
                
            if samples_captured >= target_samples:
                break
                
        self.capture_in_progress = False
        self.capture_btn.configure(state="normal", text="➕ Add New Face")
        self.capture_progress_label.configure(text="")
        
        if captured_faces:
            if name not in self.name_to_id:
                person_id = len(self.name_to_id)
                self.name_to_id[name] = person_id
                self.id_to_name[person_id] = name
            else:
                person_id = self.name_to_id[name]
                
            for face in captured_faces:
                self.face_data.append(face)
                self.face_labels.append(person_id)
                
            self.train_recognizer()
            self.save_data()
            self.update_face_list()
            
            if len(self.face_data) > 0:
                self.recognize_btn.configure(state="normal")
            
            self.update_status(f"Successfully added {len(captured_faces)} samples for {name}", True)
            messagebox.showinfo("Success", f"Added {len(captured_faces)} face samples for {name}")
        else:
            self.update_status("No face detected during capture", False)
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
            self.recognize_btn.configure(text="■ Stop Recognition")
            self.update_status("Face recognition active - Detecting faces...", True)
        else:
            self.recognize_btn.configure(text="🎯 Recognize Faces")
            self.update_status("Face recognition stopped", False)
            
    def update_face_list(self):
        self.face_listbox.delete("1.0", "end")
        
        person_counts = {}
        for label in self.face_labels:
            label_int = int(label)
            if label_int in self.id_to_name:
                name = self.id_to_name[label_int]
                person_counts[name] = person_counts.get(name, 0) + 1
        
        if not person_counts:
            self.face_listbox.insert("1.0", "No faces registered yet.\n\nAdd faces using 'Add New Face' button.")
            self.face_count_label.configure(text="0 faces")
        else:
            for name, count in person_counts.items():
                self.face_listbox.insert("end", f"👤 {name} ({count} samples)\n")
            self.face_listbox.insert("end", f"\nTotal: {len(person_counts)} people, {len(self.face_data)} samples")
            self.face_count_label.configure(text=f"{len(person_counts)} people, {len(self.face_data)} samples")
            
    def delete_selected_face(self):
        try:
            selection = self.face_listbox.tag_ranges("sel")
            if selection:
                selected_text = self.face_listbox.get("sel.first", "sel.last").strip()
                if selected_text.startswith("👤 "):
                    name = selected_text.split(" (")[0].replace("👤 ", "")
                elif selected_text.startswith("Total:"):
                    messagebox.showwarning("Warning", "Please select a person, not the total line")
                    return
                else:
                    return
            else:
                messagebox.showwarning("Warning", "Please select a person to delete")
                return
        except:
            messagebox.showwarning("Warning", "Please select a person to delete")
            return
            
        if messagebox.askyesno("Confirm", f"Delete all data for '{name}'?"):
            person_id = self.name_to_id[name]
            
            new_face_data = []
            new_face_labels = []
            
            for i, label in enumerate(self.face_labels):
                if int(label) != person_id:
                    new_face_data.append(self.face_data[i])
                    new_face_labels.append(label)
                    
            self.face_data = new_face_data
            self.face_labels = new_face_labels
            
            del self.name_to_id[name]
            del self.id_to_name[person_id]
            
            if len(self.face_data) > 0:
                self.train_recognizer()
            else:
                self.recognize_btn.configure(state="disabled")
                
            self.save_data()
            self.update_face_list()
            
            self.update_status(f"Deleted data for {name}", True)
            
    def clear_all_faces(self):
        if messagebox.askyesno("Confirm", "Delete all face data?"):
            self.face_data = []
            self.face_labels = []
            self.name_to_id = {}
            self.id_to_name = {}
            
            self.recognize_btn.configure(state="disabled", text="🎯 Recognize Faces")
            self.recognition_active = False
            
            self.save_data()
            self.update_face_list()
            
            self.update_status("All face data cleared", False)
            
    def export_data(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Export Face Data"
        )
        
        if file_path:
            try:
                export_data = {
                    'name_to_id': self.name_to_id,
                    'id_to_name': {str(k): v for k, v in self.id_to_name.items()},
                    'face_count': len(self.face_data),
                    'exported_at': time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
                with open(file_path, 'w') as f:
                    json.dump(export_data, f, indent=2)
                    
                self.update_status(f"Data exported to {os.path.basename(file_path)}", True)
                messagebox.showinfo("Success", f"Data exported successfully to:\n{file_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export data: {str(e)}")
                
    def import_data(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Import Face Data"
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    import_data = json.load(f)
                    
                self.name_to_id = import_data.get('name_to_id', {})
                self.id_to_name = {int(k): v for k, v in import_data.get('id_to_name', {}).items()}
                
                if os.path.exists('face_data_opencv.npy') and os.path.exists('face_labels_opencv.npy'):
                    face_data_loaded = np.load('face_data_opencv.npy')
                    face_labels_loaded = np.load('face_labels_opencv.npy')
                    
                    self.face_data = [face for face in face_data_loaded]
                    self.face_labels = [int(label) for label in face_labels_loaded]
                    
                    if len(self.face_data) > 0:
                        self.train_recognizer()
                        self.recognize_btn.configure(state="normal")
                        
                self.update_face_list()
                self.update_status(f"Data imported from {os.path.basename(file_path)}", True)
                messagebox.showinfo("Success", "Data imported successfully!")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to import data: {str(e)}")
                
    def save_data(self):
        try:
            data = {
                'name_to_id': self.name_to_id,
                'id_to_name': {str(k): v for k, v in self.id_to_name.items()},
                'face_count': len(self.face_data)
            }
            
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
                
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
                
                if os.path.exists('face_data_opencv.npy') and os.path.exists('face_labels_opencv.npy'):
                    face_data_loaded = np.load('face_data_opencv.npy')
                    face_labels_loaded = np.load('face_labels_opencv.npy')
                    
                    self.face_data = [face for face in face_data_loaded]
                    self.face_labels = [int(label) for label in face_labels_loaded]
                    
                    if len(self.face_data) > 0:
                        self.train_recognizer()
                        
        except Exception as e:
            print(f"Failed to load data: {str(e)}")
            self.face_data = []
            self.face_labels = []
            self.name_to_id = {}
            self.id_to_name = {}
            
    def on_closing(self):
        self.capture_in_progress = False
        self.stop_camera()
        if self.mp_face_detector:
            self.mp_face_detector.close()
        self.root.destroy()


def main():
    root = ctk.CTk()
    app = FaceRecognitionApp(root)
    
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()


if __name__ == "__main__":
    main()
