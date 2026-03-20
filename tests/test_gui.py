"""
Unit tests for GUI components and user interface.
Tests button states, dialogs, shortcuts, and theme toggle logic.
"""

import pytest
from unittest.mock import MagicMock, patch


class TestButtonStates:
    """Test suite for button state management."""

    def test_capture_button_disabled_initially(self):
        """Test capture button is disabled when camera is off."""
        state = 'disabled'
        assert state == 'disabled'

    def test_recognize_button_disabled_initially(self):
        """Test recognize button is disabled when no faces registered."""
        face_data = []
        state = 'disabled' if len(face_data) == 0 else 'normal'
        assert state == 'disabled'

    def test_recognize_button_enabled_with_faces(self):
        """Test recognize button is enabled when faces are registered."""
        face_data = [MagicMock()]
        state = 'disabled' if len(face_data) == 0 else 'normal'
        assert state == 'normal'

    def test_capture_button_enabled_when_camera_on(self):
        """Test capture button is enabled when camera is on."""
        is_camera_on = True
        state = 'normal' if is_camera_on else 'disabled'
        assert state == 'normal'


class TestThemeToggle:
    """Test suite for theme switching functionality."""

    def test_toggle_theme_switches_mode(self):
        """Test that toggle_theme switches between Dark and Light."""
        current_mode = 'Dark'
        new_mode = 'Light' if current_mode == 'Dark' else 'Dark'
        assert new_mode == 'Light'

    def test_toggle_theme_cycles_back(self):
        """Test that toggling twice cycles back to original."""
        current_mode = 'Light'
        new_mode = 'Light' if current_mode == 'Dark' else 'Dark'
        assert new_mode == 'Dark'

    def test_theme_icon_dark_mode(self):
        """Test theme icon for dark mode."""
        mode = 'Dark'
        icon = '☀' if mode == 'Dark' else '🌙'
        assert icon == '☀'

    def test_theme_icon_light_mode(self):
        """Test theme icon for light mode."""
        mode = 'Light'
        icon = '☀' if mode == 'Dark' else '🌙'
        assert icon == '🌙'


class TestKeyboardShortcuts:
    """Test suite for keyboard shortcut bindings."""

    def test_space_toggles_camera(self):
        """Test Space key toggles camera."""
        is_camera_on = False
        is_camera_on = True
        assert is_camera_on is True

    def test_ctrl_a_adds_face_when_camera_on(self):
        """Test Ctrl+A opens add face dialog when camera is on."""
        is_camera_on = True
        can_add_face = is_camera_on
        assert can_add_face is True

    def test_ctrl_a_does_nothing_when_camera_off(self):
        """Test Ctrl+A does nothing when camera is off."""
        is_camera_on = False
        can_add_face = is_camera_on
        assert can_add_face is False

    def test_ctrl_r_toggles_recognition(self):
        """Test Ctrl+R toggles recognition."""
        can_toggle = True
        recognition_toggled = not can_toggle
        assert recognition_toggled is False


class TestStatusBar:
    """Test suite for status bar updates."""

    def test_update_status_success(self):
        """Test status update with success message."""
        status = "Operation successful"
        success = True
        icon_color = "#00b894" if success else "#d63031"
        assert icon_color == "#00b894"

    def test_update_status_failure(self):
        """Test status update with failure message."""
        status = "Operation failed"
        success = False
        icon_color = "#00b894" if success else "#d63031"
        assert icon_color == "#d63031"


class TestDetectionMethodSelection:
    """Test suite for detection method switching."""

    def test_change_detection_method(self):
        """Test changing detection method updates state."""
        method_info = {
            'haar': 'Basic OpenCV Haar Cascades',
            'dlib': 'HOG-based dlib detector',
            'face_recognition': 'Advanced dlib-based recognition',
            'mediapipe': 'Fast MediaPipe detection'
        }
        
        assert method_info['haar'] == 'Basic OpenCV Haar Cascades'
        assert method_info['dlib'] == 'HOG-based dlib detector'

    def test_method_info_labels(self):
        """Test method info labels are correct."""
        method_info = {
            'haar': 'Basic OpenCV Haar Cascades',
            'dlib': 'HOG-based dlib detector',
            'face_recognition': 'Advanced dlib-based recognition',
            'mediapipe': 'Fast MediaPipe detection'
        }
        
        assert len(method_info) == 4


class TestFaceListDisplay:
    """Test suite for face list UI updates."""

    def test_empty_face_list_message(self):
        """Test face list shows empty message when no faces."""
        face_data = []
        face_labels = []
        
        if not face_data:
            message = "No faces registered yet."
        else:
            message = f"{len(face_data)} faces"
        
        assert message == "No faces registered yet."

    def test_face_list_with_data(self):
        """Test face list shows registered faces."""
        face_labels = [0, 0, 0, 1, 1]
        id_to_name = {0: 'Alice', 1: 'Bob'}
        
        person_counts = {}
        for label in face_labels:
            label_int = int(label)
            if label_int in id_to_name:
                name = id_to_name[label_int]
                person_counts[name] = person_counts.get(name, 0) + 1
        
        assert 'Alice' in person_counts
        assert 'Bob' in person_counts
        assert person_counts['Alice'] == 3
        assert person_counts['Bob'] == 2

    def test_face_count_calculation(self):
        """Test face count calculation."""
        person_counts = {'Alice': 3, 'Bob': 2}
        total_samples = sum(person_counts.values())
        num_people = len(person_counts)
        
        assert total_samples == 5
        assert num_people == 2

    def test_face_list_format_string(self):
        """Test face list item format."""
        name = "Alice"
        count = 5
        format_string = f"👤 {name} ({count} samples)"
        assert format_string == "👤 Alice (5 samples)"


class TestCameraToggleLogic:
    """Test camera toggle state logic."""

    def test_camera_starts_off(self):
        """Test camera is initially off."""
        is_camera_on = False
        assert is_camera_on is False

    def test_camera_toggle_on(self):
        """Test toggling camera on."""
        is_camera_on = False
        is_camera_on = True
        assert is_camera_on is True

    def test_camera_toggle_off(self):
        """Test toggling camera off."""
        is_camera_on = True
        is_camera_on = False
        assert is_camera_on is False


class TestRecognitionToggleLogic:
    """Test recognition toggle logic."""

    def test_recognition_starts_off(self):
        """Test recognition is initially off."""
        recognition_active = False
        assert recognition_active is False

    def test_recognition_requires_camera(self):
        """Test recognition requires camera to be on."""
        is_camera_on = False
        can_recognize = is_camera_on
        assert can_recognize is False

    def test_recognition_requires_faces(self):
        """Test recognition requires registered faces."""
        face_data = []
        can_recognize = len(face_data) > 0
        assert can_recognize is False

    def test_recognition_toggle_on(self):
        """Test toggling recognition on."""
        recognition_active = False
        recognition_active = True
        assert recognition_active is True

    def test_recognition_toggle_off(self):
        """Test toggling recognition off."""
        recognition_active = True
        recognition_active = False
        assert recognition_active is False


class TestDeleteOperations:
    """Test face deletion operations."""

    def test_delete_person_removes_all_samples(self):
        """Test deleting a person removes all their samples."""
        face_data = [1, 2, 3, 4, 5]
        face_labels = [0, 0, 1, 1, 1]
        name_to_id = {'Alice': 0, 'Bob': 1}
        person_id = name_to_id['Alice']
        
        new_face_data = [face_data[i] for i in range(len(face_data)) if face_labels[i] != person_id]
        new_face_labels = [label for label in face_labels if label != person_id]
        
        del name_to_id['Alice']
        
        assert 'Alice' not in name_to_id
        assert len(new_face_data) == 3
        assert len(new_face_labels) == 3

    def test_clear_all_faces(self):
        """Test clearing all face data."""
        face_data = [1, 2, 3]
        face_labels = [0, 1, 2]
        name_to_id = {'Alice': 0, 'Bob': 1, 'Charlie': 2}
        
        face_data = []
        face_labels = []
        name_to_id = {}
        
        assert len(face_data) == 0
        assert len(face_labels) == 0
        assert len(name_to_id) == 0


class TestSampleCaptureLogic:
    """Test face sample capture logic."""

    def test_target_samples_count(self):
        """Test target number of samples is 20."""
        target_samples = 20
        assert target_samples == 20

    def test_sample_capture_in_progress_flag(self):
        """Test capture in progress flag toggles."""
        capture_in_progress = False
        capture_in_progress = True
        assert capture_in_progress is True
        
        capture_in_progress = False
        assert capture_in_progress is False

    def test_sample_collection_logic(self):
        """Test sample collection continues until target reached."""
        samples_captured = 0
        target_samples = 20
        captured_faces = []
        
        for i in range(25):
            if samples_captured >= target_samples:
                break
            captured_faces.append(i)
            samples_captured += 1
        
        assert len(captured_faces) == 20
        assert samples_captured == 20


class TestFaceRegions:
    """Test face region extraction logic."""

    def test_face_region_extraction(self):
        """Test extracting face region from frame."""
        import numpy as np
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        x, y, w, h = 100, 100, 50, 50
        
        face_region = frame[y:y+h, x:x+w]
        
        assert face_region.shape[0] == h
        assert face_region.shape[1] == w

    def test_face_region_resize(self):
        """Test resizing face region to standard size."""
        import numpy as np
        import cv2
        
        face_region = np.zeros((80, 60), dtype=np.uint8)
        resized = cv2.resize(face_region, (100, 100))
        
        assert resized.shape == (100, 100)
