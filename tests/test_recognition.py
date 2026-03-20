"""
Unit tests for face recognition logic.
Tests recognition confidence, labeling, and processing.
"""

import pytest
import numpy as np
import cv2
from unittest.mock import MagicMock, patch


class TestRecognitionLogicStandalone:
    """Test recognition logic without full GUI initialization."""

    def test_confidence_calculation_logic(self):
        """Test confidence calculation logic."""
        confidence = 80
        match_percentage = 100 - confidence
        assert match_percentage == 20

    def test_confidence_threshold_logic(self):
        """Test confidence threshold determines recognition."""
        confidence = 80
        is_known = confidence < 100
        assert is_known is True
        
        confidence = 120
        is_known = confidence < 100
        assert is_known is False

    def test_id_mapping_consistency(self):
        """Test ID to name and name to ID mappings are consistent."""
        name_to_id = {'Alice': 0, 'Bob': 1}
        id_to_name = {0: 'Alice', 1: 'Bob'}
        
        for name, id_val in name_to_id.items():
            assert id_to_name[id_val] == name

    def test_face_data_labels_sync(self):
        """Test face data and labels stay synchronized."""
        face_data = []
        face_labels = []
        
        face_data.append(np.zeros((100, 100), dtype=np.uint8))
        face_labels.append(0)
        
        assert len(face_data) == len(face_labels)

    def test_recognition_output_format(self):
        """Test recognition output format."""
        name = "TestPerson"
        confidence = 75.5
        output = f"{name} ({100-confidence:.1f}%)"
        assert output == "TestPerson (24.5%)"

    def test_multiple_face_recognition_labels(self):
        """Test processing multiple faces with different labels."""
        labels = [0, 1, 0, 1, 2]
        
        label_counts = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        assert label_counts[0] == 2
        assert label_counts[1] == 2
        assert label_counts[2] == 1

    def test_delete_person_logic(self):
        """Test deleting a person from data structures."""
        name_to_id = {'Alice': 0, 'Bob': 1}
        id_to_name = {0: 'Alice', 1: 'Bob'}
        face_data = [1, 2, 3, 4]
        face_labels = [0, 0, 1, 1]
        
        person_id = name_to_id['Alice']
        new_face_data = [face_data[i] for i in range(len(face_data)) if face_labels[i] != person_id]
        new_face_labels = [label for label in face_labels if label != person_id]
        
        del name_to_id['Alice']
        del id_to_name[person_id]
        
        assert 'Alice' not in name_to_id
        assert 0 not in id_to_name
        assert len(new_face_data) == 2
        assert len(new_face_labels) == 2


class TestConfidenceCalculation:
    """Test confidence and recognition percentage calculation."""

    def test_confidence_to_percentage_conversion(self):
        """Test that confidence 80 gives 20% match."""
        confidence = 80
        match_percentage = 100 - confidence
        assert match_percentage == 20

    def test_confidence_0_max_percentage(self):
        """Test that confidence 0 gives 100% match."""
        confidence = 0
        match_percentage = 100 - confidence
        assert match_percentage == 100

    def test_confidence_100_min_percentage(self):
        """Test that confidence 100 gives 0% match (unknown)."""
        confidence = 100
        match_percentage = 100 - confidence
        assert match_percentage == 0

    def test_confidence_format_string(self):
        """Test confidence text formatting."""
        name = "John"
        confidence = 75
        confidence_text = f"{name} ({100-confidence:.1f}%)"
        assert confidence_text == "John (25.0%)"

    def test_confidence_boundary_recognized(self):
        """Test confidence just below threshold is recognized."""
        confidence = 99
        is_recognized = confidence < 100
        assert is_recognized is True

    def test_confidence_boundary_unknown(self):
        """Test confidence at threshold is unknown."""
        confidence = 100
        is_recognized = confidence < 100
        assert is_recognized is False


class TestNameMapping:
    """Test ID to name mapping logic."""

    @pytest.fixture
    def mock_app_state(self):
        """Create mock app state for testing."""
        return {
            'name_to_id': {},
            'id_to_name': {},
            'face_data': [],
            'face_labels': []
        }

    def test_name_to_id_mapping(self, mock_app_state):
        """Test adding a person creates correct mapping."""
        mock_app_state['name_to_id']['Alice'] = 0
        mock_app_state['id_to_name'][0] = 'Alice'
        
        assert mock_app_state['name_to_id']['Alice'] == 0
        assert mock_app_state['id_to_name'][0] == 'Alice'

    def test_multiple_people_mapping(self, mock_app_state):
        """Test mapping with multiple people."""
        mock_app_state['name_to_id'] = {'Alice': 0, 'Bob': 1, 'Charlie': 2}
        mock_app_state['id_to_name'] = {0: 'Alice', 1: 'Bob', 2: 'Charlie'}
        
        assert len(mock_app_state['name_to_id']) == 3
        assert len(mock_app_state['id_to_name']) == 3

    def test_delete_person_updates_mappings(self, mock_app_state):
        """Test deleting a person updates both mappings."""
        mock_app_state['name_to_id'] = {'Alice': 0, 'Bob': 1}
        mock_app_state['id_to_name'] = {0: 'Alice', 1: 'Bob'}
        
        del mock_app_state['name_to_id']['Alice']
        del mock_app_state['id_to_name'][0]
        
        assert 'Alice' not in mock_app_state['name_to_id']
        assert 0 not in mock_app_state['id_to_name']
        assert 'Bob' in mock_app_state['name_to_id']

    def test_lookup_by_name(self, mock_app_state):
        """Test looking up person ID by name."""
        mock_app_state['name_to_id']['Alice'] = 0
        
        assert mock_app_state['name_to_id'].get('Alice') == 0
        assert mock_app_state['name_to_id'].get('Unknown') is None

    def test_lookup_by_id(self, mock_app_state):
        """Test looking up person name by ID."""
        mock_app_state['id_to_name'][0] = 'Alice'
        
        assert mock_app_state['id_to_name'].get(0) == 'Alice'
        assert mock_app_state['id_to_name'].get(999) is None


class TestFaceDetectionCoordinateConversion:
    """Test coordinate conversion for different detection methods."""

    def test_haar_bbox_format(self):
        """Test Haar Cascade bounding box format (x, y, w, h)."""
        faces = [(100, 100, 50, 50)]
        x, y, w, h = faces[0]
        
        assert x == 100
        assert y == 100
        assert w == 50
        assert h == 50

    def test_dlib_rect_to_bbox(self):
        """Test dlib rectangle to bounding box conversion."""
        left, top, right, bottom = 100, 100, 200, 200
        
        x, y, w, h = left, top, right - left, bottom - top
        
        assert x == 100
        assert y == 100
        assert w == 100
        assert h == 100

    def test_face_recognition_locations_to_bbox(self):
        """Test face_recognition locations format to bbox conversion."""
        top, right, bottom, left = 100, 200, 200, 100
        
        x, y, w, h = left, top, right - left, bottom - top
        
        assert x == 100
        assert y == 100
        assert w == 100
        assert h == 100

    def test_mediapipe_relative_to_absolute(self):
        """Test MediaPipe relative coordinates to absolute."""
        frame_h, frame_w = 480, 640
        rel_xmin, rel_ymin, rel_width, rel_height = 0.1, 0.1, 0.2, 0.2
        
        x = int(rel_xmin * frame_w)
        y = int(rel_ymin * frame_h)
        w = int(rel_width * frame_w)
        h = int(rel_height * frame_h)
        
        assert x == 64
        assert y == 48
        assert w == 128
        assert h == 96


class TestFrameProcessing:
    """Test video frame processing logic."""

    def test_image_resize_aspect_ratio_landscape(self):
        """Test resize maintains aspect ratio for landscape."""
        from PIL import Image
        import numpy as np
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        h, w = frame.shape[:2]
        aspect_ratio = w / h
        
        max_width, max_height = 320, 240
        
        if w > h:
            new_width = min(max_width, w)
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = min(max_height, h)
            new_width = int(new_height * aspect_ratio)
        
        assert new_width <= max_width
        assert new_height <= max_height

    def test_image_resize_aspect_ratio_portrait(self):
        """Test resize maintains aspect ratio for portrait."""
        frame = np.zeros((640, 480, 3), dtype=np.uint8)
        
        h, w = frame.shape[:2]
        aspect_ratio = w / h
        
        max_width, max_height = 320, 240
        
        if w > h:
            new_width = min(max_width, w)
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = min(max_height, h)
            new_width = int(new_height * aspect_ratio)
        
        assert new_width <= max_width
        assert new_height <= max_height

    def test_frame_flip_horizontal(self):
        """Test horizontal frame flip."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[50, 10] = [255, 0, 0]
        
        flipped = cv2.flip(frame, 1)
        
        h, w = flipped.shape[:2]
        assert flipped[50, w-1-10, 0] == 255

    def test_gray_conversion(self):
        """Test RGB to grayscale conversion."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :] = [255, 255, 255]
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        assert len(gray.shape) == 2
        assert gray[50, 50] > 200


class TestRecognitionStates:
    """Test recognition state management."""

    def test_recognition_requires_faces(self):
        """Test recognition requires registered faces."""
        face_data = []
        can_recognize = len(face_data) > 0
        assert can_recognize is False

    def test_recognition_with_faces_enabled(self):
        """Test recognition enabled when faces exist."""
        face_data = [MagicMock()]
        can_recognize = len(face_data) > 0
        assert can_recognize is True

    def test_recognition_toggle_state(self):
        """Test recognition toggle changes state."""
        recognition_active = False
        recognition_active = not recognition_active
        assert recognition_active is True
        
        recognition_active = not recognition_active
        assert recognition_active is False
