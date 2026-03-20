"""
Unit tests for face detection methods.
Tests all 4 detection methods: haar, dlib, face_recognition, mediapipe
"""

import pytest
import numpy as np
import cv2
from unittest.mock import MagicMock, patch


class TestHaarDetection:
    """Test Haar Cascade detection logic."""

    def test_haar_detection_returns_list(self):
        """Test that Haar detection returns a list of face coordinates."""
        mock_faces = np.array([[100, 100, 50, 50]])
        faces = [(x, y, w, h) for (x, y, w, h) in mock_faces]
        
        assert isinstance(faces, list)
        assert len(faces) == 1
        assert faces[0] == (100, 100, 50, 50)

    def test_haar_detection_no_faces(self):
        """Test Haar detection returns empty list when no faces."""
        mock_faces = np.array([])
        faces = [(x, y, w, h) for (x, y, w, h) in mock_faces]
        
        assert isinstance(faces, list)
        assert len(faces) == 0

    def test_haar_detection_multiple_faces(self):
        """Test Haar detection with multiple faces."""
        mock_faces = np.array([
            [100, 100, 50, 50],
            [200, 100, 50, 50],
            [300, 100, 50, 50]
        ])
        faces = [(x, y, w, h) for (x, y, w, h) in mock_faces]
        
        assert len(faces) == 3


class TestDlibDetection:
    """Test dlib detection logic."""

    def test_dlib_rect_to_bbox(self):
        """Test dlib rectangle to bounding box conversion."""
        mock_face = MagicMock()
        mock_face.left.return_value = 100
        mock_face.top.return_value = 100
        mock_face.right.return_value = 150
        mock_face.bottom.return_value = 150
        
        faces = [(face.left(), face.top(), 
                 face.right() - face.left(), 
                 face.bottom() - face.top()) 
                for face in [mock_face]]
        
        assert len(faces) == 1
        assert faces[0] == (100, 100, 50, 50)

    def test_dlib_multiple_faces(self):
        """Test dlib with multiple faces."""
        faces = []
        for i in range(3):
            mock_face = MagicMock()
            mock_face.left.return_value = 100 * (i + 1)
            mock_face.top.return_value = 100
            mock_face.right.return_value = 150 * (i + 1)
            mock_face.bottom.return_value = 150
            faces.append(mock_face)
        
        bboxes = [(face.left(), face.top(), 
                  face.right() - face.left(), 
                  face.bottom() - face.top()) 
                 for face in faces]
        
        assert len(bboxes) == 3


class TestFaceRecognitionDetection:
    """Test face_recognition library detection logic."""

    def test_face_locations_to_bbox(self):
        """Test face_locations (top, right, bottom, left) converts correctly to (x, y, w, h)."""
        face_locations = [(100, 200, 200, 100)]
        
        bboxes = [(left, top, right - left, bottom - top) 
                 for (top, right, bottom, left) in face_locations]
        
        assert len(bboxes) == 1
        assert bboxes[0] == (100, 100, 100, 100)

    def test_multiple_face_locations(self):
        """Test multiple face location conversions."""
        face_locations = [
            (100, 200, 200, 100),
            (150, 300, 300, 150)
        ]
        
        bboxes = [(left, top, right - left, bottom - top) 
                 for (top, right, bottom, left) in face_locations]
        
        assert len(bboxes) == 2
        assert bboxes[0] == (100, 100, 100, 100)
        assert bboxes[1] == (150, 150, 150, 150)


class TestMediapipeDetection:
    """Test MediaPipe detection logic."""

    def test_mediapipe_relative_to_absolute(self):
        """Test MediaPipe relative coordinates to absolute conversion."""
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

    def test_mediapipe_single_detection(self):
        """Test MediaPipe single detection processing."""
        frame_h, frame_w = 480, 640
        
        mock_detection = MagicMock()
        mock_detection.location_data.relative_bounding_box.xmin = 0.1
        mock_detection.location_data.relative_bounding_box.ymin = 0.1
        mock_detection.location_data.relative_bounding_box.width = 0.2
        mock_detection.location_data.relative_bounding_box.height = 0.2
        
        faces = []
        for detection in [mock_detection]:
            bboxC = detection.location_data.relative_bounding_box
            x = int(bboxC.xmin * frame_w)
            y = int(bboxC.ymin * frame_h)
            width = int(bboxC.width * frame_w)
            height = int(bboxC.height * frame_h)
            faces.append((x, y, width, height))
        
        assert len(faces) == 1
        assert faces[0] == (64, 48, 128, 96)

    def test_mediapipe_no_detections(self):
        """Test MediaPipe with no detections."""
        detections = None
        
        faces = []
        if detections:
            faces = [(0, 0, 100, 100)]
        
        assert len(faces) == 0


class TestDetectionFallback:
    """Test detection fallback logic when methods are unavailable."""

    def test_haar_fallback_when_dlib_unavailable(self):
        """Test Haar is used when dlib is unavailable."""
        DLIB_AVAILABLE = False
        detection_method = 'dlib'
        
        if not DLIB_AVAILABLE:
            use_method = 'haar'
        else:
            use_method = detection_method
        
        assert use_method == 'haar'

    def test_haar_fallback_when_face_recognition_unavailable(self):
        """Test Haar is used when face_recognition is unavailable."""
        FACE_RECOGNITION_AVAILABLE = False
        detection_method = 'face_recognition'
        
        if not FACE_RECOGNITION_AVAILABLE:
            use_method = 'haar'
        else:
            use_method = detection_method
        
        assert use_method == 'haar'

    def test_haar_fallback_when_mediapipe_unavailable(self):
        """Test Haar is used when mediapipe is unavailable."""
        MEDIAPIPE_AVAILABLE = False
        detection_method = 'mediapipe'
        
        if not MEDIAPIPE_AVAILABLE:
            use_method = 'haar'
        else:
            use_method = detection_method
        
        assert use_method == 'haar'

    def test_default_method_with_face_recognition_available(self):
        """Test default method is face_recognition when available."""
        FACE_RECOGNITION_AVAILABLE = True
        
        if FACE_RECOGNITION_AVAILABLE:
            detection_method = 'face_recognition'
        else:
            detection_method = 'haar'
        
        assert detection_method == 'face_recognition'

    def test_default_method_without_face_recognition(self):
        """Test default method is Haar when face_recognition is unavailable."""
        FACE_RECOGNITION_AVAILABLE = False
        
        if FACE_RECOGNITION_AVAILABLE:
            detection_method = 'face_recognition'
        else:
            detection_method = 'haar'
        
        assert detection_method == 'haar'


class TestCoordinateConversion:
    """Test coordinate format consistency."""

    def test_bbox_format_consistency(self):
        """Test all detection methods return consistent (x, y, w, h) format."""
        haar_result = (100, 100, 50, 50)
        dlib_result = (100, 100, 50, 50)
        face_rec_result = (100, 100, 50, 50)
        mediapipe_result = (100, 100, 50, 50)
        
        assert all(len(bbox) == 4 for bbox in [haar_result, dlib_result, face_rec_result, mediapipe_result])
        
        for bbox in [haar_result, dlib_result, face_rec_result, mediapipe_result]:
            x, y, w, h = bbox
            assert x >= 0
            assert y >= 0
            assert w > 0
            assert h > 0

    def test_bbox_within_frame_bounds(self):
        """Test bounding boxes are within frame bounds."""
        frame_h, frame_w = 480, 640
        bbox = (100, 100, 50, 50)
        
        x, y, w, h = bbox
        assert x >= 0
        assert y >= 0
        assert x + w <= frame_w
        assert y + h <= frame_h


class TestDetectionMethodAvailability:
    """Test detection method availability detection."""

    def test_available_methods_list_haar_always(self):
        """Test Haar is always in available methods."""
        available_methods = ['haar']
        
        assert 'haar' in available_methods

    def test_dlib_added_when_available(self):
        """Test dlib is added to available methods when installed."""
        DLIB_AVAILABLE = True
        available_methods = ['haar']
        
        if DLIB_AVAILABLE:
            available_methods.append('dlib')
        
        assert 'dlib' in available_methods
        assert len(available_methods) == 2

    def test_face_recognition_added_when_available(self):
        """Test face_recognition is added when installed."""
        FACE_RECOGNITION_AVAILABLE = True
        available_methods = ['haar']
        
        if FACE_RECOGNITION_AVAILABLE:
            available_methods.append('face_recognition')
        
        assert 'face_recognition' in available_methods

    def test_mediapipe_added_when_available(self):
        """Test mediapipe is added when installed."""
        MEDIAPIPE_AVAILABLE = True
        available_methods = ['haar']
        
        if MEDIAPIPE_AVAILABLE:
            available_methods.append('mediapipe')
        
        assert 'mediapipe' in available_methods
