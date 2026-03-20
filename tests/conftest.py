"""
Pytest configuration and shared fixtures for face recognition tests.
"""

import os
import sys
import tempfile
import shutil
import pytest
import numpy as np
import cv2
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['DISPLAY'] = ':0'

@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary directory for test data files."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    original_dir = os.getcwd()
    os.chdir(data_dir)
    yield data_dir
    os.chdir(original_dir)


@pytest.fixture
def mock_camera():
    """Create a mock camera that returns synthetic frames."""
    mock_cap = MagicMock()
    
    frames = []
    for i in range(10):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, f"Frame {i}", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        frames.append(frame)
    
    mock_cap.read.side_effect = [(True, frames[i % len(frames)]) for i in range(100)]
    mock_cap.isOpened.return_value = True
    return mock_cap


@pytest.fixture
def synthetic_face_image():
    """Generate a synthetic image with a face-like region for testing."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(img, (220, 140), (420, 380), (200, 180, 160), -1)
    cv2.ellipse(img, (320, 220), (60, 70), 0, 0, 360, (255, 220, 200), -1)
    return img


@pytest.fixture
def sample_face_data():
    """Generate sample face data for testing."""
    face_data = []
    for _ in range(5):
        face = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        face_data.append(face)
    return np.array(face_data)


@pytest.fixture
def sample_face_labels():
    """Generate sample face labels."""
    return np.array([0, 0, 0, 1, 1])


@pytest.fixture
def mock_tkinter():
    """Mock tkinter components to avoid GUI initialization."""
    with patch('customtkinter.CTk') as mock_ctk:
        mock_root = MagicMock()
        mock_ctk.return_value = mock_root
        yield mock_root


@pytest.fixture
def camera_available():
    """Check if a camera is available for testing."""
    try:
        cap = cv2.VideoCapture(0)
        available = cap.isOpened()
        cap.release()
        return available
    except Exception:
        return False


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "camera: tests that require a camera"
    )
    config.addinivalue_line(
        "markers", "integration: integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: unit tests"
    )
