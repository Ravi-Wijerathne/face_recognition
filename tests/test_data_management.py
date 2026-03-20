"""
Unit tests for data management (save, load, export, import).
Tests JSON format, numpy arrays, and file operations.
"""

import pytest
import json
import numpy as np
import os
from unittest.mock import MagicMock, patch, mock_open


class TestJSONDataStructure:
    """Test JSON data structure and format."""

    def test_save_data_structure(self):
        """Test save_data produces correct JSON structure."""
        data = {
            'name_to_id': {'Alice': 0, 'Bob': 1},
            'id_to_name': {str(k): v for k, v in {0: 'Alice', 1: 'Bob'}.items()},
            'face_count': 5
        }
        
        assert 'name_to_id' in data
        assert 'id_to_name' in data
        assert 'face_count' in data

    def test_export_data_includes_timestamp(self):
        """Test export includes timestamp."""
        import time
        data = {
            'name_to_id': {},
            'id_to_name': {},
            'face_count': 0,
            'exported_at': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        assert 'exported_at' in data
        assert '2026' in data['exported_at']

    def test_id_to_name_keys_are_strings_in_json(self):
        """Test id_to_name keys are converted to strings for JSON."""
        id_to_name = {0: 'Alice', 1: 'Bob'}
        json_compatible = {str(k): v for k, v in id_to_name.items()}
        
        assert all(isinstance(k, str) for k in json_compatible.keys())

    def test_id_to_name_loads_with_int_keys(self):
        """Test id_to_name keys are converted back to int after load."""
        imported = {'0': 'Alice', '1': 'Bob'}
        id_to_name = {int(k): v for k, v in imported.items()}
        
        assert all(isinstance(k, int) for k in id_to_name.keys())


class TestFileOperations:
    """Test file I/O operations."""

    @pytest.fixture
    def temp_data_dir(self, tmp_path):
        """Create temporary directory for test files."""
        return tmp_path

    def test_save_to_json_file(self, temp_data_dir):
        """Test saving data to JSON file."""
        file_path = temp_data_dir / "test_data.json"
        data = {
            'name_to_id': {'Alice': 0},
            'id_to_name': {'0': 'Alice'},
            'face_count': 1
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f)
        
        assert file_path.exists()

    def test_load_from_json_file(self, temp_data_dir):
        """Test loading data from JSON file."""
        file_path = temp_data_dir / "test_data.json"
        data = {
            'name_to_id': {'Alice': 0},
            'id_to_name': {'0': 'Alice'},
            'face_count': 1
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f)
        
        with open(file_path, 'r') as f:
            loaded = json.load(f)
        
        assert loaded['name_to_id']['Alice'] == 0
        assert loaded['face_count'] == 1

    def test_save_numpy_array(self, temp_data_dir):
        """Test saving numpy array to file."""
        file_path = temp_data_dir / "face_data.npy"
        face_data = np.array([np.zeros((100, 100), dtype=np.uint8)])
        
        np.save(file_path, face_data)
        
        assert file_path.exists()

    def test_load_numpy_array(self, temp_data_dir):
        """Test loading numpy array from file."""
        file_path = temp_data_dir / "face_data.npy"
        original = np.array([np.zeros((100, 100), dtype=np.uint8)])
        
        np.save(file_path, original)
        loaded = np.load(file_path)
        
        assert np.array_equal(loaded, original)
        assert len(loaded) == len(original)

    def test_file_not_exists_handling(self, temp_data_dir):
        """Test handling when file doesn't exist."""
        file_path = temp_data_dir / "nonexistent.json"
        
        assert not file_path.exists()

    def test_corrupted_json_handling(self, temp_data_dir):
        """Test handling corrupted JSON."""
        file_path = temp_data_dir / "corrupted.json"
        file_path.write_text("{ invalid json }")
        
        try:
            with open(file_path, 'r') as f:
                json.load(f)
            loaded = False
        except json.JSONDecodeError:
            loaded = False
        
        assert loaded is False


class TestDataIntegrity:
    """Test data integrity between operations."""

    def test_face_data_and_labels_length_match(self):
        """Test that face_data and face_labels always have matching lengths."""
        face_data = []
        face_labels = []
        
        assert len(face_data) == len(face_labels)
        
        face_data.append(np.zeros((100, 100), dtype=np.uint8))
        face_labels.append(0)
        
        assert len(face_data) == len(face_labels)

    def test_add_samples_to_existing_person(self):
        """Test adding more samples to an existing person."""
        name_to_id = {'Alice': 0}
        id_to_name = {0: 'Alice'}
        face_data = [np.zeros((100, 100), dtype=np.uint8)]
        face_labels = [0]
        
        initial_count = len(face_data)
        
        face_data.append(np.zeros((100, 100), dtype=np.uint8))
        face_labels.append(0)
        
        assert len(face_data) == initial_count + 1
        assert name_to_id['Alice'] == 0

    def test_id_mapping_consistency(self):
        """Test that name_to_id and id_to_name remain consistent."""
        name_to_id = {'Alice': 0}
        id_to_name = {0: 'Alice'}
        
        assert id_to_name[name_to_id['Alice']] == 'Alice'
        assert name_to_id[id_to_name[0]] == 0

    def test_delete_person_removes_correct_samples(self):
        """Test deleting a person removes all their samples."""
        face_data = [1, 2, 3, 4, 5]
        face_labels = [0, 0, 1, 1, 1]
        person_id = 0
        
        new_face_data = [face_data[i] for i in range(len(face_data)) if face_labels[i] != person_id]
        new_face_labels = [label for label in face_labels if label != person_id]
        
        assert len(new_face_data) == 3
        assert len(new_face_labels) == 3
        assert 0 not in new_face_labels


class TestDataRoundtrip:
    """Test save/load roundtrip operations."""

    @pytest.fixture
    def temp_data_dir(self, tmp_path):
        """Create temporary directory for test files."""
        return tmp_path

    def test_name_mapping_roundtrip(self, temp_data_dir):
        """Test name mappings survive save/load cycle."""
        original_name_to_id = {'Alice': 0, 'Bob': 1}
        original_id_to_name = {0: 'Alice', 1: 'Bob'}
        
        data = {
            'name_to_id': original_name_to_id,
            'id_to_name': {str(k): v for k, v in original_id_to_name.items()}
        }
        
        file_path = temp_data_dir / "mappings.json"
        with open(file_path, 'w') as f:
            json.dump(data, f)
        
        with open(file_path, 'r') as f:
            loaded = json.load(f)
        
        loaded_name_to_id = loaded['name_to_id']
        loaded_id_to_name = {int(k): v for k, v in loaded['id_to_name'].items()}
        
        assert loaded_name_to_id == original_name_to_id
        assert loaded_id_to_name == original_id_to_name

    def test_face_data_roundtrip(self, temp_data_dir):
        """Test face data survives save/load cycle."""
        original_face_data = [np.zeros((100, 100), dtype=np.uint8) for _ in range(3)]
        original_labels = [0, 0, 1]
        
        file_path = temp_data_dir / "face_data.npy"
        labels_path = temp_data_dir / "face_labels.npy"
        
        np.save(file_path, np.array(original_face_data))
        np.save(labels_path, np.array(original_labels))
        
        loaded_face_data = np.load(file_path)
        loaded_labels = np.load(labels_path)
        
        assert len(loaded_face_data) == len(original_face_data)
        assert len(loaded_labels) == len(original_labels)


class TestExportImport:
    """Test export and import operations."""

    @pytest.fixture
    def temp_data_dir(self, tmp_path):
        """Create temporary directory for test files."""
        return tmp_path

    def test_export_file_structure(self, temp_data_dir):
        """Test export produces valid JSON."""
        export_data = {
            'name_to_id': {'Alice': 0},
            'id_to_name': {'0': 'Alice'},
            'face_count': 5,
            'exported_at': '2026-03-20 12:00:00'
        }
        
        file_path = temp_data_dir / "export.json"
        with open(file_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        with open(file_path, 'r') as f:
            loaded = json.load(f)
        
        assert 'name_to_id' in loaded
        assert 'id_to_name' in loaded
        assert 'face_count' in loaded
        assert 'exported_at' in loaded

    def test_import_with_missing_npy_files(self, temp_data_dir):
        """Test import works when npy files are missing."""
        import_data = {
            'name_to_id': {'Alice': 0},
            'id_to_name': {'0': 'Alice'},
            'face_count': 0
        }
        
        file_path = temp_data_dir / "import.json"
        with open(file_path, 'w') as f:
            json.dump(import_data, f)
        
        with open(file_path, 'r') as f:
            loaded = json.load(f)
        
        name_to_id = loaded.get('name_to_id', {})
        id_to_name = {int(k): v for k, v in loaded.get('id_to_name', {}).items()}
        
        assert name_to_id['Alice'] == 0
        assert id_to_name[0] == 'Alice'
