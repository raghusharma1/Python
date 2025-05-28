import pytest
import os
import glob
import shutil
from string import ascii_lowercase, digits
import random
from flip_augmentation import get_dataset

class Test_FlipAugmentationGetDataset:

    @pytest.mark.valid
    def test_get_dataset_valid_paths(self, tmpdir):
        # Arrange
        label_dir = tmpdir.mkdir("labels")
        img_dir = tmpdir.mkdir("images")
        img_file = img_dir.join("test.jpg")
        label_file = label_dir.join("test.txt")
        img_file.write("dummy image content")
        label_file.write("0 0.1 0.2 0.3 0.4")

        # Act
        img_paths, labels = get_dataset(str(label_dir), str(img_dir))

        # Assert
        assert img_paths == [str(img_file)]
        assert labels == [[[0, 0.1, 0.2, 0.3, 0.4]]]

    @pytest.mark.invalid
    def test_get_dataset_missing_label_files(self, tmpdir):
        # Arrange
        label_dir = tmpdir.mkdir("labels")
        img_dir = tmpdir.mkdir("images")
        img_file = img_dir.join("test.jpg")
        img_file.write("dummy image content")

        # Act
        img_paths, labels = get_dataset(str(label_dir), str(img_dir))

        # Assert
        assert img_paths == []
        assert labels == []

    @pytest.mark.invalid
    def test_get_dataset_empty_label_files(self, tmpdir):
        # Arrange
        label_dir = tmpdir.mkdir("labels")
        img_dir = tmpdir.mkdir("images")
        img_file = img_dir.join("test.jpg")
        label_file = label_dir.join("test.txt")
        img_file.write("dummy image content")
        label_file.write("")

        # Act
        img_paths, labels = get_dataset(str(label_dir), str(img_dir))

        # Assert
        assert img_paths == []
        assert labels == []

    @pytest.mark.valid
    def test_get_dataset_special_characters(self, tmpdir):
        # Arrange
        label_dir = tmpdir.mkdir("labels")
        img_dir = tmpdir.mkdir("images")
        img_file = img_dir.join("test@#$.jpg")
        label_file = label_dir.join("test@#$.txt")
        img_file.write("dummy image content")
        label_file.write("0 0.1 0.2 0.3 0.4")

        # Act
        img_paths, labels = get_dataset(str(label_dir), str(img_dir))

        # Assert
        assert img_paths == [str(img_file)]
        assert labels == [[[0, 0.1, 0.2, 0.3, 0.4]]]

    @pytest.mark.performance
    def test_get_dataset_large_number_of_files(self, tmpdir):
        # Arrange
        label_dir = tmpdir.mkdir("labels")
        img_dir = tmpdir.mkdir("images")
        num_files = 1000  # Change this to a higher number for more extensive testing
        for i in range(num_files):
            img_file = img_dir.join(f"test{i}.jpg")
            label_file = label_dir.join(f"test{i}.txt")
            img_file.write("dummy image content")
            label_file.write(f"{i} 0.1 0.2 0.3 0.4")

        # Act
        img_paths, labels = get_dataset(str(label_dir), str(img_dir))

        # Assert
        assert len(img_paths) == num_files
        assert len(labels) == num_files
        for i in range(num_files):
            assert img_paths[i] == str(img_dir.join(f"test{i}.jpg"))
            assert labels[i] == [[i, 0.1, 0.2, 0.3, 0.4]]

    @pytest.mark.invalid
    def test_get_dataset_non_existent_directories(self):
        # Act & Assert
        with pytest.raises(FileNotFoundError):
            get_dataset("/non/existent/path", "/non/existent/path")

    @pytest.mark.invalid
    def test_get_dataset_invalid_label_format(self, tmpdir):
        # Arrange
        label_dir = tmpdir.mkdir("labels")
        img_dir = tmpdir.mkdir("images")
        img_file = img_dir.join("test.jpg")
        label_file = label_dir.join("test.txt")
        img_file.write("dummy image content")
        label_file.write("invalid format content")

        # Act
        img_paths, labels = get_dataset(str(label_dir), str(img_dir))

        # Assert
        assert img_paths == []
        assert labels == []

    @pytest.mark.valid
    def test_get_dataset_mixed_case_file_names(self, tmpdir):
        # Arrange
        label_dir = tmpdir.mkdir("labels")
        img_dir = tmpdir.mkdir("images")
        img_file = img_dir.join("Test.jpg")
        label_file = label_dir.join("test.txt")
        img_file.write("dummy image content")
        label_file.write("0 0.1 0.2 0.3 0.4")

        # Act
        img_paths, labels = get_dataset(str(label_dir), str(img_dir))

        # Assert
        assert img_paths == [str(img_file)]
        assert labels == [[[0, 0.1, 0.2, 0.3, 0.4]]]

    @pytest.mark.valid
    def test_get_dataset_varying_boxes_per_label(self, tmpdir):
        # Arrange
        label_dir = tmpdir.mkdir("labels")
        img_dir = tmpdir.mkdir("images")
        img_file = img_dir.join("test.jpg")
        label_file = label_dir.join("test.txt")
        img_file.write("dummy image content")
        label_file.write("0 0.1 0.2 0.3 0.4\n1 0.5 0.6 0.7 0.8")

        # Act
        img_paths, labels = get_dataset(str(label_dir), str(img_dir))

        # Assert
        assert img_paths == [str(img_file)]
        assert labels == [[[0, 0.1, 0.2, 0.3, 0.4], [1, 0.5, 0.6, 0.7, 0.8]]]

    @pytest.mark.invalid
    def test_get_dataset_empty_directories(self, tmpdir):
        # Arrange
        label_dir = tmpdir.mkdir("labels")
        img_dir = tmpdir.mkdir("images")

        # Act
        img_paths, labels = get_dataset(str(label_dir), str(img_dir))

        # Assert
        assert img_paths == []
        assert labels == []
