import pytest
import os
import shutil
import glob
import random
from string import ascii_lowercase, digits
import cv2
import numpy as np
from mosaic_augmentation import get_dataset

class Test_MosaicAugmentationGetDataset:

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        # Setup code to create temporary directories for testing
        self.test_label_dir = "test_labels"
        self.test_img_dir = "test_images"
        os.makedirs(self.test_label_dir, exist_ok=True)
        os.makedirs(self.test_img_dir, exist_ok=True)
        yield
        # Teardown code to remove temporary directories after testing
        shutil.rmtree(self.test_label_dir)
        shutil.rmtree(self.test_img_dir)

    @pytest.mark.valid
    def test_valid_input_directories(self):
        # Arrange
        label_files = ["img1.txt", "img2.txt"]
        img_files = ["img1.jpg", "img2.jpg"]
        labels = ["0 0.5 0.5 0.2 0.2", "1 0.6 0.6 0.3 0.3"]
        for label_file, label in zip(label_files, labels):
            with open(os.path.join(self.test_label_dir, label_file), "w") as f:
                f.write(label)
        for img_file in img_files:
            cv2.imwrite(os.path.join(self.test_img_dir, img_file), np.zeros((100, 100, 3), dtype=np.uint8))

        # Act
        img_paths, labels = get_dataset(self.test_label_dir, self.test_img_dir)

        # Assert
        expected_img_paths = [os.path.join(self.test_img_dir, img_file) for img_file in img_files]
        expected_labels = [[[int(label.split()[0]), float(label.split()[1]) - float(label.split()[3]) / 2,
                              float(label.split()[2]) - float(label.split()[4]) / 2,
                              float(label.split()[1]) + float(label.split()[3]) / 2,
                              float(label.split()[2]) + float(label.split()[4]) / 2]] for label in labels]
        assert img_paths == expected_img_paths
        assert labels == expected_labels

    @pytest.mark.invalid
    def test_missing_label_files(self):
        # Arrange
        label_files = ["img1.txt"]
        img_files = ["img1.jpg", "img2.jpg"]
        labels = ["0 0.5 0.5 0.2 0.2"]
        for label_file, label in zip(label_files, labels):
            with open(os.path.join(self.test_label_dir, label_file), "w") as f:
                f.write(label)
        for img_file in img_files:
            cv2.imwrite(os.path.join(self.test_img_dir, img_file), np.zeros((100, 100, 3), dtype=np.uint8))

        # Act
        img_paths, labels = get_dataset(self.test_label_dir, self.test_img_dir)

        # Assert
        expected_img_paths = [os.path.join(self.test_img_dir, "img1.jpg")]
        expected_labels = [[[0, 0.4, 0.4, 0.6, 0.6]]]
        assert img_paths == expected_img_paths
        assert labels == expected_labels

    @pytest.mark.invalid
    def test_empty_label_files(self):
        # Arrange
        label_files = ["img1.txt", "img2.txt"]
        img_files = ["img1.jpg", "img2.jpg"]
        labels = ["", "1 0.6 0.6 0.3 0.3"]
        for label_file, label in zip(label_files, labels):
            with open(os.path.join(self.test_label_dir, label_file), "w") as f:
                f.write(label)
        for img_file in img_files:
            cv2.imwrite(os.path.join(self.test_img_dir, img_file), np.zeros((100, 100, 3), dtype=np.uint8))

        # Act
        img_paths, labels = get_dataset(self.test_label_dir, self.test_img_dir)

        # Assert
        expected_img_paths = [os.path.join(self.test_img_dir, "img2.jpg")]
        expected_labels = [[[1, 0.45, 0.45, 0.75, 0.75]]]
        assert img_paths == expected_img_paths
        assert labels == expected_labels

    @pytest.mark.invalid
    def test_invalid_label_format(self):
        # Arrange
        label_files = ["img1.txt", "img2.txt"]
        img_files = ["img1.jpg", "img2.jpg"]
        labels = ["invalid format", "1 0.6 0.6 0.3 0.3"]
        for label_file, label in zip(label_files, labels):
            with open(os.path.join(self.test_label_dir, label_file), "w") as f:
                f.write(label)
        for img_file in img_files:
            cv2.imwrite(os.path.join(self.test_img_dir, img_file), np.zeros((100, 100, 3), dtype=np.uint8))

        # Act & Assert
        with pytest.raises(ValueError):
            get_dataset(self.test_label_dir, self.test_img_dir)

    @pytest.mark.invalid
    def test_non_existent_directories(self):
        # Arrange
        non_existent_label_dir = "non_existent_labels"
        non_existent_img_dir = "non_existent_images"

        # Act
        img_paths, labels = get_dataset(non_existent_label_dir, non_existent_img_dir)

        # Assert
        assert img_paths == []
        assert labels == []

    @pytest.mark.performance
    def test_large_number_of_files(self):
        # Arrange
        num_files = 1000
        for i in range(num_files):
            label_file = f"img{i}.txt"
            img_file = f"img{i}.jpg"
            label = f"0 {random.random()} {random.random()} {random.random()} {random.random()}"
            with open(os.path.join(self.test_label_dir, label_file), "w") as f:
                f.write(label)
            cv2.imwrite(os.path.join(self.test_img_dir, img_file), np.zeros((100, 100, 3), dtype=np.uint8))

        # Act
        img_paths, labels = get_dataset(self.test_label_dir, self.test_img_dir)

        # Assert
        assert len(img_paths) == num_files
        assert len(labels) == num_files

    @pytest.mark.valid
    def test_special_characters_in_file_names(self):
        # Arrange
        label_files = ["img@#$%.txt", "img&*().txt"]
        img_files = ["img@#$%.jpg", "img&*().jpg"]
        labels = ["0 0.5 0.5 0.2 0.2", "1 0.6 0.6 0.3 0.3"]
        for label_file, label in zip(label_files, labels):
            with open(os.path.join(self.test_label_dir, label_file), "w") as f:
                f.write(label)
        for img_file in img_files:
            cv2.imwrite(os.path.join(self.test_img_dir, img_file), np.zeros((100, 100, 3), dtype=np.uint8))

        # Act
        img_paths, labels = get_dataset(self.test_label_dir, self.test_img_dir)

        # Assert
        expected_img_paths = [os.path.join(self.test_img_dir, img_file) for img_file in img_files]
        expected_labels = [[[int(label.split()[0]), float(label.split()[1]) - float(label.split()[3]) / 2,
                              float(label.split()[2]) - float(label.split()[4]) / 2,
                              float(label.split()[1]) + float(label.split()[3]) / 2,
                              float(label.split()[2]) + float(label.split()[4]) / 2]] for label in labels]
        assert img_paths == expected_img_paths
        assert labels == expected_labels

    @pytest.mark.valid
    def test_mixed_case_file_names(self):
        # Arrange
        label_files = ["Img1.txt", "iMg2.txt"]
        img_files = ["Img1.jpg", "iMg2.jpg"]
        labels = ["0 0.5 0.5 0.2 0.2", "1 0.6 0.6 0.3 0.3"]
        for label_file, label in zip(label_files, labels):
            with open(os.path.join(self.test_label_dir, label_file), "w") as f:
                f.write(label)
        for img_file in img_files:
            cv2.imwrite(os.path.join(self.test_img_dir, img_file), np.zeros((100, 100, 3), dtype=np.uint8))

        # Act
        img_paths, labels = get_dataset(self.test_label_dir, self.test_img_dir)

        # Assert
        expected_img_paths = [os.path.join(self.test_img_dir, img_file) for img_file in img_files]
        expected_labels = [[[int(label.split()[0]), float(label.split()[1]) - float(label.split()[3]) / 2,
                              float(label.split()[2]) - float(label.split()[4]) / 2,
                              float(label.split()[1]) + float(label.split()[3]) / 2,
                              float(label.split()[2]) + float(label.split()[4]) / 2]] for label in labels]
        assert img_paths == expected_img_paths
        assert labels == expected_labels

    @pytest.mark.valid
    def test_subdirectories_in_input_directories(self):
        # Arrange
        os.makedirs(os.path.join(self.test_label_dir, "subdir"))
        os.makedirs(os.path.join(self.test_img_dir, "subdir"))
        label_files = ["subdir/img1.txt", "subdir/img2.txt"]
        img_files = ["subdir/img1.jpg", "subdir/img2.jpg"]
        labels = ["0 0.5 0.5 0.2 0.2", "1 0.6 0.6 0.3 0.3"]
        for label_file, label in zip(label_files, labels):
            with open(os.path.join(self.test_label_dir, label_file), "w") as f:
                f.write(label)
        for img_file in img_files:
            cv2.imwrite(os.path.join(self.test_img_dir, img_file), np.zeros((100, 100, 3), dtype=np.uint8))

        # Act
        img_paths, labels = get_dataset(self.test_label_dir, self.test_img_dir)

        # Assert
        expected_img_paths = [os.path.join(self.test_img_dir, img_file) for img_file in img_files]
        expected_labels = [[[int(label.split()[0]), float(label.split()[1]) - float(label.split()[3]) / 2,
                              float(label.split()[2]) - float(label.split()[4]) / 2,
                              float(label.split()[1]) + float(label.split()[3]) / 2,
                              float(label.split()[2]) + float(label.split()[4]) / 2]] for label in labels]
        assert img_paths == expected_img_paths
        assert labels == expected_labels

    @pytest.mark.invalid
    def test_empty_input_directories(self):
        # Arrange
        # Directories are already empty due to setup_teardown fixture

        # Act
        img_paths, labels = get_dataset(self.test_label_dir, self.test_img_dir)

        # Assert
        assert img_paths == []
        assert labels == []
