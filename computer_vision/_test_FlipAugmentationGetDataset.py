# test_FlipAugmentationGetDataset.py

import pytest
import os
import shutil
import tempfile
from flip_augmentation import get_dataset

class Test_FlipAugmentationGetDataset:

    @pytest.mark.smoke
    @pytest.mark.valid
    def test_get_dataset_valid_inputs(self):
        # Arrange
        temp_dir = tempfile.mkdtemp()
        label_dir = os.path.join(temp_dir, 'labels')
        img_dir = os.path.join(temp_dir, 'images')
        os.makedirs(label_dir)
        os.makedirs(img_dir)

        # Create label and image files
        label_file = os.path.join(label_dir, 'img1.txt')
        with open(label_file, 'w') as f:
            f.write("0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.4 0.4")

        img_file = os.path.join(img_dir, 'img1.jpg')
        with open(img_file, 'wb') as f:
            f.write(b'fake image data')

        # Act
        img_paths, labels = get_dataset(label_dir, img_dir)

        # Assert
        assert len(img_paths) == 1
        assert img_paths[0] == img_file
        assert len(labels) == 1
        assert labels[0] == [
            [0, 0.5, 0.5, 0.2, 0.2],
            [1, 0.3, 0.3, 0.4, 0.4]
        ]

        # Cleanup
        shutil.rmtree(temp_dir)

    @pytest.mark.valid
    def test_get_dataset_multiple_files(self):
        # Arrange
        temp_dir = tempfile.mkdtemp()
        label_dir = os.path.join(temp_dir, 'labels')
        img_dir = os.path.join(temp_dir, 'images')
        os.makedirs(label_dir)
        os.makedirs(img_dir)

        # Create multiple label and image files
        for i in range(5):
            label_file = os.path.join(label_dir, f'img{i}.txt')
            with open(label_file, 'w') as f:
                f.write(f"0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.4 0.4")

            img_file = os.path.join(img_dir, f'img{i}.jpg')
            with open(img_file, 'wb') as f:
                f.write(b'fake image data')

        # Act
        img_paths, labels = get_dataset(label_dir, img_dir)

        # Assert
        assert len(img_paths) == 5
        for i in range(5):
            assert img_paths[i] == os.path.join(img_dir, f'img{i}.jpg')
            assert labels[i] == [
                [0, 0.5, 0.5, 0.2, 0.2],
                [1, 0.3, 0.3, 0.4, 0.4]
            ]

        # Cleanup
        shutil.rmtree(temp_dir)

    @pytest.mark.invalid
    def test_get_dataset_empty_directories(self):
        # Arrange
        temp_dir = tempfile.mkdtemp()
        label_dir = os.path.join(temp_dir, 'labels')
        img_dir = os.path.join(temp_dir, 'images')
        os.makedirs(label_dir)
        os.makedirs(img_dir)

        # Act
        img_paths, labels = get_dataset(label_dir, img_dir)

        # Assert
        assert len(img_paths) == 0
        assert len(labels) == 0

        # Cleanup
        shutil.rmtree(temp_dir)

    @pytest.mark.invalid
    def test_get_dataset_no_matching_images(self):
        # Arrange
        temp_dir = tempfile.mkdtemp()
        label_dir = os.path.join(temp_dir, 'labels')
        img_dir = os.path.join(temp_dir, 'images')
        os.makedirs(label_dir)
        os.makedirs(img_dir)

        # Create label files without corresponding image files
        for i in range(5):
            label_file = os.path.join(label_dir, f'img{i}.txt')
            with open(label_file, 'w') as f:
                f.write(f"0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.4 0.4")

        # Act
        img_paths, labels = get_dataset(label_dir, img_dir)

        # Assert
        assert len(img_paths) == 0
        assert len(labels) == 0

        # Cleanup
        shutil.rmtree(temp_dir)

    @pytest.mark.invalid
    def test_get_dataset_empty_labels(self):
        # Arrange
        temp_dir = tempfile.mkdtemp()
        label_dir = os.path.join(temp_dir, 'labels')
        img_dir = os.path.join(temp_dir, 'images')
        os.makedirs(label_dir)
        os.makedirs(img_dir)

        # Create label files with empty content
        for i in range(5):
            label_file = os.path.join(label_dir, f'img{i}.txt')
            with open(label_file, 'w') as f:
                f.write("")

            img_file = os.path.join(img_dir, f'img{i}.jpg')
            with open(img_file, 'wb') as f:
                f.write(b'fake image data')

        # Act
        img_paths, labels = get_dataset(label_dir, img_dir)

        # Assert
        assert len(img_paths) == 0
        assert len(labels) == 0

        # Cleanup
        shutil.rmtree(temp_dir)
