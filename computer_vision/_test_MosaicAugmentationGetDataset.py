import pytest
import glob
import os
import random
from string import ascii_lowercase, digits
import cv2
import numpy as np
from mosaic_augmentation import get_dataset

class Test_MosaicAugmentationGetDataset:

    @pytest.mark.smoke
    @pytest.mark.valid
    def test_valid_directories(self):
        # Arrange
        label_dir = 'test_data/valid_labels'
        img_dir = 'test_data/valid_images'
        os.makedirs(label_dir, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)

        label_file = os.path.join(label_dir, 'test_label.txt')
        img_file = os.path.join(img_dir, 'test_label.jpg')
        with open(label_file, 'w') as f:
            f.write('0 0.5 0.5 0.2 0.2\n')
        cv2.imwrite(img_file, np.zeros((100, 100, 3), dtype=np.uint8))

        # Act
        img_paths, labels = get_dataset(label_dir, img_dir)

        # Assert
        assert len(img_paths) == 1
        assert len(labels) == 1
        assert img_paths[0] == img_file
        assert labels[0] == [[0, 0.4, 0.4, 0.6, 0.6]]

        # Cleanup
        os.remove(label_file)
        os.remove(img_file)
        os.rmdir(label_dir)
        os.rmdir(img_dir)

    @pytest.mark.regression
    @pytest.mark.valid
    def test_empty_label_directory(self):
        # Arrange
        label_dir = 'test_data/empty_labels'
        img_dir = 'test_data/images'
        os.makedirs(label_dir, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)

        img_file = os.path.join(img_dir, 'test_image.jpg')
        cv2.imwrite(img_file, np.zeros((100, 100, 3), dtype=np.uint8))

        # Act
        img_paths, labels = get_dataset(label_dir, img_dir)

        # Assert
        assert len(img_paths) == 0
        assert len(labels) == 0

        # Cleanup
        os.remove(img_file)
        os.rmdir(img_dir)
        os.rmdir(label_dir)

    @pytest.mark.regression
    @pytest.mark.valid
    def test_no_corresponding_images(self):
        # Arrange
        label_dir = 'test_data/labels'
        img_dir = 'test_data/images'
        os.makedirs(label_dir, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)

        label_file = os.path.join(label_dir, 'test_label.txt')
        with open(label_file, 'w') as f:
            f.write('0 0.5 0.5 0.2 0.2\n')

        # Act
        img_paths, labels = get_dataset(label_dir, img_dir)

        # Assert
        assert len(img_paths) == 0
        assert len(labels) == 0

        # Cleanup
        os.remove(label_file)
        os.rmdir(img_dir)
        os.rmdir(label_dir)

    @pytest.mark.regression
    @pytest.mark.valid
    def test_label_files_with_no_objects(self):
        # Arrange
        label_dir = 'test_data/labels'
        img_dir = 'test_data/images'
        os.makedirs(label_dir, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)

        label_file = os.path.join(label_dir, 'test_label.txt')
        img_file = os.path.join(img_dir, 'test_label.jpg')
        with open(label_file, 'w') as f:
            f.write('')
        cv2.imwrite(img_file, np.zeros((100, 100, 3), dtype=np.uint8))

        # Act
        img_paths, labels = get_dataset(label_dir, img_dir)

        # Assert
        assert len(img_paths) == 0
        assert len(labels) == 0

        # Cleanup
        os.remove(label_file)
        os.remove(img_file)
        os.rmdir(img_dir)
        os.rmdir(label_dir)

    @pytest.mark.regression
    @pytest.mark.valid
    def test_label_files_with_multiple_objects(self):
        # Arrange
        label_dir = 'test_data/labels'
        img_dir = 'test_data/images'
        os.makedirs(label_dir, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)

        label_file = os.path.join(label_dir, 'test_label.txt')
        img_file = os.path.join(img_dir, 'test_label.jpg')
        with open(label_file, 'w') as f:
            f.write('0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n')
        cv2.imwrite(img_file, np.zeros((100, 100, 3), dtype=np.uint8))

        # Act
        img_paths, labels = get_dataset(label_dir, img_dir)

        # Assert
        assert len(img_paths) == 1
        assert len(labels) == 1
        assert img_paths[0] == img_file
        assert labels[0] == [[0, 0.4, 0.4, 0.6, 0.6], [1, 0.25, 0.25, 0.35, 0.35]]

        # Cleanup
        os.remove(label_file)
        os.remove(img_file)
        os.rmdir(img_dir)
        os.rmdir(label_dir)

    @pytest.mark.regression
    @pytest.mark.invalid
    def test_label_files_with_invalid_object_data(self):
        # Arrange
        label_dir = 'test_data/labels'
        img_dir = 'test_data/images'
        os.makedirs(label_dir, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)

        label_file = os.path.join(label_dir, 'test_label.txt')
        img_file = os.path.join(img_dir, 'test_label.jpg')
        with open(label_file, 'w') as f:
            f.write('0 0.5 0.5 invalid 0.2\n')
        cv2.imwrite(img_file, np.zeros((100, 100, 3), dtype=np.uint8))

        # Act & Assert
        with pytest.raises(ValueError):
            get_dataset(label_dir, img_dir)

        # Cleanup
        os.remove(label_file)
        os.remove(img_file)
        os.rmdir(img_dir)
        os.rmdir(label_dir)

    @pytest.mark.performance
    @pytest.mark.valid
    def test_large_number_of_files(self):
        # Arrange
        label_dir = 'test_data/labels'
        img_dir = 'test_data/images'
        os.makedirs(label_dir, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)

        num_files = 1000  # TODO: Adjust the number of files as needed
        for i in range(num_files):
            label_file = os.path.join(label_dir, f'test_label_{i}.txt')
            img_file = os.path.join(img_dir, f'test_label_{i}.jpg')
            with open(label_file, 'w') as f:
                f.write('0 0.5 0.5 0.2 0.2\n')
            cv2.imwrite(img_file, np.zeros((100, 100, 3), dtype=np.uint8))

        # Act
        img_paths, labels = get_dataset(label_dir, img_dir)

        # Assert
        assert len(img_paths) == num_files
        assert len(labels) == num_files

        # Cleanup
        for i in range(num_files):
            os.remove(os.path.join(label_dir, f'test_label_{i}.txt'))
            os.remove(os.path.join(img_dir, f'test_label_{i}.jpg'))
        os.rmdir(img_dir)
        os.rmdir(label_dir)
