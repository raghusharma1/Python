import pytest
import os
import glob
import random
from string import ascii_lowercase, digits
import cv2
from flip_augmentation import get_dataset

class Test_FlipAugmentationGetDataset:

    @pytest.mark.valid
    def test_get_dataset_valid_inputs(self):
        # Arrange
        label_dir = 'test_data/labels'
        img_dir = 'test_data/images'
        os.makedirs(label_dir, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)

        label_content = "0 0.5 0.5 0.2 0.2"
        with open(os.path.join(label_dir, 'image1.txt'), 'w') as f:
            f.write(label_content)
        cv2.imwrite(os.path.join(img_dir, 'image1.jpg'), cv2.imread('test_image.jpg'))

        expected_img_paths = [os.path.join(img_dir, 'image1.jpg')]
        expected_labels = [[[0, 0.5, 0.5, 0.2, 0.2]]]

        # Act
        img_paths, labels = get_dataset(label_dir, img_dir)

        # Assert
        assert img_paths == expected_img_paths
        assert labels == expected_labels

    @pytest.mark.invalid
    def test_get_dataset_no_corresponding_image(self):
        # Arrange
        label_dir = 'test_data/labels'
        img_dir = 'test_data/images'
        os.makedirs(label_dir, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)

        label_content = "0 0.5 0.5 0.2 0.2"
        with open(os.path.join(label_dir, 'image2.txt'), 'w') as f:
            f.write(label_content)

        # Act
        img_paths, labels = get_dataset(label_dir, img_dir)

        # Assert
        assert img_paths == []
        assert labels == []

    @pytest.mark.invalid
    def test_get_dataset_empty_label_file(self):
        # Arrange
        label_dir = 'test_data/labels'
        img_dir = 'test_data/images'
        os.makedirs(label_dir, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)

        with open(os.path.join(label_dir, 'image3.txt'), 'w') as f:
            pass

        # Act
        img_paths, labels = get_dataset(label_dir, img_dir)

        # Assert
        assert img_paths == []
        assert labels == []

    @pytest.mark.valid
    def test_get_dataset_multiple_label_files(self):
        # Arrange
        label_dir = 'test_data/labels'
        img_dir = 'test_data/images'
        os.makedirs(label_dir, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)

        label_content1 = "0 0.5 0.5 0.2 0.2"
        label_content2 = "1 0.3 0.3 0.4 0.4"
        with open(os.path.join(label_dir, 'image4.txt'), 'w') as f:
            f.write(label_content1)
        with open(os.path.join(label_dir, 'image5.txt'), 'w') as f:
            f.write(label_content2)
        cv2.imwrite(os.path.join(img_dir, 'image4.jpg'), cv2.imread('test_image.jpg'))
        cv2.imwrite(os.path.join(img_dir, 'image5.jpg'), cv2.imread('test_image.jpg'))

        expected_img_paths = [os.path.join(img_dir, 'image4.jpg'), os.path.join(img_dir, 'image5.jpg')]
        expected_labels = [[[0, 0.5, 0.5, 0.2, 0.2]], [[1, 0.3, 0.3, 0.4, 0.4]]]

        # Act
        img_paths, labels = get_dataset(label_dir, img_dir)

        # Assert
        assert img_paths == expected_img_paths
        assert labels == expected_labels

    @pytest.mark.invalid
    def test_get_dataset_invalid_label_format(self):
        # Arrange
        label_dir = 'test_data/labels'
        img_dir = 'test_data/images'
        os.makedirs(label_dir, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)

        label_content = "invalid format"
        with open(os.path.join(label_dir, 'image6.txt'), 'w') as f:
            f.write(label_content)

        # Act & Assert
        with pytest.raises(ValueError):
            get_dataset(label_dir, img_dir)

    @pytest.mark.invalid
    def test_get_dataset_non_existent_directories(self):
        # Arrange
        label_dir = 'non_existent_labels'
        img_dir = 'non_existent_images'

        # Act & Assert
        with pytest.raises(FileNotFoundError):
            get_dataset(label_dir, img_dir)

    @pytest.mark.performance
    def test_get_dataset_large_number_of_files(self):
        # Arrange
        label_dir = 'test_data/labels'
        img_dir = 'test_data/images'
        os.makedirs(label_dir, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)

        label_content = "0 0.5 0.5 0.2 0.2"
        for i in range(1000):
            with open(os.path.join(label_dir, f'image{i}.txt'), 'w') as f:
                f.write(label_content)
            cv2.imwrite(os.path.join(img_dir, f'image{i}.jpg'), cv2.imread('test_image.jpg'))

        # Act
        img_paths, labels = get_dataset(label_dir, img_dir)

        # Assert
        assert len(img_paths) == 1000
        assert len(labels) == 1000

    @pytest.mark.valid
    def test_get_dataset_special_characters_in_filenames(self):
        # Arrange
        label_dir = 'test_data/labels'
        img_dir = 'test_data/images'
        os.makedirs(label_dir, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)

        label_content = "0 0.5 0.5 0.2 0.2"
        special_chars = '!@#$%^&*()_+[]{}|;:,.<>?'
        filename = ''.join(random.choices(ascii_lowercase + digits + special_chars, k=10))
        with open(os.path.join(label_dir, f'{filename}.txt'), 'w') as f:
            f.write(label_content)
        cv2.imwrite(os.path.join(img_dir, f'{filename}.jpg'), cv2.imread('test_image.jpg'))

        expected_img_paths = [os.path.join(img_dir, f'{filename}.jpg')]
        expected_labels = [[[0, 0.5, 0.5, 0.2, 0.2]]]

        # Act
        img_paths, labels = get_dataset(label_dir, img_dir)

        # Assert
        assert img_paths == expected_img_paths
        assert labels == expected_labels

    @pytest.mark.valid
    def test_get_dataset_mixed_case_filenames(self):
        # Arrange
        label_dir = 'test_data/labels'
        img_dir = 'test_data/images'
        os.makedirs(label_dir, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)

        label_content = "0 0.5 0.5 0.2 0.2"
        filename = ''.join(random.choices(ascii_lowercase + digits, k=10))
        mixed_case_filename = ''.join(random.choice((str.upper, str.lower))(c) for c in filename)
        with open(os.path.join(label_dir, f'{mixed_case_filename}.txt'), 'w') as f:
            f.write(label_content)
        cv2.imwrite(os.path.join(img_dir, f'{mixed_case_filename}.jpg'), cv2.imread('test_image.jpg'))

        expected_img_paths = [os.path.join(img_dir, f'{mixed_case_filename}.jpg')]
        expected_labels = [[[0, 0.5, 0.5, 0.2, 0.2]]]

        # Act
        img_paths, labels = get_dataset(label_dir, img_dir)

        # Assert
        assert img_paths == expected_img_paths
        assert labels == expected_labels

    @pytest.mark.invalid
    def test_get_dataset_empty_directories(self):
        # Arrange
        label_dir = 'test_data/labels'
        img_dir = 'test_data/images'
        os.makedirs(label_dir, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)

        # Act
        img_paths, labels = get_dataset(label_dir, img_dir)

        # Assert
        assert img_paths == []
        assert labels == []
