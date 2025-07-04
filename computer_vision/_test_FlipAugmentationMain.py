import pytest
import os
import shutil
import glob
import random
from string import ascii_lowercase, digits
import cv2
from flip_augmentation import main

# Define the test class
class Test_FlipAugmentationMain:

    @pytest.fixture(autouse=True)
    def setup(self):
        # Create temporary directories for testing
        self.label_dir = 'test_labels'
        self.image_dir = 'test_images'
        self.output_dir = 'test_output'
        os.makedirs(self.label_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        yield
        # Clean up after tests
        shutil.rmtree(self.label_dir)
        shutil.rmtree(self.image_dir)
        shutil.rmtree(self.output_dir)

    @pytest.mark.valid
    def test_main_with_valid_input_directories(self):
        # Arrange
        global LABEL_DIR, IMAGE_DIR, OUTPUT_DIR, FLIP_TYPE
        LABEL_DIR = self.label_dir
        IMAGE_DIR = self.image_dir
        OUTPUT_DIR = self.output_dir
        FLIP_TYPE = 1

        # Create test data
        with open(os.path.join(self.label_dir, 'image1.txt'), 'w') as f:
            f.write('0 0.5 0.5 0.1 0.1\n')
        cv2.imwrite(os.path.join(self.image_dir, 'image1.jpg'), cv2.imread('path_to_test_image.jpg'))

        # Act
        main()

        # Assert
        output_files = glob.glob(os.path.join(self.output_dir, '*'))
        assert len(output_files) == 2  # One image and one annotation file
        assert any('image1_FLIP_' in file for file in output_files)

    @pytest.mark.invalid
    def test_main_with_empty_input_directories(self):
        # Arrange
        global LABEL_DIR, IMAGE_DIR, OUTPUT_DIR, FLIP_TYPE
        LABEL_DIR = self.label_dir
        IMAGE_DIR = self.image_dir
        OUTPUT_DIR = self.output_dir
        FLIP_TYPE = 1

        # Act
        main()

        # Assert
        output_files = glob.glob(os.path.join(self.output_dir, '*'))
        assert len(output_files) == 0

    @pytest.mark.invalid
    def test_main_with_missing_annotation_files(self):
        # Arrange
        global LABEL_DIR, IMAGE_DIR, OUTPUT_DIR, FLIP_TYPE
        LABEL_DIR = self.label_dir
        IMAGE_DIR = self.image_dir
        OUTPUT_DIR = self.output_dir
        FLIP_TYPE = 1

        # Create test data
        cv2.imwrite(os.path.join(self.image_dir, 'image1.jpg'), cv2.imread('path_to_test_image.jpg'))
        cv2.imwrite(os.path.join(self.image_dir, 'image2.jpg'), cv2.imread('path_to_test_image.jpg'))
        with open(os.path.join(self.label_dir, 'image1.txt'), 'w') as f:
            f.write('0 0.5 0.5 0.1 0.1\n')

        # Act
        main()

        # Assert
        output_files = glob.glob(os.path.join(self.output_dir, '*'))
        assert len(output_files) == 2  # Only one image and annotation file should be processed

    @pytest.mark.invalid
    def test_main_with_invalid_flip_type(self):
        # Arrange
        global LABEL_DIR, IMAGE_DIR, OUTPUT_DIR, FLIP_TYPE
        LABEL_DIR = self.label_dir
        IMAGE_DIR = self.image_dir
        OUTPUT_DIR = self.output_dir
        FLIP_TYPE = 2  # Invalid flip type

        # Create test data
        with open(os.path.join(self.label_dir, 'image1.txt'), 'w') as f:
            f.write('0 0.5 0.5 0.1 0.1\n')
        cv2.imwrite(os.path.join(self.image_dir, 'image1.jpg'), cv2.imread('path_to_test_image.jpg'))

        # Act & Assert
        with pytest.raises(ValueError):  # TODO: Update with the actual exception raised by the function
            main()

    @pytest.mark.performance
    def test_main_with_large_number_of_images(self):
        # Arrange
        global LABEL_DIR, IMAGE_DIR, OUTPUT_DIR, FLIP_TYPE
        LABEL_DIR = self.label_dir
        IMAGE_DIR = self.image_dir
        OUTPUT_DIR = self.output_dir
        FLIP_TYPE = 1

        # Create test data
        for i in range(1000):
            with open(os.path.join(self.label_dir, f'image{i}.txt'), 'w') as f:
                f.write('0 0.5 0.5 0.1 0.1\n')
            cv2.imwrite(os.path.join(self.image_dir, f'image{i}.jpg'), cv2.imread('path_to_test_image.jpg'))

        # Act
        main()

        # Assert
        output_files = glob.glob(os.path.join(self.output_dir, '*'))
        assert len(output_files) == 2000  # 1000 images and 1000 annotation files

    @pytest.mark.valid
    def test_main_with_special_characters_in_filenames(self):
        # Arrange
        global LABEL_DIR, IMAGE_DIR, OUTPUT_DIR, FLIP_TYPE
        LABEL_DIR = self.label_dir
        IMAGE_DIR = self.image_dir
        OUTPUT_DIR = self.output_dir
        FLIP_TYPE = 1

        # Create test data
        with open(os.path.join(self.label_dir, 'image@#$.txt'), 'w') as f:
            f.write('0 0.5 0.5 0.1 0.1\n')
        cv2.imwrite(os.path.join(self.image_dir, 'image@#$.jpg'), cv2.imread('path_to_test_image.jpg'))

        # Act
        main()

        # Assert
        output_files = glob.glob(os.path.join(self.output_dir, '*'))
        assert len(output_files) == 2  # One image and one annotation file
        assert any('image@#$_FLIP_' in file for file in output_files)

    @pytest.mark.valid
    def test_main_with_non_existent_output_directory(self):
        # Arrange
        global LABEL_DIR, IMAGE_DIR, OUTPUT_DIR, FLIP_TYPE
        LABEL_DIR = self.label_dir
        IMAGE_DIR = self.image_dir
        OUTPUT_DIR = 'non_existent_output'
        FLIP_TYPE = 1

        # Create test data
        with open(os.path.join(self.label_dir, 'image1.txt'), 'w') as f:
            f.write('0 0.5 0.5 0.1 0.1\n')
        cv2.imwrite(os.path.join(self.image_dir, 'image1.jpg'), cv2.imread('path_to_test_image.jpg'))

        # Act
        main()

        # Assert
        assert os.path.exists(OUTPUT_DIR)
        output_files = glob.glob(os.path.join(OUTPUT_DIR, '*'))
        assert len(output_files) == 2  # One image and one annotation file

    @pytest.mark.invalid
    def test_main_with_read_only_output_directory(self):
        # Arrange
        global LABEL_DIR, IMAGE_DIR, OUTPUT_DIR, FLIP_TYPE
        LABEL_DIR = self.label_dir
        IMAGE_DIR = self.image_dir
        OUTPUT_DIR = 'read_only_output'
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.chmod(OUTPUT_DIR, 0o444)  # Make the directory read-only
        FLIP_TYPE = 1

        # Create test data
        with open(os.path.join(self.label_dir, 'image1.txt'), 'w') as f:
            f.write('0 0.5 0.5 0.1 0.1\n')
        cv2.imwrite(os.path.join(self.image_dir, 'image1.jpg'), cv2.imread('path_to_test_image.jpg'))

        # Act & Assert
        with pytest.raises(PermissionError):
            main()

        # Clean up
        os.chmod(OUTPUT_DIR, 0o777)  # Restore write permissions
        shutil.rmtree(OUTPUT_DIR)
