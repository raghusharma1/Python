import pytest
import os
import shutil
import random
from string import ascii_lowercase, digits
import cv2
import numpy as np
from mosaic_augmentation import main, get_dataset, update_image_and_anno, random_chars
from _test_MosaicAugmentationMain import Test_MosaicAugmentationMain

class Test_TestMosaicAugmentationMainTestMainMissingImageFiles(Test_MosaicAugmentationMain):

    def test_main_no_image_files(self):
        # Arrange
        LABEL_DIR = 'test_data/labels'
        IMG_DIR = 'test_data/no_images'
        OUTPUT_DIR = 'test_output'
        NUMBER_IMAGES = 10
        OUTPUT_SIZE = (720, 1280)
        SCALE_RANGE = (0.4, 0.6)
        FILTER_TINY_SCALE = 1 / 100

        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        # Create empty image and label directories
        os.makedirs(IMG_DIR, exist_ok=True)
        os.makedirs(LABEL_DIR, exist_ok=True)

        # Act
        main()

        # Assert
        output_files = os.listdir(OUTPUT_DIR)
        assert len(output_files) == 0

        # Clean up
        shutil.rmtree(OUTPUT_DIR)
        shutil.rmtree(IMG_DIR)
        shutil.rmtree(LABEL_DIR)

    def test_main_missing_annotation_files(self):
        # Arrange
        LABEL_DIR = 'test_data/labels'
        IMG_DIR = 'test_data/images'
        OUTPUT_DIR = 'test_output'
        NUMBER_IMAGES = 10
        OUTPUT_SIZE = (720, 1280)
        SCALE_RANGE = (0.4, 0.6)
        FILTER_TINY_SCALE = 1 / 100

        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        # Create image and label directories with missing annotation files
        os.makedirs(IMG_DIR, exist_ok=True)
        os.makedirs(LABEL_DIR, exist_ok=True)

        # Create some image files
        for i in range(5):
            image_path = os.path.join(IMG_DIR, f'image_{i}.jpg')
            cv2.imwrite(image_path, np.zeros((256, 256, 3), dtype=np.uint8))

        # Create some annotation files
        for i in range(3):
            label_path = os.path.join(LABEL_DIR, f'image_{i}.txt')
            with open(label_path, 'w') as f:
                f.write('0 0.5 0.5 0.1 0.1')

        # Act
        main()

        # Assert
        output_files = os.listdir(OUTPUT_DIR)
        assert len(output_files) == 6  # 3 images with annotations * 2 files per image

        # Clean up
        shutil.rmtree(OUTPUT_DIR)
        shutil.rmtree(IMG_DIR)
        shutil.rmtree(LABEL_DIR)

    def test_main_valid_inputs(self):
        # Arrange
        LABEL_DIR = 'test_data/labels'
        IMG_DIR = 'test_data/images'
        OUTPUT_DIR = 'test_output'
        NUMBER_IMAGES = 10
        OUTPUT_SIZE = (720, 1280)
        SCALE_RANGE = (0.4, 0.6)
        FILTER_TINY_SCALE = 1 / 100

        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        # Create image and label directories with valid inputs
        os.makedirs(IMG_DIR, exist_ok=True)
        os.makedirs(LABEL_DIR, exist_ok=True)

        # Create some image files
        for i in range(5):
            image_path = os.path.join(IMG_DIR, f'image_{i}.jpg')
            cv2.imwrite(image_path, np.zeros((256, 256, 3), dtype=np.uint8))

        # Create corresponding annotation files
        for i in range(5):
            label_path = os.path.join(LABEL_DIR, f'image_{i}.txt')
            with open(label_path, 'w') as f:
                f.write('0 0.5 0.5 0.1 0.1')

        # Act
        main()

        # Assert
        output_files = os.listdir(OUTPUT_DIR)
        assert len(output_files) == 20  # 10 images * 2 files per image

        # Clean up
        shutil.rmtree(OUTPUT_DIR)
        shutil.rmtree(IMG_DIR)
        shutil.rmtree(LABEL_DIR)

    def test_main_large_number_of_images(self):
        # Arrange
        LABEL_DIR = 'test_data/labels'
        IMG_DIR = 'test_data/images'
        OUTPUT_DIR = 'test_output'
        NUMBER_IMAGES = 100  # // TODO: Change this value if needed
        OUTPUT_SIZE = (720, 1280)
        SCALE_RANGE = (0.4, 0.6)
        FILTER_TINY_SCALE = 1 / 100

        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        # Create image and label directories with a large number of images
        os.makedirs(IMG_DIR, exist_ok=True)
        os.makedirs(LABEL_DIR, exist_ok=True)

        # Create a large number of image files
        for i in range(NUMBER_IMAGES):
            image_path = os.path.join(IMG_DIR, f'image_{i}.jpg')
            cv2.imwrite(image_path, np.zeros((256, 256, 3), dtype=np.uint8))

        # Create corresponding annotation files
        for i in range(NUMBER_IMAGES):
            label_path = os.path.join(LABEL_DIR, f'image_{i}.txt')
            with open(label_path, 'w') as f:
                f.write('0 0.5 0.5 0.1 0.1')

        # Act
        main()

        # Assert
        output_files = os.listdir(OUTPUT_DIR)
        assert len(output_files) == 200  # 100 images * 2 files per image

        # Clean up
        shutil.rmtree(OUTPUT_DIR)
        shutil.rmtree(IMG_DIR)
        shutil.rmtree(LABEL_DIR)

    def test_main_invalid_annotation_formats(self):
        # Arrange
        LABEL_DIR = 'test_data/labels'
        IMG_DIR = 'test_data/images'
        OUTPUT_DIR = 'test_output'
        NUMBER_IMAGES = 10
        OUTPUT_SIZE = (720, 1280)
        SCALE_RANGE = (0.4, 0.6)
        FILTER_TINY_SCALE = 1 / 100

        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        # Create image and label directories with invalid annotation formats
        os.makedirs(IMG_DIR, exist_ok=True)
        os.makedirs(LABEL_DIR, exist_ok=True)

        # Create some image files
        for i in range(5):
            image_path = os.path.join(IMG_DIR, f'image_{i}.jpg')
            cv2.imwrite(image_path, np.zeros((256, 256, 3), dtype=np.uint8))

        # Create some annotation files with invalid formats
        for i in range(5):
            label_path = os.path.join(LABEL_DIR, f'image_{i}.txt')
            with open(label_path, 'w') as f:
                f.write('invalid format')

        # Act
        main()

        # Assert
        output_files = os.listdir(OUTPUT_DIR)
        assert len(output_files) == 0  # No valid annotations, so no output files

        # Clean up
        shutil.rmtree(OUTPUT_DIR)
        shutil.rmtree(IMG_DIR)
        shutil.rmtree(LABEL_DIR)

    def test_main_filter_tiny_scale(self):
        # Arrange
        LABEL_DIR = 'test_data/labels'
        IMG_DIR = 'test_data/images'
        OUTPUT_DIR = 'test_output'
        NUMBER_IMAGES = 10
        OUTPUT_SIZE = (720, 1280)
        SCALE_RANGE = (0.4, 0.6)
        FILTER_TINY_SCALE = 0.1

        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        # Create image and label directories with valid inputs
        os.makedirs(IMG_DIR, exist_ok=True)
        os.makedirs(LABEL_DIR, exist_ok=True)

        # Create some image files
        for i in range(5):
            image_path = os.path.join(IMG_DIR, f'image_{i}.jpg')
            cv2.imwrite(image_path, np.zeros((256, 256, 3), dtype=np.uint8))

        # Create corresponding annotation files with tiny bounding boxes
        for i in range(5):
            label_path = os.path.join(LABEL_DIR, f'image_{i}.txt')
            with open(label_path, 'w') as f:
                f.write('0 0.5 0.5 0.05 0.05')

        # Act
        main()

        # Assert
        output_files = os.listdir(OUTPUT_DIR)
        assert len(output_files) == 0  # Tiny bounding boxes filtered out

        # Clean up
        shutil.rmtree(OUTPUT_DIR)
        shutil.rmtree(IMG_DIR)
        shutil.rmtree(LABEL_DIR)

    def test_main_output_file_naming(self):
        # Arrange
        LABEL_DIR = 'test_data/labels'
        IMG_DIR = 'test_data/images'
        OUTPUT_DIR = 'test_output'
        NUMBER_IMAGES = 10
        OUTPUT_SIZE = (720, 1280)
        SCALE_RANGE = (0.4, 0.6)
        FILTER_TINY_SCALE = 1 / 100

        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        # Create image and label directories with valid inputs
        os.makedirs(IMG_DIR, exist_ok=True)
        os.makedirs(LABEL_DIR, exist_ok=True)

        # Create some image files
        for i in range(5):
            image_path = os.path.join(IMG_DIR, f'image_{i}.jpg')
            cv2.imwrite(image_path, np.zeros((256, 256, 3), dtype=np.uint8))

        # Create corresponding annotation files
        for i in range(5):
            label_path = os.path.join(LABEL_DIR, f'image_{i}.txt')
            with open(label_path, 'w') as f:
                f.write('0 0.5 0.5 0.1 0.1')

        # Act
        main()

        # Assert
        output_files = os.listdir(OUTPUT_DIR)
        for file in output_files:
            assert file.endswith('.jpg') or file.endswith('.txt')
            if file.endswith('.jpg'):
                assert 'MOSAIC_' in file
            if file.endswith('.txt'):
                assert 'MOSAIC_' in file

        # Clean up
        shutil.rmtree(OUTPUT_DIR)
        shutil.rmtree(IMG_DIR)
        shutil.rmtree(LABEL_DIR)

    def test_main_missing_image_files(self):
        # Arrange
        LABEL_DIR = 'test_data/labels'
        IMG_DIR = 'test_data/missing_images'
        OUTPUT_DIR = 'test_output'
        NUMBER_IMAGES = 10
        OUTPUT_SIZE = (720, 1280)
        SCALE_RANGE = (0.4, 0.6)
        FILTER_TINY_SCALE = 1 / 100

        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        # Create label directory with annotation files
        os.makedirs(LABEL_DIR, exist_ok=True)
        # IMG_DIR is intentionally missing

        # Create some annotation files
        for i in range(5):
            label_path = os.path.join(LABEL_DIR, f'image_{i}.txt')
            with open(label_path, 'w') as f:
                f.write('0 0.5 0.5 0.1 0.1')

        # Act
        main()

        # Assert
        output_files = os.listdir(OUTPUT_DIR)
        assert len(output_files) == 0

        # Clean up
        shutil.rmtree(OUTPUT_DIR)
        shutil.rmtree(LABEL_DIR)
