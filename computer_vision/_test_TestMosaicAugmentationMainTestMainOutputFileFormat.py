import pytest
import os
import shutil
import random
from string import ascii_lowercase, digits
import cv2
import numpy as np
from mosaic_augmentation import main, get_dataset, update_image_and_anno, random_chars
from _test_MosaicAugmentationMain import Test_MosaicAugmentationMain

class Test_TestMosaicAugmentationMainTestMainOutputFileFormat:

    @pytest.mark.valid
    @pytest.mark.smoke
    def test_correct_file_creation(self):
        # Arrange
        LABEL_DIR = 'test_data/labels'
        IMG_DIR = 'test_data/images'
        OUTPUT_DIR = 'test_output'
        NUMBER_IMAGES = 10
        OUTPUT_SIZE = (720, 1280)
        SCALE_RANGE = (0.4, 0.6)
        FILTER_TINY_SCALE = 1 / 100

        if not os.path.exists(LABEL_DIR):
            os.makedirs(LABEL_DIR)
        if not os.path.exists(IMG_DIR):
            os.makedirs(IMG_DIR)

        # Create sample images and annotations
        for i in range(20):  # Create more than NUMBER_IMAGES to ensure enough data for mosaics
            img_path = os.path.join(IMG_DIR, f'image_{i}.jpg')
            label_path = os.path.join(LABEL_DIR, f'image_{i}.txt')

            img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(img_path, img)

            with open(label_path, 'w') as f:
                f.write('0 50 50 80 80\n')

        # Act
        main()

        # Assert
        img_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.jpg')]
        anno_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.txt')]

        assert len(img_files) == NUMBER_IMAGES
        assert len(anno_files) == NUMBER_IMAGES

        # Clean up
        shutil.rmtree(LABEL_DIR)
        shutil.rmtree(IMG_DIR)
        shutil.rmtree(OUTPUT_DIR)

    @pytest.mark.invalid
    @pytest.mark.regression
    def test_main_output_file_format(self):
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

        # Act
        main()

        # Assert
        img_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.jpg')]
        for img_file in img_files:
            img = cv2.imread(os.path.join(OUTPUT_DIR, img_file))
            assert img is not None

        anno_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.txt')]
        for anno_file in anno_files:
            with open(os.path.join(OUTPUT_DIR, anno_file), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.split()
                    assert len(parts) == 5
                    assert all(isinstance(float(part), float) for part in parts[1:])

        # Clean up
        shutil.rmtree(OUTPUT_DIR)
