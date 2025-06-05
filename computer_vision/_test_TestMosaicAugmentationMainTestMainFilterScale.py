import pytest
import os
import shutil
import random
from string import ascii_lowercase, digits
import cv2
import numpy as np
from mosaic_augmentation import main, get_dataset, update_image_and_anno, random_chars
from _test_MosaicAugmentationMain import Test_MosaicAugmentationMain

class Test_TestMosaicAugmentationMainTestMainFilterScale(Test_MosaicAugmentationMain):

    @pytest.mark.smoke
    @pytest.mark.valid
    @pytest.mark.positive
    def test_filter_scale_applied(self):
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

        # Act
        main()

        # Assert
        anno_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.txt')]
        for anno_file in anno_files:
            with open(os.path.join(OUTPUT_DIR, anno_file), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    _, x_center, y_center, width, height = line.split()
                    assert float(width) >= FILTER_TINY_SCALE
                    assert float(height) >= FILTER_TINY_SCALE

        # Clean up
        shutil.rmtree(OUTPUT_DIR)
