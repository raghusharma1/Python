import pytest
import os
import shutil
import random
from string import ascii_lowercase, digits
import cv2
import numpy as np
from _test_MosaicAugmentationMain import Test_MosaicAugmentationMain
from mosaic_augmentation import main, get_dataset, update_image_and_anno, random_chars

class Test_TestMosaicAugmentationMainTestMainNonexistentOutputDir(Test_MosaicAugmentationMain):

    @pytest.mark.invalid
    @pytest.mark.negative
    @pytest.mark.smoke
    def test_main_nonexistent_output_dir(self):
        # Arrange
        LABEL_DIR = 'test_data/labels'
        IMG_DIR = 'test_data/images'
        OUTPUT_DIR = 'nonexistent_output'
        NUMBER_IMAGES = 10
        OUTPUT_SIZE = (720, 1280)
        SCALE_RANGE = (0.4, 0.6)
        FILTER_TINY_SCALE = 1 / 100

        # Ensure that OUTPUT_DIR does not exist before the test runs
        if os.path.exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR)

        # Act & Assert
        with pytest.raises(FileNotFoundError):
            main()
