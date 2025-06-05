import pytest
import os
import shutil
import random
from string import ascii_lowercase, digits
import cv2
import numpy as np
from mosaic_augmentation import main, get_dataset, update_image_and_anno, random_chars
from _test_MosaicAugmentationMain import Test_MosaicAugmentationMain

class Test_TestMosaicAugmentationMainTestMainLargeNumberOfImages(Test_MosaicAugmentationMain):

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        # Common setup and teardown code for the test class
        self.LABEL_DIR = 'test_data/labels'
        self.IMG_DIR = 'test_data/images'
        self.OUTPUT_DIR = 'test_output'
        self.NUMBER_IMAGES = 1000
        self.OUTPUT_SIZE = (720, 1280)
        self.SCALE_RANGE = (0.4, 0.6)
        self.FILTER_TINY_SCALE = 1 / 100

        # Setup: Create directories and dummy data if needed
        if not os.path.exists(self.LABEL_DIR):
            os.makedirs(self.LABEL_DIR)
            os.makedirs(self.IMG_DIR)
            os.makedirs(self.OUTPUT_DIR)

        # Generate dummy images and annotations
        for i in range(self.NUMBER_IMAGES):
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(self.IMG_DIR, f'image_{i}.jpg'), img)
            with open(os.path.join(self.LABEL_DIR, f'anno_{i}.txt'), 'w') as f:
                f.write('0 50 50 100 100\n')

        yield

        # Clean up
        shutil.rmtree(self.OUTPUT_DIR)
        shutil.rmtree(self.LABEL_DIR)
        shutil.rmtree(self.IMG_DIR)

    @pytest.mark.smoke
    def test_main_large_number_of_images(self):
        # Act
        main()

        # Assert
        output_files = os.listdir(self.OUTPUT_DIR)
        assert len(output_files) == self.NUMBER_IMAGES * 2
        assert all(file.endswith(('.jpg', '.txt')) for file in output_files)
