import pytest
import os
import shutil
import random
from string import ascii_lowercase, digits
import cv2
import numpy as np
from mosaic_augmentation import main, get_dataset, update_image_and_anno, random_chars

class Test_TestMosaicAugmentationMainTestMainAnnotationFormat:

    @pytest.mark.valid
    @pytest.mark.smoke
    def test_main_annotation_format(self):
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
        anno_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.txt')]
        for anno_file in anno_files:
            with open(os.path.join(OUTPUT_DIR, anno_file), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.split()
                    assert len(parts) == 5
                    assert parts[0].isdigit(), f"Expected an integer but got {parts[0]}"
                    assert all(parts[i].replace('.', '', 1).isdigit() for i in range(1, 5)), f"Expected floats but got {parts[1:]}"

        # Clean up
        shutil.rmtree(OUTPUT_DIR)
