import pytest
import os
import shutil
import random
from string import ascii_lowercase, digits
import cv2
import numpy as np
from mosaic_augmentation import main, get_dataset, update_image_and_anno, random_chars

from _test_MosaicAugmentationMain import Test_MosaicAugmentationMain

class Test_TestMosaicAugmentationMainTestMainEmptyLabelDir(Test_MosaicAugmentationMain):

    @pytest.mark.smoke
    @pytest.mark.valid
    def test_main_no_images(self):
        # Arrange
        LABEL_DIR = 'test_data/labels'
        IMG_DIR = 'test_data/empty_images'
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
        output_files = os.listdir(OUTPUT_DIR)
        assert len(output_files) == 0

        # Clean up
        shutil.rmtree(OUTPUT_DIR)

    @pytest.mark.smoke
    @pytest.mark.valid
    def test_main_no_labels(self):
        # Arrange
        LABEL_DIR = 'test_data/empty_labels'
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
        output_files = os.listdir(OUTPUT_DIR)
        assert len(output_files) == 0

        # Clean up
        shutil.rmtree(OUTPUT_DIR)

    @pytest.mark.smoke
    @pytest.mark.negative
    def test_main_mismatched_files(self):
        # Arrange
        LABEL_DIR = 'test_data/labels'
        IMG_DIR = 'test_data/mismatched_images'
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
        output_files = os.listdir(OUTPUT_DIR)
        assert len(output_files) == 0

        # Clean up
        shutil.rmtree(OUTPUT_DIR)

    @pytest.mark.regression
    @pytest.mark.valid
    def test_main_large_number_of_images(self):
        # Arrange
        LABEL_DIR = 'test_data/large_labels'
        IMG_DIR = 'test_data/large_images'
        OUTPUT_DIR = 'test_output'
        NUMBER_IMAGES = 1000
        OUTPUT_SIZE = (720, 1280)
        SCALE_RANGE = (0.4, 0.6)
        FILTER_TINY_SCALE = 1 / 100

        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        # Act
        main()

        # Assert
        output_files = os.listdir(OUTPUT_DIR)
        assert len(output_files) == NUMBER_IMAGES * 2  # Assuming each image generates one txt and one jpg file

        # Clean up
        shutil.rmtree(OUTPUT_DIR)

    @pytest.mark.regression
    @pytest.mark.valid
    def test_main_filter_tiny_scale_zero(self):
        # Arrange
        LABEL_DIR = 'test_data/labels'
        IMG_DIR = 'test_data/images'
        OUTPUT_DIR = 'test_output'
        NUMBER_IMAGES = 10
        OUTPUT_SIZE = (720, 1280)
        SCALE_RANGE = (0.4, 0.6)
        FILTER_TINY_SCALE = 0

        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        # Act
        main()

        # Assert
        output_files = os.listdir(OUTPUT_DIR)
        assert len(output_files) == NUMBER_IMAGES * 2  # Assuming each image generates one txt and one jpg file

        # Clean up
        shutil.rmtree(OUTPUT_DIR)

    @pytest.mark.regression
    @pytest.mark.valid
    def test_main_specific_scale_range(self):
        # Arrange
        LABEL_DIR = 'test_data/labels'
        IMG_DIR = 'test_data/images'
        OUTPUT_DIR = 'test_output'
        NUMBER_IMAGES = 10
        OUTPUT_SIZE = (720, 1280)
        SCALE_RANGE = (0.5, 0.5)
        FILTER_TINY_SCALE = 1 / 100

        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        # Act
        main()

        # Assert
        output_files = os.listdir(OUTPUT_DIR)
        assert len(output_files) == NUMBER_IMAGES * 2  # Assuming each image generates one txt and one jpg file

        # Clean up
        shutil.rmtree(OUTPUT_DIR)

    @pytest.mark.regression
    @pytest.mark.negative
    def test_main_invalid_output_size(self):
        # Arrange
        LABEL_DIR = 'test_data/labels'
        IMG_DIR = 'test_data/images'
        OUTPUT_DIR = 'test_output'
        NUMBER_IMAGES = 10
        OUTPUT_SIZE = (-720, 1280)  # Invalid size
        SCALE_RANGE = (0.4, 0.6)
        FILTER_TINY_SCALE = 1 / 100

        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        # Act & Assert
        with pytest.raises(ValueError):
            main()

        # Clean up
        shutil.rmtree(OUTPUT_DIR)

    @pytest.mark.regression
    @pytest.mark.valid
    def test_main_specific_output_directory(self):
        # Arrange
        LABEL_DIR = 'test_data/labels'
        IMG_DIR = 'test_data/images'
        OUTPUT_DIR = 'specific_output_directory'
        NUMBER_IMAGES = 10
        OUTPUT_SIZE = (720, 1280)
        SCALE_RANGE = (0.4, 0.6)
        FILTER_TINY_SCALE = 1 / 100

        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        # Act
        main()

        # Assert
        output_files = os.listdir(OUTPUT_DIR)
        assert len(output_files) == NUMBER_IMAGES * 2  # Assuming each image generates one txt and one jpg file

        # Clean up
        shutil.rmtree(OUTPUT_DIR)

    @pytest.mark.regression
    @pytest.mark.valid
    def test_main_existing_output_directory(self):
        # Arrange
        LABEL_DIR = 'test_data/labels'
        IMG_DIR = 'test_data/images'
        OUTPUT_DIR = 'existing_output_directory'
        NUMBER_IMAGES = 10
        OUTPUT_SIZE = (720, 1280)
        SCALE_RANGE = (0.4, 0.6)
        FILTER_TINY_SCALE = 1 / 100

        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        # Act
        main()

        # Assert
        output_files = os.listdir(OUTPUT_DIR)
        assert len(output_files) == NUMBER_IMAGES * 2  # Assuming each image generates one txt and one jpg file

        # Clean up
        shutil.rmtree(OUTPUT_DIR)

    @pytest.mark.regression
    @pytest.mark.valid
    def test_main_random_number_of_images(self):
        # Arrange
        LABEL_DIR = 'test_data/labels'
        IMG_DIR = 'test_data/images'
        OUTPUT_DIR = 'test_output'
        NUMBER_IMAGES = random.randint(1, 100)
        OUTPUT_SIZE = (720, 1280)
        SCALE_RANGE = (0.4, 0.6)
        FILTER_TINY_SCALE = 1 / 100

        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        # Act
        main()

        # Assert
        output_files = os.listdir(OUTPUT_DIR)
        assert len(output_files) == NUMBER_IMAGES * 2  # Assuming each image generates one txt and one jpg file

        # Clean up
        shutil.rmtree(OUTPUT_DIR)

    def test_main_empty_label_dir(self):
        # Arrange
        LABEL_DIR = 'test_data/empty_labels'
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
        output_files = os.listdir(OUTPUT_DIR)
        assert len(output_files) == 0

        # Clean up
        shutil.rmtree(OUTPUT_DIR)
