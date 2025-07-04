import pytest
import os
import random
import cv2
import numpy as np
from mosaic_augmentation import main
from mosaic_augmentation import get_dataset, update_image_and_anno, random_chars

class Test_MosaicAugmentationMain:

    @pytest.mark.valid
    @pytest.mark.smoke
    def test_successful_image_and_annotation_update(self, capsys):
        # Arrange
        LABEL_DIR = 'test_data/labels'
        IMG_DIR = 'test_data/images'
        OUTPUT_DIR = 'test_output'
        NUMBER_IMAGES = 2
        OUTPUT_SIZE = (720, 1280)
        SCALE_RANGE = (0.4, 0.6)
        FILTER_TINY_SCALE = 1 / 100

        # Act
        main()

        # Assert
        output_files = os.listdir(OUTPUT_DIR)
        assert len(output_files) == NUMBER_IMAGES * 2  # Each image has a corresponding annotation file
        captured = capsys.readouterr()
        assert f"Succeeded {NUMBER_IMAGES}/{NUMBER_IMAGES}" in captured.out

    @pytest.mark.invalid
    @pytest.mark.negative
    def test_empty_annotation_files(self):
        # Arrange
        LABEL_DIR = 'test_data/empty_labels'
        IMG_DIR = 'test_data/images'
        OUTPUT_DIR = 'test_output'
        NUMBER_IMAGES = 2
        OUTPUT_SIZE = (720, 1280)
        SCALE_RANGE = (0.4, 0.6)
        FILTER_TINY_SCALE = 1 / 100

        # Act & Assert
        try:
            main()
        except Exception as e:
            pytest.fail(f"main() raised an exception: {e}")

    @pytest.mark.valid
    @pytest.mark.regression
    def test_random_image_selection(self):
        # Arrange
        LABEL_DIR = 'test_data/labels'
        IMG_DIR = 'test_data/images'
        OUTPUT_DIR = 'test_output'
        NUMBER_IMAGES = 2
        OUTPUT_SIZE = (720, 1280)
        SCALE_RANGE = (0.4, 0.6)
        FILTER_TINY_SCALE = 1 / 100

        # Act
        main()
        output_files_run1 = set(os.listdir(OUTPUT_DIR))

        # Clear output directory for the second run
        for file in output_files_run1:
            os.remove(os.path.join(OUTPUT_DIR, file))

        main()
        output_files_run2 = set(os.listdir(OUTPUT_DIR))

        # Assert
        assert output_files_run1 != output_files_run2

    @pytest.mark.valid
    @pytest.mark.smoke
    def test_output_file_naming_convention(self):
        # Arrange
        LABEL_DIR = 'test_data/labels'
        IMG_DIR = 'test_data/images'
        OUTPUT_DIR = 'test_output'
        NUMBER_IMAGES = 2
        OUTPUT_SIZE = (720, 1280)
        SCALE_RANGE = (0.4, 0.6)
        FILTER_TINY_SCALE = 1 / 100

        # Act
        main()

        # Assert
        output_files = os.listdir(OUTPUT_DIR)
        for file in output_files:
            if file.endswith('.jpg'):
                assert file.startswith('test_image_MOSAIC_') and len(file.split('_')[-1].split('.')[0]) == 32
            elif file.endswith('.txt'):
                assert file.startswith('test_image_MOSAIC_') and len(file.split('_')[-1].split('.')[0]) == 32

    @pytest.mark.valid
    @pytest.mark.regression
    def test_filter_scale_application(self):
        # Arrange
        LABEL_DIR = 'test_data/labels'
        IMG_DIR = 'test_data/images'
        OUTPUT_DIR = 'test_output'
        NUMBER_IMAGES = 2
        OUTPUT_SIZE = (720, 1280)
        SCALE_RANGE = (0.4, 0.6)
        FILTER_TINY_SCALE = 0.1

        # Act
        main()

        # Assert
        for file in os.listdir(OUTPUT_DIR):
            if file.endswith('.txt'):
                with open(os.path.join(OUTPUT_DIR, file), 'r') as f:
                    for line in f:
                        bbox = list(map(float, line.split()[1:]))
                        width = bbox[2]
                        height = bbox[3]
                        assert width > FILTER_TINY_SCALE and height > FILTER_TINY_SCALE

    @pytest.mark.invalid
    @pytest.mark.negative
    def test_insufficient_images_for_mosaicking(self):
        # Arrange
        LABEL_DIR = 'test_data/insufficient_labels'
        IMG_DIR = 'test_data/insufficient_images'
        OUTPUT_DIR = 'test_output'
        NUMBER_IMAGES = 2
        OUTPUT_SIZE = (720, 1280)
        SCALE_RANGE = (0.4, 0.6)
        FILTER_TINY_SCALE = 1 / 100

        # Act & Assert
        try:
            main()
        except Exception as e:
            pytest.fail(f"main() raised an exception: {e}")

    @pytest.mark.valid
    @pytest.mark.regression
    def test_image_and_annotation_consistency(self):
        # Arrange
        LABEL_DIR = 'test_data/labels'
        IMG_DIR = 'test_data/images'
        OUTPUT_DIR = 'test_output'
        NUMBER_IMAGES = 2
        OUTPUT_SIZE = (720, 1280)
        SCALE_RANGE = (0.4, 0.6)
        FILTER_TINY_SCALE = 1 / 100

        # Act
        main()

        # Assert
        for file in os.listdir(OUTPUT_DIR):
            if file.endswith('.jpg'):
                img = cv2.imread(os.path.join(OUTPUT_DIR, file))
                assert img.shape == (OUTPUT_SIZE[0], OUTPUT_SIZE[1], 3)
            elif file.endswith('.txt'):
                with open(os.path.join(OUTPUT_DIR, file), 'r') as f:
                    for line in f:
                        bbox = list(map(float, line.split()[1:]))
                        assert 0 <= bbox[0] <= 1 and 0 <= bbox[1] <= 1 and 0 <= bbox[2] <= 1 and 0 <= bbox[3] <= 1

    @pytest.mark.performance
    def test_performance_with_large_datasets(self):
        # Arrange
        LABEL_DIR = 'test_data/large_labels'
        IMG_DIR = 'test_data/large_images'
        OUTPUT_DIR = 'test_output'
        NUMBER_IMAGES = 100
        OUTPUT_SIZE = (720, 1280)
        SCALE_RANGE = (0.4, 0.6)
        FILTER_TINY_SCALE = 1 / 100

        # Act
        import time
        start_time = time.time()
        main()
        end_time = time.time()

        # Assert
        assert end_time - start_time < 60  # TODO: Adjust the acceptable time frame based on performance requirements
