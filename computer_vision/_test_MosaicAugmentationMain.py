import pytest
import os
import shutil
import random
from string import ascii_lowercase, digits
import cv2
import numpy as np
from mosaic_augmentation import main, get_dataset, update_image_and_anno, random_chars

class Test_MosaicAugmentationMain:
    @pytest.mark.smoke
    @pytest.mark.positive
    def test_main_successful_execution(self):
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
        output_files = os.listdir(OUTPUT_DIR)
        assert len(output_files) == NUMBER_IMAGES * 2

        img_files = [f for f in output_files if f.endswith('.jpg')]
        assert len(img_files) == NUMBER_IMAGES

        anno_files = [f for f in output_files if f.endswith('.txt')]
        assert len(anno_files) == NUMBER_IMAGES

        for img_file in img_files:
            img_path = os.path.join(OUTPUT_DIR, img_file)
            img = cv2.imread(img_path)
            assert img.shape[:2] == OUTPUT_SIZE

        # Clean up
        shutil.rmtree(OUTPUT_DIR)

    @pytest.mark.regression
    @pytest.mark.negative
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

    @pytest.mark.regression
    @pytest.mark.negative
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

        # Act
        main()

        # Assert
        output_files = os.listdir(OUTPUT_DIR)
        assert len(output_files) > 0

        # Clean up
        shutil.rmtree(OUTPUT_DIR)

    @pytest.mark.regression
    @pytest.mark.positive
    def test_main_filter_scale(self):
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

    @pytest.mark.performance
    @pytest.mark.positive
    def test_main_large_number_of_images(self):
        # Arrange
        LABEL_DIR = 'test_data/labels'
        IMG_DIR = 'test_data/images'
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
        assert len(output_files) == NUMBER_IMAGES * 2

        # Clean up
        shutil.rmtree(OUTPUT_DIR)

    @pytest.mark.regression
    @pytest.mark.positive
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

    @pytest.mark.regression
    @pytest.mark.negative
    def test_main_invalid_output_size(self):
        # Arrange
        LABEL_DIR = 'test_data/labels'
        IMG_DIR = 'test_data/images'
        OUTPUT_DIR = 'test_output'
        NUMBER_IMAGES = 10
        OUTPUT_SIZE = (0, 0)
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
    @pytest.mark.positive
    def test_main_random_string_generation(self):
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
            file_name, _ = os.path.splitext(img_file)
            random_string = file_name.split('_')[-1]
            assert len(random_string) == 32
            assert set(random_string).issubset(set(ascii_lowercase + digits))

        # Clean up
        shutil.rmtree(OUTPUT_DIR)

    @pytest.mark.regression
    @pytest.mark.negative
    def test_main_nonexistent_output_dir(self):
        # Arrange
        LABEL_DIR = 'test_data/labels'
        IMG_DIR = 'test_data/images'
        OUTPUT_DIR = 'nonexistent_output'
        NUMBER_IMAGES = 10
        OUTPUT_SIZE = (720, 1280)
        SCALE_RANGE = (0.4, 0.6)
        FILTER_TINY_SCALE = 1 / 100

        # Act & Assert
        with pytest.raises(FileNotFoundError):
            main()

    @pytest.mark.regression
    @pytest.mark.positive
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
                    assert all(isinstance(float(part), float) for part in parts[1:])

        # Clean up
        shutil.rmtree(OUTPUT_DIR)
