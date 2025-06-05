import pytest
import os
import shutil
import glob
import random
from string import ascii_lowercase, digits
import cv2
import numpy
from flip_augmentation import main, get_dataset, update_image_and_anno, random_chars

class Test_FlipAugmentationMain:
    @pytest.fixture(autouse=True)
    def setup(self):
        # Create temporary directories for testing
        self.label_dir = 'test_labels'
        self.img_dir = 'test_images'
        self.output_dir = 'test_output'
        os.makedirs(self.label_dir, exist_ok=True)
        os.makedirs(self.img_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        # Clean up directories after tests
        yield
        shutil.rmtree(self.label_dir)
        shutil.rmtree(self.img_dir)
        shutil.rmtree(self.output_dir)

    @pytest.mark.valid
    @pytest.mark.positive
    def test_process_images_and_annotations_correctly(self):
        # Arrange
        # Prepare valid directories for labels and images with corresponding annotations.
        # TODO: Update with actual test data paths
        LABEL_DIR = self.label_dir
        IMAGE_DIR = self.img_dir
        OUTPUT_DIR = self.output_dir
        FLIP_TYPE = 1

        # Create dummy data for testing
        img_path = os.path.join(IMAGE_DIR, 'test_image.jpg')
        label_path = os.path.join(LABEL_DIR, 'test_image.txt')
        cv2.imwrite(img_path, numpy.zeros((100, 100, 3), dtype=numpy.uint8))  # Create a dummy image
        with open(label_path, 'w') as f:
            f.write('0 0.5 0.5 0.1 0.1\n')  # Create a dummy annotation

        # Act
        main()  # Invoke the function

        # Assert
        # Verify that the new images and annotations are saved in the output directory with the correct format and content.
        output_img_path = glob.glob(os.path.join(OUTPUT_DIR, 'test_image_FLIP_*.jpg'))[0]
        output_label_path = glob.glob(os.path.join(OUTPUT_DIR, 'test_image_FLIP_*.txt'))[0]

        assert os.path.exists(output_img_path)
        assert os.path.exists(output_label_path)

        with open(output_label_path, 'r') as f:
            output_annotation = f.readline().strip()
            assert output_annotation == '0 0.5 0.5 0.1 0.1'  # Verify the annotation content

        # Clean up test data
        os.remove(img_path)
        os.remove(label_path)
