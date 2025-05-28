import pytest
import os
import shutil
import cv2
import glob
import numpy as np
import random
from string import ascii_lowercase, digits
from flip_augmentation import main, get_dataset, update_image_and_anno, random_chars

# Global variables for testing
LABEL_DIR = ''
IMAGE_DIR = ''
OUTPUT_DIR = ''
FLIP_TYPE = 1

@pytest.fixture(autouse=True)
def setup_teardown():
    # Setup: Create temporary directories for testing
    global LABEL_DIR, IMAGE_DIR, OUTPUT_DIR
    LABEL_DIR = 'test_labels'
    IMAGE_DIR = 'test_images'
    OUTPUT_DIR = 'test_output'
    os.makedirs(LABEL_DIR, exist_ok=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    yield

    # Teardown: Clean up temporary directories
    shutil.rmtree(LABEL_DIR)
    shutil.rmtree(IMAGE_DIR)
    shutil.rmtree(OUTPUT_DIR)

class Test_FlipAugmentationMain:
    @pytest.mark.positive
    def test_main_successful_execution(self):
        # Arrange
        img_paths = [f'{IMAGE_DIR}/img1.jpg', f'{IMAGE_DIR}/img2.jpg']
        annos = [[[0, 0.5, 0.5, 0.1, 0.1]], [[0, 0.5, 0.5, 0.1, 0.1]]]
        for img_path in img_paths:
            cv2.imwrite(img_path, np.zeros((100, 100, 3), dtype=np.uint8))
        for i, anno in enumerate(annos):
            with open(f'{LABEL_DIR}/label{i+1}.txt', 'w') as f:
                for a in anno:
                    f.write(f"{a[0]} {a[1]} {a[2]} {a[3]} {a[4]}\n")

        # Act
        main()

        # Assert
        output_images = glob.glob(f'{OUTPUT_DIR}/*.jpg')
        output_labels = glob.glob(f'{OUTPUT_DIR}/*.txt')
        assert len(output_images) == 2
        assert len(output_labels) == 2
        for img in output_images:
            assert cv2.imread(img) is not None

    @pytest.mark.negative
    def test_main_missing_images(self):
        # Arrange
        img_paths = [f'{IMAGE_DIR}/img1.jpg']
        annos = [[[0, 0.5, 0.5, 0.1, 0.1]]]
        cv2.imwrite(img_paths[0], np.zeros((100, 100, 3), dtype=np.uint8))
        with open(f'{LABEL_DIR}/label1.txt', 'w') as f:
            f.write("0 0.5 0.5 0.1 0.1\n")

        # Act
        main()

        # Assert
        output_images = glob.glob(f'{OUTPUT_DIR}/*.jpg')
        output_labels = glob.glob(f'{OUTPUT_DIR}/*.txt')
        assert len(output_images) == 1
        assert len(output_labels) == 1

    @pytest.mark.negative
    def test_main_empty_annotations(self):
        # Arrange
        img_paths = [f'{IMAGE_DIR}/img1.jpg']
        annos = [[]]
        cv2.imwrite(img_paths[0], np.zeros((100, 100, 3), dtype=np.uint8))
        open(f'{LABEL_DIR}/label1.txt', 'w').close()

        # Act
        main()

        # Assert
        output_images = glob.glob(f'{OUTPUT_DIR}/*.jpg')
        output_labels = glob.glob(f'{OUTPUT_DIR}/*.txt')
        assert len(output_images) == 0
        assert len(output_labels) == 0

    @pytest.mark.negative
    def test_main_invalid_flip_type(self):
        # Arrange
        global FLIP_TYPE
        FLIP_TYPE = -1  # Invalid flip type

        # Act & Assert
        with pytest.raises(ValueError):
            main()

    @pytest.mark.performance
    def test_main_large_dataset(self):
        # Arrange
        num_images = 1000
        for i in range(num_images):
            img_path = f'{IMAGE_DIR}/img{i+1}.jpg'
            cv2.imwrite(img_path, np.zeros((100, 100, 3), dtype=np.uint8))
            with open(f'{LABEL_DIR}/label{i+1}.txt', 'w') as f:
                f.write("0 0.5 0.5 0.1 0.1\n")

        # Act
        import time
        start_time = time.time()
        main()
        end_time = time.time()

        # Assert
        assert end_time - start_time < 60  # TODO: Adjust the time limit based on performance requirements

    @pytest.mark.positive
    def test_main_special_characters_in_filenames(self):
        # Arrange
        img_paths = [f'{IMAGE_DIR}/img@#$%.jpg']
        annos = [[[0, 0.5, 0.5, 0.1, 0.1]]]
        cv2.imwrite(img_paths[0], np.zeros((100, 100, 3), dtype=np.uint8))
        with open(f'{LABEL_DIR}/label@#$%.txt', 'w') as f:
            f.write("0 0.5 0.5 0.1 0.1\n")

        # Act
        main()

        # Assert
        output_images = glob.glob(f'{OUTPUT_DIR}/*.jpg')
        output_labels = glob.glob(f'{OUTPUT_DIR}/*.txt')
        assert len(output_images) == 1
        assert len(output_labels) == 1

    @pytest.mark.positive
    def test_main_non_jpeg_images(self):
        # Arrange
        img_paths = [f'{IMAGE_DIR}/img1.png']
        annos = [[[0, 0.5, 0.5, 0.1, 0.1]]]
        cv2.imwrite(img_paths[0], np.zeros((100, 100, 3), dtype=np.uint8), [cv2.IMWRITE_PNG_COMPRESSION, 0])
        with open(f'{LABEL_DIR}/label1.txt', 'w') as f:
            f.write("0 0.5 0.5 0.1 0.1\n")

        # Act
        main()

        # Assert
        output_images = glob.glob(f'{OUTPUT_DIR}/*.jpg')
        output_labels = glob.glob(f'{OUTPUT_DIR}/*.txt')
        assert len(output_images) == 1
        assert len(output_labels) == 1

    @pytest.mark.negative
    def test_main_corrupted_images(self):
        # Arrange
        img_paths = [f'{IMAGE_DIR}/img1.jpg']
        annos = [[[0, 0.5, 0.5, 0.1, 0.1]]]
        with open(img_paths[0], 'wb') as f:
            f.write(b'corrupted data')
        with open(f'{LABEL_DIR}/label1.txt', 'w') as f:
            f.write("0 0.5 0.5 0.1 0.1\n")

        # Act
        main()

        # Assert
        output_images = glob.glob(f'{OUTPUT_DIR}/*.jpg')
        output_labels = glob.glob(f'{OUTPUT_DIR}/*.txt')
        assert len(output_images) == 0
        assert len(output_labels) == 0

    @pytest.mark.performance
    def test_main_large_images(self):
        # Arrange
        img_paths = [f'{IMAGE_DIR}/img1.jpg']
        annos = [[[0, 0.5, 0.5, 0.1, 0.1]]]
        cv2.imwrite(img_paths[0], np.zeros((2000, 2000, 3), dtype=np.uint8))
        with open(f'{LABEL_DIR}/label1.txt', 'w') as f:
            f.write("0 0.5 0.5 0.1 0.1\n")

        # Act
        import time
        start_time = time.time()
        main()
        end_time = time.time()

        # Assert
        assert end_time - start_time < 60  # TODO: Adjust the time limit based on performance requirements

    @pytest.mark.positive
    def test_main_multiple_annotation_files(self):
        # Arrange
        img_paths = [f'{IMAGE_DIR}/img1.jpg']
        annos = [[[0, 0.5, 0.5, 0.1, 0.1]]]
        cv2.imwrite(img_paths[0], np.zeros((100, 100, 3), dtype=np.uint8))
        with open(f'{LABEL_DIR}/label1.txt', 'w') as f:
            f.write("0 0.5 0.5 0.1 0.1\n")
        with open(f'{LABEL_DIR}/label2.txt', 'w') as f:
            f.write("0 0.6 0.6 0.1 0.1\n")

        # Act
        main()

        # Assert
        output_images = glob.glob(f'{OUTPUT_DIR}/*.jpg')
        output_labels = glob.glob(f'{OUTPUT_DIR}/*.txt')
        assert len(output_images) == 1
        assert len(output_labels) == 1

    @pytest.mark.positive
    def test_main_non_existing_output_directory(self):
        # Arrange
        global OUTPUT_DIR
        OUTPUT_DIR = 'non_existing_output_dir'

        # Act
        main()

        # Assert
        assert os.path.exists(OUTPUT_DIR)
        output_images = glob.glob(f'{OUTPUT_DIR}/*.jpg')
        output_labels = glob.glob(f'{OUTPUT_DIR}/*.txt')
        assert len(output_images) > 0
        assert len(output_labels) > 0

    @pytest.mark.negative
    def test_main_read_only_output_directory(self):
        # Arrange
        global OUTPUT_DIR
        OUTPUT_DIR = 'read_only_output_dir'
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.chmod(OUTPUT_DIR, 0o444)  # Set read-only permissions

        # Act & Assert
        with pytest.raises(PermissionError):
            main()
