import pytest
import glob
import os
import random
from string import ascii_lowercase, digits
import cv2
import numpy as np
from mosaic_augmentation import main, update_image_and_anno, random_chars, get_dataset

class Test_MosaicAugmentationMain:

    @pytest.mark.valid
    def test_main_successful_update(self, tmp_path):
        # Arrange
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        label_dir = tmp_path / "labels"
        label_dir.mkdir()
        img_paths = [image_dir / f"image{i}.jpg" for i in range(4)]
        label_paths = [label_dir / f"image{i}.txt" for i in range(4)]
        for img_path in img_paths:
            cv2.imwrite(str(img_path), np.zeros((100, 100, 3), dtype=np.uint8))
        for label_path in label_paths:
            with open(label_path, "w") as f:
                f.write("0 0.5 0.5 0.1 0.1\n")

        global OUTPUT_SIZE, SCALE_RANGE, FILTER_TINY_SCALE, LABEL_DIR, IMG_DIR, OUTPUT_DIR, NUMBER_IMAGES
        OUTPUT_SIZE = (720, 1280)
        SCALE_RANGE = (0.4, 0.6)
        FILTER_TINY_SCALE = 1 / 100
        LABEL_DIR = str(label_dir)
        IMG_DIR = str(image_dir)
        OUTPUT_DIR = str(output_dir)
        NUMBER_IMAGES = 1

        # Act
        main()

        # Assert
        output_images = list(glob.glob(os.path.join(OUTPUT_DIR, "*.jpg")))
        output_labels = list(glob.glob(os.path.join(OUTPUT_DIR, "*.txt")))
        assert len(output_images) == 1
        assert len(output_labels) == 1

    @pytest.mark.invalid
    def test_main_empty_image_list(self, tmp_path, capsys):
        # Arrange
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        label_dir = tmp_path / "labels"
        label_dir.mkdir()

        global OUTPUT_SIZE, SCALE_RANGE, FILTER_TINY_SCALE, LABEL_DIR, IMG_DIR, OUTPUT_DIR, NUMBER_IMAGES
        OUTPUT_SIZE = (720, 1280)
        SCALE_RANGE = (0.4, 0.6)
        FILTER_TINY_SCALE = 1 / 100
        LABEL_DIR = str(label_dir)
        IMG_DIR = str(image_dir)
        OUTPUT_DIR = str(output_dir)
        NUMBER_IMAGES = 1

        # Act
        main()

        # Assert
        captured = capsys.readouterr()
        assert "No images were processed" in captured.out

    @pytest.mark.valid
    def test_main_filter_scale(self, tmp_path):
        # Arrange
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        label_dir = tmp_path / "labels"
        label_dir.mkdir()
        img_paths = [image_dir / f"image{i}.jpg" for i in range(4)]
        label_paths = [label_dir / f"image{i}.txt" for i in range(4)]
        for img_path in img_paths:
            cv2.imwrite(str(img_path), np.zeros((100, 100, 3), dtype=np.uint8))
        for label_path in label_paths:
            with open(label_path, "w") as f:
                f.write("0 0.5 0.5 0.01 0.01\n")  # Small bounding box

        global OUTPUT_SIZE, SCALE_RANGE, FILTER_TINY_SCALE, LABEL_DIR, IMG_DIR, OUTPUT_DIR, NUMBER_IMAGES
        OUTPUT_SIZE = (720, 1280)
        SCALE_RANGE = (0.4, 0.6)
        FILTER_TINY_SCALE = 0.1
        LABEL_DIR = str(label_dir)
        IMG_DIR = str(image_dir)
        OUTPUT_DIR = str(output_dir)
        NUMBER_IMAGES = 1

        # Act
        main()

        # Assert
        output_labels = list(glob.glob(os.path.join(OUTPUT_DIR, "*.txt")))
        assert len(output_labels) == 1
        with open(output_labels[0], "r") as f:
            content = f.read()
            assert "0 0.5 0.5 0.01 0.01" not in content

    @pytest.mark.invalid
    def test_main_non_existent_image_files(self, tmp_path):
        # Arrange
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        label_dir = tmp_path / "labels"
        label_dir.mkdir()
        img_paths = [image_dir / f"image{i}.jpg" for i in range(3)]
        label_paths = [label_dir / f"image{i}.txt" for i in range(3)]
        for img_path in img_paths:
            cv2.imwrite(str(img_path), np.zeros((100, 100, 3), dtype=np.uint8))
        for label_path in label_paths:
            with open(label_path, "w") as f:
                f.write("0 0.5 0.5 0.1 0.1\n")

        global OUTPUT_SIZE, SCALE_RANGE, FILTER_TINY_SCALE, LABEL_DIR, IMG_DIR, OUTPUT_DIR, NUMBER_IMAGES
        OUTPUT_SIZE = (720, 1280)
        SCALE_RANGE = (0.4, 0.6)
        FILTER_TINY_SCALE = 1 / 100
        LABEL_DIR = str(label_dir)
        IMG_DIR = str(image_dir)
        OUTPUT_DIR = str(output_dir)
        NUMBER_IMAGES = 1

        # Act
        main()

        # Assert
        output_images = list(glob.glob(os.path.join(OUTPUT_DIR, "*.jpg")))
        output_labels = list(glob.glob(os.path.join(OUTPUT_DIR, "*.txt")))
        assert len(output_images) == 1
        assert len(output_labels) == 1

    @pytest.mark.valid
    def test_main_randomness_in_mosaic(self, tmp_path):
        # Arrange
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        label_dir = tmp_path / "labels"
        label_dir.mkdir()
        img_paths = [image_dir / f"image{i}.jpg" for i in range(4)]
        label_paths = [label_dir / f"image{i}.txt" for i in range(4)]
        for img_path in img_paths:
            cv2.imwrite(str(img_path), np.zeros((100, 100, 3), dtype=np.uint8))
        for label_path in label_paths:
            with open(label_path, "w") as f:
                f.write("0 0.5 0.5 0.1 0.1\n")

        global OUTPUT_SIZE, SCALE_RANGE, FILTER_TINY_SCALE, LABEL_DIR, IMG_DIR, OUTPUT_DIR, NUMBER_IMAGES
        OUTPUT_SIZE = (720, 1280)
        SCALE_RANGE = (0.4, 0.6)
        FILTER_TINY_SCALE = 1 / 100
        LABEL_DIR = str(label_dir)
        IMG_DIR = str(image_dir)
        OUTPUT_DIR = str(output_dir)
        NUMBER_IMAGES = 1

        # Act
        main()
        output_images1 = list(glob.glob(os.path.join(OUTPUT_DIR, "*.jpg")))
        output_labels1 = list(glob.glob(os.path.join(OUTPUT_DIR, "*.txt")))

        main()
        output_images2 = list(glob.glob(os.path.join(OUTPUT_DIR, "*.jpg")))
        output_labels2 = list(glob.glob(os.path.join(OUTPUT_DIR, "*.txt")))

        # Assert
        assert output_images1 != output_images2
        assert output_labels1 != output_labels2

    @pytest.mark.performance
    def test_main_large_number_of_images(self, tmp_path):
        # Arrange
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        label_dir = tmp_path / "labels"
        label_dir.mkdir()
        img_paths = [image_dir / f"image{i}.jpg" for i in range(100)]
        label_paths = [label_dir / f"image{i}.txt" for i in range(100)]
        for img_path in img_paths:
            cv2.imwrite(str(img_path), np.zeros((100, 100, 3), dtype=np.uint8))
        for label_path in label_paths:
            with open(label_path, "w") as f:
                f.write("0 0.5 0.5 0.1 0.1\n")

        global OUTPUT_SIZE, SCALE_RANGE, FILTER_TINY_SCALE, LABEL_DIR, IMG_DIR, OUTPUT_DIR, NUMBER_IMAGES
        OUTPUT_SIZE = (720, 1280)
        SCALE_RANGE = (0.4, 0.6)
        FILTER_TINY_SCALE = 1 / 100
        LABEL_DIR = str(label_dir)
        IMG_DIR = str(image_dir)
        OUTPUT_DIR = str(output_dir)
        NUMBER_IMAGES = 100

        # Act
        main()

        # Assert
        output_images = list(glob.glob(os.path.join(OUTPUT_DIR, "*.jpg")))
        output_labels = list(glob.glob(os.path.join(OUTPUT_DIR, "*.txt")))
        assert len(output_images) == 100
        assert len(output_labels) == 100

    @pytest.mark.valid
    def test_main_output_file_naming(self, tmp_path):
        # Arrange
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        label_dir = tmp_path / "labels"
        label_dir.mkdir()
        img_paths = [image_dir / f"image{i}.jpg" for i in range(4)]
        label_paths = [label_dir / f"image{i}.txt" for i in range(4)]
        for img_path in img_paths:
            cv2.imwrite(str(img_path), np.zeros((100, 100, 3), dtype=np.uint8))
        for label_path in label_paths:
            with open(label_path, "w") as f:
                f.write("0 0.5 0.5 0.1 0.1\n")

        global OUTPUT_SIZE, SCALE_RANGE, FILTER_TINY_SCALE, LABEL_DIR, IMG_DIR, OUTPUT_DIR, NUMBER_IMAGES
        OUTPUT_SIZE = (720, 1280)
        SCALE_RANGE = (0.4, 0.6)
        FILTER_TINY_SCALE = 1 / 100
        LABEL_DIR = str(label_dir)
        IMG_DIR = str(image_dir)
        OUTPUT_DIR = str(output_dir)
        NUMBER_IMAGES = 1

        # Act
        main()

        # Assert
        output_images = list(glob.glob(os.path.join(OUTPUT_DIR, "*.jpg")))
        output_labels = list(glob.glob(os.path.join(OUTPUT_DIR, "*.txt")))
        assert len(output_images) == 1
        assert len(output_labels) == 1
        file_name = output_images[0].split(os.sep)[-1].rsplit(".", 1)[0]
        assert "MOSAIC_" in file_name

    @pytest.mark.invalid
    def test_main_invalid_annotation_format(self, tmp_path):
        # Arrange
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        label_dir = tmp_path / "labels"
        label_dir.mkdir()
        img_paths = [image_dir / f"image{i}.jpg" for i in range(4)]
        label_paths = [label_dir / f"image{i}.txt" for i in range(4)]
        for img_path in img_paths:
            cv2.imwrite(str(img_path), np.zeros((100, 100, 3), dtype=np.uint8))
        for label_path in label_paths:
            with open(label_path, "w") as f:
                f.write("0 0.5 0.5\n")  # Invalid format

        global OUTPUT_SIZE, SCALE_RANGE, FILTER_TINY_SCALE, LABEL_DIR, IMG_DIR, OUTPUT_DIR, NUMBER_IMAGES
        OUTPUT_SIZE = (720, 1280)
        SCALE_RANGE = (0.4, 0.6)
        FILTER_TINY_SCALE = 1 / 100
        LABEL_DIR = str(label_dir)
        IMG_DIR = str(image_dir)
        OUTPUT_DIR = str(output_dir)
        NUMBER_IMAGES = 1

        # Act
        main()

        # Assert
        output_labels = list(glob.glob(os.path.join(OUTPUT_DIR, "*.txt")))
        assert len(output_labels) == 1
        with open(output_labels[0], "r") as f:
            content = f.read()
            assert "0 0.5 0.5" not in content

    @pytest.mark.valid
    def test_main_image_resizing_and_scaling(self, tmp_path):
        # Arrange
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        label_dir = tmp_path / "labels"
        label_dir.mkdir()
        img_paths = [image_dir / f"image{i}.jpg" for i in range(4)]
        label_paths = [label_dir / f"image{i}.txt" for i in range(4)]
        for img_path in img_paths:
            cv2.imwrite(str(img_path), np.zeros((100, 100, 3), dtype=np.uint8))
        for label_path in label_paths:
            with open(label_path, "w") as f:
                f.write("0 0.5 0.5 0.1 0.1\n")

        global OUTPUT_SIZE, SCALE_RANGE, FILTER_TINY_SCALE, LABEL_DIR, IMG_DIR, OUTPUT_DIR, NUMBER_IMAGES
        OUTPUT_SIZE = (720, 1280)
        SCALE_RANGE = (0.4, 0.6)
        FILTER_TINY_SCALE = 1 / 100
        LABEL_DIR = str(label_dir)
        IMG_DIR = str(image_dir)
        OUTPUT_DIR = str(output_dir)
        NUMBER_IMAGES = 1

        # Act
        main()

        # Assert
        output_images = list(glob.glob(os.path.join(OUTPUT_DIR, "*.jpg")))
        output_labels = list(glob.glob(os.path.join(OUTPUT_DIR, "*.txt")))
        assert len(output_images) == 1
        assert len(output_labels) == 1
        img = cv2.imread(output_images[0])
        assert img.shape == (720, 1280, 3)
