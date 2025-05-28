import pytest
import cv2
import numpy as np
from harris_corner import HarrisCorner

class Test_HarrisCornerDetect:

    @pytest.mark.valid
    def test_valid_image_path(self):
        # Arrange
        edge_detect = HarrisCorner(0.04, 3)
        valid_image_path = 'path_to_valid_image.jpg'  # TODO: Provide a valid image path

        # Act
        color_img, corner_list = edge_detect.detect(valid_image_path)

        # Assert
        assert color_img is not None
        assert len(corner_list) > 0

    @pytest.mark.invalid
    def test_invalid_image_path(self):
        # Arrange
        edge_detect = HarrisCorner(0.04, 3)
        invalid_image_path = 'path_to_invalid_image.jpg'  # TODO: Provide an invalid image path

        # Act & Assert
        with pytest.raises(FileNotFoundError):
            edge_detect.detect(invalid_image_path)

    @pytest.mark.valid
    def test_non_grayscale_image(self):
        # Arrange
        edge_detect = HarrisCorner(0.04, 3)
        non_grayscale_image_path = 'path_to_non_grayscale_image.jpg'  # TODO: Provide a non-grayscale image path

        # Act
        color_img, corner_list = edge_detect.detect(non_grayscale_image_path)

        # Assert
        assert color_img is not None
        assert len(corner_list) > 0

    @pytest.mark.valid
    def test_extremely_small_image(self):
        # Arrange
        edge_detect = HarrisCorner(0.04, 3)
        extremely_small_image_path = 'path_to_extremely_small_image.jpg'  # TODO: Provide an extremely small image path

        # Act
        color_img, corner_list = edge_detect.detect(extremely_small_image_path)

        # Assert
        assert color_img is not None
        assert len(corner_list) == 0

    @pytest.mark.performance
    def test_extremely_large_image(self, benchmark):
        # Arrange
        edge_detect = HarrisCorner(0.04, 3)
        extremely_large_image_path = 'path_to_extremely_large_image.jpg'  # TODO: Provide an extremely large image path

        # Act
        color_img, corner_list = benchmark(edge_detect.detect, extremely_large_image_path)

        # Assert
        assert color_img is not None

    @pytest.mark.valid
    def test_image_with_no_corners(self):
        # Arrange
        edge_detect = HarrisCorner(0.04, 3)
        image_with_no_corners_path = 'path_to_image_with_no_corners.jpg'  # TODO: Provide an image with no corners path

        # Act
        color_img, corner_list = edge_detect.detect(image_with_no_corners_path)

        # Assert
        assert color_img is not None
        assert len(corner_list) == 0

    @pytest.mark.valid
    def test_image_with_multiple_corners(self):
        # Arrange
        edge_detect = HarrisCorner(0.04, 3)
        image_with_multiple_corners_path = 'path_to_image_with_multiple_corners.jpg'  # TODO: Provide an image with multiple corners path

        # Act
        color_img, corner_list = edge_detect.detect(image_with_multiple_corners_path)

        # Assert
        assert color_img is not None
        assert len(corner_list) > 1  # TODO: Change the expected number of corners if necessary

    @pytest.mark.valid
    def test_image_with_low_contrast(self):
        # Arrange
        edge_detect = HarrisCorner(0.04, 3)
        image_with_low_contrast_path = 'path_to_image_with_low_contrast.jpg'  # TODO: Provide an image with low contrast path

        # Act
        color_img, corner_list = edge_detect.detect(image_with_low_contrast_path)

        # Assert
        assert color_img is not None
        assert len(corner_list) == 0

    @pytest.mark.valid
    def test_image_with_high_contrast(self):
        # Arrange
        edge_detect = HarrisCorner(0.04, 3)
        image_with_high_contrast_path = 'path_to_image_with_high_contrast.jpg'  # TODO: Provide an image with high contrast path

        # Act
        color_img, corner_list = edge_detect.detect(image_with_high_contrast_path)

        # Assert
        assert color_img is not None
        assert len(corner_list) > 0  # TODO: Change the expected number of corners if necessary

    @pytest.mark.valid
    def test_image_with_different_background(self):
        # Arrange
        edge_detect = HarrisCorner(0.04, 3)
        image_with_different_background_path = 'path_to_image_with_different_background.jpg'  # TODO: Provide an image with a different background path

        # Act
        color_img, corner_list = edge_detect.detect(image_with_different_background_path)

        # Assert
        assert color_img is not None
        assert len(corner_list) > 0  # TODO: Change the expected number of corners if necessary
