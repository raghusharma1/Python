import pytest
import cv2
import numpy as np
from harris_corner import HarrisCorner

class Test_HarrisCornerDetect:

    @pytest.mark.valid
    def test_valid_image_path_with_corners(self):
        # Arrange
        edge_detect = HarrisCorner(0.04, 3)
        path_to_image = "path_to_image_with_corners.jpg"  # TODO: Change to a valid image path

        # Act
        color_img, corner_list = edge_detect.detect(path_to_image)

        # Assert
        assert len(corner_list) > 0, "No corners detected in the image."
        assert cv2.countNonZero(cv2.cvtColor(color_img, cv2.COLOR_RGB2GRAY)) > 0, "No corners highlighted in the image."

    @pytest.mark.invalid
    def test_invalid_image_path(self):
        # Arrange
        edge_detect = HarrisCorner(0.04, 3)
        invalid_path_to_image = "path_to_invalid_image.jpg"  # TODO: Change to an invalid image path

        # Act & Assert
        with pytest.raises(FileNotFoundError):
            edge_detect.detect(invalid_path_to_image)

    @pytest.mark.valid
    def test_image_with_no_corners(self):
        # Arrange
        edge_detect = HarrisCorner(0.04, 3)
        path_to_image = "path_to_image_with_no_corners.jpg"  # TODO: Change to a valid image path with no corners

        # Act
        color_img, corner_list = edge_detect.detect(path_to_image)

        # Assert
        assert len(corner_list) == 0, "Corners detected in an image with no corners."

    @pytest.mark.valid
    def test_image_with_low_contrast(self):
        # Arrange
        edge_detect = HarrisCorner(0.04, 3)
        path_to_image = "path_to_image_with_low_contrast.jpg"  # TODO: Change to a valid image path with low contrast

        # Act
        color_img, corner_list = edge_detect.detect(path_to_image)

        # Assert
        assert len(corner_list) == 0, "Corners detected in an image with low contrast."

    @pytest.mark.valid
    def test_image_with_high_noise(self):
        # Arrange
        edge_detect = HarrisCorner(0.04, 3)
        path_to_image = "path_to_image_with_high_noise.jpg"  # TODO: Change to a valid image path with high noise

        # Act
        color_img, corner_list = edge_detect.detect(path_to_image)

        # Assert
        assert len(corner_list) == 0, "Corners detected in an image with high noise."

    @pytest.mark.valid
    def test_image_with_different_k_value(self):
        # Arrange
        edge_detect = HarrisCorner(0.06, 3)
        path_to_image = "path_to_image_with_corners.jpg"  # TODO: Change to a valid image path

        # Act
        color_img, corner_list = edge_detect.detect(path_to_image)

        # Assert
        assert len(corner_list) > 0, "No corners detected in the image with different k value."
        assert cv2.countNonZero(cv2.cvtColor(color_img, cv2.COLOR_RGB2GRAY)) > 0, "No corners highlighted in the image with different k value."

    @pytest.mark.valid
    def test_image_with_different_window_size(self):
        # Arrange
        edge_detect = HarrisCorner(0.04, 5)
        path_to_image = "path_to_image_with_corners.jpg"  # TODO: Change to a valid image path

        # Act
        color_img, corner_list = edge_detect.detect(path_to_image)

        # Assert
        assert len(corner_list) > 0, "No corners detected in the image with different window size."
        assert cv2.countNonZero(cv2.cvtColor(color_img, cv2.COLOR_RGB2GRAY)) > 0, "No corners highlighted in the image with different window size."

    @pytest.mark.performance
    def test_image_with_large_dimensions(self):
        # Arrange
        edge_detect = HarrisCorner(0.04, 3)
        path_to_image = "path_to_image_with_large_dimensions.jpg"  # TODO: Change to a valid image path with large dimensions

        # Act
        color_img, corner_list = edge_detect.detect(path_to_image)

        # Assert
        assert len(corner_list) > 0, "No corners detected in the large image."
        assert cv2.countNonZero(cv2.cvtColor(color_img, cv2.COLOR_RGB2GRAY)) > 0, "No corners highlighted in the large image."

    @pytest.mark.performance
    def test_image_with_small_dimensions(self):
        # Arrange
        edge_detect = HarrisCorner(0.04, 3)
        path_to_image = "path_to_image_with_small_dimensions.jpg"  # TODO: Change to a valid image path with small dimensions

        # Act
        color_img, corner_list = edge_detect.detect(path_to_image)

        # Assert
        assert len(corner_list) > 0, "No corners detected in the small image."
        assert cv2.countNonZero(cv2.cvtColor(color_img, cv2.COLOR_RGB2GRAY)) > 0, "No corners highlighted in the small image."
