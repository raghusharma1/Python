import cv2
import numpy as np
import pytest
from harris_corner import HarrisCorner

class Test_HarrisCornerDetect:
    @pytest.mark.valid
    @pytest.mark.positive
    def test_valid_image_path_with_corners(self):
        # Arrange
        k = 0.04  # TODO: Change k if necessary
        window_size = 3
        edge_detect = HarrisCorner(k, window_size)
        img_path = "path_to_valid_image_with_corners.jpg"  # TODO: Provide a valid image path

        # Act
        color_img, corner_list = edge_detect.detect(img_path)

        # Assert
        assert isinstance(color_img, np.ndarray)
        assert len(corner_list) > 0
        for corner in corner_list:
            assert len(corner) == 3
            x, y, _ = corner
            assert 0 <= x < color_img.shape[1]
            assert 0 <= y < color_img.shape[0]

    @pytest.mark.invalid
    @pytest.mark.negative
    def test_invalid_image_path(self):
        # Arrange
        k = 0.04
        window_size = 3
        edge_detect = HarrisCorner(k, window_size)
        img_path = "path_to_invalid_image.jpg"  # TODO: Provide an invalid image path

        # Act & Assert
        with pytest.raises(cv2.error):
            edge_detect.detect(img_path)

    @pytest.mark.invalid
    @pytest.mark.negative
    def test_invalid_k_value(self):
        # Arrange
        k = 0.1  # Invalid k value
        window_size = 3

        # Act & Assert
        with pytest.raises(ValueError):
            HarrisCorner(k, window_size)

    @pytest.mark.valid
    @pytest.mark.positive
    def test_large_window_size(self):
        # Arrange
        k = 0.04
        window_size = 10
        edge_detect = HarrisCorner(k, window_size)
        img_path = "path_to_valid_image_with_corners.jpg"  # TODO: Provide a valid image path

        # Act
        color_img, corner_list = edge_detect.detect(img_path)

        # Assert
        assert isinstance(color_img, np.ndarray)
        assert len(corner_list) > 0
        for corner in corner_list:
            assert len(corner) == 3
            x, y, _ = corner
            assert 0 <= x < color_img.shape[1]
            assert 0 <= y < color_img.shape[0]

    @pytest.mark.valid
    @pytest.mark.positive
    def test_small_window_size(self):
        # Arrange
        k = 0.04
        window_size = 1
        edge_detect = HarrisCorner(k, window_size)
        img_path = "path_to_valid_image_with_corners.jpg"  # TODO: Provide a valid image path

        # Act
        color_img, corner_list = edge_detect.detect(img_path)

        # Assert
        assert isinstance(color_img, np.ndarray)
        assert len(corner_list) > 0
        for corner in corner_list:
            assert len(corner) == 3
            x, y, _ = corner
            assert 0 <= x < color_img.shape[1]
            assert 0 <= y < color_img.shape[0]

