import pytest
import cv2
import numpy as np
from harris_corner import HarrisCorner

class Test_HarrisCornerDetect:

    @pytest.mark.valid
    def test_detect_valid_image_with_defaults(self):
        # Arrange
        edge_detect = HarrisCorner(0.04, 3)
        img_path = "path_to_valid_image.jpg"  # TODO: Provide a valid image path

        # Act
        color_img, corner_list = edge_detect.detect(img_path)

        # Assert
        assert isinstance(color_img, np.ndarray)
        assert isinstance(corner_list, list)
        assert len(corner_list) > 0
        h, w, _ = color_img.shape
        for corner in corner_list:
            x, y, r = corner
            assert 0 <= x < w
            assert 0 <= y < h

    @pytest.mark.invalid
    def test_detect_image_with_no_corners(self):
        # Arrange
        edge_detect = HarrisCorner(0.04, 3)
        img_path = "path_to_image_with_no_corners.jpg"  # TODO: Provide a path to an image known to have no corners

        # Act
        color_img, corner_list = edge_detect.detect(img_path)

        # Assert
        assert isinstance(color_img, np.ndarray)
        assert isinstance(corner_list, list)
        assert len(corner_list) == 0

    @pytest.mark.valid
    def test_detect_image_with_high_k(self):
        # Arrange
        edge_detect = HarrisCorner(0.06, 3)
        img_path = "path_to_valid_image.jpg"  # TODO: Provide a valid image path

        # Act
        color_img, corner_list = edge_detect.detect(img_path)

        # Assert
        assert isinstance(color_img, np.ndarray)
        assert isinstance(corner_list, list)
        assert len(corner_list) > 0
        h, w, _ = color_img.shape
        for corner in corner_list:
            x, y, r = corner
            assert 0 <= x < w
            assert 0 <= y < h

    @pytest.mark.valid
    def test_detect_image_with_low_k(self):
        # Arrange
        edge_detect = HarrisCorner(0.04, 3)
        img_path = "path_to_valid_image.jpg"  # TODO: Provide a valid image path

        # Act
        color_img, corner_list = edge_detect.detect(img_path)

        # Assert
        assert isinstance(color_img, np.ndarray)
        assert isinstance(corner_list, list)
        assert len(corner_list) > 0
        h, w, _ = color_img.shape
        for corner in corner_list:
            x, y, r = corner
            assert 0 <= x < w
            assert 0 <= y < h

    @pytest.mark.valid
    def test_detect_image_with_low_contrast(self):
        # Arrange
        edge_detect = HarrisCorner(0.04, 3)
        img_path = "path_to_low_contrast_image.jpg"  # TODO: Provide a path to an image with low contrast

        # Act
        color_img, corner_list = edge_detect.detect(img_path)

        # Assert
        assert isinstance(color_img, np.ndarray)
        assert isinstance(corner_list, list)
        assert len(corner_list) > 0
        h, w, _ = color_img.shape
        for corner in corner_list:
            x, y, r = corner
            assert 0 <= x < w
            assert 0 <= y < h

    @pytest.mark.valid
    def test_detect_image_with_large_window_size(self):
        # Arrange
        edge_detect = HarrisCorner(0.04, 20)
        img_path = "path_to_valid_image.jpg"  # TODO: Provide a valid image path

        # Act
        color_img, corner_list = edge_detect.detect(img_path)

        # Assert
        assert isinstance(color_img, np.ndarray)
        assert isinstance(corner_list, list)
        assert len(corner_list) > 0
        h, w, _ = color_img.shape
        for corner in corner_list:
            x, y, r = corner
            assert 0 <= x < w
            assert 0 <= y < h

    @pytest.mark.valid
    def test_detect_image_with_small_window_size(self):
        # Arrange
        edge_detect = HarrisCorner(0.04, 1)
        img_path = "path_to_valid_image.jpg"  # TODO: Provide a valid image path

        # Act
        color_img, corner_list = edge_detect.detect(img_path)

        # Assert
        assert isinstance(color_img, np.ndarray)
        assert isinstance(corner_list, list)
        assert len(corner_list) > 0
        h, w, _ = color_img.shape
        for corner in corner_list:
            x, y, r = corner
            assert 0 <= x < w
            assert 0 <= y < h

    @pytest.mark.invalid
    def test_detect_image_with_nonexistent_path(self):
        # Arrange
        edge_detect = HarrisCorner(0.04, 3)
        img_path = "path_to_nonexistent_image.jpg"  # TODO: Provide a non-existent image path

        # Act & Assert
        with pytest.raises(cv2.error):
            edge_detect.detect(img_path)
