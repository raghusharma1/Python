import cv2
import numpy as np
import pytest
from harris_corner import HarrisCorner
from _test_HarrisCornerDetect import Test_HarrisCornerDetect

class Test_TestHarrisCornerDetectTestLargeWindowSize(Test_HarrisCornerDetect):

    @pytest.mark.valid
    @pytest.mark.smoke
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
    @pytest.mark.smoke
    def test_no_corners_in_image(self):
        # Arrange
        k = 0.04
        window_size = 5
        edge_detect = HarrisCorner(k, window_size)
        img_path = "path_to_image_with_no_corners.jpg"  # TODO: Provide a valid image path with no corners

        # Act
        color_img, corner_list = edge_detect.detect(img_path)

        # Assert
        assert isinstance(color_img, np.ndarray)
        assert len(corner_list) == 0

    @pytest.mark.valid
    @pytest.mark.smoke
    def test_maximum_window_size_limit(self):
        # Arrange
        k = 0.04
        window_size = 25
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
    @pytest.mark.smoke
    def test_different_k_value(self):
        # Arrange
        k = 0.06
        window_size = 5
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
    @pytest.mark.smoke
    @pytest.mark.invalid
    def test_invalid_k_value(self):
        # Arrange
        k = 0.03
        window_size = 5

        # Act & Assert
        with pytest.raises(ValueError) as excinfo:
            edge_detect = HarrisCorner(k, window_size)
        assert str(excinfo.value) == "invalid k value"

    @pytest.mark.valid
    @pytest.mark.performance
    @pytest.mark.smoke
    def test_large_image_performance(self):
        # Arrange
        k = 0.04
        window_size = 5
        edge_detect = HarrisCorner(k, window_size)
        img_path = "path_to_large_image.jpg"  # TODO: Provide a valid image path for high resolution images

        # Act
        import time
        start_time = time.time()
        color_img, corner_list = edge_detect.detect(img_path)
        end_time = time.time()

        # Assert
        assert isinstance(color_img, np.ndarray)
        assert len(corner_list) > 0
        assert end_time - start_time < 5
        for corner in corner_list:
            assert len(corner) == 3
            x, y, _ = corner
            assert 0 <= x < color_img.shape[1]
            assert 0 <= y < color_img.shape[0]
