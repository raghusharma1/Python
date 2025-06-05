import cv2
import numpy as np
import pytest
from harris_corner import HarrisCorner

class Test_TestHarrisCornerDetectTestValidImagePathWithCorners:
    @pytest.mark.smoke
    @pytest.mark.valid
    def test_valid_image_with_corners(self):
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

    @pytest.mark.smoke
    @pytest.mark.valid
    def test_image_with_no_corners(self):
        # Arrange
        k = 0.04
        window_size = 3
        edge_detect = HarrisCorner(k, window_size)
        img_path = "path_to_image_with_no_corners.jpg"  # TODO: Provide a valid image path

        # Act
        color_img, corner_list = edge_detect.detect(img_path)

        # Assert
        assert isinstance(color_img, np.ndarray)
        assert len(corner_list) == 0

    @pytest.mark.regression
    @pytest.mark.valid
    def test_image_with_different_corner_response_thresholds(self):
        # Arrange
        window_size = 3
        img_path = "path_to_valid_image_with_corners.jpg"  # TODO: Provide a valid image path

        # Act & Assert for k=0.04
        edge_detect = HarrisCorner(0.04, window_size)
        color_img, corner_list = edge_detect.detect(img_path)
        assert isinstance(color_img, np.ndarray)
        assert len(corner_list) > 0
        for corner in corner_list:
            assert len(corner) == 3
            x, y, _ = corner
            assert 0 <= x < color_img.shape[1]
            assert 0 <= y < color_img.shape[0]

        # Act & Assert for k=0.06
        edge_detect = HarrisCorner(0.06, window_size)
        color_img, corner_list = edge_detect.detect(img_path)
        assert isinstance(color_img, np.ndarray)
        assert len(corner_list) > 0
        for corner in corner_list:
            assert len(corner) == 3
            x, y, _ = corner
            assert 0 <= x < color_img.shape[1]
            assert 0 <= y < color_img.shape[0]

    @pytest.mark.performance
    @pytest.mark.valid
    def test_large_image_with_corners(self):
        # Arrange
        k = 0.04
        window_size = 3
        edge_detect = HarrisCorner(k, window_size)
        img_path = "path_to_large_image_with_corners.jpg"  # TODO: Provide a valid large image path

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

    @pytest.mark.regression
    @pytest.mark.valid
    def test_image_with_different_window_sizes(self):
        # Arrange
        k = 0.04
        img_path = "path_to_valid_image_with_corners.jpg"  # TODO: Provide a valid image path

        # Act & Assert for window_size=3
        edge_detect = HarrisCorner(k, 3)
        color_img, corner_list = edge_detect.detect(img_path)
        assert isinstance(color_img, np.ndarray)
        assert len(corner_list) > 0
        for corner in corner_list:
            assert len(corner) == 3
            x, y, _ = corner
            assert 0 <= x < color_img.shape[1]
            assert 0 <= y < color_img.shape[0]

        # Act & Assert for window_size=5
        edge_detect = HarrisCorner(k, 5)
        color_img, corner_list = edge_detect.detect(img_path)
        assert isinstance(color_img, np.ndarray)
        assert len(corner_list) > 0
        for corner in corner_list:
            assert len(corner) == 3
            x, y, _ = corner
            assert 0 <= x < color_img.shape[1]
            assert 0 <= y < color_img.shape[0]

        # Act & Assert for window_size=7
        edge_detect = HarrisCorner(k, 7)
        color_img, corner_list = edge_detect.detect(img_path)
        assert isinstance(color_img, np.ndarray)
        assert len(corner_list) > 0
        for corner in corner_list:
            assert len(corner) == 3
            x, y, _ = corner
            assert 0 <= x < color_img.shape[1]
            assert 0 <= y < color_img.shape[0]

    @pytest.mark.negative
    @pytest.mark.invalid
    def test_non_existent_image_path(self):
        # Arrange
        k = 0.04
        window_size = 3
        edge_detect = HarrisCorner(k, window_size)
        img_path = "non_existent_image_path.jpg"  # TODO: Provide a non-existent image path

        # Act & Assert
        with pytest.raises(FileNotFoundError):
            edge_detect.detect(img_path)

    @pytest.mark.regression
    @pytest.mark.valid
    def test_image_with_different_formats(self):
        # Arrange
        k = 0.04
        window_size = 3
        edge_detect = HarrisCorner(k, window_size)

        # Act & Assert for .jpg format
        img_path = "path_to_valid_image_with_corners.jpg"  # TODO: Provide a valid image path
        color_img, corner_list = edge_detect.detect(img_path)
        assert isinstance(color_img, np.ndarray)
        assert len(corner_list) > 0
        for corner in corner_list:
            assert len(corner) == 3
            x, y, _ = corner
            assert 0 <= x < color_img.shape[1]
            assert 0 <= y < color_img.shape[0]

        # Act & Assert for .png format
        img_path = "path_to_valid_image_with_corners.png"  # TODO: Provide a valid image path
        color_img, corner_list = edge_detect.detect(img_path)
        assert isinstance(color_img, np.ndarray)
        assert len(corner_list) > 0
        for corner in corner_list:
            assert len(corner) == 3
            x, y, _ = corner
            assert 0 <= x < color_img.shape[1]
            assert 0 <= y < color_img.shape[0]

        # Act & Assert for .bmp format
        img_path = "path_to_valid_image_with_corners.bmp"  # TODO: Provide a valid image path
        color_img, corner_list = edge_detect.detect(img_path)
        assert isinstance(color_img, np.ndarray)
        assert len(corner_list) > 0
        for corner in corner_list:
            assert len(corner) == 3
            x, y, _ = corner
            assert 0 <= x < color_img.shape[1]
            assert 0 <= y < color_img.shape[0]

    @pytest.mark.regression
    @pytest.mark.valid
    def test_image_with_different_scales(self):
        # Arrange
        k = 0.04
        window_size = 3
        edge_detect = HarrisCorner(k, window_size)

        # Act & Assert for high resolution
        img_path = "path_to_high_res_image_with_corners.jpg"  # TODO: Provide a valid high resolution image path
        color_img, corner_list = edge_detect.detect(img_path)
        assert isinstance(color_img, np.ndarray)
        assert len(corner_list) > 0
        for corner in corner_list:
            assert len(corner) == 3
            x, y, _ = corner
            assert 0 <= x < color_img.shape[1]
            assert 0 <= y < color_img.shape[0]

        # Act & Assert for medium resolution
        img_path = "path_to_medium_res_image_with_corners.jpg"  # TODO: Provide a valid medium resolution image path
        color_img, corner_list = edge_detect.detect(img_path)
        assert isinstance(color_img, np.ndarray)
        assert len(corner_list) > 0
        for corner in corner_list:
            assert len(corner) == 3
            x, y, _ = corner
            assert 0 <= x < color_img.shape[1]
            assert 0 <= y < color_img.shape[0]

        # Act & Assert for low resolution
        img_path = "path_to_low_res_image_with_corners.jpg"  # TODO: Provide a valid low resolution image path
        color_img, corner_list = edge_detect.detect(img_path)
        assert isinstance(color_img, np.ndarray)
        assert len(corner_list) > 0
        for corner in corner_list:
            assert len(corner) == 3
            x, y, _ = corner
            assert 0 <= x < color_img.shape[1]
            assert 0 <= y < color_img.shape[0]
