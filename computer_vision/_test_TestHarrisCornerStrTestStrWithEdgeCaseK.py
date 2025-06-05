import cv2
import numpy as np
import pytest
from harris_corner import HarrisCorner

class Test_TestHarrisCornerStrTestStrWithEdgeCaseK:

    @pytest.mark.smoke
    @pytest.mark.positive
    @pytest.mark.valid
    def test_str_with_edge_case_k(self):
        # Arrange
        k_values = [0.04, 0.06]
        expected_results = ["0.04", "0.06"]
        results = []

        # Act
        for k in k_values:
            harris_corner = HarrisCorner(k, 3)
            results.append(str(harris_corner))

        # Assert
        for i in range(len(k_values)):
            assert results[i] == expected_results[i], f"Expected {expected_results[i]}, but got {results[i]}"

    @pytest.mark.negative
    @pytest.mark.invalid
    def test_str_with_invalid_k(self):
        # Arrange
        k_values = [0.01, 0.07, 0.1]
        expected_exception = ValueError

        # Act & Assert
        for k in k_values:
            with pytest.raises(expected_exception, match="invalid k value"):
                harris_corner = HarrisCorner(k, 3)
                str(harris_corner)

    @pytest.mark.smoke
    @pytest.mark.positive
    @pytest.mark.valid
    def test_detect_corners(self):
        # Arrange
        k = 0.04
        window_size = 3
        img_path = 'test_image.jpg'  # TODO: Change this to a valid image path

        # Act
        harris_corner = HarrisCorner(k, window_size)
        result_img, corner_list = harris_corner.detect(img_path)

        # Assert
        assert isinstance(result_img, np.ndarray), "Expected result_img to be a numpy array"
        assert isinstance(corner_list, list), "Expected corner_list to be a list"
        for corner in corner_list:
            assert isinstance(corner, list), "Expected each corner to be a list"
            assert len(corner) == 3, "Expected each corner to have three elements"
            assert all(isinstance(x, int) for x in corner), "Expected corner elements to be integers"

    @pytest.mark.negative
    @pytest.mark.invalid
    def test_detect_corners_invalid_image(self):
        # Arrange
        k = 0.04
        window_size = 3
        invalid_img_path = 'invalid_image.jpg'  # TODO: Change this to an invalid image path

        # Act & Assert
        harris_corner = HarrisCorner(k, window_size)
        with pytest.raises(cv2.error, match="!_src.empty"):
            harris_corner.detect(invalid_img_path)

