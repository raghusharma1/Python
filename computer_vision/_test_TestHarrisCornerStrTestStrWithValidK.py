import cv2
import numpy as np
import pytest
from harris_corner import HarrisCorner

class Test_TestHarrisCornerStrTestStrWithValidK:
    @pytest.mark.valid
    @pytest.mark.positive
    def test_str_with_valid_k_004(self):
        # Arrange
        k_value = 0.04
        harris_corner = HarrisCorner(k_value, 3)

        # Act
        result = str(harris_corner)

        # Assert
        assert result == str(k_value), f"Expected {str(k_value)}, but got {result}"

    @pytest.mark.valid
    @pytest.mark.positive
    def test_str_with_valid_k_006(self):
        # Arrange
        k_value = 0.06
        harris_corner = HarrisCorner(k_value, 3)

        # Act
        result = str(harris_corner)

        # Assert
        assert result == str(k_value), f"Expected {str(k_value)}, but got {result}"

    @pytest.mark.invalid
    @pytest.mark.negative
    def test_str_with_invalid_k_below_range(self):
        # Arrange
        k_value = 0.03

        # Act & Assert
        with pytest.raises(ValueError, match="invalid k value"):
            HarrisCorner(k_value, 3)

    @pytest.mark.invalid
    @pytest.mark.negative
    def test_str_with_invalid_k_above_range(self):
        # Arrange
        k_value = 0.07

        # Act & Assert
        with pytest.raises(ValueError, match="invalid k value"):
            HarrisCorner(k_value, 3)

    @pytest.mark.regression
    @pytest.mark.valid
    def test_window_size_impact(self):
        # Arrange
        k_value = 0.04
        window_sizes = [3, 5, 7]
        harris_corners = [HarrisCorner(k_value, size) for size in window_sizes]

        # Act
        image_path = "sample_image.jpg"  # TODO: Update with actual image path
        results = [harris_corner.detect(image_path) for harris_corner in harris_corners]

        # Assert
        num_corners = [len(corners) for _, corners in results]
        assert num_corners != [num_corners[0]] * len(num_corners), "Window size did not impact corner detection"

    @pytest.mark.regression
    @pytest.mark.valid
    def test_corner_detection_on_different_images(self):
        # Arrange
        k_value = 0.04
        window_size = 3
        harris_corner = HarrisCorner(k_value, window_size)
        image_paths = ["simple_shape.jpg", "natural_scene.jpg", "texture.jpg"]  # TODO: Update with actual image paths

        # Act
        results = [harris_corner.detect(image_path) for image_path in image_paths]

        # Assert
        for image_path, (_, corners) in zip(image_paths, results):
            assert len(corners) > 0, f"No corners detected in {image_path}"
