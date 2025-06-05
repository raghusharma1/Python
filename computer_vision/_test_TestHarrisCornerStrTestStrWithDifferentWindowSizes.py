import pytest
import numpy as np
import cv2
from harris_corner import HarrisCorner

class Test_TestHarrisCornerStrTestStrWithDifferentWindowSizes:
    def test_str_with_different_window_sizes(self):
        # Arrange
        k_value = 0.04
        window_sizes = [3, 5, 7]
        results = []

        # Act
        for window_size in window_sizes:
            harris_corner = HarrisCorner(k_value, window_size)
            results.append(str(harris_corner))

        # Assert
        assert len(set(results)) == 1, f"Expected all results to be the same, but got {results}"

