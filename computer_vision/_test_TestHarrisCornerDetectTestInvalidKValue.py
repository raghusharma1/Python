import cv2
import numpy as np
import pytest
from harris_corner import HarrisCorner
from _test_HarrisCornerDetect import Test_HarrisCornerDetect

class Test_TestHarrisCornerDetectTestInvalidKValue:
    def test_k_value_below_lower_bound(self):
        # Arrange
        k = 0.03  # below the lower bound
        window_size = 3

        # Act & Assert
        with pytest.raises(ValueError):
            HarrisCorner(k, window_size)

    def test_k_value_above_upper_bound(self):
        # Arrange
        k = 0.07  # above the upper bound
        window_size = 3

        # Act & Assert
        with pytest.raises(ValueError):
            HarrisCorner(k, window_size)

    def test_k_value_at_lower_bound(self):
        # Arrange
        k = 0.04  # exactly at the lower bound
        window_size = 3

        # Act & Assert
        try:
            HarrisCorner(k, window_size)
        except ValueError:
            pytest.fail("HarrisCorner raised ValueError unexpectedly!")

    def test_k_value_at_upper_bound(self):
        # Arrange
        k = 0.06  # exactly at the upper bound
        window_size = 3

        # Act & Assert
        try:
            HarrisCorner(k, window_size)
        except ValueError:
            pytest.fail("HarrisCorner raised ValueError unexpectedly!")

    def test_negative_k_value(self):
        # Arrange
        k = -0.05  # negative k value
        window_size = 3

        # Act & Assert
        with pytest.raises(ValueError):
            HarrisCorner(k, window_size)

    def test_non_float_k_value(self):
        # Arrange
        k = "0.05"  # non-float k value
        window_size = 3

        # Act & Assert
        with pytest.raises(ValueError):
            HarrisCorner(k, window_size)

    def test_large_k_value(self):
        # Arrange
        k = 1000  # significantly larger than the upper bound
        window_size = 3

        # Act & Assert
        with pytest.raises(ValueError):
            HarrisCorner(k, window_size)

    def test_small_k_value(self):
        # Arrange
        k = -1000  # significantly smaller than the lower bound
        window_size = 3

        # Act & Assert
        with pytest.raises(ValueError):
            HarrisCorner(k, window_size)

    def test_invalid_k_value(self):
        # Arrange
        k = 0.1  # Invalid k value
        window_size = 3

        # Act & Assert
        with pytest.raises(ValueError):
            HarrisCorner(k, window_size)
