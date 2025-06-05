import pytest
import cv2
import numpy as np
from harris_corner import HarrisCorner

class Test_TestHarrisCornerInitTestNegativeWindowSize:

    @pytest.mark.negative
    @pytest.mark.invalid
    def test_zero_window_size(self):
        k_values = [0.04, 0.06]
        for k in k_values:
            with pytest.raises(ValueError):  # Expecting ValueError for zero window size
                HarrisCorner(k, 0)

    @pytest.mark.negative
    @pytest.mark.invalid
    def test_large_negative_window_size(self):
        k_values = [0.04, 0.06]
        for k in k_values:
            with pytest.raises(ValueError):  # Expecting ValueError for large negative window size
                HarrisCorner(k, -1000)

    @pytest.mark.positive
    @pytest.mark.valid
    def test_valid_window_size_range(self):
        k_values = [0.04, 0.06]
        valid_window_sizes = list(range(3, 10))  # TODO: Modify as per valid range
        for k in k_values:
            for window_size in valid_window_sizes:
                harris_corner = HarrisCorner(k, window_size)
                assert harris_corner.k == k
                assert harris_corner.window_size == window_size

    @pytest.mark.positive
    @pytest.mark.valid
    def test_even_and_odd_window_sizes(self):
        k_values = [0.04, 0.06]
        window_sizes = [3, 4]  # One odd, one even
        for k in k_values:
            for window_size in window_sizes:
                harris_corner = HarrisCorner(k, window_size)
                assert harris_corner.k == k
                assert harris_corner.window_size == window_size

    @pytest.mark.positive
    @pytest.mark.valid
    def test_boundary_window_sizes(self):
        k_values = [0.04, 0.06]
        boundary_window_sizes = [1, 20]  # Minimum and maximum valid window sizes
        for k in k_values:
            for window_size in boundary_window_sizes:
                harris_corner = HarrisCorner(k, window_size)
                assert harris_corner.k == k
                assert harris_corner.window_size == window_size

    @pytest.mark.negative
    @pytest.mark.invalid
    def test_negative_window_size(self):
        k_values = [0.04, 0.06]

        for k in k_values:
            with pytest.raises(ValueError):  # Expecting ValueError for negative window size
                HarrisCorner(k, -3)

