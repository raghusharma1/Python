import pytest
import cv2
from harris_corner import HarrisCorner

class Test_HarrisCornerInit:

    @pytest.mark.valid
    def test_valid_k_and_window_size(self):
        k_values = [0.04, 0.06]
        window_size = 5

        for k in k_values:
            harris_corner = HarrisCorner(k, window_size)
            assert harris_corner.k == k
            assert harris_corner.window_size == window_size

    @pytest.mark.invalid
    def test_invalid_k_value(self):
        invalid_k_values = [0.03, 0.07]

        for k in invalid_k_values:
            with pytest.raises(ValueError, match="invalid k value"):
                HarrisCorner(k, 5)

    @pytest.mark.valid
    def test_zero_window_size(self):
        k_values = [0.04, 0.06]

        for k in k_values:
            harris_corner = HarrisCorner(k, 0)
            assert harris_corner.k == k
            assert harris_corner.window_size == 0

    @pytest.mark.valid
    def test_negative_window_size(self):
        k_values = [0.04, 0.06]

        for k in k_values:
            harris_corner = HarrisCorner(k, -3)
            assert harris_corner.k == k
            assert harris_corner.window_size == -3

    @pytest.mark.valid
    def test_boundary_k_values(self):
        boundary_k_values = [0.04, 0.06]

        for k in boundary_k_values:
            harris_corner = HarrisCorner(k, 5)
            assert harris_corner.k == k
            assert harris_corner.window_size == 5

    @pytest.mark.valid
    def test_large_window_size(self):
        k_values = [0.04, 0.06]
        large_window_size = 1000

        for k in k_values:
            harris_corner = HarrisCorner(k, large_window_size)
            assert harris_corner.k == k
            assert harris_corner.window_size == large_window_size

    @pytest.mark.invalid
    def test_floating_point_k_values_within_range(self):
        floating_point_k_values = [0.045, 0.055]

        for k in floating_point_k_values:
            with pytest.raises(ValueError, match="invalid k value"):
                HarrisCorner(k, 5)
