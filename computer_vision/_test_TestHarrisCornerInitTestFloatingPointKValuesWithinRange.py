import pytest
import cv2
import numpy as np
from harris_corner import HarrisCorner
from _test_HarrisCornerInit import Test_HarrisCornerInit

class Test_TestHarrisCornerInitTestFloatingPointKValuesWithinRange(Test_HarrisCornerInit):

    @pytest.mark.invalid
    @pytest.mark.negative
    def test_k_value_just_below_lower_boundary(self):
        with pytest.raises(ValueError, match="invalid k value"):
            HarrisCorner(0.039, 5)

    @pytest.mark.invalid
    @pytest.mark.negative
    def test_k_value_just_above_upper_boundary(self):
        with pytest.raises(ValueError, match="invalid k value"):
            HarrisCorner(0.061, 5)

    @pytest.mark.valid
    @pytest.mark.positive
    def test_k_value_within_valid_range(self):
        floating_point_k_values = [0.045, 0.055]

        for k in floating_point_k_values:
            harris_corner = HarrisCorner(k, 5)
            assert harris_corner.k == k
            assert harris_corner.window_size == 5

    @pytest.mark.valid
    @pytest.mark.positive
    def test_k_value_at_lower_boundary(self):
        harris_corner = HarrisCorner(0.04, 5)
        assert harris_corner.k == 0.04
        assert harris_corner.window_size == 5

    @pytest.mark.valid
    @pytest.mark.positive
    def test_k_value_at_upper_boundary(self):
        harris_corner = HarrisCorner(0.06, 5)
        assert harris_corner.k == 0.06
        assert harris_corner.window_size == 5

    @pytest.mark.negative
    def test_invalid_window_size(self):
        with pytest.raises(ValueError, match="invalid k value"):
            HarrisCorner(0.05, -1)

    @pytest.mark.negative
    def test_invalid_window_size(self):
        with pytest.raises(ValueError, match="invalid k value"):
            HarrisCorner(0.05, 0)
