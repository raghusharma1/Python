import pytest
import cv2
import numpy as np
from _test_HarrisCornerInit import Test_HarrisCornerInit
from harris_corner import HarrisCorner

class Test_TestHarrisCornerInitTestBoundaryKValues(Test_HarrisCornerInit):

    @pytest.mark.valid
    @pytest.mark.smoke
    def test_initialization_with_boundary_k_values(self):
        # Arrange
        boundary_k_values = [0.04, 0.06]

        # Act & Assert
        for k in boundary_k_values:
            harris_corner = HarrisCorner(k, 5)
            assert harris_corner.k == k
            assert harris_corner.window_size == 5

    def test_boundary_k_values(self):
        boundary_k_values = [0.04, 0.06]

        for k in boundary_k_values:
            harris_corner = HarrisCorner(k, 5)
            assert harris_corner.k == k
            assert harris_corner.window_size == 5
