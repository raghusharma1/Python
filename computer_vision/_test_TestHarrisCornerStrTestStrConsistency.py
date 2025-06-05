import cv2
import numpy as np
import pytest
from harris_corner import HarrisCorner

class Test_TestHarrisCornerStrTestStrConsistency:
    @pytest.mark.valid
    def test_str_consistency(self):
        # Arrange
        k_value = 0.04
        harris_corner = HarrisCorner(k_value, 3)

        # Act
        result1 = str(harris_corner)
        result2 = str(harris_corner)

        # Assert
        assert result1 == result2, f"Expected consistency, but got {result1} and {result2}"
