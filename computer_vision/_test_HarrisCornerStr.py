import cv2
import numpy as np
import pytest
from harris_corner import HarrisCorner

class Test_HarrisCornerStr:

    @pytest.mark.valid
    def test_str_representation_k_0_04(self):
        # Arrange
        harris_corner = HarrisCorner(0.04, 3)
        # Act
        result = harris_corner.__str__()
        # Assert
        assert result == "0.04"

    @pytest.mark.valid
    def test_str_representation_k_0_06(self):
        # Arrange
        harris_corner = HarrisCorner(0.06, 3)
        # Act
        result = harris_corner.__str__()
        # Assert
        assert result == "0.06"

    @pytest.mark.valid
    def test_str_representation_k_within_valid_range(self):
        # Arrange
        harris_corner = HarrisCorner(0.05, 3)
        # Act
        result = harris_corner.__str__()
        # Assert
        assert result == "0.05"

    @pytest.mark.valid
    def test_str_representation_k_at_lower_boundary(self):
        # Arrange
        harris_corner = HarrisCorner(0.04, 3)
        # Act
        result = harris_corner.__str__()
        # Assert
        assert result == "0.04"

    @pytest.mark.valid
    def test_str_representation_k_at_upper_boundary(self):
        # Arrange
        harris_corner = HarrisCorner(0.06, 3)
        # Act
        result = harris_corner.__str__()
        # Assert
        assert result == "0.06"
