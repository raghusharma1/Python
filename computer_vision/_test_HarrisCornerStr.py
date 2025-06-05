import cv2
import numpy as np
import pytest
from harris_corner import HarrisCorner

class Test_HarrisCornerStr:
    @pytest.mark.valid
    def test_str_with_valid_k(self):
        # Arrange
        k_value = 0.04
        harris_corner = HarrisCorner(k_value, 3)

        # Act
        result = str(harris_corner)

        # Assert
        assert result == str(k_value), f"Expected {str(k_value)}, but got {result}"

    @pytest.mark.valid
    def test_str_formatting(self):
        # Arrange
        k_value = 0.04
        harris_corner = HarrisCorner(k_value, 3)

        # Act
        result = str(harris_corner)

        # Assert
        assert result == "0.04", f"Expected '0.04', but got {result}"

    @pytest.mark.edge_case
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

    @pytest.mark.valid
    def test_str_comparison_with_conversion(self):
        # Arrange
        k_value = 0.04
        harris_corner = HarrisCorner(k_value, 3)

        # Act
        result = str(harris_corner)
        direct_conversion = str(k_value)

        # Assert
        assert result == direct_conversion, f"Expected {direct_conversion}, but got {result}"

    @pytest.mark.valid
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
