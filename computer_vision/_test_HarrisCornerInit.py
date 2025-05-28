import pytest
import cv2
from harris_corner import HarrisCorner

class Test_HarrisCornerInit:

    @pytest.mark.valid
    def test_valid_initialization_k_0_04(self):
        # Arrange
        k = 0.04
        window_size = 3  # TODO: Change the value as needed

        # Act
        harris_corner = HarrisCorner(k, window_size)

        # Assert
        assert harris_corner.k == k
        assert harris_corner.window_size == window_size

    @pytest.mark.valid
    def test_valid_initialization_k_0_06(self):
        # Arrange
        k = 0.06
        window_size = 3  # TODO: Change the value as needed

        # Act
        harris_corner = HarrisCorner(k, window_size)

        # Assert
        assert harris_corner.k == k
        assert harris_corner.window_size == window_size

    @pytest.mark.invalid
    def test_invalid_initialization_k_0_03(self):
        # Arrange
        k = 0.03
        window_size = 3  # TODO: Change the value as needed

        # Act & Assert
        with pytest.raises(ValueError, match="invalid k value"):
            HarrisCorner(k, window_size)

    @pytest.mark.invalid
    def test_invalid_initialization_k_0_07(self):
        # Arrange
        k = 0.07
        window_size = 3  # TODO: Change the value as needed

        # Act & Assert
        with pytest.raises(ValueError, match="invalid k value"):
            HarrisCorner(k, window_size)

    @pytest.mark.valid
    def test_valid_initialization_min_window_size(self):
        # Arrange
        k = 0.04
        window_size = 1

        # Act
        harris_corner = HarrisCorner(k, window_size)

        # Assert
        assert harris_corner.k == k
        assert harris_corner.window_size == window_size

    @pytest.mark.valid
    def test_valid_initialization_large_window_size(self):
        # Arrange
        k = 0.04
        window_size = 100

        # Act
        harris_corner = HarrisCorner(k, window_size)

        # Assert
        assert harris_corner.k == k
        assert harris_corner.window_size == window_size

    @pytest.mark.valid
    def test_valid_initialization_typical_window_size(self):
        # Arrange
        k = 0.04
        window_size = 3

        # Act
        harris_corner = HarrisCorner(k, window_size)

        # Assert
        assert harris_corner.k == k
        assert harris_corner.window_size == window_size

    @pytest.mark.edge
    def test_edge_case_initialization_window_size_zero(self):
        # Arrange
        k = 0.04
        window_size = 0

        # Act
        harris_corner = HarrisCorner(k, window_size)

        # Assert
        assert harris_corner.k == k
        assert harris_corner.window_size == window_size

    @pytest.mark.edge
    def test_edge_case_initialization_negative_window_size(self):
        # Arrange
        k = 0.04
        window_size = -3  # TODO: Change the value as needed

        # Act
        harris_corner = HarrisCorner(k, window_size)

        # Assert
        assert harris_corner.k == k
        assert harris_corner.window_size == window_size

    @pytest.mark.invalid
    def test_initialization_k_boundary_float(self):
        # Arrange
        k_values = [0.039999, 0.060001]
        window_size = 3  # TODO: Change the value as needed

        for k in k_values:
            # Act & Assert
            with pytest.raises(ValueError, match="invalid k value"):
                HarrisCorner(k, window_size)
