import pytest
from harris_corner import HarrisCorner

class Test_HarrisCornerStr:
    @pytest.mark.valid
    def test_str_representation_valid_k(self):
        # Arrange
        k_value = 0.04
        window_size = 5
        harris_corner = HarrisCorner(k_value, window_size)

        # Act
        result = str(harris_corner)

        # Assert
        assert result == str(k_value)

    @pytest.mark.valid
    def test_str_representation_boundary_k(self):
        # Arrange
        k_values = [0.04, 0.06]
        window_size = 5

        for k in k_values:
            harris_corner = HarrisCorner(k, window_size)

            # Act
            result = str(harris_corner)

            # Assert
            assert result == str(k)

    @pytest.mark.invalid
    def test_str_representation_invalid_k(self):
        # Arrange
        k_value = 0.03
        window_size = 5

        with pytest.raises(ValueError) as excinfo:
            # Act
            harris_corner = HarrisCorner(k_value, window_size)

        # Assert
        assert "invalid k value" in str(excinfo.value)

    @pytest.mark.valid
    def test_str_representation_floating_k(self):
        # Arrange
        k_value = 0.045
        window_size = 5
        harris_corner = HarrisCorner(k_value, window_size)

        # Act
        result = str(harris_corner)

        # Assert
        assert result == str(k_value)

    @pytest.mark.valid
    def test_str_representation_different_window_size(self):
        # Arrange
        k_value = 0.04
        window_sizes = [3, 5, 7]

        for window_size in window_sizes:
            harris_corner = HarrisCorner(k_value, window_size)

            # Act
            result = str(harris_corner)

            # Assert
            assert result == str(k_value)
