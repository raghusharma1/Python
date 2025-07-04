import pytest
from harris_corner import HarrisCorner

class Test_HarrisCornerInit:

    @pytest.mark.valid
    @pytest.mark.positive
    def test_valid_initialization_k_0_04(self):
        harris = HarrisCorner(0.04, 3)
        assert harris.k == 0.04
        assert harris.window_size == 3

    @pytest.mark.valid
    @pytest.mark.positive
    def test_valid_initialization_k_0_06(self):
        harris = HarrisCorner(0.06, 3)
        assert harris.k == 0.06
        assert harris.window_size == 3

    @pytest.mark.invalid
    @pytest.mark.negative
    def test_invalid_initialization_k_0_03(self):
        with pytest.raises(ValueError, match="invalid k value"):
            HarrisCorner(0.03, 3)

    @pytest.mark.invalid
    @pytest.mark.negative
    def test_invalid_initialization_k_0_07(self):
        with pytest.raises(ValueError, match="invalid k value"):
            HarrisCorner(0.07, 3)

    @pytest.mark.valid
    @pytest.mark.positive
    def test_valid_initialization_min_window_size(self):
        harris = HarrisCorner(0.04, 1)
        assert harris.k == 0.04
        assert harris.window_size == 1

    @pytest.mark.valid
    @pytest.mark.positive
    def test_valid_initialization_large_window_size(self):
        harris = HarrisCorner(0.04, 100)
        assert harris.k == 0.04
        assert harris.window_size == 100

    @pytest.mark.invalid
    @pytest.mark.negative
    def test_invalid_initialization_negative_window_size(self):
        with pytest.raises(ValueError, match="invalid window size"):
            HarrisCorner(0.04, -5)

    @pytest.mark.invalid
    @pytest.mark.negative
    def test_invalid_initialization_zero_window_size(self):
        with pytest.raises(ValueError, match="invalid window size"):
            HarrisCorner(0.04, 0)
