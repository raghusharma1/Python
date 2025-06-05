import pytest
from _test_HarrisCornerInit import Test_HarrisCornerInit
from harris_corner import HarrisCorner

# This is a test class for testing the HarrisCorner class with invalid k values
class Test_TestHarrisCornerInitTestInvalidKValue(Test_HarrisCornerInit):

    # Test case to check if the k value is below the lower boundary
    @pytest.mark.invalid
    @pytest.mark.negative
    @pytest.mark.parametrize("k", [0.03])
    def test_k_value_below_lower_boundary(self, k):
        with pytest.raises(ValueError, match="invalid k value"):
            HarrisCorner(k, 5)

    # Test case to check if the k value is above the upper boundary
    @pytest.mark.invalid
    @pytest.mark.negative
    @pytest.mark.parametrize("k", [0.07])
    def test_k_value_above_upper_boundary(self, k):
        with pytest.raises(ValueError, match="invalid k value"):
            HarrisCorner(k, 5)

    # Test case to check if the k value is at the lower boundary
    @pytest.mark.valid
    @pytest.mark.positive
    @pytest.mark.parametrize("k", [0.04])
    def test_k_value_at_lower_boundary(self, k):
        try:
            HarrisCorner(k, 5)
        except ValueError:
            pytest.fail("HarrisCorner raised ValueError unexpectedly!")

    # Test case to check if the k value is at the upper boundary
    @pytest.mark.valid
    @pytest.mark.positive
    @pytest.mark.parametrize("k", [0.06])
    def test_k_value_at_upper_boundary(self, k):
        try:
            HarrisCorner(k, 5)
        except ValueError:
            pytest.fail("HarrisCorner raised ValueError unexpectedly!")

    # Test case to check if the k value is within the valid range
    @pytest.mark.valid
    @pytest.mark.positive
    @pytest.mark.parametrize("k", [0.05])
    def test_k_value_within_valid_range(self, k):
        try:
            HarrisCorner(k, 5)
        except ValueError:
            pytest.fail("HarrisCorner raised ValueError unexpectedly!")

    # Test case to check if the k value is non-numeric
    @pytest.mark.invalid
    @pytest.mark.negative
    @pytest.mark.parametrize("k", ["invalid"])
    def test_non_numeric_k_value(self, k):
        with pytest.raises(TypeError):
            HarrisCorner(k, 5)

    # Test case to check if the k value is negative
    @pytest.mark.invalid
    @pytest.mark.negative
    @pytest.mark.parametrize("k", [-0.05])
    def test_negative_k_value(self, k):
        with pytest.raises(ValueError, match="invalid k value"):
            HarrisCorner(k, 5)

    # Test case to check if the k value is zero
    @pytest.mark.invalid
    @pytest.mark.negative
    @pytest.mark.parametrize("k", [0])
    def test_zero_k_value(self, k):
        with pytest.raises(ValueError, match="invalid k value"):
            HarrisCorner(k, 5)

    # Test case to check if the k value is None
    @pytest.mark.invalid
    @pytest.mark.negative
    @pytest.mark.parametrize("k", [None])
    def test_none_k_value(self, k):
        with pytest.raises(TypeError):
            HarrisCorner(k, 5)

    # Test case to check if the k value is an empty string
    @pytest.mark.invalid
    @pytest.mark.negative
    @pytest.mark.parametrize("k", [""])
    def test_empty_string_k_value(self, k):
        with pytest.raises(TypeError):
            HarrisCorner(k, 5)

    # Test case to check if the k value is a boolean
    @pytest.mark.invalid
    @pytest.mark.negative
    @pytest.mark.parametrize("k", [True])
    def test_boolean_k_value(self, k):
        with pytest.raises(TypeError):
            HarrisCorner(k, 5)

    # Test case to check if the k value is a list
    @pytest.mark.invalid
    @pytest.mark.negative
    @pytest.mark.parametrize("k", [[0.04]])
    def test_list_k_value(self, k):
        with pytest.raises(TypeError):
            HarrisCorner(k, 5)

    # Test case to check if the k value is a dictionary
    @pytest.mark.invalid
    @pytest.mark.negative
    @pytest.mark.parametrize("k", [{"k": 0.04}])
    def test_dict_k_value(self, k):
        with pytest.raises(TypeError):
            HarrisCorner(k, 5)

    # Test case to check if the k value is a tuple
    @pytest.mark.invalid
    @pytest.mark.negative
    @pytest.mark.parametrize("k", [(0.04,)])
    def test_tuple_k_value(self, k):
        with pytest.raises(TypeError):
            HarrisCorner(k, 5)

    # Test case to check if the k value is an object
    @pytest.mark.invalid
    @pytest.mark.negative
    @pytest.mark.parametrize("k", [object()])
    def test_object_k_value(self, k):
        with pytest.raises(TypeError):
            HarrisCorner(k, 5)
