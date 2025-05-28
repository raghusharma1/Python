import pytest
import numpy as np
from haralick_descriptors import normalize_array

class Test_HaralickDescriptorsNormalizeArray:

    @pytest.mark.valid
    def test_normalize_array_simple(self):
        array = np.array([2, 3, 5, 7])
        expected = np.array([0., 0.2, 0.6, 1.])
        result = normalize_array(array)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.valid
    def test_normalize_array_2d(self):
        array = np.array([[5], [7], [11], [13]])
        expected = np.array([[0.], [0.25], [0.75], [1.]])
        result = normalize_array(array)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.valid
    def test_normalize_array_custom_cap(self):
        array = np.array([2, 3, 5, 7])
        expected = np.array([0., 2., 6., 10.])
        result = normalize_array(array, cap=10)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.valid
    def test_normalize_array_equal_elements(self):
        array = np.array([5, 5, 5, 5])
        expected = np.array([0., 0., 0., 0.])
        result = normalize_array(array)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.valid
    def test_normalize_array_zero_difference(self):
        array = np.array([3, 3, 3, 3])
        expected = np.array([0., 0., 0., 0.])
        result = normalize_array(array)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.valid
    def test_normalize_array_negative_values(self):
        array = np.array([-2, -1, 0, 1, 2])
        expected = np.array([0., 0.25, 0.5, 0.75, 1.])
        result = normalize_array(array)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.valid
    def test_normalize_array_mixed_values(self):
        array = np.array([-5, -3, 0, 3, 5])
        expected = np.array([0., 0.25, 0.5, 0.75, 1.])
        result = normalize_array(array)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.valid
    def test_normalize_array_large(self):
        array = np.random.rand(1000) * 100  # TODO: Change the value of the array to test with different size of data
        result = normalize_array(array)
        assert np.all(result >= 0) and np.all(result <= 1)

