import pytest
from data_structures.arrays.kth_largest_element import partition

class Test_KthLargestElementPartition:

    @pytest.mark.valid
    @pytest.mark.smoke
    def test_partition_unique_elements(self):
        arr = [3, 1, 4, 2, 5]
        low = 0
        high = len(arr) - 1
        pivot_index = partition(arr, low, high)
        assert all(arr[i] <= arr[pivot_index] for i in range(low, pivot_index))
        assert all(arr[i] >= arr[pivot_index] for i in range(pivot_index + 1, high + 1)

    @pytest.mark.valid
    @pytest.mark.regression
    def test_partition_duplicate_elements(self):
        arr = [5, 3, 8, 3, 6, 3]
        low = 0
        high = len(arr) - 1
        pivot_index = partition(arr, low, high)
        assert all(arr[i] <= arr[pivot_index] for i in range(low, pivot_index))
        assert all(arr[i] >= arr[pivot_index] for i in range(pivot_index + 1, high + 1)

    @pytest.mark.valid
    @pytest.mark.regression
    def test_partition_sorted_array(self):
        arr = [1, 2, 3, 4, 5]
        low = 0
        high = len(arr) - 1
        pivot_index = partition(arr, low, high)
        assert all(arr[i] <= arr[pivot_index] for i in range(low, pivot_index))
        assert all(arr[i] >= arr[pivot_index] for i in range(pivot_index + 1, high + 1)

    @pytest.mark.valid
    @pytest.mark.regression
    def test_partition_reverse_sorted_array(self):
        arr = [5, 4, 3, 2, 1]
        low = 0
        high = len(arr) - 1
        pivot_index = partition(arr, low, high)
        assert all(arr[i] <= arr[pivot_index] for i in range(low, pivot_index))
        assert all(arr[i] >= arr[pivot_index] for i in range(pivot_index + 1, high + 1)

    @pytest.mark.valid
    @pytest.mark.security
    def test_partition_mixed_sign_elements(self):
        arr = [-3, 0, 4, -1, 2]
        low = 0
        high = len(arr) - 1
        pivot_index = partition(arr, low, high)
        assert all(arr[i] <= arr[pivot_index] for i in range(low, pivot_index))
        assert all(arr[i] >= arr[pivot_index] for i in range(pivot_index + 1, high + 1)

    @pytest.mark.valid
    @pytest.mark.security
    def test_partition_single_element_array(self):
        arr = [42]
        low = 0
        high = 0
        pivot_index = partition(arr, low, high)
        assert pivot_index == 0
        assert arr == [42]

    @pytest.mark.invalid
    @pytest.mark.security
    def test_partition_mixed_data_types(self):
        arr = [1, 'string', 3]
        low = 0
        high = len(arr) - 1
        with pytest.raises(TypeError):
            partition(arr, low, high)

    @pytest.mark.invalid
    @pytest.mark.security
    def test_partition_empty_array(self):
        arr = []
        low = 0
        high = -1
        with pytest.raises(IndexError):
            partition(arr, low, high)

    @pytest.mark.valid
    @pytest.mark.performance
    def test_partition_floating_point_numbers(self):
        arr = [3.1, 1.2, 4.7, 5.6]
        low = 0
        high = len(arr) - 1
        pivot_index = partition(arr, low, high)
        assert all(arr[i] <= arr[pivot_index] for i in range(low, pivot_index))
        assert all(arr[i] >= arr[pivot_index] for i in range(pivot_index + 1, high + 1)

    @pytest.mark.valid
    @pytest.mark.performance
    def test_partition_extreme_pivot_values(self):
        # Pivot is the smallest element
        arr = [10, 5, 1, 6, 2]
        low = 0
        high = len(arr) - 1
        pivot_index = partition(arr, low, high)
        assert all(arr[i] <= arr[pivot_index] for i in range(low, pivot_index))
        assert all(arr[i] >= arr[pivot_index] for i in range(pivot_index + 1, high + 1)

        # Pivot is the largest element
        arr = [1, 5, 10, 6, 3]
        pivot_index = partition(arr, low, high)
        assert all(arr[i] <= arr[pivot_index] for i in range(low, pivot_index))
        assert all(arr[i] >= arr[pivot_index] for i in range(pivot_index + 1, high + 1)
