# Corrected Test Cases File: test_PermutationsBacktrack.py
import pytest
from data_structures.arrays.permutations import backtrack
from math import factorial

class TestPermutationsBacktrack:
    @pytest.mark.valid
    @pytest.mark.smoke
    def test_generate_all_permutations_distinct_elements(self):
        nums = [1, 2, 3]
        output = []
        backtrack(nums, output, 0)
        expected_permutations = [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]
        assert sorted(output) == sorted(expected_permutations)

    @pytest.mark.invalid
    @pytest.mark.regression
    def test_empty_list_permutation(self):
        nums = []
        output = []
        backtrack(nums, output, 0)
        assert output == []

    @pytest.mark.valid
    @pytest.mark.edge
    def test_identical_elements_permutation(self):
        nums = [1, 1, 1]
        output = []
        backtrack(nums, output, 0)
        expected_permutations = [[1, 1, 1]]
        assert len(output) == factorial(len(nums))
        assert all(perm == [1, 1, 1] for perm in output)

    @pytest.mark.valid
    @pytest.mark.regression
    def test_negative_and_positive_integers(self):
        nums = [-1, 0, 1]
        output = []
        backtrack(nums, output, 0)
        expected_permutations = [[-1, 0, 1], [-1, 1, 0], [0, -1, 1], [0, 1, -1], [1, -1, 0], [1, 0, -1]]
        assert sorted(output) == sorted(expected_permutations)

    @pytest.mark.performance
    @pytest.mark.valid
    def test_large_input_list_permutations(self):
        nums = [1, 2, 3, 4, 5, 6]
        output = []
        backtrack(nums, output, 0)
        expected_count = factorial(len(nums))
        assert len(output) == expected_count
        assert len(set(map(tuple, output))) == expected_count

    @pytest.mark.valid
    @pytest.mark.regression
    def test_mixed_data_types(self):
        nums = [1, 'a', 3]
        output = []
        backtrack(nums, output, 0)
        expected_permutations = [[1, 'a', 3], [1, 3, 'a'], ['a', 1, 3], ['a', 3, 1], [3, 1, 'a'], [3, 'a', 1]]
        assert sorted(output) == sorted(expected_permutations)

    @pytest.mark.valid
    @pytest.mark.edge
    def test_single_element_list_permutation(self):
        nums = [42]
        output = []
        backtrack(nums, output, 0)
        expected_permutations = [[42]]
        assert output == expected_permutations

    @pytest.mark.valid
    @pytest.mark.edge
    def test_mixed_duplicate_and_distinct_elements(self):
        nums = [1, 1, 2]
        output = []
        backtrack(nums, output, 0)
        expected_permutations = [[1, 1, 2], [1, 2, 1], [2, 1, 1]]
        assert sorted(output) == sorted(expected_permutations)
        assert len(output) == len(expected_permutations)
