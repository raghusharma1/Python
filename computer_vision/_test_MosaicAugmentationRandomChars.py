import pytest
from mosaic_augmentation import random_chars
import time
from string import ascii_lowercase, digits
from unittest.mock import patch

class Test_MosaicAugmentationRandomChars:

    @pytest.mark.valid
    def test_generates_correct_number_of_characters(self):
        result = random_chars(10)
        assert len(result) == 10

    @pytest.mark.valid
    def test_minimum_boundary_value(self):
        result = random_chars(2)
        assert len(result) == 2

    @pytest.mark.performance
    def test_large_input_value_efficiency(self):
        start_time = time.time()
        result = random_chars(1000)
        end_time = time.time()
        assert len(result) == 1000
        assert end_time - start_time < 1  # TODO: Adjust the time limit as necessary.

    @pytest.mark.valid
    def test_generates_unique_strings(self):
        results = [random_chars(10) for _ in range(100)]
        assert len(set(results)) == 100

    @pytest.mark.invalid
    def test_invalid_input_values(self):
        with pytest.raises(AssertionError, match="The number of character should greater than 1"):
            random_chars(1)

    @pytest.mark.valid
    def test_uses_specified_character_set(self):
        result = random_chars(10)
        assert all(char in ascii_lowercase + digits for char in result)

    @pytest.mark.invalid
    def test_unicode_input(self):
        with pytest.raises(AssertionError, match="The number of character should greater than 1"):
            random_chars('十')

    @pytest.mark.invalid
    def test_special_characters(self):
        with pytest.raises(AssertionError, match="The number of character should greater than 1"):
            random_chars('!@#')

    @pytest.mark.valid
    def test_random_module_usage(self):
        with patch('random.choice', return_value='a'):
            result = random_chars(10)
            assert result == 'aaaaaaaaaa'

    @pytest.mark.invalid
    def test_floating_point_input(self):
        result = random_chars(5.5)
        assert len(result) == 5
