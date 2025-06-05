# Import necessary modules
import pytest
import random
from string import ascii_lowercase, digits
from mosaic_augmentation import random_chars

class Test_TestMosaicAugmentationRandomCharsTestRandomCharsMinLength:

    def test_random_chars_min_length(self):
        with pytest.raises(AssertionError, match="The number of character should greater than 1"):
            random_chars(1)

    def test_random_chars_exact_length(self):
        test_length = 15
        result = random_chars(test_length)
        assert len(result) == test_length

    def test_random_chars_valid_characters(self):
        test_length = 20
        valid_characters = ascii_lowercase + digits
        result = random_chars(test_length)
        assert all(char in valid_characters for char in result)

    def test_random_chars_different_outputs(self):
        test_length = 10
        results = [random_chars(test_length) for _ in range(10)]
        assert len(set(results)) == len(results)

    def test_random_chars_large_input(self):
        test_length = 10000
        result = random_chars(test_length)
        assert len(result) == test_length
