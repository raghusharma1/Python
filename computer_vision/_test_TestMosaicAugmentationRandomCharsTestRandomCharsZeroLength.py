import pytest
import random
from string import ascii_lowercase, digits
from mosaic_augmentation import random_chars

class Test_TestMosaicAugmentationRandomCharsTestRandomCharsZeroLength:
    @pytest.mark.invalid
    def test_random_chars_zero_length(self):
        with pytest.raises(AssertionError, match="The number of character should greater than 1"):
            random_chars(0)

    @pytest.mark.invalid
    def test_random_chars_negative_length(self):
        with pytest.raises(AssertionError, match="The number of character should greater than 1"):
            random_chars(-5)

    @pytest.mark.valid
    def test_random_chars_valid_length(self):
        valid_length = 10
        result = random_chars(valid_length)
        assert len(result) == valid_length

    @pytest.mark.valid
    def test_random_chars_valid_characters(self):
        valid_length = 10
        result = random_chars(valid_length)
        valid_characters = ascii_lowercase + digits
        for char in result:
            assert char in valid_characters

    @pytest.mark.valid
    def test_random_chars_unique_strings(self):
        valid_length = 10
        results = [random_chars(valid_length) for _ in range(100)]
        assert len(set(results)) == len(results)

    @pytest.mark.valid
    def test_random_chars_minimum_length(self):
        minimum_length = 2
        result = random_chars(minimum_length)
        assert len(result) == minimum_length
        valid_characters = ascii_lowercase + digits
        for char in result:
            assert char in valid_characters
