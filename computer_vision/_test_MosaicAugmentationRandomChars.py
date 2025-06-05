from string import ascii_lowercase, digits
from mosaic_augmentation import random_chars

import pytest

class Test_MosaicAugmentationRandomChars:

    @pytest.mark.valid
    @pytest.mark.smoke
    def test_random_chars_length(self):
        result = random_chars(10)
        assert len(result) == 10

    @pytest.mark.valid
    @pytest.mark.smoke
    def test_random_chars_charset(self):
        result = random_chars(32)
        allowed_chars = set(ascii_lowercase + digits)
        assert all(char in allowed_chars for char in result)

    @pytest.mark.invalid
    @pytest.mark.negative
    def test_random_chars_min_length(self):
        with pytest.raises(AssertionError, match="The number of character should greater than 1"):
            random_chars(1)

    @pytest.mark.valid
    def test_random_chars_randomness(self):
        results = [random_chars(32) for _ in range(10)]
        assert len(set(results)) == 10

    @pytest.mark.valid
    def test_random_chars_large_length(self):
        result = random_chars(1000)
        assert len(result) == 1000

    @pytest.mark.invalid
    @pytest.mark.negative
    def test_random_chars_negative_length(self):
        with pytest.raises(AssertionError, match="The number of character should greater than 1"):
            random_chars(-5)

    @pytest.mark.invalid
    @pytest.mark.negative
    def test_random_chars_zero_length(self):
        with pytest.raises(AssertionError, match="The number of character should greater than 1"):
            random_chars(0)
