import pytest
import random
from string import ascii_lowercase, digits
from flip_augmentation import random_chars

class Test_TestFlipAugmentationRandomCharsTestRandomCharsCustomLength:

    @pytest.mark.valid
    def test_random_chars_custom_length(self):
        custom_length = 16
        result = random_chars(custom_length)
        assert len(result) == custom_length

    @pytest.mark.valid
    def test_random_chars_content(self):
        result = random_chars()
        allowed_chars = ascii_lowercase + digits
        assert all(char in allowed_chars for char in result)

    @pytest.mark.valid
    def test_random_chars_edge_length(self):
        custom_length = 2
        result = random_chars(custom_length)
        assert len(result) == custom_length

    @pytest.mark.invalid
    def test_random_chars_invalid_length(self):
        custom_length = 1
        with pytest.raises(AssertionError, match="The number of character should greater than 1"):
            random_chars(custom_length)

    @pytest.mark.valid
    def test_random_chars_randomness(self):
        custom_length = 16
        results = [random_chars(custom_length) for _ in range(10)]
        assert len(set(results)) == 10

    @pytest.mark.performance
    def test_random_chars_large_length(self):
        custom_length = 1000
        result = random_chars(custom_length)
        assert len(result) == custom_length
