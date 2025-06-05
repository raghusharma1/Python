import pytest
import random
from string import ascii_lowercase, digits
from flip_augmentation import random_chars

class Test_TestFlipAugmentationRandomCharsTestRandomCharsRandomness:
    @pytest.mark.valid
    @pytest.mark.smoke
    def test_random_chars_consistency(self):
        number_char = 32
        result = random_chars(number_char)
        assert len(result) == number_char

    @pytest.mark.valid
    @pytest.mark.regression
    def test_random_chars_randomness(self):
        results = [random_chars(32) for _ in range(10)]
        assert len(set(results)) == 10

    @pytest.mark.valid
    @pytest.mark.regression
    def test_random_chars_different_lengths(self):
        lengths = [5, 10, 15, 20, 25]
        for length in lengths:
            result = random_chars(length)
            assert len(result) == length

    @pytest.mark.invalid
    @pytest.mark.negative
    def test_random_chars_minimum_character_assertion(self):
        with pytest.raises(AssertionError):
            random_chars(1)

    @pytest.mark.valid
    @pytest.mark.smoke
    def test_random_chars_character_set(self):
        valid_chars = set(ascii_lowercase + digits)
        result = random_chars(32)
        assert all(char in valid_chars for char in result)
