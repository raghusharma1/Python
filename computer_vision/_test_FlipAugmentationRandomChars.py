import pytest
import random
from string import ascii_lowercase, digits
from flip_augmentation import random_chars

class TestFlipAugmentationRandomChars:

    @pytest.mark.smoke
    @pytest.mark.valid
    @pytest.mark.positive
    def test_random_chars_default(self):
        result = random_chars()
        assert len(result) == 32

    @pytest.mark.valid
    @pytest.mark.positive
    def test_random_chars_custom_length(self):
        custom_length = 16
        result = random_chars(custom_length)
        assert len(result) == custom_length

    @pytest.mark.valid
    @pytest.mark.positive
    def test_random_chars_min_length(self):
        result = random_chars(2)
        assert len(result) == 2

    @pytest.mark.invalid
    @pytest.mark.negative
    def test_random_chars_below_min_length(self):
        with pytest.raises(AssertionError, match="The number of character should greater than 1"):
            random_chars(1)

    @pytest.mark.valid
    @pytest.mark.positive
    def test_random_chars_character_set(self):
        result = random_chars(16)
        assert all(char in ascii_lowercase + digits for char in result)

    @pytest.mark.valid
    @pytest.mark.positive
    def test_random_chars_randomness(self):
        results = [random_chars(32) for _ in range(10)]
        assert len(set(results)) == 10

    @pytest.mark.performance
    @pytest.mark.valid
    @pytest.mark.positive
    def test_random_chars_large_input(self):
        result = random_chars(1000)
        assert len(result) == 1000

    @pytest.mark.invalid
    @pytest.mark.negative
    def test_random_chars_negative_length(self):
        with pytest.raises(AssertionError, match="The number of character should greater than 1"):
            random_chars(-10)
