import pytest
import random
from string import ascii_lowercase, digits
from flip_augmentation import random_chars

class Test_TestFlipAugmentationRandomCharsTestRandomCharsCharacterSet:

    @pytest.mark.valid
    def test_random_chars_character_set_consistency(self):
        result = random_chars(16)
        assert all(char in ascii_lowercase + digits for char in result)

    @pytest.mark.valid
    def test_random_chars_length(self):
        result = random_chars(10)
        assert len(result) == 10

    @pytest.mark.valid
    def test_random_chars_minimum_length(self):
        result = random_chars(2)
        assert len(result) == 2
        assert all(char in ascii_lowercase + digits for char in result)

    @pytest.mark.valid
    def test_random_chars_randomness(self):
        results = [random_chars(16) for _ in range(10)]
        assert len(set(results)) == 10

    @pytest.mark.valid
    def test_random_chars_default_parameter(self):
        result = random_chars()
        assert len(result) == 32
        assert all(char in ascii_lowercase + digits for char in result)

    @pytest.mark.invalid
    def test_random_chars_invalid_input(self):
        with pytest.raises(AssertionError, match="The number of character should greater than 1"):
            random_chars(1)

    def test_random_chars_character_set(self):
        result = random_chars(16)
        assert all(char in ascii_lowercase + digits for char in result)
