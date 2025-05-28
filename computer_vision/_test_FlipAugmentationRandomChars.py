import pytest
import random
from string import ascii_lowercase, digits
from flip_augmentation import random_chars

class Test_FlipAugmentationRandomChars:

    @pytest.mark.smoke
    @pytest.mark.valid
    def test_default_character_count(self):
        result = random_chars()
        assert len(result) == 32

    @pytest.mark.smoke
    @pytest.mark.valid
    def test_specific_character_count(self):
        result = random_chars(10)
        assert len(result) == 10

    @pytest.mark.invalid
    @pytest.mark.negative
    def test_minimum_character_count(self):
        with pytest.raises(AssertionError, match="The number of character should greater than 1"):
            random_chars(1)

    @pytest.mark.valid
    def test_character_set_composition(self):
        result = random_chars(32)
        character_set = set(ascii_lowercase + digits)
        assert all(char in character_set for char in result)

    @pytest.mark.valid
    def test_repeated_character_generation(self):
        results = [random_chars(32) for _ in range(100)]
        assert len(set(results)) == 100

    @pytest.mark.performance
    @pytest.mark.valid
    def test_large_character_count(self):
        result = random_chars(10000)
        assert len(result) == 10000

    @pytest.mark.invalid
    @pytest.mark.negative
    def test_negative_character_count(self):
        with pytest.raises(AssertionError, match="The number of character should greater than 1"):
            random_chars(-5)

    @pytest.mark.invalid
    @pytest.mark.negative
    def test_zero_character_count(self):
        with pytest.raises(AssertionError, match="The number of character should greater than 1"):
            random_chars(0)
