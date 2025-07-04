import pytest
from flip_augmentation import random_chars
import random
from string import ascii_lowercase, digits

class Test_FlipAugmentationRandomChars:

    @pytest.mark.smoke
    @pytest.mark.valid
    def test_default_number_of_characters(self):
        result = random_chars()
        assert len(result) == 32

    @pytest.mark.valid
    def test_specified_number_of_characters(self):
        result = random_chars(10)
        assert len(result) == 10

    @pytest.mark.valid
    def test_minimum_number_of_characters(self):
        result = random_chars(2)
        assert len(result) == 2

    @pytest.mark.invalid
    @pytest.mark.negative
    def test_invalid_number_of_characters_less_than_2(self):
        with pytest.raises(AssertionError, match="The number of character should greater than 1"):
            random_chars(1)

    @pytest.mark.valid
    def test_randomness_of_generated_characters(self):
        results = [random_chars(10) for _ in range(5)]
        assert len(set(results)) == 5

    @pytest.mark.valid
    def test_character_set_inclusion(self):
        result = random_chars(10)
        allowed_chars = ascii_lowercase + digits
        assert all(char in allowed_chars for char in result)

    @pytest.mark.performance
    def test_large_number_of_characters(self):
        result = random_chars(10000)
        assert len(result) == 10000

    @pytest.mark.invalid
    @pytest.mark.negative
    def test_non_integer_input(self):
        with pytest.raises(TypeError):
            random_chars('a string')
