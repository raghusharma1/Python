import pytest
import random
from string import ascii_lowercase, digits
from flip_augmentation import random_chars

class Test_TestFlipAugmentationRandomCharsTestRandomCharsDefault:

    @pytest.mark.smoke
    @pytest.mark.valid
    def test_random_chars_default_length(self):
        result = random_chars()
        assert len(result) == 32

    @pytest.mark.valid
    @pytest.mark.regression
    def test_random_chars_custom_length(self):
        result = random_chars(40)
        assert len(result) == 40

    @pytest.mark.invalid
    @pytest.mark.negative
    def test_random_chars_min_length_constraint(self):
        with pytest.raises(AssertionError, match="The number of character should greater than 1"):
            random_chars(1)

    @pytest.mark.valid
    @pytest.mark.regression
    def test_random_chars_randomness(self):
        result = random_chars()
        assert any(char in ascii_lowercase for char in result)
        assert any(char in digits for char in result)

    @pytest.mark.valid
    @pytest.mark.regression
    def test_random_chars_consistency(self):
        results = [random_chars(20) for _ in range(10)]
        assert len(results) == len(set(results)), "Duplicate strings found in results"
