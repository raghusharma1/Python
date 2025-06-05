import pytest
import random
from string import ascii_lowercase, digits
from flip_augmentation import random_chars

class Test_TestFlipAugmentationRandomCharsTestRandomCharsBelowMinLength:

    @pytest.mark.invalid
    @pytest.mark.negative
    @pytest.mark.smoke
    def test_random_chars_below_min_length(self):
        with pytest.raises(AssertionError, match="The number of character should greater than 1"):
            random_chars(1)

    @pytest.mark.invalid
    @pytest.mark.negative
    @pytest.mark.smoke
    def test_random_chars_negative_length(self):
        with pytest.raises(AssertionError, match="The number of character should greater than 1"):
            random_chars(-5)

    @pytest.mark.invalid
    @pytest.mark.negative
    @pytest.mark.smoke
    def test_random_chars_zero_length(self):
        with pytest.raises(AssertionError, match="The number of character should greater than 1"):
            random_chars(0)

    @pytest.mark.invalid
    @pytest.mark.negative
    @pytest.mark.smoke
    def test_random_chars_non_integer_length(self):
        with pytest.raises(AssertionError, match="The number of character should greater than 1"):
            random_chars("five")

    @pytest.mark.invalid
    @pytest.mark.negative
    @pytest.mark.smoke
    def test_random_chars_float_length(self):
        with pytest.raises(AssertionError, match="The number of character should greater than 1"):
            random_chars(5.5)

    @pytest.mark.valid
    @pytest.mark.positive
    def test_random_chars_large_integer_length(self):
        result = random_chars(1000000)
        assert len(result) == 1000000

    @pytest.mark.valid
    @pytest.mark.positive
    def test_random_chars_default_length(self):
        result = random_chars()
        assert len(result) == 32
