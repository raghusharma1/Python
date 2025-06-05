import pytest
import random
from string import ascii_lowercase, digits
from flip_augmentation import random_chars

class Test_TestFlipAugmentationRandomCharsTestRandomCharsMinLength:
    @pytest.mark.smoke
    @pytest.mark.valid
    @pytest.mark.positive
    def test_random_chars_min_length(self):
        result = random_chars(2)
        assert len(result) == 2

    @pytest.mark.positive
    @pytest.mark.regression
    def test_random_chars_boundary_minimum(self):
        result = random_chars(3)
        assert len(result) == 3

    @pytest.mark.positive
    @pytest.mark.regression
    def test_random_chars_within_valid_range(self):
        expected_length = random.randint(2, 100)
        result = random_chars(expected_length)
        assert len(result) == expected_length

    @pytest.mark.positive
    @pytest.mark.smoke
    @pytest.mark.regression
    def test_random_chars_max_length(self):
        result = random_chars(100)
        assert len(result) == 100

    @pytest.mark.positive
    @pytest.mark.regression
    def test_random_chars_boundary_maximum(self):
        result = random_chars(99)
        assert len(result) == 99

    @pytest.mark.negative
    @pytest.mark.invalid
    def test_random_chars_invalid_length_below_minimum(self):
        with pytest.raises(AssertionError):
            random_chars(1)

    @pytest.mark.positive
    @pytest.mark.regression
    def test_random_chars_character_set(self):
        result = random_chars(10)
        assert all(char in ascii_lowercase + digits for char in result)

    @pytest.mark.positive
    @pytest.mark.regression
    def test_random_chars_deterministic_output(self):
        random.seed(42)
        result1 = random_chars(10)
        random.seed(42)
        result2 = random_chars(10)
        assert result1 == result2
