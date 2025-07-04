import pytest
import random
from string import ascii_lowercase, digits
from mosaic_augmentation import random_chars

class Test_MosaicAugmentationRandomChars:

    @pytest.mark.valid
    @pytest.mark.smoke
    def test_random_chars_basic(self):
        result = random_chars(32)
        assert len(result) == 32

    @pytest.mark.valid
    @pytest.mark.smoke
    def test_random_chars_min_length(self):
        result = random_chars(2)
        assert len(result) == 2

    @pytest.mark.invalid
    @pytest.mark.negative
    def test_random_chars_invalid_input(self):
        with pytest.raises(AssertionError, match="The number of character should greater than 1"):
            random_chars(1)

    @pytest.mark.valid
    @pytest.mark.smoke
    def test_random_chars_character_set(self):
        result = random_chars(32)
        assert all(c in ascii_lowercase + digits for c in result)

    @pytest.mark.valid
    @pytest.mark.smoke
    def test_random_chars_uniqueness(self):
        results = [random_chars(32) for _ in range(10)]
        assert len(set(results)) == 10

    @pytest.mark.performance
    def test_random_chars_performance(self):
        result = random_chars(1000000)
        assert len(result) == 1000000

    @pytest.mark.valid
    @pytest.mark.smoke
    def test_random_chars_large_input(self):
        result = random_chars(10**6)
        assert len(result) == 10**6

    @pytest.mark.valid
    @pytest.mark.smoke
    def test_random_chars_consistency(self):
        results = [random_chars(32) for _ in range(10)]
        assert all(len(result) == 32 for result in results)
