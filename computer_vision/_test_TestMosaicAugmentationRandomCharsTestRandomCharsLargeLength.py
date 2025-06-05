import pytest
import random
from string import ascii_lowercase, digits
from _test_MosaicAugmentationRandomChars import Test_MosaicAugmentationRandomChars
from mosaic_augmentation import random_chars

class Test_TestMosaicAugmentationRandomCharsTestRandomCharsLargeLength(Test_MosaicAugmentationRandomChars):
    @pytest.mark.valid
    def test_random_chars_large_length_exact_length(self):
        result = random_chars(1000)
        assert len(result) == 1000

    @pytest.mark.valid
    def test_random_chars_large_length_valid_characters(self):
        result = random_chars(1000)
        valid_chars = set(ascii_lowercase + digits)
        assert all(char in valid_chars for char in result)

    @pytest.mark.valid
    def test_random_chars_large_length_minimum_valid_length(self):
        result = random_chars(2)
        assert len(result) == 2

    @pytest.mark.invalid
    def test_random_chars_large_length_invalid_minimum_length(self):
        with pytest.raises(AssertionError, match="The number of character should greater than 1"):
            random_chars(1)

    @pytest.mark.performance
    def test_random_chars_large_length_efficiency(self):
        import time
        start_time = time.time()
        result = random_chars(1000000)
        end_time = time.time()
        assert end_time - start_time < 5.0

    @pytest.mark.valid
    def test_random_chars_large_length_determinism(self):
        results = [random_chars(1000) for _ in range(10)]
        assert len(set(results)) == 10
