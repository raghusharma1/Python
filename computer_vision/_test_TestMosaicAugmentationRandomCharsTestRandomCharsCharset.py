import pytest
import random
from string import ascii_lowercase, digits
from mosaic_augmentation import random_chars
from _test_MosaicAugmentationRandomChars import Test_MosaicAugmentationRandomChars

class Test_TestMosaicAugmentationRandomCharsTestRandomCharsCharset(Test_MosaicAugmentationRandomChars):

    @pytest.mark.valid
    @pytest.mark.smoke
    def test_random_chars_charset(self):
        result = random_chars(32)
        allowed_chars = set(ascii_lowercase + digits)
        assert all(char in allowed_chars for char in result)

    @pytest.mark.valid
    @pytest.mark.regression
    def test_random_chars_length(self):
        result = random_chars(32)
        assert len(result) == 32

    @pytest.mark.valid
    @pytest.mark.regression
    def test_random_chars_different_lengths(self):
        lengths = [16, 64, 128]
        for length in lengths:
            result = random_chars(length)
            assert len(result) == length

    @pytest.mark.valid
    @pytest.mark.regression
    def test_random_chars_uniqueness(self):
        generated_strings = set()
        for _ in range(100):  # TODO: Change the number of strings to generate based on requirements.
            result = random_chars(32)
            assert result not in generated_strings
            generated_strings.add(result)

    @pytest.mark.invalid
    @pytest.mark.negative
    def test_random_chars_invalid_input(self):
        invalid_lengths = [0, 1, -1]
        for length in invalid_lengths:
            with pytest.raises(AssertionError, match="The number of character should greater than 1"):
                random_chars(length)
