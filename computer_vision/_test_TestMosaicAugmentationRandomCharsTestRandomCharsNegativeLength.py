import pytest
from mosaic_augmentation import random_chars

class Test_MosaicAugmentationRandomCharsTestRandomCharsNegativeLength:
    @pytest.mark.invalid
    @pytest.mark.negative
    def test_random_chars_negative_length(self):
        with pytest.raises(AssertionError, match="The number of character should greater than 1"):
            random_chars(-5)

    @pytest.mark.invalid
    @pytest.mark.negative
    def test_random_chars_zero_length(self):
        with pytest.raises(AssertionError, match="The number of character should greater than 1"):
            random_chars(0)

    @pytest.mark.invalid
    @pytest.mark.negative
    def test_random_chars_one_length(self):
        with pytest.raises(AssertionError, match="The number of character should greater than 1"):
            random_chars(1)

    @pytest.mark.invalid
    @pytest.mark.negative
    def test_random_chars_large_negative_length(self):
        with pytest.raises(AssertionError, match="The number of character should greater than 1"):
            random_chars(-1000)

    @pytest.mark.invalid
    @pytest.mark.negative
    def test_random_chars_negative_one_length(self):
        with pytest.raises(AssertionError, match="The number of character should greater than 1"):
            random_chars(-1)
