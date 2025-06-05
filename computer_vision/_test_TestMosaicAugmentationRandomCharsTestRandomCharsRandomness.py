import pytest
import random
from string import ascii_lowercase, digits
from mosaic_augmentation import random_chars

class Test_TestMosaicAugmentationRandomCharsTestRandomCharsRandomness:

    @pytest.mark.positive
    @pytest.mark.smoke
    def test_random_chars_generates_different_characters(self):
        # Arrange
        number_of_calls = 10  # TODO: change if required

        # Act
        results = [random_chars(32) for _ in range(number_of_calls)]

        # Assert
        assert len(set(results)) == number_of_calls

    def test_random_chars_randomness(self):
        results = [random_chars(32) for _ in range(10)]
        assert len(set(results)) == 10
