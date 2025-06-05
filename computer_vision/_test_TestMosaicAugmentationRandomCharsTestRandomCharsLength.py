import pytest
import random
from string import ascii_lowercase, digits
from mosaic_augmentation import random_chars

class Test_TestMosaicAugmentationRandomCharsTestRandomCharsLength:

    def test_correct_number_of_characters(self):
        result = random_chars(10)
        assert len(result) == 10

    def test_minimum_boundary_condition(self):
        result = random_chars(2)
        assert len(result) == 2

    def test_large_input_values(self):
        result = random_chars(1000)
        assert len(result) == 1000

    def test_character_uniqueness(self):
        results = [random_chars(10) for _ in range(5)]
        assert len(set(results)) == 5  # TODO: Change the number of iterations as needed

    def test_invalid_input_types(self):
        with pytest.raises(TypeError):
            random_chars("10")
        with pytest.raises(TypeError):
            random_chars(10.5)

    def test_character_set_constraints(self):
        result = random_chars(20)
        assert all(char in ascii_lowercase + digits for char in result)

    def test_upper_boundary_condition(self):
        result = random_chars(10000)
        assert len(result) == 10000
