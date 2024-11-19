import pytest
from conversions.decimal_to_binary import decimal_to_binary_iterative
import doctest

def test_decimal_to_binary_iterative_with_zero():
    result = decimal_to_binary_iterative(0)
    # Assuming the function should return '0' without the '0b' prefix
    assert result == '0', f"Expected '0', got {result}"

def test_decimal_to_binary_iterative_with_positive_integer():
    result = decimal_to_binary_iterative(5)
    assert result == '101', f"Expected '101', got {result}"

def test_decimal_to_binary_iterative_with_large_positive_integer():
    result = decimal_to_binary_iterative(1023)
    assert result == '1111111111', f"Expected '1111111111', got {result}"

def test_decimal_to_binary_iterative_with_negative_integer():
    with pytest.raises(ValueError):
        decimal_to_binary_iterative(-5)

def test_decimal_to_binary_iterative_with_boundary_positive_integer():
    result = decimal_to_binary_iterative(1)
    assert result == '1', f"Expected '1', got {result}"

def test_decimal_to_binary_iterative_with_power_of_two():
    result = decimal_to_binary_iterative(8)
    assert result == '1000', f"Expected '1000', got {result}"

def test_decimal_to_binary_iterative_with_odd_number():
    result = decimal_to_binary_iterative(7)
    assert result == '111', f"Expected '111', got {result}"

if __name__ == '__main__':
    doctest.testmod()
