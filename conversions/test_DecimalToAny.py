import pytest
from conversions.decimal_to_any import decimal_to_any
from string import ascii_uppercase
import doctest

class Test_DecimalToAny:
    
    @pytest.mark.valid
    @pytest.mark.smoke
    def test_conversion_of_zero(self):
        # Test for conversion of zero to various bases
        bases = [2, 10, 16]
        for base in bases:
            # Act
            result = decimal_to_any(0, base)
            # Assert
            assert result == "0", f"Expected '0' for base {base}, got {result}"

    @pytest.mark.valid
    @pytest.mark.regression
    def test_decimal_to_binary(self):
        # Test for positive decimal conversion to binary
        # Act
        result = decimal_to_any(5, 2)
        # Assert
        assert result == "101", f"Expected '101', got {result}"

    @pytest.mark.valid
    @pytest.mark.regression
    def test_decimal_to_hexadecimal(self):
        # Test for decimal conversion to hexadecimal
        # Act
        result = decimal_to_any(255, 16)
        # Assert
        assert result == "FF", f"Expected 'FF', got {result}"

    @pytest.mark.valid
    def test_conversion_to_max_base(self):
        # Test for conversion to maximum supported base (base 36)
        # Act
        result = decimal_to_any(1295, 36)
        # Assert (pre-determined correct result for the input 1295)
        expected_result = 'ZZ'  # TODO: Replace with manually calculated expected result of decimal 1295 to base 36
        assert result == expected_result, f"Expected '{expected_result}', got {result}"

    @pytest.mark.invalid
    @pytest.mark.security
    def test_conversion_with_non_standard_base(self):
        # Test for handling non-standard bases
        invalid_bases = [1, 0, 37]  # Bases 1, 0, and greater than 36 are invalid
        for base in invalid_bases:
            # Act and Assert
            with pytest.raises(ValueError):
                decimal_to_any(10, base)
