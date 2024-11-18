import pytest
from conversions.decimal_to_binary import decimal_to_binary_recursive
import doctest

class Test_DecimalToBinaryRecursive:

    @pytest.mark.positive
    @pytest.mark.regression
    def test_single_digit_decimal_conversion(self):
        # Arrange
        input_decimal = "5"
        
        # Act
        result = decimal_to_binary_recursive(input_decimal)

        # Assert
        assert result == "101"

    @pytest.mark.positive
    @pytest.mark.regression
    def test_multi_digit_decimal_conversion(self):
        # Arrange
        input_decimal = "18"
        
        # Act
        result = decimal_to_binary_recursive(input_decimal)

        # Assert
        assert result == "10010"

    @pytest.mark.positive
    @pytest.mark.regression
    def test_zero_conversion(self):
        # Arrange
        input_decimal = "0"
        
        # Act
        result = decimal_to_binary_recursive(input_decimal)

        # Assert
        assert result == "0"

    @pytest.mark.positive
    @pytest.mark.regression
    def test_one_conversion(self):
        # Arrange
        input_decimal = "1"
        
        # Act
        result = decimal_to_binary_recursive(input_decimal)

        # Assert
        assert result == "1"

    @pytest.mark.positive
    @pytest.mark.performance
    def test_large_decimal_number_conversion(self):
        # Arrange
        input_decimal = "1023"
        
        # Act
        result = decimal_to_binary_recursive(input_decimal)

        # Assert
        assert result == "1111111111"

    @pytest.mark.negative
    @pytest.mark.security
    def test_negative_number_input(self):
        # Arrange
        input_decimal = "-5"
        
        # Act and Assert
        with pytest.raises(ValueError):
            decimal_to_binary_recursive(input_decimal)

    @pytest.mark.negative
    @pytest.mark.security
    def test_non_numeric_input(self):
        # Arrange
        input_decimal = "abc"
        
        # Act and Assert
        with pytest.raises(ValueError):
            decimal_to_binary_recursive(input_decimal)

    @pytest.mark.negative
    @pytest.mark.security
    def test_empty_string_input(self):
        # Arrange
        input_decimal = ""
        
        # Act and Assert
        with pytest.raises(ValueError):
            decimal_to_binary_recursive(input_decimal)

# The following line can be used to run the tests directly if needed
# pytest.main(["-v", "-s"])  # TODO: Uncomment this line to execute tests directly
