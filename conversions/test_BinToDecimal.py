import pytest
from doctest import testmod
from conversions.binary_to_decimal import bin_to_decimal

# Test class for bin_to_decimal function
class Test_BinToDecimal:
    
    @pytest.mark.valid
    def test_simple_binary_conversion(self):
        # Arrange
        binary_string = "101"
        
        # Act
        result = bin_to_decimal(binary_string)
        
        # Assert
        assert result == 5
    
    @pytest.mark.valid
    def test_binary_to_zero(self):
        # Arrange
        binary_string = "0"
        
        # Act
        result = bin_to_decimal(binary_string)
        
        # Assert
        assert result == 0
    
    @pytest.mark.valid
    def test_large_binary_conversion(self):
        # Arrange
        binary_string = "11011011101"
        
        # Act
        result = bin_to_decimal(binary_string)
        
        # Assert
        assert result == 1757
    
    @pytest.mark.invalid
    def test_invalid_characters_in_binary(self):
        # Arrange
        invalid_binary_string = "102"
        
        # Act & Assert
        with pytest.raises(ValueError):
            bin_to_decimal(invalid_binary_string)
    
    @pytest.mark.valid
    def test_leading_zeros(self):
        # Arrange
        binary_string = "00101"
        
        # Act
        result = bin_to_decimal(binary_string)
        
        # Assert
        assert result == 5
    
    @pytest.mark.invalid
    def test_empty_string(self):
        # Arrange
        empty_string = ""
        
        # Act & Assert
        with pytest.raises(ValueError):  # Assuming ValueError should be raised
            bin_to_decimal(empty_string)

# You can add a method to run tests if this script is executed directly.
if __name__ == '__main__':
    pytest.main()
