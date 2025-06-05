import cv2
import numpy as np
import pytest
from harris_corner import HarrisCorner

class Test_TestHarrisCornerStrTestStrFormatting:

    @pytest.mark.valid
    @pytest.mark.smoke
    def test_valid_initialization(self):
        # Arrange: None
        # Act: Initialize a HarrisCorner object with k = 0.04 and window size = 3.
        harris_corner = HarrisCorner(0.04, 3)

        # Assert: Check that the object's __str__ method returns "0.04".
        assert str(harris_corner) == "0.04", f"Expected '0.04', but got {str(harris_corner)}"

    @pytest.mark.valid
    @pytest.mark.smoke
    def test_str_formatting(self):
        # Arrange
        k_value = 0.04
        harris_corner = HarrisCorner(k_value, 3)

        # Act
        result = str(harris_corner)

        # Assert
        assert result == "0.04", f"Expected '0.04', but got {result}"

    @pytest.mark.invalid
    @pytest.mark.negative
    def test_invalid_k_value(self):
        # Arrange: None
        # Act and Assert: Initialize a HarrisCorner object with an invalid k value.
        with pytest.raises(ValueError, match="invalid k value"):
            harris_corner = HarrisCorner(0.1, 3)

    @pytest.mark.valid
    @pytest.mark.smoke
    def test_valid_k_value_upper_bound(self):
        # Arrange: None
        # Act: Initialize a HarrisCorner object with k = 0.06 and window size = 3.
        harris_corner = HarrisCorner(0.06, 3)

        # Assert: Check that the object's __str__ method returns "0.06".
        assert str(harris_corner) == "0.06", f"Expected '0.06', but got {str(harris_corner)}"

    @pytest.mark.valid
    @pytest.mark.smoke
    def test_detect_corners(self):
        # Arrange: None
        # Act: Initialize a HarrisCorner object and detect corners in a test image.
        k_value = 0.04
        window_size = 3
        harris_corner = HarrisCorner(k_value, window_size)

        # TODO: Change the image path to a valid image path for testing.
        img_path = 'path/to/test/image.jpg'
        color_img, corner_list = harris_corner.detect(img_path)

        # Assert: Check that the output image and corner list are not empty.
        assert color_img is not None, "Expected a valid image, but got None"
        assert isinstance(corner_list, list), "Expected a list of corners, but got something else"
        assert len(corner_list) > 0, "Expected to find corners, but found none"
