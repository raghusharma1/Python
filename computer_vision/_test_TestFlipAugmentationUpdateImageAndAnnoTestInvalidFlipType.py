import pytest
import cv2
import numpy as np
import os
from flip_augmentation import update_image_and_anno

class Test_TestFlipAugmentationUpdateImageAndAnnoTestInvalidFlipType:

    @pytest.mark.invalid
    @pytest.mark.smoke
    def test_invalid_flip_type_value(self):
        # Arrange
        img_path = 'test_image.jpg'
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(img_path, img)
        annotation = [[0, 0.5, 0.5, 0.2, 0.2]]  # [class, x_center, y_center, width, height]

        img_list = [img_path]
        anno_list = [annotation]
        flip_type = 2  # Invalid flip type

        # Act & Assert
        try:
            update_image_and_anno(img_list, anno_list, flip_type)
        except ValueError as e:
            assert str(e) == "Invalid flip_type"

        # Clean up
        os.remove(img_path)
