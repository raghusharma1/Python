import pytest
import cv2
import numpy as np
import os
from flip_augmentation import update_image_and_anno

class TestFlipAugmentationUpdateImageAndAnno:

    @pytest.mark.valid
    @pytest.mark.positive
    def test_vertical_flip_single_image(self):
        # Arrange
        img_path = 'test_image.jpg'
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(img_path, img)
        annotation = [[0, 0.5, 0.5, 0.2, 0.2]]  # [class, x_center, y_center, width, height]

        img_list = [img_path]
        anno_list = [annotation]
        flip_type = 0

        # Act
        new_imgs_list, new_annos_lists, path_list = update_image_and_anno(img_list, anno_list, flip_type)

        # Assert
        assert len(new_imgs_list) == 1
        assert len(new_annos_lists) == 1
        assert path_list == img_list

        flipped_img = new_imgs_list[0]
        flipped_annotation = new_annos_lists[0][0]

        # Check if the image is flipped vertically
        expected_flipped_img = cv2.flip(img, flip_type)
        assert np.array_equal(flipped_img, expected_flipped_img)

        # Check if the annotation is updated correctly
        expected_flipped_annotation = [0, 0.5, 1 - annotation[0][2], 0.2, 0.2]
        assert flipped_annotation == pytest.approx(expected_flipped_annotation)

        # Clean up
        os.remove(img_path)
