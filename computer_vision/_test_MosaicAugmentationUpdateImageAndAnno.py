import pytest
import numpy as np
import cv2
import random
import os
from mosaic_augmentation import update_image_and_anno

# Test class for the update_image_and_anno function
class Test_MosaicAugmentationUpdateImageAndAnno:

    @pytest.mark.valid
    def test_update_image_and_anno_basic(self):
        # Arrange
        all_img_list = ['image1.jpg', 'image2.jpg', 'image3.jpg', 'image4.jpg']
        all_annos = [
            [[0, 0, 0, 10, 10], [0, 20, 20, 30, 30]],
            [[0, 0, 0, 10, 10], [0, 20, 20, 30, 30]],
            [[0, 0, 0, 10, 10], [0, 20, 20, 30, 30]],
            [[0, 0, 0, 10, 10], [0, 20, 20, 30, 30]]
        ]
        idxs = [0, 1, 2, 3]
        output_size = (720, 1280)
        scale_range = (0.4, 0.6)
        filter_scale = 0.0

        # Act
        output_img, new_anno, _ = update_image_and_anno(all_img_list, all_annos, idxs, output_size, scale_range, filter_scale)

        # Assert
        assert output_img.shape == (720, 1280, 3)
        assert len(new_anno) == 8  # 4 images * 2 annotations per image

    @pytest.mark.valid
    def test_update_image_and_anno_filter_scale(self):
        # Arrange
        all_img_list = ['image1.jpg', 'image2.jpg']
        all_annos = [
            [[0, 0, 0, 10, 10], [0, 20, 20, 30, 30]],  # small annotations
            [[0, 200, 200, 300, 300], [0, 400, 400, 600, 600]]  # large annotations
        ]
        idxs = [0, 1]
        output_size = (720, 1280)
        scale_range = (0.4, 0.6)
        filter_scale = 0.1

        # Act
        output_img, new_anno, _ = update_image_and_anno(all_img_list, all_annos, idxs, output_size, scale_range, filter_scale)

        # Assert
        assert output_img.shape == (720, 1280, 3)
        assert len(new_anno) == 2  # only large annotations should be retained

    @pytest.mark.valid
    def test_update_image_and_anno_boundary_scale_values(self):
        # Arrange
        all_img_list = ['image1.jpg', 'image2.jpg']
        all_annos = [
            [[0, 0, 0, 10, 10], [0, 20, 20, 30, 30]],
            [[0, 0, 0, 10, 10], [0, 20, 20, 30, 30]]
        ]
        idxs = [0, 1]
        output_size = (720, 1280)
        scale_range = (0.0, 1.0)
        filter_scale = 0.0

        # Act
        output_img, new_anno, _ = update_image_and_anno(all_img_list, all_annos, idxs, output_size, scale_range, filter_scale)

        # Assert
        assert output_img.shape == (720, 1280, 3)
        assert len(new_anno) == 4  # 2 images * 2 annotations per image

    @pytest.mark.valid
    def test_update_image_and_anno_multiple_images(self):
        # Arrange
        all_img_list = ['image1.jpg', 'image2.jpg', 'image3.jpg', 'image4.jpg']
        all_annos = [
            [[0, 0, 0, 10, 10], [0, 20, 20, 30, 30]],
            [[0, 0, 0, 10, 10], [0, 20, 20, 30, 30]],
            [[0, 0, 0, 10, 10], [0, 20, 20, 30, 30]],
            [[0, 0, 0, 10, 10], [0, 20, 20, 30, 30]]
        ]
        idxs = [0, 1, 2, 3]
        output_size = (720, 1280)
        scale_range = (0.4, 0.6)
        filter_scale = 0.0

        # Act
        output_img, new_anno, _ = update_image_and_anno(all_img_list, all_annos, idxs, output_size, scale_range, filter_scale)

        # Assert
        assert output_img.shape == (720, 1280, 3)
        assert len(new_anno) == 8  # 4 images * 2 annotations per image

    @pytest.mark.valid
    def test_update_image_and_anno_empty_annotations(self):
        # Arrange
        all_img_list = ['image1.jpg', 'image2.jpg']
        all_annos = [[], []]  # Empty annotations
        idxs = [0, 1]
        output_size = (720, 1280)
        scale_range = (0.4, 0.6)
        filter_scale = 0.0

        # Act
        output_img, new_anno, _ = update_image_and_anno(all_img_list, all_annos, idxs, output_size, scale_range, filter_scale)

        # Assert
        assert output_img.shape == (720, 1280, 3)
        assert len(new_anno) == 0  # No annotations should be present

    @pytest.mark.valid
    def test_update_image_and_anno_large_output_size(self):
        # Arrange
        all_img_list = ['image1.jpg', 'image2.jpg']
        all_annos = [
            [[0, 0, 0, 10, 10], [0, 20, 20, 30, 30]],
            [[0, 0, 0, 10, 10], [0, 20, 20, 30, 30]]
        ]
        idxs = [0, 1]
        output_size = (4096, 4096)
        scale_range = (0.4, 0.6)
        filter_scale = 0.0

        # Act
        output_img, new_anno, _ = update_image_and_anno(all_img_list, all_annos, idxs, output_size, scale_range, filter_scale)

        # Assert
        assert output_img.shape == (4096, 4096, 3)
        assert len(new_anno) == 4  # 2 images * 2 annotations per image

    @pytest.mark.valid
    def test_update_image_and_anno_zero_filter_scale(self):
        # Arrange
        all_img_list = ['image1.jpg', 'image2.jpg']
        all_annos = [
            [[0, 0, 0, 10, 10], [0, 20, 20, 30, 30]],
            [[0, 0, 0, 10, 10], [0, 20, 20, 30, 30]]
        ]
        idxs = [0, 1]
        output_size = (720, 1280)
        scale_range = (0.4, 0.6)
        filter_scale = 0.0

        # Act
        output_img, new_anno, _ = update_image_and_anno(all_img_list, all_annos, idxs, output_size, scale_range, filter_scale)

        # Assert
        assert output_img.shape == (720, 1280, 3)
        assert len(new_anno) == 4  # 2 images * 2 annotations per image

    @pytest.mark.invalid
    def test_update_image_and_anno_nonexistent_image_paths(self):
        # Arrange
        all_img_list = ['nonexistent1.jpg', 'nonexistent2.jpg']
        all_annos = [
            [[0, 0, 0, 10, 10], [0, 20, 20, 30, 30]],
            [[0, 0, 0, 10, 10], [0, 20, 20, 30, 30]]
        ]
        idxs = [0, 1]
        output_size = (720, 1280)
        scale_range = (0.4, 0.6)
        filter_scale = 0.0

        # Act & Assert
        with pytest.raises(cv2.error):
            update_image_and_anno(all_img_list, all_annos, idxs, output_size, scale_range, filter_scale)

    @pytest.mark.invalid
    def test_update_image_and_anno_invalid_scale_range(self):
        # Arrange
        all_img_list = ['image1.jpg', 'image2.jpg']
        all_annos = [
            [[0, 0, 0, 10, 10], [0, 20, 20, 30, 30]],
            [[0, 0, 0, 10, 10], [0, 20, 20, 30, 30]]
        ]
        idxs = [0, 1]
        output_size = (720, 1280)
        scale_range = (-0.4, 1.6)  # Invalid scale range
        filter_scale = 0.0

        # Act & Assert
        # TODO: Handle invalid scale range in the function or adjust the test
        with pytest.raises(ValueError):
            update_image_and_anno(all_img_list, all_annos, idxs, output_size, scale_range, filter_scale)

    @pytest.mark.invalid
    def test_update_image_and_anno_mixed_valid_invalid_inputs(self):
        # Arrange
        all_img_list = ['image1.jpg', 'nonexistent.jpg', 'image3.jpg', 'nonexistent2.jpg']
        all_annos = [
            [[0, 0, 0, 10, 10], [0, 20, 20, 30, 30]],
            [[0, 0, 0, 10, 10], [0, 20, 20, 30, 30]],
            [[0, 0, 0, 10, 10], [0, 20, 20, 30, 30]],
            [[0, 0, 0, 10, 10], [0, 20, 20, 30, 30]]
        ]
        idxs = [0, 1, 2, 3]
        output_size = (720, 1280)
        scale_range = (0.4, 0.6)
        filter_scale = 0.0

        # Act & Assert
        # TODO: Handle mixed valid and invalid inputs in the function or adjust the test
        with pytest.raises(cv2.error):
            update_image_and_anno(all_img_list, all_annos, idxs, output_size, scale_range, filter_scale)
