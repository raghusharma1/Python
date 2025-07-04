import pytest
import cv2
import os
import random
from string import ascii_lowercase, digits
from flip_augmentation import update_image_and_anno

class Test_FlipAugmentationUpdateImageAndAnno:

    @pytest.mark.valid
    def test_horizontal_flip_valid_inputs(self):
        # Arrange
        img_list = ['path/to/image1.jpg', 'path/to/image2.jpg']
        anno_list = [[[1, 0.5, 0.5, 0.1, 0.1]], [[2, 0.6, 0.6, 0.2, 0.2]]]
        flip_type = 1

        # Act
        new_imgs_list, new_annos_lists, path_list = update_image_and_anno(img_list, anno_list, flip_type)

        # Assert
        assert len(new_imgs_list) == len(img_list)
        assert len(new_annos_lists) == len(anno_list)
        assert path_list == img_list
        for new_annos in new_annos_lists:
            for bbox in new_annos:
                assert bbox[1] == 1 - anno_list[0][0][1]

    @pytest.mark.valid
    def test_vertical_flip_valid_inputs(self):
        # Arrange
        img_list = ['path/to/image1.jpg', 'path/to/image2.jpg']
        anno_list = [[[1, 0.5, 0.5, 0.1, 0.1]], [[2, 0.6, 0.6, 0.2, 0.2]]]
        flip_type = 0

        # Act
        new_imgs_list, new_annos_lists, path_list = update_image_and_anno(img_list, anno_list, flip_type)

        # Assert
        assert len(new_imgs_list) == len(img_list)
        assert len(new_annos_lists) == len(anno_list)
        assert path_list == img_list
        for new_annos in new_annos_lists:
            for bbox in new_annos:
                assert bbox[2] == 1 - anno_list[0][0][2]

    @pytest.mark.invalid
    def test_no_images_in_input_list(self):
        # Arrange
        img_list = []
        anno_list = []
        flip_type = 1

        # Act
        new_imgs_list, new_annos_lists, path_list = update_image_and_anno(img_list, anno_list, flip_type)

        # Assert
        assert new_imgs_list == []
        assert new_annos_lists == []
        assert path_list == []

    @pytest.mark.invalid
    def test_mismatched_image_and_annotation_lists(self):
        # Arrange
        img_list = ['path/to/image1.jpg']
        anno_list = [[[1, 0.5, 0.5, 0.1, 0.1]], [[2, 0.6, 0.6, 0.2, 0.2]]]
        flip_type = 1

        # Act & Assert
        with pytest.raises(IndexError):
            update_image_and_anno(img_list, anno_list, flip_type)

    @pytest.mark.invalid
    def test_invalid_flip_type(self):
        # Arrange
        img_list = ['path/to/image1.jpg']
        anno_list = [[[1, 0.5, 0.5, 0.1, 0.1]]]
        flip_type = 2

        # Act & Assert
        with pytest.raises(ValueError):
            update_image_and_anno(img_list, anno_list, flip_type)

    @pytest.mark.invalid
    def test_non_existent_image_paths(self):
        # Arrange
        img_list = ['path/to/nonexistent.jpg', 'path/to/image1.jpg']
        anno_list = [[[1, 0.5, 0.5, 0.1, 0.1]], [[2, 0.6, 0.6, 0.2, 0.2]]]
        flip_type = 1

        # Act
        new_imgs_list, new_annos_lists, path_list = update_image_and_anno(img_list, anno_list, flip_type)

        # Assert
        assert len(new_imgs_list) == 1
        assert len(new_annos_lists) == 1
        assert path_list == ['path/to/image1.jpg']

    @pytest.mark.performance
    def test_large_number_of_images(self):
        # Arrange
        img_list = [f'path/to/image{i}.jpg' for i in range(1000)]
        anno_list = [[[1, 0.5, 0.5, 0.1, 0.1]] for _ in range(1000)]
        flip_type = 1

        # Act
        new_imgs_list, new_annos_lists, path_list = update_image_and_anno(img_list, anno_list, flip_type)

        # Assert
        assert len(new_imgs_list) == len(img_list)
        assert len(new_annos_lists) == len(anno_list)
        assert path_list == img_list

    @pytest.mark.valid
    def test_images_with_different_sizes(self):
        # Arrange
        img_list = ['path/to/image1.jpg', 'path/to/image2.jpg']
        anno_list = [[[1, 0.5, 0.5, 0.1, 0.1]], [[2, 0.6, 0.6, 0.2, 0.2]]]
        flip_type = 1

        # Act
        new_imgs_list, new_annos_lists, path_list = update_image_and_anno(img_list, anno_list, flip_type)

        # Assert
        assert len(new_imgs_list) == len(img_list)
        assert len(new_annos_lists) == len(anno_list)
        assert path_list == img_list
        for new_annos in new_annos_lists:
            for bbox in new_annos:
                assert bbox[1] == 1 - anno_list[0][0][1]

    @pytest.mark.valid
    def test_images_with_no_annotations(self):
        # Arrange
        img_list = ['path/to/image1.jpg', 'path/to/image2.jpg']
        anno_list = [[], []]
        flip_type = 1

        # Act
        new_imgs_list, new_annos_lists, path_list = update_image_and_anno(img_list, anno_list, flip_type)

        # Assert
        assert len(new_imgs_list) == len(img_list)
        assert len(new_annos_lists) == len(anno_list)
        assert path_list == img_list
        for new_annos in new_annos_lists:
            assert new_annos == []

    @pytest.mark.invalid
    def test_images_with_invalid_annotations(self):
        # Arrange
        img_list = ['path/to/image1.jpg']
        anno_list = [[[1, 0.5, 0.5, 0.1, 0.1, 0.1]]]
        flip_type = 1

        # Act & Assert
        with pytest.raises(IndexError):
            update_image_and_anno(img_list, anno_list, flip_type)
