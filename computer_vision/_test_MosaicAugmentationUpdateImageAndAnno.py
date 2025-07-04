import pytest
import numpy as np
import cv2
import random
import os
from mosaic_augmentation import update_image_and_anno

class Test_MosaicAugmentationUpdateImageAndAnno:

    @pytest.mark.smoke
    @pytest.mark.valid
    def test_basic_functionality(self):
        # Arrange
        all_img_list = ["test_images/image1.jpg", "test_images/image2.jpg"]
        all_annos = [[[1, 10, 10, 50, 50]], [[2, 20, 20, 60, 60]]]
        idxs = [0, 1]
        output_size = (720, 1280)
        scale_range = (0.4, 0.6)
        filter_scale = 0.0

        # Act
        output_img, new_anno, path = update_image_and_anno(all_img_list, all_annos, idxs, output_size, scale_range, filter_scale)

        # Assert
        assert output_img.shape == (720, 1280, 3)
        assert len(new_anno) == 2
        assert path in all_img_list

    @pytest.mark.regression
    @pytest.mark.valid
    def test_multiple_images(self):
        # Arrange
        all_img_list = ["test_images/image1.jpg", "test_images/image2.jpg", "test_images/image3.jpg", "test_images/image4.jpg"]
        all_annos = [[[1, 10, 10, 50, 50]], [[2, 20, 20, 60, 60]], [[3, 30, 30, 70, 70]], [[4, 40, 40, 80, 80]]]
        idxs = [0, 1, 2, 3]
        output_size = (720, 1280)
        scale_range = (0.4, 0.6)
        filter_scale = 0.0

        # Act
        output_img, new_anno, path = update_image_and_anno(all_img_list, all_annos, idxs, output_size, scale_range, filter_scale)

        # Assert
        assert output_img.shape == (720, 1280, 3)
        assert len(new_anno) == 4
        assert path in all_img_list

    @pytest.mark.regression
    @pytest.mark.valid
    def test_filter_scale(self):
        # Arrange
        all_img_list = ["test_images/image1.jpg", "test_images/image2.jpg"]
        all_annos = [[[1, 10, 10, 50, 50]], [[2, 20, 20, 60, 60]]]
        idxs = [0, 1]
        output_size = (720, 1280)
        scale_range = (0.4, 0.6)
        filter_scale = 0.2

        # Act
        output_img, new_anno, path = update_image_and_anno(all_img_list, all_annos, idxs, output_size, scale_range, filter_scale)

        # Assert
        assert output_img.shape == (720, 1280, 3)
        assert len(new_anno) == 2
        for anno in new_anno:
            assert (anno[3] - anno[1]) >= filter_scale
            assert (anno[4] - anno[2]) >= filter_scale

    @pytest.mark.negative
    @pytest.mark.valid
    def test_single_image(self):
        # Arrange
        all_img_list = ["test_images/image1.jpg"]
        all_annos = [[[1, 10, 10, 50, 50]]]
        idxs = [0]
        output_size = (720, 1280)
        scale_range = (0.4, 0.6)
        filter_scale = 0.0

        # Act
        output_img, new_anno, path = update_image_and_anno(all_img_list, all_annos, idxs, output_size, scale_range, filter_scale)

        # Assert
        assert output_img.shape == (720, 1280, 3)
        assert len(new_anno) == 1
        assert path == all_img_list[0]

    @pytest.mark.negative
    @pytest.mark.valid
    def test_empty_annotations(self):
        # Arrange
        all_img_list = ["test_images/image1.jpg", "test_images/image2.jpg"]
        all_annos = [[], []]
        idxs = [0, 1]
        output_size = (720, 1280)
        scale_range = (0.4, 0.6)
        filter_scale = 0.0

        # Act
        output_img, new_anno, path = update_image_and_anno(all_img_list, all_annos, idxs, output_size, scale_range, filter_scale)

        # Assert
        assert output_img.shape == (720, 1280, 3)
        assert len(new_anno) == 0
        assert path in all_img_list

    @pytest.mark.regression
    @pytest.mark.valid
    def test_scale_range_boundary(self):
        # Arrange
        all_img_list = ["test_images/image1.jpg", "test_images/image2.jpg"]
        all_annos = [[[1, 10, 10, 50, 50]], [[2, 20, 20, 60, 60]]]
        idxs = [0, 1]
        output_size = (720, 1280)
        scale_range = (0.0, 1.0)
        filter_scale = 0.0

        # Act
        output_img, new_anno, path = update_image_and_anno(all_img_list, all_annos, idxs, output_size, scale_range, filter_scale)

        # Assert
        assert output_img.shape == (720, 1280, 3)
        assert len(new_anno) == 2
        assert path in all_img_list

    @pytest.mark.performance
    @pytest.mark.valid
    def test_large_output_size(self):
        # Arrange
        all_img_list = ["test_images/image1.jpg", "test_images/image2.jpg"]
        all_annos = [[[1, 10, 10, 50, 50]], [[2, 20, 20, 60, 60]]]
        idxs = [0, 1]
        output_size = (2000, 3000)
        scale_range = (0.4, 0.6)
        filter_scale = 0.0

        # Act
        output_img, new_anno, path = update_image_and_anno(all_img_list, all_annos, idxs, output_size, scale_range, filter_scale)

        # Assert
        assert output_img.shape == (2000, 3000, 3)
        assert len(new_anno) == 2
        assert path in all_img_list

    @pytest.mark.negative
    @pytest.mark.invalid
    def test_invalid_image_path(self):
        # Arrange
        all_img_list = ["test_images/image1.jpg", "invalid_path.jpg"]
        all_annos = [[[1, 10, 10, 50, 50]], [[2, 20, 20, 60, 60]]]
        idxs = [0, 1]
        output_size = (720, 1280)
        scale_range = (0.4, 0.6)
        filter_scale = 0.0

        # Act & Assert
        with pytest.raises(Exception):
            update_image_and_anno(all_img_list, all_annos, idxs, output_size, scale_range, filter_scale)

    @pytest.mark.negative
    @pytest.mark.valid
    def test_non_existent_annotations(self):
        # Arrange
        all_img_list = ["test_images/image1.jpg", "test_images/image2.jpg"]
        all_annos = [[[1, 10, 10, 50, 50]], []]
        idxs = [0, 1]
        output_size = (720, 1280)
        scale_range = (0.4, 0.6)
        filter_scale = 0.0

        # Act
        output_img, new_anno, path = update_image_and_anno(all_img_list, all_annos, idxs, output_size, scale_range, filter_scale)

        # Assert
        assert output_img.shape == (720, 1280, 3)
        assert len(new_anno) == 1
        assert path in all_img_list

    @pytest.mark.regression
    @pytest.mark.valid
    def test_random_scale_range(self):
        # Arrange
        all_img_list = ["test_images/image1.jpg", "test_images/image2.jpg"]
        all_annos = [[[1, 10, 10, 50, 50]], [[2, 20, 20, 60, 60]]]
        idxs = [0, 1]
        output_size = (720, 1280)
        scale_range = (random.uniform(0.1, 0.9), random.uniform(0.1, 0.9))
        filter_scale = 0.0

        # Act
        output_img, new_anno, path = update_image_and_anno(all_img_list, all_annos, idxs, output_size, scale_range, filter_scale)

        # Assert
        assert output_img.shape == (720, 1280, 3)
        assert len(new_anno) == 2
        assert path in all_img_list
