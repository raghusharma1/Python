import pytest
import glob
import os
import random
from string import ascii_lowercase, digits
import numpy as np
from mosaic_augmentation import update_image_and_anno

class Test_MosaicAugmentationUpdateImageAndAnno:

    @pytest.mark.valid
    def test_update_image_and_anno_basic_functionality(self):
        # Arrange
        all_img_list = [f"test_image_{i}.jpg" for i in range(4)]
        all_annos = [
            [[0, 10, 10, 50, 50]],
            [[0, 20, 20, 60, 60]],
            [[0, 30, 30, 70, 70]],
            [[0, 40, 40, 80, 80]]
        ]
        idxs = [0, 1, 2, 3]
        output_size = (720, 1280)
        scale_range = (0.4, 0.6)
        filter_scale = 0.0

        # Act
        output_img, new_anno, img_path = update_image_and_anno(
            all_img_list, all_annos, idxs, output_size, scale_range, filter_scale
        )

        # Assert
        assert output_img.shape == (output_size[0], output_size[1], 3)
        assert len(new_anno) == 4
        for anno in new_anno:
            assert len(anno) == 5
        assert img_path == all_img_list[0]

    @pytest.mark.valid
    def test_update_image_and_anno_with_filter_scale(self):
        # Arrange
        all_img_list = [f"test_image_{i}.jpg" for i in range(4)]
        all_annos = [
            [[0, 10, 10, 50, 50]],
            [[0, 20, 20, 60, 60]],
            [[0, 30, 30, 50, 50]],  # This annotation should be filtered out
            [[0, 40, 40, 80, 80]]
        ]
        idxs = [0, 1, 2, 3]
        output_size = (720, 1280)
        scale_range = (0.4, 0.6)
        filter_scale = 0.5

        # Act
        output_img, new_anno, img_path = update_image_and_anno(
            all_img_list, all_annos, idxs, output_size, scale_range, filter_scale
        )

        # Assert
        assert output_img.shape == (output_size[0], output_size[1], 3)
        assert len(new_anno) == 3
        for anno in new_anno:
            assert len(anno) == 5
            assert (anno[3] - anno[1]) > filter_scale
            assert (anno[4] - anno[2]) > filter_scale
        assert img_path == all_img_list[0]

    @pytest.mark.valid
    def test_update_image_and_anno_single_image(self):
        # Arrange
        all_img_list = ["test_image_0.jpg"]
        all_annos = [[[0, 10, 10, 50, 50]]]
        idxs = [0]
        output_size = (720, 1280)
        scale_range = (0.4, 0.6)
        filter_scale = 0.0

        # Act
        output_img, new_anno, img_path = update_image_and_anno(
            all_img_list, all_annos, idxs, output_size, scale_range, filter_scale
        )

        # Assert
        assert output_img.shape == (output_size[0], output_size[1], 3)
        assert len(new_anno) == 1
        assert new_anno[0] == [0, new_anno[0][1], new_anno[0][2], new_anno[0][3], new_anno[0][4]]
        assert img_path == all_img_list[0]

    @pytest.mark.valid
    def test_update_image_and_anno_large_scale_range(self):
        # Arrange
        all_img_list = [f"test_image_{i}.jpg" for i in range(4)]
        all_annos = [
            [[0, 10, 10, 50, 50]],
            [[0, 20, 20, 60, 60]],
            [[0, 30, 30, 70, 70]],
            [[0, 40, 40, 80, 80]]
        ]
        idxs = [0, 1, 2, 3]
        output_size = (720, 1280)
        scale_range = (0.8, 1.2)
        filter_scale = 0.0

        # Act
        output_img, new_anno, img_path = update_image_and_anno(
            all_img_list, all_annos, idxs, output_size, scale_range, filter_scale
        )

        # Assert
        assert output_img.shape == (output_size[0], output_size[1], 3)
        assert len(new_anno) == 4
        for anno in new_anno:
            assert len(anno) == 5
        assert img_path == all_img_list[0]

    @pytest.mark.valid
    def test_update_image_and_anno_empty_annotations(self):
        # Arrange
        all_img_list = [f"test_image_{i}.jpg" for i in range(4)]
        all_annos = [[], [], [], []]
        idxs = [0, 1, 2, 3]
        output_size = (720, 1280)
        scale_range = (0.4, 0.6)
        filter_scale = 0.0

        # Act
        output_img, new_anno, img_path = update_image_and_anno(
            all_img_list, all_annos, idxs, output_size, scale_range, filter_scale
        )

        # Assert
        assert output_img.shape == (output_size[0], output_size[1], 3)
        assert len(new_anno) == 0
        assert img_path == all_img_list[0]

    @pytest.mark.invalid
    def test_update_image_and_anno_non_existent_image_paths(self):
        # Arrange
        all_img_list = [f"test_image_{i}.jpg" for i in range(4)]
        all_annos = [
            [[0, 10, 10, 50, 50]],
            [[0, 20, 20, 60, 60]],
            [[0, 30, 30, 70, 70]],
            [[0, 40, 40, 80, 80]]
        ]
        idxs = [0, 1, 2, 3]
        output_size = (720, 1280)
        scale_range = (0.4, 0.6)
        filter_scale = 0.0

        # Act & Assert
        with pytest.raises(FileNotFoundError):
            update_image_and_anno(
                all_img_list, all_annos, idxs, output_size, scale_range, filter_scale
            )
