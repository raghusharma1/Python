import pytest
import cv2
import os
import glob
from flip_augmentation import update_image_and_anno

IMAGE_DIR = '/path/to/image/directory'  # Update with the actual path

class Test_FlipAugmentationUpdateImageAndAnno:

    @pytest.mark.smoke
    @pytest.mark.positive
    def test_horizontal_flip(self):
        # Arrange
        img_list = glob.glob(os.path.join(IMAGE_DIR, '*.jpg'))
        if not img_list:
            pytest.skip("No images found in the directory")
        anno_list = [[[1, 0.5, 0.5, 100, 100]] for _ in img_list]  # Adjust annotations as per actual data

        # Act
        new_imgs_list, new_annos_lists, path_list = update_image_and_anno(img_list, anno_list, flip_type=1)

        # Assert
        for idx, img in enumerate(new_imgs_list):
            assert img.shape == cv2.imread(img_list[idx]).shape  # Check image dimensions
            assert new_annos_lists[idx][0][1] == 1 - anno_list[idx][0][1]  # Check horizontal flip annotation update

    @pytest.mark.smoke
    @pytest.mark.positive
    def test_vertical_flip(self):
        # Arrange
        img_list = glob.glob(os.path.join(IMAGE_DIR, '*.jpg'))
        if not img_list:
            pytest.skip("No images found in the directory")
        anno_list = [[[1, 0.5, 0.5, 100, 100]] for _ in img_list]  # Adjust annotations as per actual data

        # Act
        new_imgs_list, new_annos_lists, path_list = update_image_and_anno(img_list, anno_list, flip_type=0)

        # Assert
        for idx, img in enumerate(new_imgs_list):
            assert img.shape == cv2.imread(img_list[idx]).shape  # Check image dimensions
            assert new_annos_lists[idx][0][2] == 1 - anno_list[idx][0][2]  # Check vertical flip annotation update

    @pytest.mark.regression
    @pytest.mark.positive
    def test_multiple_images(self):
        # Arrange
        img_list = glob.glob(os.path.join(IMAGE_DIR, '*.jpg'))
        if not img_list:
            pytest.skip("No images found in the directory")
        anno_list = [[[1, 0.5, 0.5, 100, 100]] for _ in img_list]  # Adjust annotations as per actual data

        # Act
        new_imgs_list, new_annos_lists, path_list = update_image_and_anno(img_list, anno_list, flip_type=1)

        # Assert
        assert len(new_imgs_list) == len(img_list)
        assert len(new_annos_lists) == len(anno_list)
        assert len(path_list) == len(img_list)

    @pytest.mark.regression
    @pytest.mark.negative
    def test_empty_input_lists(self):
        # Arrange
        img_list = []
        anno_list = []

        # Act
        new_imgs_list, new_annos_lists, path_list = update_image_and_anno(img_list, anno_list, flip_type=1)

        # Assert
        assert new_imgs_list == []
        assert new_annos_lists == []
        assert path_list == []

    @pytest.mark.regression
    @pytest.mark.positive
    def test_annotation_updates(self):
        # Arrange
        img_list = glob.glob(os.path.join(IMAGE_DIR, '*.jpg'))
        if not img_list:
            pytest.skip("No images found in the directory")
        anno_list = [[[1, 0.5, 0.5, 100, 100]] for _ in img_list]  # Adjust annotations as per actual data

        # Act
        new_imgs_list, new_annos_lists, path_list = update_image_and_anno(img_list, anno_list, flip_type=1)

        # Assert
        for idx, new_annos in enumerate(new_annos_lists):
            assert new_annos[0][1] == 1 - anno_list[idx][0][1]  # Check horizontal flip annotation update

        new_imgs_list, new_annos_lists, path_list = update_image_and_anno(img_list, anno_list, flip_type=0)

        # Assert
        for idx, new_annos in enumerate(new_annos_lists):
            assert new_annos[0][2] == 1 - anno_list[idx][0][2]  # Check vertical flip annotation update

    @pytest.mark.regression
    @pytest.mark.negative
    def test_nonexistent_image_paths(self):
        # Arrange
        img_list = ['nonexistent.jpg'] + glob.glob(os.path.join(IMAGE_DIR, '*.jpg'))
        if not img_list:
            pytest.skip("No images found in the directory")
        anno_list = [[[1, 0.5, 0.5, 100, 100]] for _ in img_list]  # Adjust annotations as per actual data

        # Act
        new_imgs_list, new_annos_lists, path_list = update_image_and_anno(img_list, anno_list, flip_type=1)

        # Assert
        assert len(new_imgs_list) == len(img_list) - 1  # One image should be skipped
        assert len(new_annos_lists) == len(img_list) - 1
        assert len(path_list) == len(img_list) - 1

    @pytest.mark.regression
    @pytest.mark.negative
    def test_invalid_flip_type(self):
        # Arrange
        img_list = glob.glob(os.path.join(IMAGE_DIR, '*.jpg'))
        if not img_list:
            pytest.skip("No images found in the directory")
        anno_list = [[[1, 0.5, 0.5, 100, 100]] for _ in img_list]  # Adjust annotations as per actual data

        # Act & Assert
        with pytest.raises(ValueError):  # Update to the actual exception raised by the function
            update_image_and_anno(img_list, anno_list, flip_type=2)

    @pytest.mark.performance
    @pytest.mark.positive
    def test_large_input(self, benchmark):
        # Arrange
        img_list = glob.glob(os.path.join(IMAGE_DIR, '*.jpg')) * 100  # Adjust multiplier for performance testing
        if not img_list:
            pytest.skip("No images found in the directory")
        anno_list = [[[1, 0.5, 0.5, 100, 100]] for _ in img_list]  # Adjust annotations as per actual data

        # Act
        new_imgs_list, new_annos_lists, path_list = benchmark(update_image_and_anno, img_list, anno_list, flip_type=1)

        # Assert
        assert len(new_imgs_list) == len(img_list)
        assert len(new_annos_lists) == len(anno_list)
        assert len(path_list) == len(img_list)
