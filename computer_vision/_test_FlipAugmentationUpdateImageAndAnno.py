import pytest
import cv2
import numpy as np
import os
from flip_augmentation import update_image_and_anno

# Added os import to fix the cleanup error

class Test_FlipAugmentationUpdateImageAndAnno:
    @pytest.mark.valid
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

    @pytest.mark.valid
    def test_horizontal_flip_single_image(self):
        # Arrange
        img_path = 'test_image.jpg'
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(img_path, img)
        annotation = [[0, 0.5, 0.5, 0.2, 0.2]]  # [class, x_center, y_center, width, height]

        img_list = [img_path]
        anno_list = [annotation]
        flip_type = 1

        # Act
        new_imgs_list, new_annos_lists, path_list = update_image_and_anno(img_list, anno_list, flip_type)

        # Assert
        assert len(new_imgs_list) == 1
        assert len(new_annos_lists) == 1
        assert path_list == img_list

        flipped_img = new_imgs_list[0]
        flipped_annotation = new_annos_lists[0][0]

        # Check if the image is flipped horizontally
        expected_flipped_img = cv2.flip(img, flip_type)
        assert np.array_equal(flipped_img, expected_flipped_img)

        # Check if the annotation is updated correctly
        expected_flipped_annotation = [0, 1 - annotation[0][1], 0.5, 0.2, 0.2]
        assert flipped_annotation == pytest.approx(expected_flipped_annotation)

        # Clean up
        os.remove(img_path)

    @pytest.mark.invalid
    def test_invalid_flip_type(self):
        # Arrange
        img_path = 'test_image.jpg'
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(img_path, img)
        annotation = [[0, 0.5, 0.5, 0.2, 0.2]]  # [class, x_center, y_center, width, height]

        img_list = [img_path]
        anno_list = [annotation]
        flip_type = 2  # Invalid flip type

        # Act & Assert
        with pytest.raises(ValueError, match="Invalid flip_type"):
            update_image_and_anno(img_list, anno_list, flip_type)

        # Clean up
        os.remove(img_path)

    @pytest.mark.valid
    def test_multiple_images_and_annotations(self):
        # Arrange
        img_paths = ['test_image1.jpg', 'test_image2.jpg']
        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        img2 = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(img_paths[0], img1)
        cv2.imwrite(img_paths[1], img2)
        annotations = [
            [[0, 0.5, 0.5, 0.2, 0.2]],  # [class, x_center, y_center, width, height]
            [[1, 0.2, 0.3, 0.1, 0.1]]
        ]

        img_list = img_paths
        anno_list = annotations
        flip_type = 0

        # Act
        new_imgs_list, new_annos_lists, path_list = update_image_and_anno(img_list, anno_list, flip_type)

        # Assert
        assert len(new_imgs_list) == 2
        assert len(new_annos_lists) == 2
        assert path_list == img_list

        for i in range(2):
            flipped_img = new_imgs_list[i]
            flipped_annotation = new_annos_lists[i][0]

            # Check if the image is flipped vertically
            expected_flipped_img = cv2.flip(cv2.imread(img_paths[i]), flip_type)
            assert np.array_equal(flipped_img, expected_flipped_img)

            # Check if the annotation is updated correctly
            expected_flipped_annotation = [annotations[i][0][0], annotations[i][0][1], 1 - annotations[i][0][2], annotations[i][0][3], annotations[i][0][4]]
            assert flipped_annotation == pytest.approx(expected_flipped_annotation)

        # Clean up
        for img_path in img_paths:
            os.remove(img_path)
