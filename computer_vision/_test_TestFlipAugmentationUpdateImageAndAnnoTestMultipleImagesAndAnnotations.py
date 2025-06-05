import pytest
import cv2
import numpy as np
import os
from flip_augmentation import update_image_and_anno

class Test_TestFlipAugmentationUpdateImageAndAnnoTestMultipleImagesAndAnnotations:
    @pytest.mark.valid
    def test_basic_vertical_flip(self):
        # Arrange
        img_paths = ['test_image1.jpg', 'test_image2.jpg']
        img1 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
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

    @pytest.mark.valid
    def test_basic_horizontal_flip(self):
        # Arrange
        img_paths = ['test_image1.jpg', 'test_image2.jpg']
        img1 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(img_paths[0], img1)
        cv2.imwrite(img_paths[1], img2)
        annotations = [
            [[0, 0.5, 0.5, 0.2, 0.2]],  # [class, x_center, y_center, width, height]
            [[1, 0.2, 0.3, 0.1, 0.1]]
        ]

        img_list = img_paths
        anno_list = annotations
        flip_type = 1

        # Act
        new_imgs_list, new_annos_lists, path_list = update_image_and_anno(img_list, anno_list, flip_type)

        # Assert
        assert len(new_imgs_list) == 2
        assert len(new_annos_lists) == 2
        assert path_list == img_list

        for i in range(2):
            flipped_img = new_imgs_list[i]
            flipped_annotation = new_annos_lists[i][0]

            # Check if the image is flipped horizontally
            expected_flipped_img = cv2.flip(cv2.imread(img_paths[i]), flip_type)
            assert np.array_equal(flipped_img, expected_flipped_img)

            # Check if the annotation is updated correctly
            expected_flipped_annotation = [annotations[i][0][0], 1 - annotations[i][0][1], annotations[i][0][2], annotations[i][0][3], annotations[i][0][4]]
            assert flipped_annotation == pytest.approx(expected_flipped_annotation)

        # Clean up
        for img_path in img_paths:
            os.remove(img_path)

    @pytest.mark.invalid
    def test_no_images_or_annotations(self):
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

    @pytest.mark.valid
    def test_single_image_and_annotation(self):
        # Arrange
        img_path = 'test_image.jpg'
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(img_path, img)
        annotations = [[[0, 0.5, 0.5, 0.2, 0.2]]]  # [class, x_center, y_center, width, height]

        img_list = [img_path]
        anno_list = annotations
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
        expected_flipped_img = cv2.flip(cv2.imread(img_path), flip_type)
        assert np.array_equal(flipped_img, expected_flipped_img)

        # Check if the annotation is updated correctly
        expected_flipped_annotation = [annotations[0][0][0], annotations[0][0][1], 1 - annotations[0][0][2], annotations[0][0][3], annotations[0][0][4]]
        assert flipped_annotation == pytest.approx(expected_flipped_annotation)

        # Clean up
        os.remove(img_path)

    @pytest.mark.valid
    def test_multiple_images_with_different_sizes(self):
        # Arrange
        img_paths = ['test_image1.jpg', 'test_image2.jpg']
        img1 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        cv2.imwrite(img_paths[0], img1)
        cv2.imwrite(img_paths[1], img2)
        annotations = [
            [[0, 0.5, 0.5, 0.2, 0.2]],  # [class, x_center, y_center, width, height]
            [[1, 0.2, 0.3, 0.1, 0.1]]
        ]

        img_list = img_paths
        anno_list = annotations
        flip_type = 1

        # Act
        new_imgs_list, new_annos_lists, path_list = update_image_and_anno(img_list, anno_list, flip_type)

        # Assert
        assert len(new_imgs_list) == 2
        assert len(new_annos_lists) == 2
        assert path_list == img_list

        for i in range(2):
            flipped_img = new_imgs_list[i]
            flipped_annotation = new_annos_lists[i][0]

            # Check if the image is flipped horizontally
            expected_flipped_img = cv2.flip(cv2.imread(img_paths[i]), flip_type)
            assert np.array_equal(flipped_img, expected_flipped_img)

            # Check if the annotation is updated correctly
            expected_flipped_annotation = [annotations[i][0][0], 1 - annotations[i][0][1], annotations[i][0][2], annotations[i][0][3], annotations[i][0][4]]
            assert flipped_annotation == pytest.approx(expected_flipped_annotation)

        # Clean up
        for img_path in img_paths:
            os.remove(img_path)

    @pytest.mark.invalid
    def test_invalid_flip_type(self):
        # Arrange
        img_paths = ['test_image1.jpg', 'test_image2.jpg']
        img1 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(img_paths[0], img1)
        cv2.imwrite(img_paths[1], img2)
        annotations = [
            [[0, 0.5, 0.5, 0.2, 0.2]],  # [class, x_center, y_center, width, height]
            [[1, 0.2, 0.3, 0.1, 0.1]]
        ]

        img_list = img_paths
        anno_list = annotations
        flip_type = -1

        # Act & Assert
        with pytest.raises(ValueError):
            update_image_and_anno(img_list, anno_list, flip_type)

        # Clean up
        for img_path in img_paths:
            os.remove(img_path)

    @pytest.mark.performance
    def test_large_number_of_images_and_annotations(self):
        # Arrange
        num_images = 1000
        img_paths = [f'test_image{i}.jpg' for i in range(num_images)]
        annotations = [[[0, 0.5, 0.5, 0.2, 0.2]] for _ in range(num_images)]

        for i in range(num_images):
            img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(img_paths[i], img)

        img_list = img_paths
        anno_list = annotations
        flip_type = 0

        # Act
        new_imgs_list, new_annos_lists, path_list = update_image_and_anno(img_list, anno_list, flip_type)

        # Assert
        assert len(new_imgs_list) == num_images
        assert len(new_annos_lists) == num_images
        assert path_list == img_list

        for i in range(num_images):
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

    @pytest.mark.valid
    def test_annotation_out_of_bounds(self):
        # Arrange
        img_paths = ['test_image1.jpg', 'test_image2.jpg']
        img1 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(img_paths[0], img1)
        cv2.imwrite(img_paths[1], img2)
        annotations = [
            [[0, 1.5, 0.5, 0.2, 0.2]],  # out of bounds annotation
            [[1, 0.2, 0.3, 0.1, 0.1]]
        ]

        img_list = img_paths
        anno_list = annotations
        flip_type = 1

        # Act
        new_imgs_list, new_annos_lists, path_list = update_image_and_anno(img_list, anno_list, flip_type)

        # Assert
        assert len(new_imgs_list) == 2
        assert len(new_annos_lists) == 2
        assert path_list == img_list

        for i in range(2):
            flipped_img = new_imgs_list[i]
            flipped_annotation = new_annos_lists[i][0]

            # Check if the image is flipped horizontally
            expected_flipped_img = cv2.flip(cv2.imread(img_paths[i]), flip_type)
            assert np.array_equal(flipped_img, expected_flipped_img)

            # Check if the annotation is updated correctly
            expected_flipped_annotation = [annotations[i][0][0], 1 - annotations[i][0][1], annotations[i][0][2], annotations[i][0][3], annotations[i][0][4]]
            assert flipped_annotation == pytest.approx(expected_flipped_annotation)

        # Clean up
        for img_path in img_paths:
            os.remove(img_path)

    @pytest.mark.valid
    def test_empty_annotations(self):
        # Arrange
        img_paths = ['test_image1.jpg', 'test_image2.jpg']
        img1 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(img_paths[0], img1)
        cv2.imwrite(img_paths[1], img2)
        annotations = [
            [],  # empty annotation
            [[1, 0.2, 0.3, 0.1, 0.1]]
        ]

        img_list = img_paths
        anno_list = annotations
        flip_type = 1

        # Act
        new_imgs_list, new_annos_lists, path_list = update_image_and_anno(img_list, anno_list, flip_type)

        # Assert
        assert len(new_imgs_list) == 2
        assert len(new_annos_lists) == 2
        assert path_list == img_list

        for i in range(2):
            flipped_img = new_imgs_list[i]
            flipped_annotations = new_annos_lists[i]

            # Check if the image is flipped horizontally
            expected_flipped_img = cv2.flip(cv2.imread(img_paths[i]), flip_type)
            assert np.array_equal(flipped_img, expected_flipped_img)

            # Check if the annotations are updated correctly
            if annotations[i]:
                expected_flipped_annotation = [annotations[i][0][0], 1 - annotations[i][0][1], annotations[i][0][2], annotations[i][0][3], annotations[i][0][4]]
                assert flipped_annotations[0] == pytest.approx(expected_flipped_annotation)
            else:
                assert flipped_annotations == []

        # Clean up
        for img_path in img_paths:
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
