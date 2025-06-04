import pytest
import os
import glob
import shutil
import tempfile
from mosaic_augmentation import get_dataset

class Test_MosaicAugmentationGetDataset:

    @pytest.mark.valid
    @pytest.mark.regression
    @pytest.mark.smoke
    def test_valid_directory_structure(self):
        # Arrange
        temp_dir = tempfile.mkdtemp()
        label_dir = os.path.join(temp_dir, 'labels')
        img_dir = os.path.join(temp_dir, 'images')
        os.makedirs(label_dir, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)

        # Create label and image files
        label_files = ['img1.txt', 'img2.txt']
        img_files = ['img1.jpg', 'img2.jpg']

        for label_file in label_files:
            with open(os.path.join(label_dir, label_file), 'w') as f:
                f.write("0 0.5 0.5 0.2 0.2\n")
                f.write("1 0.3 0.3 0.1 0.1\n")

        for img_file in img_files:
            with open(os.path.join(img_dir, img_file), 'w') as f:
                f.write("dummy image data")

        # Act
        img_paths, labels = get_dataset(label_dir, img_dir)

        # Assert
        expected_img_paths = [os.path.join(img_dir, img_file) for img_file in img_files]
        expected_labels = [
            [
                [0, 0.4, 0.4, 0.6, 0.6],
                [1, 0.25, 0.25, 0.35, 0.35]
            ],
            [
                [0, 0.4, 0.4, 0.6, 0.6],
                [1, 0.25, 0.25, 0.35, 0.35]
            ]
        ]

        assert img_paths == expected_img_paths
        assert labels == expected_labels

        # Cleanup
        shutil.rmtree(temp_dir)
