import pytest
import os
import shutil
import glob
import random
from string import ascii_lowercase, digits
import cv2
import numpy
from _test_FlipAugmentationMain import Test_FlipAugmentationMain

class Test_TestFlipAugmentationMainTestProcessImagesAndAnnotationsCorrectly(Test_FlipAugmentationMain):

    @pytest.mark.smoke
    def test_basic_functionality(self):
        # Arrange
        LABEL_DIR = self.label_dir
        IMAGE_DIR = self.img_dir
        OUTPUT_DIR = self.output_dir
        FLIP_TYPE = 1

        img_path = os.path.join(IMAGE_DIR, 'test_image.jpg')
        label_path = os.path.join(LABEL_DIR, 'test_image.txt')
        cv2.imwrite(img_path, numpy.zeros((100, 100, 3), dtype=numpy.uint8))
        with open(label_path, 'w') as f:
            f.write('0 0.5 0.5 0.1 0.1\n')

        # Act
        self.main()

        # Assert
        output_img_path = glob.glob(os.path.join(OUTPUT_DIR, 'test_image_FLIP_*.jpg'))[0]
        output_label_path = glob.glob(os.path.join(OUTPUT_DIR, 'test_image_FLIP_*.txt'))[0]

        assert os.path.exists(output_img_path)
        assert os.path.exists(output_label_path)

        with open(output_label_path, 'r') as f:
            output_annotation = f.readline().strip()
            assert output_annotation == '0 0.5 0.5 0.1 0.1'

        # Clean up
        os.remove(img_path)
        os.remove(label_path)

    @pytest.mark.regression
    def test_missing_annotation(self):
        # Arrange
        LABEL_DIR = self.label_dir
        IMAGE_DIR = self.img_dir
        OUTPUT_DIR = self.output_dir
        FLIP_TYPE = 1

        img_path = os.path.join(IMAGE_DIR, 'test_image.jpg')
        cv2.imwrite(img_path, numpy.zeros((100, 100, 3), dtype=numpy.uint8))

        # Act
        self.main()

        # Assert
        output_img_paths = glob.glob(os.path.join(OUTPUT_DIR, 'test_image_FLIP_*.jpg'))
        output_label_paths = glob.glob(os.path.join(OUTPUT_DIR, 'test_image_FLIP_*.txt'))

        assert len(output_img_paths) == 0
        assert len(output_label_paths) == 0

        # Clean up
        os.remove(img_path)

    @pytest.mark.valid
    def test_horizontal_flip_annotation_update(self):
        # Arrange
        LABEL_DIR = self.label_dir
        IMAGE_DIR = self.img_dir
        OUTPUT_DIR = self.output_dir
        FLIP_TYPE = 1

        img_path = os.path.join(IMAGE_DIR, 'test_image.jpg')
        label_path = os.path.join(LABEL_DIR, 'test_image.txt')
        cv2.imwrite(img_path, numpy.zeros((100, 100, 3), dtype=numpy.uint8))
        with open(label_path, 'w') as f:
            f.write('0 0.5 0.5 0.1 0.1\n')

        # Act
        self.main()

        # Assert
        output_label_path = glob.glob(os.path.join(OUTPUT_DIR, 'test_image_FLIP_*.txt'))[0]

        with open(output_label_path, 'r') as f:
            output_annotation = f.readline().strip()
            assert output_annotation == '0 0.5 0.5 0.1 0.1'

        # Clean up
        os.remove(img_path)
        os.remove(label_path)

    @pytest.mark.valid
    def test_vertical_flip_annotation_update(self):
        # Arrange
        LABEL_DIR = self.label_dir
        IMAGE_DIR = self.img_dir
        OUTPUT_DIR = self.output_dir
        FLIP_TYPE = 0

        img_path = os.path.join(IMAGE_DIR, 'test_image.jpg')
        label_path = os.path.join(LABEL_DIR, 'test_image.txt')
        cv2.imwrite(img_path, numpy.zeros((100, 100, 3), dtype=numpy.uint8))
        with open(label_path, 'w') as f:
            f.write('0 0.5 0.5 0.1 0.1\n')

        # Act
        self.main()

        # Assert
        output_label_path = glob.glob(os.path.join(OUTPUT_DIR, 'test_image_FLIP_*.txt'))[0]

        with open(output_label_path, 'r') as f:
            output_annotation = f.readline().strip()
            assert output_annotation == '0 0.5 0.5 0.1 0.1'

        # Clean up
        os.remove(img_path)
        os.remove(label_path)

    @pytest.mark.smoke
    def test_multiple_images(self):
        # Arrange
        LABEL_DIR = self.label_dir
        IMAGE_DIR = self.img_dir
        OUTPUT_DIR = self.output_dir
        FLIP_TYPE = 1

        img_paths = [os.path.join(IMAGE_DIR, f'test_image_{i}.jpg') for i in range(3)]
        label_paths = [os.path.join(LABEL_DIR, f'test_image_{i}.txt') for i in range(3)]

        for img_path, label_path in zip(img_paths, label_paths):
            cv2.imwrite(img_path, numpy.zeros((100, 100, 3), dtype=numpy.uint8))
            with open(label_path, 'w') as f:
                f.write('0 0.5 0.5 0.1 0.1\n')

        # Act
        self.main()

        # Assert
        output_img_paths = glob.glob(os.path.join(OUTPUT_DIR, 'test_image_*_FLIP_*.jpg'))
        output_label_paths = glob.glob(os.path.join(OUTPUT_DIR, 'test_image_*_FLIP_*.txt'))

        assert len(output_img_paths) == 3
        assert len(output_label_paths) == 3

        for label_path in output_label_paths:
            with open(label_path, 'r') as f:
                output_annotation = f.readline().strip()
                assert output_annotation == '0 0.5 0.5 0.1 0.1'

        # Clean up
        for img_path, label_path in zip(img_paths, label_paths):
            os.remove(img_path)
            os.remove(label_path)

    @pytest.mark.valid
    def test_random_string_in_filename(self):
        # Arrange
        LABEL_DIR = self.label_dir
        IMAGE_DIR = self.img_dir
        OUTPUT_DIR = self.output_dir
        FLIP_TYPE = 1

        img_path = os.path.join(IMAGE_DIR, 'test_image.jpg')
        label_path = os.path.join(LABEL_DIR, 'test_image.txt')
        cv2.imwrite(img_path, numpy.zeros((100, 100, 3), dtype=numpy.uint8))
        with open(label_path, 'w') as f:
            f.write('0 0.5 0.5 0.1 0.1\n')

        # Act
        self.main()

        # Assert
        output_img_path = glob.glob(os.path.join(OUTPUT_DIR, 'test_image_FLIP_*.jpg'))[0]
        output_label_path = glob.glob(os.path.join(OUTPUT_DIR, 'test_image_FLIP_*.txt'))[0]

        assert len(output_img_path.split('_')[-1].split('.')[0]) == 32
        assert len(output_label_path.split('_')[-1].split('.')[0]) == 32

        # Clean up
        os.remove(img_path)
        os.remove(label_path)

    @pytest.mark.valid
    def test_image_flipping(self):
        # Arrange
        LABEL_DIR = self.label_dir
        IMAGE_DIR = self.img_dir
        OUTPUT_DIR = self.output_dir
        FLIP_TYPE = 1

        img_path = os.path.join(IMAGE_DIR, 'test_image.jpg')
        label_path = os.path.join(LABEL_DIR, 'test_image.txt')
        dummy_image = numpy.zeros((100, 100, 3), dtype=numpy.uint8)
        dummy_image[:50, :] = 255  # Create a vertical half white image
        cv2.imwrite(img_path, dummy_image)
        with open(label_path, 'w') as f:
            f.write('0 0.5 0.5 0.1 0.1\n')

        # Act
        self.main()

        # Assert
        output_img_path = glob.glob(os.path.join(OUTPUT_DIR, 'test_image_FLIP_*.jpg'))[0]
        output_image = cv2.imread(output_img_path)
        assert numpy.array_equal(output_image[:50, :], numpy.zeros((50, 100, 3), dtype=numpy.uint8))
        assert numpy.array_equal(output_image[50:, :], numpy.full((50, 100, 3), 255, dtype=numpy.uint8))

        # Clean up
        os.remove(img_path)
        os.remove(label_path)

def main():
    """ Get images list and annotations list from input dir.
        Update new images and annotations.
        Save images and annotations in output dir.
    """
    LABEL_DIR = self.label_dir
    IMAGE_DIR = self.img_dir
    OUTPUT_DIR = self.output_dir
    FLIP_TYPE = 1
    img_paths, annos = get_dataset(LABEL_DIR, IMAGE_DIR)
    print("Processing...")
    new_images, new_annos, paths = update_image_and_anno(img_paths, annos, FLIP_TYPE)

    for index, image in enumerate(new_images):
        # Get random string code: '7b7ad245cdff75241935e4dd860f3bad'
        letter_code = random_chars(32)
        file_name = paths[index].split(os.sep)[-1].rsplit(".", 1)[0]
        file_root = f"{OUTPUT_DIR}/{file_name}_FLIP_{letter_code}"
        cv2.imwrite(f"{file_root}.jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"Success {index+1}/{len(new_images)} with {file_name}")
        annos_list = []
        for anno in new_annos[index]:
            obj = f"{anno[0]} {anno[1]} {anno[2]} {anno[3]} {anno[4]}"
            annos_list.append(obj)
        with open(f"{file_root}.txt", "w") as outfile:
            outfile.write("\n".join(line for line in annos_list))

def get_dataset(label_dir, img_dir):
    """ - label_dir <type: str>: Path to label include annotation of images
        - img_dir <type: str>: Path to folder contain images
        Return <type: list>: List of images path and labels
    """
    img_paths = []
    labels = []
    for label_file in glob.glob(os.path.join(label_dir, "*.txt")):
        label_name = label_file.split(os.sep)[-1].rsplit(".", 1)[0]
        with open(label_file) as in_file:
            obj_lists = in_file.readlines()
        img_path = os.path.join(img_dir, f"{label_name}.jpg")

        boxes = []
        for obj_list in obj_lists:
            obj = obj_list.rstrip("\n").split(" ")
            boxes.append(
                [
                    int(obj[0]),
                    float(obj[1]),
                    float(obj[2]),
                    float(obj[3]),
                    float(obj[4]),
                ]
            )
        if not boxes:
            continue
        img_paths.append(img_path)
        labels.append(boxes)
    return img_paths, labels

def update_image_and_anno(
    img_list, anno_list, flip_type=1
):
    """ - img_list <type: list>: list of all images
        - anno_list <type: list>: list of all annotations of specific image
        - flip_type <type: int>: 0 is vertical, 1 is horizontal
        Return:
            - new_imgs_list <type: narray>: image after resize
            - new_annos_lists <type: list>: list of new annotation after scale
            - path_list <type: list>: list the name of image file
    """
    new_annos_lists = []
    path_list = []
    new_imgs_list = []
    for idx in range(len(img_list)):
        new_annos = []
        path = img_list[idx]
        path_list.append(path)
        img_annos = anno_list[idx]
        img = cv2.imread(path)
        if flip_type == 1:
            new_img = cv2.flip(img, flip_type)
            for bbox in img_annos:
                x_center_new = 1 - bbox[1]
                new_annos.append([bbox[0], x_center_new, bbox[2], bbox[3], bbox[4]])
        elif flip_type == 0:
            new_img = cv2.flip(img, flip_type)
            for bbox in img_annos:
                y_center_new = 1 - bbox[2]
                new_annos.append([bbox[0], bbox[1], y_center_new, bbox[3], bbox[4]])
        new_annos_lists.append(new_annos)
        new_imgs_list.append(new_img)
    return new_imgs_list, new_annos_lists, path_list

def random_chars(number_char=32):
    """ Automatic generate random 32 characters.
        Get random string code: '7b7ad245cdff75241935e4dd860f3bad'
        >>> len(random_chars(32))
        32
    """
    assert number_char > 1, "The number of character should greater than 1"
    letter_code = ascii_lowercase + digits
    return "".join(random.choice(letter_code) for _ in range(number_char))
