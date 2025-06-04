import pytest
import os
import shutil
import random
from string import ascii_lowercase, digits
import cv2
import numpy as np
from mosaic_augmentation import main, get_dataset, update_image_and_anno, random_chars
import glob

class Test_MosaicAugmentationMain:
    @pytest.mark.smoke
    @pytest.mark.positive
    def test_main_successful_execution(self):
        # Arrange
        LABEL_DIR = 'test_data/labels'
        IMG_DIR = 'test_data/images'
        OUTPUT_DIR = 'test_output'
        NUMBER_IMAGES = 10
        OUTPUT_SIZE = (720, 1280)
        SCALE_RANGE = (0.4, 0.6)
        FILTER_TINY_SCALE = 1 / 100

        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        # Act
        main()

        # Assert
        output_files = os.listdir(OUTPUT_DIR)
        assert len(output_files) == NUMBER_IMAGES * 2

        img_files = [f for f in output_files if f.endswith('.jpg')]
        assert len(img_files) == NUMBER_IMAGES

        anno_files = [f for f in output_files if f.endswith('.txt')]
        assert len(anno_files) == NUMBER_IMAGES

        for img_file in img_files:
            img_path = os.path.join(OUTPUT_DIR, img_file)
            img = cv2.imread(img_path)
            assert img.shape[:2] == OUTPUT_SIZE

        # Clean up
        shutil.rmtree(OUTPUT_DIR)

    @pytest.mark.regression
    @pytest.mark.negative
    def test_main_empty_label_dir(self):
        # Arrange
        LABEL_DIR = 'test_data/empty_labels'
        IMG_DIR = 'test_data/images'
        OUTPUT_DIR = 'test_output'
        NUMBER_IMAGES = 10
        OUTPUT_SIZE = (720, 1280)
        SCALE_RANGE = (0.4, 0.6)
        FILTER_TINY_SCALE = 1 / 100

        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        # Act
        main()

        # Assert
        output_files = os.listdir(OUTPUT_DIR)
        assert len(output_files) == 0

        # Clean up
        shutil.rmtree(OUTPUT_DIR)

    @pytest.mark.regression
    @pytest.mark.negative
    def test_main_missing_image_files(self):
        # Arrange
        LABEL_DIR = 'test_data/labels'
        IMG_DIR = 'test_data/missing_images'
        OUTPUT_DIR = 'test_output'
        NUMBER_IMAGES = 10
        OUTPUT_SIZE = (720, 1280)
        SCALE_RANGE = (0.4, 0.6)
        FILTER_TINY_SCALE = 1 / 100

        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        # Act
        main()

        # Assert
        output_files = os.listdir(OUTPUT_DIR)
        assert len(output_files) > 0

        # Clean up
        shutil.rmtree(OUTPUT_DIR)

    @pytest.mark.regression
    @pytest.mark.positive
    def test_main_filter_scale(self):
        # Arrange
        LABEL_DIR = 'test_data/labels'
        IMG_DIR = 'test_data/images'
        OUTPUT_DIR = 'test_output'
        NUMBER_IMAGES = 10
        OUTPUT_SIZE = (720, 1280)
        SCALE_RANGE = (0.4, 0.6)
        FILTER_TINY_SCALE = 0.1

        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        # Act
        main()

        # Assert
        anno_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.txt')]
        for anno_file in anno_files:
            with open(os.path.join(OUTPUT_DIR, anno_file), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    _, x_center, y_center, width, height = line.split()
                    assert float(width) >= FILTER_TINY_SCALE
                    assert float(height) >= FILTER_TINY_SCALE

        # Clean up
        shutil.rmtree(OUTPUT_DIR)

    @pytest.mark.performance
    @pytest.mark.positive
    def test_main_large_number_of_images(self):
        # Arrange
        LABEL_DIR = 'test_data/labels'
        IMG_DIR = 'test_data/images'
        OUTPUT_DIR = 'test_output'
        NUMBER_IMAGES = 1000
        OUTPUT_SIZE = (720, 1280)
        SCALE_RANGE = (0.4, 0.6)
        FILTER_TINY_SCALE = 1 / 100

        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        # Act
        main()

        # Assert
        output_files = os.listdir(OUTPUT_DIR)
        assert len(output_files) == NUMBER_IMAGES * 2

        # Clean up
        shutil.rmtree(OUTPUT_DIR)

    @pytest.mark.regression
    @pytest.mark.positive
    def test_main_output_file_format(self):
        # Arrange
        LABEL_DIR = 'test_data/labels'
        IMG_DIR = 'test_data/images'
        OUTPUT_DIR = 'test_output'
        NUMBER_IMAGES = 10
        OUTPUT_SIZE = (720, 1280)
        SCALE_RANGE = (0.4, 0.6)
        FILTER_TINY_SCALE = 1 / 100

        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        # Act
        main()

        # Assert
        img_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.jpg')]
        for img_file in img_files:
            img = cv2.imread(os.path.join(OUTPUT_DIR, img_file))
            assert img is not None

        anno_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.txt')]
        for anno_file in anno_files:
            with open(os.path.join(OUTPUT_DIR, anno_file), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.split()
                    assert len(parts) == 5
                    assert all(isinstance(float(part), float) for part in parts[1:])

        # Clean up
        shutil.rmtree(OUTPUT_DIR)

    @pytest.mark.regression
    @pytest.mark.negative
    def test_main_invalid_output_size(self):
        # Arrange
        LABEL_DIR = 'test_data/labels'
        IMG_DIR = 'test_data/images'
        OUTPUT_DIR = 'test_output'
        NUMBER_IMAGES = 10
        OUTPUT_SIZE = (0, 0)
        SCALE_RANGE = (0.4, 0.6)
        FILTER_TINY_SCALE = 1 / 100

        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        # Act & Assert
        with pytest.raises(ValueError):
            main()

        # Clean up
        shutil.rmtree(OUTPUT_DIR)

    @pytest.mark.regression
    @pytest.mark.positive
    def test_main_random_string_generation(self):
        # Arrange
        LABEL_DIR = 'test_data/labels'
        IMG_DIR = 'test_data/images'
        OUTPUT_DIR = 'test_output'
        NUMBER_IMAGES = 10
        OUTPUT_SIZE = (720, 1280)
        SCALE_RANGE = (0.4, 0.6)
        FILTER_TINY_SCALE = 1 / 100

        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        # Act
        main()

        # Assert
        img_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.jpg')]
        for img_file in img_files:
            file_name, _ = os.path.splitext(img_file)
            random_string = file_name.split('_')[-1]
            assert len(random_string) == 32
            assert set(random_string).issubset(set(ascii_lowercase + digits))

        # Clean up
        shutil.rmtree(OUTPUT_DIR)

    @pytest.mark.regression
    @pytest.mark.negative
    def test_main_nonexistent_output_dir(self):
        # Arrange
        LABEL_DIR = 'test_data/labels'
        IMG_DIR = 'test_data/images'
        OUTPUT_DIR = 'nonexistent_output'
        NUMBER_IMAGES = 10
        OUTPUT_SIZE = (720, 1280)
        SCALE_RANGE = (0.4, 0.6)
        FILTER_TINY_SCALE = 1 / 100

        # Act & Assert
        with pytest.raises(FileNotFoundError):
            main()

    @pytest.mark.regression
    @pytest.mark.positive
    def test_main_annotation_format(self):
        # Arrange
        LABEL_DIR = 'test_data/labels'
        IMG_DIR = 'test_data/images'
        OUTPUT_DIR = 'test_output'
        NUMBER_IMAGES = 10
        OUTPUT_SIZE = (720, 1280)
        SCALE_RANGE = (0.4, 0.6)
        FILTER_TINY_SCALE = 1 / 100

        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        # Act
        main()

        # Assert
        anno_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.txt')]
        for anno_file in anno_files:
            with open(os.path.join(OUTPUT_DIR, anno_file), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.split()
                    assert len(parts) == 5
                    assert all(isinstance(float(part), float) for part in parts[1:])

        # Clean up
        shutil.rmtree(OUTPUT_DIR)

def get_dataset(label_dir: str, img_dir: str) -> tuple[list, list]:
    """
    - label_dir <type: str>: Path to label include annotation of images
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
            xmin = float(obj[1]) - float(obj[3]) / 2
            ymin = float(obj[2]) - float(obj[4]) / 2
            xmax = float(obj[1]) + float(obj[3]) / 2
            ymax = float(obj[2]) + float(obj[4]) / 2

            boxes.append([int(obj[0]), xmin, ymin, xmax, ymax])
        if not boxes:
            continue
        img_paths.append(img_path)
        labels.append(boxes)
    return img_paths, labels

def update_image_and_anno(
    all_img_list: list,
    all_annos: list,
    idxs: list[int],
    output_size: tuple[int, int],
    scale_range: tuple[float, float],
    filter_scale: float = 0.0,
) -> tuple[list, list, str]:
    """
    - all_img_list <type: list>: list of all images
    - all_annos <type: list>: list of all annotations of specific image
    - idxs <type: list>: index of image in list
    - output_size <type: tuple>: size of output image (Height, Width)
    - scale_range <type: tuple>: range of scale image
    - filter_scale <type: float>: the condition of downscale image and bounding box
    Return:
        - output_img <type: narray>: image after resize
        - new_anno <type: list>: list of new annotation after scale
        - path[0] <type: string>: get the name of image file
    """
    output_img = np.zeros([output_size[0], output_size[1], 3], dtype=np.uint8)
    scale_x = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])
    scale_y = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])
    divid_point_x = int(scale_x * output_size[1])
    divid_point_y = int(scale_y * output_size[0])

    new_anno = []
    path_list = []
    for i, index in enumerate(idxs):
        path = all_img_list[index]
        path_list.append(path)
        img_annos = all_annos[index]
        img = cv2.imread(path)
        if i == 0:  # top-left
            img = cv2.resize(img, (divid_point_x, divid_point_y))
            output_img[:divid_point_y, :divid_point_x, :] = img
            for bbox in img_annos:
                xmin = bbox[1] * scale_x
                ymin = bbox[2] * scale_y
                xmax = bbox[3] * scale_x
                ymax = bbox[4] * scale_y
                new_anno.append([bbox[0], xmin, ymin, xmax, ymax])
        elif i == 1:  # top-right
            img = cv2.resize(img, (output_size[1] - divid_point_x, divid_point_y))
            output_img[:divid_point_y, divid_point_x: output_size[1], :] = img
            for bbox in img_annos:
                xmin = scale_x + bbox[1] * (1 - scale_x)
                ymin = bbox[2] * scale_y
                xmax = scale_x + bbox[3] * (1 - scale_x)
                ymax = bbox[4] * scale_y
                new_anno.append([bbox[0], xmin, ymin, xmax, ymax])
        elif i == 2:  # bottom-left
            img = cv2.resize(img, (divid_point_x, output_size[0] - divid_point_y))
            output_img[divid_point_y: output_size[0], :divid_point_x, :] = img
            for bbox in img_annos:
                xmin = bbox[1] * scale_x
                ymin = scale_y + bbox[2] * (1 - scale_y)
                xmax = bbox[3] * scale_x
                ymax = scale_y + bbox[4] * (1 - scale_y)
                new_anno.append([bbox[0], xmin, ymin, xmax, ymax])
        else:  # bottom-right
            img = cv2.resize(
                img, (output_size[1] - divid_point_x, output_size[0] - divid_point_y)
            )
            output_img[
                divid_point_y: output_size[0], divid_point_x: output_size[1], :
            ] = img
            for bbox in img_annos:
                xmin = scale_x + bbox[1] * (1 - scale_x)
                ymin = scale_y + bbox[2] * (1 - scale_y)
                xmax = scale_x + bbox[3] * (1 - scale_x)
                ymax = scale_y + bbox[4] * (1 - scale_y)
                new_anno.append([bbox[0], xmin, ymin, xmax, ymax])

    # Remove bounding box small than scale of filter
    if filter_scale > 0:
        new_anno = [
            anno
            for anno in new_anno
            if filter_scale < (anno[3] - anno[1]) and filter_scale < (anno[4] - anno[2])
        ]

    return output_img, new_anno, path_list[0]

def random_chars(number_char: int) -> str:
    """
    Automatic generate random 32 characters.
    Get random string code: '7b7ad245cdff75241935e4dd860f3bad'
    >>> len(random_chars(32))
    32
    """
    assert number_char > 1, "The number of character should greater than 1"
    letter_code = ascii_lowercase + digits
    return "".join(random.choice(letter_code) for _ in range(number_char))

def main() -> None:
    """
    Get images list and annotations list from input dir.
    Update new images and annotations.
    Save images and annotations in output dir.
    """
    img_paths, annos = get_dataset(LABEL_DIR, IMG_DIR)
    for index in range(NUMBER_IMAGES):
        idxs = random.sample(range(len(annos)), 4)
        new_image, new_annos, path = update_image_and_anno(
            img_paths,
            annos,
            idxs,
            OUTPUT_SIZE,
            SCALE_RANGE,
            filter_scale=FILTER_TINY_SCALE,
        )

        # Get random string code: '7b7ad245cdff75241935e4dd860f3bad'
        letter_code = random_chars(32)
        file_name = path.split(os.sep)[-1].rsplit(".", 1)[0]
        file_root = f"{OUTPUT_DIR}/{file_name}_MOSAIC_{letter_code}"
        cv2.imwrite(f"{file_root}.jpg", new_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"Succeeded {index+1}/{NUMBER_IMAGES} with {file_name}")
        annos_list = []
        for anno in new_annos:
            width = anno[3] - anno[1]
            height = anno[4] - anno[2]
            x_center = anno[1] + width / 2
            y_center = anno[2] + height / 2
            obj = f"{anno[0]} {x_center} {y_center} {width} {height}"
            annos_list.append(obj)
        with open(f"{file_root}.txt", "w") as outfile:
            outfile.write("\n".join(line for line in annos_list))
