import pytest
import os
import random
import shutil
import cv2
import numpy as np
import glob
from mosaic_augmentation import main
from string import ascii_lowercase, digits

# Define necessary dependencies
def random_chars(number_char: int) -> str:
    """Automatic generate random 32 characters."""
    return "".join(random.choice(ascii_lowercase + digits) for _ in range(number_char))

def get_dataset(label_dir: str, img_dir: str) -> tuple[list, list]:
    """Return list of images path and labels."""
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
) -> tuple[np.ndarray, list, str]:
    """Update image and annotation."""
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
            output_img[:divid_point_y, divid_point_x:output_size[1], :] = img
            for bbox in img_annos:
                xmin = scale_x + bbox[1] * (1 - scale_x)
                ymin = bbox[2] * scale_y
                xmax = scale_x + bbox[3] * (1 - scale_x)
                ymax = bbox[4] * scale_y
                new_anno.append([bbox[0], xmin, ymin, xmax, ymax])
        elif i == 2:  # bottom-left
            img = cv2.resize(img, (divid_point_x, output_size[0] - divid_point_y))
            output_img[divid_point_y:output_size[0], :divid_point_x, :] = img
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
                divid_point_y:output_size[0], divid_point_x:output_size[1], :
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

# Mock configuration for testing
OUTPUT_SIZE = (720, 1280)
SCALE_RANGE = (0.4, 0.6)
FILTER_TINY_SCALE = 1 / 100
NUMBER_IMAGES = 250

class Test_MosaicAugmentationMain:

    @pytest.mark.smoke
    @pytest.mark.valid
    def test_main_basic_functionality(self):
        # Arrange: Prepare a set of sample images and corresponding annotations in a temporary directory.
        temp_dir = os.path.join(os.getcwd(), "temp_test_dir")
        os.makedirs(temp_dir, exist_ok=True)

        # TODO: Update these values according to your test setup
        label_dir = os.path.join(temp_dir, "labels")
        img_dir = os.path.join(temp_dir, "images")
        output_dir = os.path.join(temp_dir, "output")

        os.makedirs(label_dir, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # Create dummy images and annotations
        for i in range(5):
            img_path = os.path.join(img_dir, f"image_{i}.jpg")
            label_path = os.path.join(label_dir, f"image_{i}.txt")
            dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(img_path, dummy_img)
            with open(label_path, "w") as f:
                f.write("0 0.5 0.5 1.0 1.0\n")

        # Act: Call the main function.
        import mosaic_augmentation
        mosaic_augmentation.LABEL_DIR = label_dir
        mosaic_augmentation.IMG_DIR = img_dir
        mosaic_augmentation.OUTPUT_DIR = output_dir
        mosaic_augmentation.main()

        # Assert: Check that the output directory contains the expected number of new images and annotation files.
        output_images = glob.glob(os.path.join(output_dir, "*.jpg"))
        output_labels = glob.glob(os.path.join(output_dir, "*.txt"))

        assert len(output_images) == NUMBER_IMAGES
        assert len(output_labels) == NUMBER_IMAGES

        # Clean up
        shutil.rmtree(temp_dir)
