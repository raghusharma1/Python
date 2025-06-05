import cv2
import numpy as np
import pytest
from harris_corner import HarrisCorner

class Test_TestHarrisCornerDetectTestSmallWindowSize:

    @pytest.mark.smoke
    @pytest.mark.valid
    def test_corner_detection_with_minimum_window_size(self):
        # Arrange
        k = 0.04
        window_size = 1
        edge_detect = HarrisCorner(k, window_size)
        img_path = "path_to_valid_image_with_corners.jpg"  # TODO: Provide a valid image path

        # Act
        color_img, corner_list = edge_detect.detect(img_path)

        # Assert
        assert isinstance(color_img, np.ndarray)
        assert len(corner_list) > 0
        for corner in corner_list:
            assert len(corner) == 3
            x, y, _ = corner
            assert 0 <= x < color_img.shape[1]
            assert 0 <= y < color_img.shape[0]

class HarrisCorner:
    def __init__(self, k: float, window_size: int):
        """

        k : is an empirically determined constant in [0.04,0.06]
        window_size : neighbourhoods considered

        """

        if k in (0.04, 0.06):
            self.k = k
            self.window_size = window_size
        else:
            raise ValueError("invalid k value")

    def __str__(self) -> str:
        return str(self.k)

    def detect(self, img_path: str) -> tuple[cv2.Mat, list[list[int]]]:
        """

        Returns the image with corners identified
        img_path  : path of the image
        output : list of the corner positions, image

        """

        img = cv2.imread(img_path, 0)
        h, w = img.shape
        corner_list: list[list[int]] = []
        color_img = img.copy()
        color_img = cv2.cvtColor(color_img, cv2.COLOR_GRAY2RGB)
        dy, dx = np.gradient(img)
        ixx = dx**2
        iyy = dy**2
        ixy = dx * dy
        k = 0.04
        offset = self.window_size // 2
        for y in range(offset, h - offset):
            for x in range(offset, w - offset):
                wxx = ixx[
                    y - offset : y + offset + 1, x - offset : x + offset + 1
                ].sum()
                wyy = iyy[
                    y - offset : y + offset + 1, x - offset : x + offset + 1
                ].sum()
                wxy = ixy[
                    y - offset : y + offset + 1, x - offset : x + offset + 1
                ].sum()

                det = (wxx * wyy) - (wxy**2)
                trace = wxx + wyy
                r = det - k * (trace**2)
                # Can change the value
                if r > 0.5:
                    corner_list.append([x, y, r])
                    color_img.itemset((y, x, 0), 0)
                    color_img.itemset((y, x, 1), 0)
                    color_img.itemset((y, x, 2), 255)
        return color_img, corner_list
