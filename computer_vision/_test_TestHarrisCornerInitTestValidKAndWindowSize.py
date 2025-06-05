import pytest
import cv2
import numpy as np
from harris_corner import HarrisCorner

class Test_TestHarrisCornerInitTestValidKAndWindowSize:
    @pytest.mark.valid
    @pytest.mark.positive
    def test_valid_k_and_window_size(self):
        k_values = [0.04, 0.06]
        window_size = 5

        for k in k_values:
            harris_corner = HarrisCorner(k, window_size)
            assert harris_corner.k == k
            assert harris_corner.window_size == window_size

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
                r = det - self.k * (trace**2)
                if r > 0.5:
                    corner_list.append([x, y, r])
                    color_img.itemset((y, x, 0), 0)
                    color_img.itemset((y, x, 1), 0)
                    color_img.itemset((y, x, 2), 255)
        return color_img, corner_list
