"""This module contains utility function that handles logic which deal with the
whole frame / image."""
from typing import Tuple

import cv2
import numpy as np


def rotate_image(img: np.ndarray, center: Tuple[int, int], angle: float) -> np.ndarray:
    """Rotate the image based on the given `center` and `angle`. A positive angle means to
    rotate counter-clockwise.
    """
    # Create rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)

    # Get image width and height
    img_height, img_width = img.shape[:2]

    # Rotate image
    return cv2.warpAffine(img, rotation_matrix, (img_width, img_height))
