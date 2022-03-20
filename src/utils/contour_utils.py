"""This module contains utility functions that deal with contour and threshold image."""
from typing import List

import cv2
import numpy as np


def filter_out_small_contours(thresh_img: np.ndarray, min_contour_area: float) -> np.ndarray:
    """Filter out contours that are less than the `min_contour_area` by filling the `thresh_img`
    as black where the contours are found.

    Expects `thresh_img` to be a grayscale image with bg as black and fg as white.
    """
    # Make copy of thresh_img
    thresh_img = thresh_img.copy()

    # Find the contours
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours
    contours_to_fill = []
    for contour in contours:
        if cv2.contourArea(contour) < min_contour_area:
            contours_to_fill.append(contour)

    # Fill small contours
    cv2.drawContours(thresh_img, contours_to_fill, -1, 0, -1)

    return thresh_img


def filter_for_largest_contour(thresh_img: np.ndarray, min_contour_area: float) -> np.ndarray:
    """Filter for the largest contour in `thresh_img`. Any contour that is not the largest
    contour will be filled as black.

    The largest contour must be larger than specified `min_contour_area` otherwise
    a simple black image will be returned.

    Expects `thresh_img` to be a grayscale image with bg as black as fg as white.
    """
    # Make a black image
    black_img = np.zeros(thresh_img.shape, np.uint8)

    # Find the contours
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the max contour
    max_contour = max(contours, key=cv2.contourArea)

    # Fill the black image with white at the max_contour if greater than min_contour_are
    if cv2.contourArea(max_contour) >= min_contour_area:
        cv2.drawContours(black_img, [max_contour], -1, 255, -1)

    return black_img


def get_contour_center_x_position(contour: np.ndarray) -> float:
    """Get the x position of the center of the given `contour`. A high value means the
    contour center is positioned more towards the right side.
    """
    contour_rect = cv2.minAreaRect(contour)

    return contour_rect[0][0]


def get_contour_center_y_position(contour: np.ndarray) -> float:
    """Get the y position of the center of the given `contour`. A high value means the
    contour center is positioned more towards the bottom side.
    """
    contour_rect = cv2.minAreaRect(contour)

    return contour_rect[0][1]


def sort_contours_by_axis(contours: List[np.ndarray], by_x_axis: bool) -> List[np.ndarray]:
    """Sort the `contours` given based on their centers according to a given axis.

    If sort by x axis, the contours will be sorted from left to right.
    If sort by y axis, the contours will be sorted from top to bottom.
    """
    if by_x_axis:
        return sorted(contours, key=get_contour_center_x_position)

    return sorted(contours, key=get_contour_center_y_position)
