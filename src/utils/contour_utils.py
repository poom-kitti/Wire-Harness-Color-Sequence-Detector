"""This module contains utility functions that deal with contour and threshold image."""
import cv2
import numpy as np


def filter_out_small_contours(
    thresh_img: np.ndarray, min_contour_area: float
) -> np.ndarray:
    """Filter out contours that are less than the `min_contour_area` by filling the `thresh_img`
    as black where the contours are found.

    Expects `thresh_img` to be a grayscale image with bg as black and fg as white.
    """
    # Make copy of thresh_img
    thresh_img = thresh_img.copy()

    # Find the contours
    contours, _ = cv2.findContours(
        thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Filter out small contours
    contours_to_fill = []
    for contour in contours:
        if cv2.contourArea(contour) < min_contour_area:
            contours_to_fill.append(contour)

    # Fill small contours
    cv2.drawContours(thresh_img, contours_to_fill, -1, 0, -1)

    return thresh_img


def filter_for_largest_contour(
    thresh_img: np.ndarray, min_contour_area: float
) -> np.ndarray:
    """Filter for the largest contour in `thresh_img`. Any contour that is not the largest
    contour will be filled as black.

    The largest contour must be larger than specified `min_contour_area` otherwise
    a simple black image will be returned.

    Expects `thresh_img` to be a grayscale image with bg as black as fg as white.
    """
    # Make a black image
    black_img = np.zeros(thresh_img.shape, np.uint8)

    # Find the contours
    contours, _ = cv2.findContours(
        thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Get the max contour
    max_contour = max(contours, key=cv2.contourArea)

    # Fill the black image with white at the max_contour if greater than min_contour_are
    if cv2.contourArea(max_contour) >= min_contour_area:
        cv2.drawContours(black_img, [max_contour], -1, 255, -1)

    return black_img