"""This module contains functions to find the wire connector."""
from typing import Tuple

import cv2
import numpy as np

from ..utils import contour_utils


def remove_wires_from_treshold(thresh_img: np.ndarray) -> np.ndarray:
    """Remove the wires from the wire connector in the `thresh_img`.

    This functions takes advantage that wire connector is much larger than the wires,
    so it is possible to perform erosion to remove the wires then dilate back to get
    connector shape.

    The shape of the wire connector in the returned threshold image will be distorted
    partially due to erosion and dilation.
    """
    # Set the kernel and iterations for morphological transformations
    kernel = np.ones((5, 5), np.uint8)
    iterations = (
        10  # Should be enough to erode the section where wires intersect connector
    )

    # Erode the thresh_img
    erode_thresh = cv2.erode(thresh_img, kernel, iterations=iterations)

    # Filter for only wire connector (assume is largest contour)
    connector_thresh = contour_utils.filter_for_largest_contour(erode_thresh, 10000)

    # Dilate back to get connector shape
    return cv2.dilate(connector_thresh, kernel, iterations=iterations)


def fill_wire_connector_holes(thresh_img: np.ndarray) -> np.ndarray:
    """Perform closing morphological transformation to fill in possible holes inside the
    wire connector.
    """
    # Set the kernel and iterations for close morphological transformation
    kernel = np.ones((5, 5), np.uint8)
    iterations = 5

    # Perform close
    return cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel, iterations=iterations)


def find_connector_contour(thresh_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Find the wire connector contour. This function assumes that
    the connector is the largest contour found in the `thresh_img`.

    This is the main function used to find the connector contour.

    Returns:
        - connector_contour: The contour of the wire connector.
        - connector_thresh: The threshold image containing only the
            wire connector.
    """
    connector_thresh = remove_wires_from_treshold(thresh_img)
    connector_thresh = fill_wire_connector_holes(connector_thresh)

    # Find contours
    contours, _ = cv2.findContours(
        connector_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Should only have one contour (connector), but find max for safety
    connector_contour = max(contours, key=cv2.contourArea)

    return connector_contour, connector_thresh
