"""This module contains functions to find the wires residing in the
wire region of interest."""
from typing import List

import cv2
import numpy as np

from ..utils import contour_utils, frame_utils, rect_utils


def get_wire_from_contour(wire_roi_img: np.ndarray, wire_contour: np.ndarray) -> np.ndarray:
    """Crop out the wire found in `wire_contour` from the `wire_roi_img`.

    This function assumes the height of the section of wire present in the contour
    is greater than its width.
    """
    # Get the rectangle surrounding the wire contour
    wire_rect = cv2.minAreaRect(wire_contour)

    # Get the rotation angle to straigten the wire rectangle
    rotation_angle = rect_utils.get_straigten_rotation_angle(wire_rect, True)

    # Rotate the wire_roi_img
    rotated_img = frame_utils.rotate_image(wire_roi_img, wire_rect[0], rotation_angle)

    # Get the width and height of the section of wire we want to crop
    wire_width, wire_height = rect_utils.get_obj_actual_width_and_height(wire_rect, True)
    wire_width = int(wire_width)
    wire_height = int(wire_height)

    # Crop the section of wire from the rotated image
    return cv2.getRectSubPix(
        rotated_img,
        (wire_width, wire_height),
        wire_rect[0],
    )


def find_wires(wire_roi_img: np.ndarray) -> List[np.ndarray]:
    """Find the wires located inside the wire region of interest."""
    # Add padding to the wire region of interest, so no information will be lost if roi is rotated
    wire_roi_img = cv2.copyMakeBorder(wire_roi_img, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    # Threshold the roi using information that the background is purely white
    wire_roi_gray = cv2.cvtColor(wire_roi_img, cv2.COLOR_BGR2GRAY)
    _, wire_roi_thresh = cv2.threshold(wire_roi_gray, 254, 255, cv2.THRESH_BINARY_INV)

    # Set up configuration for erosion
    kernel = np.ones((3, 3), np.uint8)
    iterations = 3

    # Perform erosion to reduce probability that wires will touch each other
    # Erosion also reduces the roi of each wire, ensuring we are inspecting the wire and not its edge
    erode_thresh = cv2.erode(wire_roi_thresh, kernel, iterations=iterations)

    # Filter out contours that are likely not wires
    wires_thresh = contour_utils.filter_out_small_contours(erode_thresh, 200)

    # Find the contours
    contours, _ = cv2.findContours(wires_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort the contour based on its x axis
    contours = contour_utils.sort_contours_by_axis(contours, True)

    # Crop a section of wire from each wire contour
    return [get_wire_from_contour(wire_roi_img, contour) for contour in contours]
