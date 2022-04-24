"""This module contains functions to find the wires residing in the
wire region of interest."""
from typing import List

import cv2
import numpy as np

from ..utils import contour_utils, frame_utils, rect_utils

DISPLAY_WIRES_WINDOW_NAME = "wires"
MINIMUM_VALID_WIRE_CONTOUR_AFTER_ERODE = 100

# The pixel distance from background in proportion to the max pixel distance
# to consider that pixel as foreground.
# E.g., If max pixel distance from background is 10 px, then the pixel distance
#       from background must be  >= WIRE_SURE_FG_DISTANCE_PROPORTION * 10 for that
#       pixel to be considered a foreground.
WIRE_SURE_FG_DISTANCE_PROPORTION = 0.7


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


def get_watershed_labels(wire_roi_img: np.ndarray, wire_roi_thresh: np.ndarray) -> np.ndarray:
    """Get the watershed labeling separating the background and each foreground object.

    Logic is adapted from https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html
    """
    # Set up configuration and noise removal
    kernel = np.ones((3, 3), np.uint8)
    wire_roi_thresh_noise_remove = cv2.morphologyEx(wire_roi_thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Expand the foreground in order to get the sure bg as black
    sure_bg = cv2.dilate(wire_roi_thresh_noise_remove, kernel)

    # Find the sure foreground area
    distance_transform: np.ndarray = cv2.distanceTransform(wire_roi_thresh_noise_remove, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(distance_transform, WIRE_SURE_FG_DISTANCE_PROPORTION * distance_transform.max(), 255, 0)
    sure_fg: np.ndarray = sure_fg.astype(np.uint8)

    # Find unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Watershed markers labelling
    _, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Mark the region of unknown with zero
    markers[unknown == 255] = 0

    # Apply watershed
    watershed_labels: np.ndarray = cv2.watershed(wire_roi_img, markers)

    return watershed_labels


def find_wires(wire_roi_img: np.ndarray, do_display_wires_thresh: bool = False) -> List[np.ndarray]:
    """Find the wires located inside the wire region of interest."""
    # Add padding to the wire region of interest, so no information will be lost if roi is rotated
    wire_roi_img = cv2.copyMakeBorder(wire_roi_img, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    # Threshold the roi using information that the background is purely white
    wire_roi_gray = cv2.cvtColor(wire_roi_img, cv2.COLOR_BGR2GRAY)
    _, wire_roi_thresh = cv2.threshold(wire_roi_gray, 254, 255, cv2.THRESH_BINARY_INV)

    watershed_labels = get_watershed_labels(wire_roi_img, wire_roi_thresh)

    # Initiate list to keep found wire contours
    wire_contours = []

    # Initiate display wire threshold got from performing watershed
    display_wire_thresh = np.zeros(watershed_labels.shape, dtype=np.uint8)

    # Loop over the unique label returned by watershed algorithm
    for label in np.unique(watershed_labels):
        # Bg will have label 1, while outline have label -1, so we ignore all labels below 1
        if label <= 1:
            continue

        # Create a mask and draw the found wire on it
        mask = np.zeros(watershed_labels.shape, dtype=np.uint8)
        mask[watershed_labels == label] = 255

        # Draw the found wire for display
        display_wire_thresh[watershed_labels == label] = 215 if label % 2 == 0 else 125

        # Detect the contours in the mask and get the largest contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        wire_contour = max(contours, key=cv2.contourArea)

        wire_contours.append(wire_contour)

    # Perform display if needed
    if do_display_wires_thresh:
        cv2.imshow(DISPLAY_WIRES_WINDOW_NAME, display_wire_thresh)

    wire_contours = contour_utils.sort_contours_by_axis(wire_contours, True)

    return [get_wire_from_contour(wire_roi_img, contour) for contour in wire_contours]
