"""This module contains utility function that handles logic regarding rectangle (minAreaRect) 
and box."""
from typing import Tuple

import cv2
import numpy as np


def get_straigten_rotation_angle(min_area_rect: Tuple, is_height_longer_than_width: bool) -> float:
    """Get the rotation angle to rotate the image such that the rectangle defined by
    `min_area_rect` will be straigthen.

    The angle returned should be used with cv2.getRotationMatrix2D() where positive
    angle means to rotate counter-clockwise.

    Reference for understanding angle from cv2.minAreaRect():
    https://namkeenman.wordpress.com/2015/12/18/open-cv-determine-angle-of-rotatedrect-minarearect/
        - Width is distance between 0th & 1st vertices. And height is distance between 1st & 2nd vertices.
        - Angle is calculated from the horizontal to the first edge of rectangle, clockwise.

    Args:
        min_area_rect: The returned result of cv2.minAreaRect() of a contour.
        is_height_longer_than_width: Whether the height of the object should be longer than the width
            in real life.
    """
    # Angle from min_area_rect is from 0 to 90.
    angle = min_area_rect[2]

    # If angle is 0 or 90, it is already straight
    if angle == 0 or angle == 90:
        return 0

    # Get the corners of the min_are_rect
    box: np.ndarray = cv2.boxPoints(min_area_rect)
    box = box.astype(np.int32)

    # Sort the corners by y coordinates order by most bottom first
    corners_sorted_by_y = sorted(box, key=lambda corner: corner[1], reverse=True)

    # Get two most bottom corners
    lowest_corner = corners_sorted_by_y[0]
    second_lowest_corner = corners_sorted_by_y[1]

    # Is the most bottom corner left of second most bottom corner
    is_most_btm_corner_on_left = lowest_corner[0] < second_lowest_corner[0]

    # Distance between the most bottom corners
    most_btm_corners_distance = np.linalg.norm(lowest_corner - second_lowest_corner)

    # Find whether rectangle is tilted left or right
    if is_height_longer_than_width:
        if is_most_btm_corner_on_left:
            is_tilted_left = True if min_area_rect[1][0] > most_btm_corners_distance else False
        else:
            is_tilted_left = True if min_area_rect[1][1] < most_btm_corners_distance else False
    else:
        if is_most_btm_corner_on_left:
            is_tilted_left = True if min_area_rect[1][0] < most_btm_corners_distance else False
        else:
            is_tilted_left = True if min_area_rect[1][1] > most_btm_corners_distance else False

    # Assume rotation is not greater than maximum (handle relatively square wire connector)
    angle_to_tilt = angle
    if is_tilted_left:
        angle_to_tilt = angle - 90
    else:
        angle_to_tilt = angle

    return angle_to_tilt


def get_obj_actual_width_and_height(min_area_rect: Tuple, is_height_longer_than_width: bool) -> Tuple[float, float]:
    """Get the width and height of the object inside the `min_area_rect`.

    The width and height calculated from cv2.minAreaRect() is based on the bottom most corner, so
    it may not reflect the actual width and height as desired. This function will give the correct
    width and height based on whether the user consider height is longer than width dimensions in
    real life or not.
    """
    # Get size reported by cv2.minAreaRect()
    size = min_area_rect[1]

    # Find the width and height
    if is_height_longer_than_width:
        width = min(size)
        height = max(size)
    else:
        width = max(size)
        height = min(size)

    return width, height
