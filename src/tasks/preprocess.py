"""This module contains functions that will perform preprocessing of frame before
finding the connector or wires."""
from dataclasses import dataclass

import cv2
import numpy as np

from ..utils import contour_utils

DEFAULT_LOWEST_V_VALUE = 150  # Default lowest V value in HSV for being considered background
MINIMUM_VALID_CONTOUR_AREA = 2000  # Minimum area to be considered a contour of an object

# Proportion of frame to disregard when searching if wire connector candidate is present
DISREGARD_WIDTH_PERCENTAGE = 0.3
DISREGARD_HEIGHT_PERCENTAGE = 0.3
# Minimum area of contour to consider as possible candidate for wire connector
MINIMUM_VALID_CONNECTOR_AREA = 50000


@dataclass
class CheckSectionPositions:
    """A dataclass to hold information on the section of frame to
    check for presence of wire connector candidate.
    """

    start_x_pos: int
    end_x_pos: int
    start_y_pos: int
    end_y_pos: int


def threshold_with_inRange(frame_hsv: np.ndarray, bg_img_hsv: np.ndarray) -> np.ndarray:
    """Perform thresholing by using cv2.inRange. Since the background is much brighter than
    the wire housing, we will filter for the bright pixels and consider them as background.

    The returned threshold image will have the background appears as black and the wire housing
    appears as white.

    This method may work better than threshold_with_otsu() in certain circumstances, but is
    more prone to error if the background in `bg_img_hsv` is not purely blank (has some
    scratches or particles shown).
    """
    # Finding the lowest v value in bg_img_hsv
    lowest_v_value = int(np.min(bg_img_hsv[:, :, 2]))
    lowest_v_value = max(DEFAULT_LOWEST_V_VALUE, lowest_v_value)

    # Threshold by filtering with only v value
    thresh_img = cv2.inRange(frame_hsv, (0, 0, lowest_v_value), (180, 255, 255))

    # Swap the color such that bg appears black and fg appears white
    thresh_img = cv2.bitwise_not(thresh_img)

    # Filter small contours
    return contour_utils.filter_out_small_contours(thresh_img, MINIMUM_VALID_CONTOUR_AREA)


def threshold_with_otsu(frame: np.ndarray) -> np.ndarray:
    """Perform thresholing by using otsu thresholding.

    The returned threshold image will have the background appears as black and the
    wire housing appears as white.
    """
    # Convert frame to gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply otsu thresholding, ignore otsu threshold calculated
    _, thresh_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Swap the color such that bg appears black and fg appears white
    thresh_img = cv2.bitwise_not(thresh_img)

    # Filter small contours
    return contour_utils.filter_out_small_contours(thresh_img, MINIMUM_VALID_CONTOUR_AREA)


def _get_check_section_positions(threshold_img: np.ndarray) -> CheckSectionPositions:
    """Get the section of the frame to check for the presence of the wire connector
    candidate.
    """
    img_width = threshold_img.shape[1]
    img_height = threshold_img.shape[0]

    start_x = int(img_width * DISREGARD_WIDTH_PERCENTAGE / 2)
    end_x = img_width - int(img_width * DISREGARD_WIDTH_PERCENTAGE / 2)
    start_y = int(img_height * DISREGARD_HEIGHT_PERCENTAGE / 2)
    end_y = img_height - int(img_height * DISREGARD_HEIGHT_PERCENTAGE / 2)

    return CheckSectionPositions(start_x, end_x, start_y, end_y)


def has_wire_connector_candidate(threshold_img: np.ndarray) -> bool:
    """Check whether there is an object present in the screen with large enough contour
    to be considered a wire connector.

    This function will only check the area relatively in the middle of the frame
    to ensure that the whole wire connector along with wires are present. This is
    done by disregarding a proportion of the sides and bottom of the frame.
    """
    # Cropping only the section of the frame to check for wire connector
    check_section_positions = _get_check_section_positions(threshold_img)
    check_section = threshold_img[
        check_section_positions.start_y_pos : check_section_positions.end_y_pos,
        check_section_positions.start_x_pos : check_section_positions.end_x_pos,
    ]
    # Find contours
    contours, _ = cv2.findContours(check_section, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Return true if any contour is deemed big enough to be considered a wire connector
    for contour in contours:
        if cv2.contourArea(contour) > MINIMUM_VALID_CONNECTOR_AREA:
            return True

    return False


def get_check_section_display(display_img: np.ndarray) -> np.ndarray:
    """Highlights the section of frame where wire connector candidate is checked.

    This function expects the display image to have the same size as the threshold image
    used to check if has wire connector candidate.
    """
    check_section_positions = _get_check_section_positions(display_img)

    # Configure the rectangle
    rect_color = (100, 117, 239)  # Light red
    top_left_point = (check_section_positions.start_x_pos, check_section_positions.start_y_pos)
    bottom_right_point = (check_section_positions.end_x_pos, check_section_positions.end_y_pos)

    return cv2.rectangle(display_img, top_left_point, bottom_right_point, rect_color, 1)


def fill_bg_as_white(frame: np.ndarray, threshold_img: np.ndarray) -> np.ndarray:
    """Fill the bg of the frame as white color. Everything that appears black
    in the `threshold_img` will be considered as bg."""
    # Create white bg
    white_bg = np.full(frame.shape, 255, dtype=np.uint8)

    # Only show the paste the fg (white area identified by threshold_img) to white_bg
    return cv2.bitwise_and(frame, frame, white_bg, mask=threshold_img)
