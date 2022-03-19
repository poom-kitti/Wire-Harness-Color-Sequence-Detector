"""This module contains functions that will perform preprocessing of frame before
finding the connector or wires."""
import cv2
import numpy as np

from ..utils import contour_utils


def threshold_with_inRange(frame_hsv: np.ndarray, bg_img_hsv: np.ndarray) -> np.ndarray:
    """Perform thresholing by using cv2.inRange. Since the background is much brighter than
    the wire harness, we will filter for the bright pixels and consider them as background.

    The returned threshold image will have the background appears as black and the wire harness
    appears as white.

    This method may work better than threshold_with_otsu() in certain circumstances, but is
    more prone to error if the background in `bg_img_hsv` is not purely blank (has some
    scratches or particles shown).
    """
    # Finding the lowest v value in bg_img_hsv
    lowest_v_value = int(np.min(bg_img_hsv[:, :, 2]))
    lowest_v_value = max(150, lowest_v_value)

    # Threshold by filtering with only v value
    thresh_img = cv2.inRange(frame_hsv, (0, 0, lowest_v_value), (180, 255, 255))

    # Swap the color such that bg appears black and fg appears white
    thresh_img = cv2.bitwise_not(thresh_img)

    # Filter small contours
    return contour_utils.filter_out_small_contours(thresh_img, 2000)


def threshold_with_otsu(frame: np.ndarray) -> np.ndarray:
    """Perform thresholing by using otsu thresholding.

    The returned threshold image will have the background appears as black and the 
    wire harness appears as white.
    """
    # Convert frame to gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply otsu thresholding, ignore otsu threshold calculated
    _, thresh_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Swap the color such that bg appears black and fg appears white
    thresh_img = cv2.bitwise_not(thresh_img)

    # Filter small contours
    return contour_utils.filter_out_small_contours(thresh_img, 2000)


def fill_bg_as_white(frame: np.ndarray, threshold_img: np.ndarray) -> np.ndarray:
    """Fill the bg of the frame as white color. Everything that appears black
    in the `threshold_img` will be considered as bg."""
    # Create white bg
    white_bg = np.full(frame.shape, 255, dtype=np.uint8)

    # Only show the paste the fg (white area identified by threshold_img) to white_bg
    return cv2.bitwise_and(frame, frame, white_bg, mask=threshold_img)
