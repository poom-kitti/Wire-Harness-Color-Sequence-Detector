"""This module contains functions to find the wire LAB color."""
from typing import List, Tuple

import cv2
import numpy as np
from colormath import color_diff
from colormath.color_objects import LabColor

# LAB representation of white color
WHITE_LAB_COLOR = cv2.cvtColor(np.full((1, 1, 3), 255, dtype=np.uint8), cv2.COLOR_BGR2LAB)[0, 0, :]

DEFAULT_ACCEPTABLE_DELTA_E_THRESHOLD = 5  # Default threshold to consider two colors equal


def find_wire_lab_color(wire_img: np.ndarray) -> Tuple[float, float, float]:
    """Find the representative LAB value for the wire color.

    The representative LAB value is the median of the mean LAB values of the bins,
    where a bin is taken as one pixel column in the `wire_img`.
    """
    # Convert the wire image to LAB color space
    wire_lab: np.ndarray = cv2.cvtColor(wire_img, cv2.COLOR_BGR2LAB)

    # Change wire image value type to float to be able to use NaN
    wire_lab = wire_lab.astype(np.float16)

    # Fill in the background color (complete white) as NaN value
    wire_lab[wire_lab == WHITE_LAB_COLOR] = np.nan

    # Initiate lists to store the mean LAB values of all the bins
    l_bin_vals = []
    a_bin_vals = []
    b_bin_vals = []

    # Iterate through each bin (1 px column) in the wire image
    for bin in range(wire_lab.shape[1]):
        # Find the average LAB value of the bin
        bin_avg_l_val = np.nanmean(wire_lab[:, bin, 0])
        bin_avg_a_val = np.nanmean(wire_lab[:, bin, 1])
        bin_avg_b_val = np.nanmean(wire_lab[:, bin, 2])

        # Add the LAB value to the bin values lists
        l_bin_vals.append(bin_avg_l_val)
        a_bin_vals.append(bin_avg_a_val)
        b_bin_vals.append(bin_avg_b_val)

    # Calculate the median of the mean LAB values to get representative LAB value
    representative_l_val = np.nanmedian(l_bin_vals)
    representative_a_val = np.nanmedian(a_bin_vals)
    representative_b_val = np.nanmedian(b_bin_vals)

    return representative_l_val, representative_a_val, representative_b_val


def is_same_wire_color(
    wire_1_lab: Tuple[float, float, float], wire_2_lab: Tuple[float, float, float], delta_e_threshold: float
) -> bool:
    """Checks whether the first wire has same color as the second wire.

    This function assumes the colors of the wires are the same if the delta E 2000 between the two LAB colors
    is below the given `delta_e_threshold`.
    """
    # Convert from opencv LAB to actual LAB color
    # Reference: https://docs.opencv.org/4.5.5/de/d25/imgproc_color_conversions.html
    wire_1_lab_color = LabColor(lab_l=wire_1_lab[0] * 100 / 255, lab_a=wire_1_lab[1] - 128, lab_b=wire_1_lab[2] - 128)
    wire_2_lab_color = LabColor(lab_l=wire_2_lab[0] * 100 / 255, lab_a=wire_2_lab[1] - 128, lab_b=wire_2_lab[2] - 128)

    # Calculate delta e
    delta_e_2000 = color_diff.delta_e_cie2000(wire_1_lab_color, wire_2_lab_color)

    # Check if delta e is less than threshold
    return delta_e_2000 <= delta_e_threshold


def is_same_color_sequence(
    ref_wire_housing_colors: List[Tuple[float, float, float]],
    target_wire_housing_colors: List[Tuple[float, float, float]],
    acceptable_delta_e_threshold: float,
) -> bool:
    """Checks whether the color sequence found in `target_wire_housing_colors` is the same as the reference
    `ref_wire_housing_colors`.

    If the number of colors, the actual colors or sequence of colors is different, then the function will
    return False.
    """
    # Checks if the number of colors are the same
    if len(ref_wire_housing_colors) != len(target_wire_housing_colors):
        return False

    # Checks if each color in the sequence is the same
    for i in range(len(ref_wire_housing_colors)):
        if not is_same_wire_color(
            ref_wire_housing_colors[i], target_wire_housing_colors[i], acceptable_delta_e_threshold
        ):
            return False

    # Has same number of colors, and each color in sequence is the same
    return True
