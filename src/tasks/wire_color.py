"""This module contains functions to find the wire HSV color."""
from typing import Tuple

import cv2
import numpy as np


def find_wire_lab_color(wire_img: np.ndarray) -> Tuple[float, float, float]:
    """Find the representative LAB value for the wire color.

    The representative LAB value is the median of the mean LAB values of the bins,
    where a bin is taken as one pixel column in the `wire_img`.
    """
    # Convert the wire image to hsv color space
    wire_lab: np.ndarray = cv2.cvtColor(wire_img, cv2.COLOR_BGR2LAB)

    # Initiate lists to store the mean hsv values of all the bins
    l_bin_vals = []
    a_bin_vals = []
    b_bin_vals = []

    # Iterate through each bin (1 px column) in the wire image
    for bin in range(wire_lab.shape[1]):
        # Find the average hsv value of the bin
        bin_avg_l_val = np.mean(wire_lab[bin, :, 0])
        bin_avg_a_val = np.mean(wire_lab[bin, :, 1])
        bin_avg_b_val = np.mean(wire_lab[bin, :, 2])

        # Add the hsv value to the bin values lists
        l_bin_vals.append(bin_avg_l_val)
        a_bin_vals.append(bin_avg_a_val)
        b_bin_vals.append(bin_avg_b_val)

    # Calculate the median of the mean hsv values to get representative hsv value
    representative_l_val = np.median(l_bin_vals)
    representative_a_val = np.median(a_bin_vals)
    representative_b_val = np.median(b_bin_vals)

    return representative_l_val, representative_a_val, representative_b_val
