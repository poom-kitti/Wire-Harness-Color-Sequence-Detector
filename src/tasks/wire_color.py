"""This module contains functions to find the wire HSV color."""
from typing import Tuple

import cv2
import numpy as np


def find_wire_hsv_color(wire_img: np.ndarray) -> Tuple[float, float, float]:
    """Find the representative HSV value for the wire color.

    The representative HSV value is the median of the mean HSV values of the bins,
    where a bin is taken as one pixel column in the `wire_img`.
    """
    # Convert the wire image to hsv color space
    wire_hsv: np.ndarray = cv2.cvtColor(wire_img, cv2.COLOR_BGR2HSV)

    # Initiate lists to store the mean hsv values of all the bins
    h_bin_vals = []
    s_bin_vals = []
    v_bin_vals = []

    # Iterate through each bin (1 px column) in the wire image
    for bin in range(wire_hsv.shape[1]):
        # Find the average hsv value of the bin
        bin_avg_h_val = np.mean(wire_hsv[bin, :, 0])
        bin_avg_s_val = np.mean(wire_hsv[bin, :, 1])
        bin_avg_v_val = np.mean(wire_hsv[bin, :, 2])

        # Add the hsv value to the bin values lists
        h_bin_vals.append(bin_avg_h_val)
        s_bin_vals.append(bin_avg_s_val)
        v_bin_vals.append(bin_avg_v_val)

    # Calculate the median of the mean hsv values to get representative hsv value
    representative_h_val = np.median(h_bin_vals)
    representative_s_val = np.median(s_bin_vals)
    representative_v_val = np.median(v_bin_vals)

    return representative_h_val, representative_s_val, representative_v_val
