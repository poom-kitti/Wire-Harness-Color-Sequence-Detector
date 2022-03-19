"""This module contains functions to find the region of interest where the wires
should be. The ROI is taken as an area beneath the wire connector."""
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from ..utils import frame_utils, rect_utils


@dataclass
class WireRoiConfig:
    roi_width: int
    roi_height: int
    roi_center_distance_from_connector_btm: int


def get_roi_center(
    connector_rect: Tuple,
    connector_height: float,
    roi_center_distance_from_connector_btm: int,
) -> Tuple[int, int]:
    """Find the center of the wire region of interest based on the center 
    of connector min area rectangle and the given configurations of roi."""
    # Find bottom point of connector
    connector_center = connector_rect[0]
    bottom_of_connector = (
        connector_center[0],
        int(connector_center[1] + connector_height / 2),
    )

    # Get roi center
    return (
        bottom_of_connector[0],
        bottom_of_connector[1] + roi_center_distance_from_connector_btm,
    )


def get_display_image(
    original_frame: np.ndarray,
    connector_rect: Tuple,
    wire_roi_config: WireRoiConfig,
    wire_roi_center: Tuple[int, int],
    rotation_angle: float,
) -> np.ndarray:
    """Get the image to display for the user highlighting the rectangles 
    surrounding the wire connector and wire region of interest."""
    # Create a mask of where the rectangles should be drawn
    # The white section of the mask will be filled with color
    mask = np.zeros(original_frame.shape[:2], np.uint8)

    # Get the corners of the wire connector rectangle
    connector_box: np.ndarray = cv2.boxPoints(connector_rect)
    connector_box = connector_box.astype(np.int32)

    # Draw the rectangle surrounding wire connector to mask
    cv2.drawContours(mask, [connector_box], -1, 255, 5)

    # Rotate the mask for wire roi
    mask = frame_utils.rotate_image(mask, connector_rect[0], rotation_angle)

    # Get the corners of the wire roi rectangle
    wire_roi_rect = (
        wire_roi_center,
        (wire_roi_config.roi_width, wire_roi_config.roi_height),
        0,
    )
    wire_roi_box: np.ndarray = cv2.boxPoints(wire_roi_rect)
    wire_roi_box = wire_roi_box.astype(np.int32)

    # Draw the rectangle surrounding wire roi to mask
    cv2.drawContours(mask, [wire_roi_box], -1, 255, 5)

    # Rotate the mask back to original position
    mask = frame_utils.rotate_image(mask, connector_rect[0], -rotation_angle)

    # Create a display image
    display_img = original_frame.copy()

    # Fill the display image with red color based on where the mask is white (the rectangles)
    display_img[mask == 255] = (114, 128, 250) # Red salmon color

    return display_img


def find_wire_roi(
    ori_frame: np.ndarray,
    frame_white_bg: np.ndarray,
    connector_contour: np.ndarray,
    is_height_greater_than_width: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Find the region of interest where the wires should reside. The ROI is 
    calculated as the area slightly below the wire connector.

    Returns:
        wire_roi: The region of interest holding the wires.
        display_img: The display image to show the user highlighting the rectangles 
            surrounding the wire connector and wire region of interest.
    """
    # Get the wire connector rotated rectangle and the connector size
    connector_rect = cv2.minAreaRect(connector_contour)
    connector_width, connector_height = rect_utils.get_obj_actual_width_and_height(
        connector_rect, is_height_greater_than_width
    )

    # Rotate the frame to straighten the connector
    rotation_angle = rect_utils.get_straigten_rotation_angle(connector_rect, is_height_greater_than_width)
    rotated_frame = frame_utils.rotate_image(frame_white_bg, connector_rect[0], rotation_angle)

    # Configure roi
    wire_roi_config = WireRoiConfig(
        roi_width=int(connector_width),
        roi_height=30,
        roi_center_distance_from_connector_btm=25,
    )

    # Find wire roi center
    wire_roi_center = get_roi_center(
        connector_rect,
        connector_height,
        wire_roi_config.roi_center_distance_from_connector_btm,
    )

    # Get wire roi
    wire_roi = cv2.getRectSubPix(
        rotated_frame,
        (wire_roi_config.roi_width, wire_roi_config.roi_height),
        wire_roi_center,
    )

    # Get display image
    display_img = get_display_image(ori_frame, connector_rect, wire_roi_config, wire_roi_center, rotation_angle)

    return wire_roi, display_img
