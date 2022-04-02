"""This module contains functions for displaying to user."""
import cv2
import numpy as np

TEXT_COLOR = (247, 252, 251)  # White


def draw_color_sequence_result(display_img: np.ndarray, is_same_sequence: bool) -> np.ndarray:
    """Draw a rectangle on upper right corner to display whether the color sequence
    of the wire housing is the same as the reference.

    If the color sequence is the same, `OK` will be written.
    If the color sequence is different, `NG` will be written.
    """
    # Configure rectangle location in frame
    start_point = (1100, 25)
    end_point = (1250, 125)
    color = (41, 92, 0) if is_same_sequence else (19, 25, 122)  # Green or same, else red

    # Draw rectangle
    display_img = cv2.rectangle(display_img, start_point, end_point, color, -1)

    # Configure text
    text = "OK" if is_same_sequence else "NG"
    text_point = (1110, 105)

    # Draw text
    display_img = cv2.putText(display_img, text, text_point, cv2.FONT_HERSHEY_DUPLEX, 3, TEXT_COLOR, 2)

    return display_img


def draw_title_and_command(display_img: np.ndarray, title: str, *commands: str) -> np.ndarray:
    """Draw the title that will be shown at the top and command that will be shown at the bottom
    of the `display_img` with black background and white text.

    Each command will be five spaces apart from each other.
    """
    # Add padding
    background_color = (20, 20, 20)
    display_img = cv2.copyMakeBorder(display_img, 30, 50, 0, 0, cv2.BORDER_CONSTANT, value=background_color)

    # Add title text
    title_point = (15, 20)
    display_img = cv2.putText(display_img, title, title_point, cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1)

    # Add command text
    height = display_img.shape[0]
    text_point = (25, height - 18)
    command_text = "     ".join(commands)
    display_img = cv2.putText(display_img, command_text, text_point, cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_COLOR, 1)

    return display_img
