"""This module contains functions for displaying to user."""
from typing import Tuple

import cv2
import numpy as np

TEXT_WHITE_COLOR = (247, 252, 251)
TEXT_BLACK_COLOR = (20, 20, 20)


def get_text_only_frame(frame_size: Tuple[int, int], text: str) -> np.ndarray:
    """Get a frame with white background and text written.

    The text will appear in multiple lines if it is longer than the number
    characters a single line can hold. To manually move to new line, use `"\\n"`.

    Example:
    text = "This is a line. \\n This is another line."

    Args:
        frame_size: Size of the frame given in `(width, height)`.
        text: Text to be written on the frame.
    """
    # Prepare frame
    text_frame = np.full((frame_size[1], frame_size[0], 3), 255, dtype=np.uint8)

    # Break long text to multiple lines
    words = text.split()
    characters_per_line = 45
    text_y_position = 50  # Starting position of first line

    line_words = []
    line_char_length = 0

    for word in words:
        # Write line if length is sufficient
        if line_char_length + len(word) > characters_per_line or word == "\\n":
            line_text = " ".join(line_words)
            text_frame = cv2.putText(
                text_frame, line_text, (10, text_y_position), cv2.FONT_HERSHEY_SIMPLEX, 1.5, TEXT_BLACK_COLOR, 2
            )

            line_words = []
            line_char_length = 0
            text_y_position += 60

        if word == "\\n":
            continue

        line_words.append(word)
        line_char_length += len(word) + 1  # Additional char for space

    # Write any leftover words
    text_frame = cv2.putText(
        text_frame, " ".join(line_words), (10, text_y_position), cv2.FONT_HERSHEY_SIMPLEX, 1.5, TEXT_BLACK_COLOR, 2
    )

    return text_frame


def draw_color_sequence_result(display_img: np.ndarray, is_same_sequence: bool) -> np.ndarray:
    """Draw a rectangle on upper right corner to display whether the color sequence
    of the wire assy is the same as the reference.

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
    display_img = cv2.putText(display_img, text, text_point, cv2.FONT_HERSHEY_DUPLEX, 3, TEXT_WHITE_COLOR, 2)

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
    display_img = cv2.putText(display_img, title, title_point, cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_WHITE_COLOR, 1)

    # Add command text
    height = display_img.shape[0]
    text_point = (25, height - 18)
    command_text = "     ".join(commands)
    display_img = cv2.putText(display_img, command_text, text_point, cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_WHITE_COLOR, 1)

    return display_img
