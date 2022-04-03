"""This module contains the class to the initialize stage where user can
configure the detector.
"""
from typing import Sequence

import cv2
import numpy as np

from ..tasks import display
from . import (
    DEFAULT_FRAME_HEIGHT,
    DEFAULT_FRAME_WIDTH,
    WINDOW_NAME,
    DetectorConfig,
    Stage,
)
from .keys import ACCEPTABLE_KEYS_FOR_INITIALIZE_STAGE, ESC_KEY, Q_KEY, Y_KEY


class InitializeStage(Stage):
    def __create_display_img(self, frame) -> np.ndarray:
        """Add in title and commands for the text frame."""
        title = "Detector Configuration"
        commands = ["[y] Yes", "[n] No", "[q / ESC] Quit"]
        display_img = display.draw_title_and_command(frame, title, *commands)

        return display_img

    def __wait_for_input(self, display_img: np.ndarray, acceptable_keys: Sequence[int]) -> str:
        """Wait for the user to input any of the acceptable keys."""
        input_key = None
        while input_key not in acceptable_keys:
            cv2.imshow(WINDOW_NAME, display_img)
            input_key = cv2.waitKey(0)

        return input_key

    def __get_connector_height_greater_than_width_config(self) -> bool:
        """Get the `is_connector_height_greater_than_width` config from user input."""
        text = "Is wire connector height greater than width?"
        text_frame = display.get_text_only_frame((DEFAULT_FRAME_WIDTH, DEFAULT_FRAME_HEIGHT), text)
        display_img = self.__create_display_img(text_frame)

        input_key = self.__wait_for_input(display_img, ACCEPTABLE_KEYS_FOR_INITIALIZE_STAGE)

        if input_key in (ESC_KEY, Q_KEY):
            self.quit()

        if input_key == Y_KEY:
            return True

        return False

    def __get_do_threshold_with_bg_config(self) -> bool:
        """Get the `do_threshold_with_bg` config from user input."""
        text = (
            "Perform thresholding using background image?"
            " \\n Note: Thresholding with background image will likely be more accurate."
        )
        text_frame = display.get_text_only_frame((DEFAULT_FRAME_WIDTH, DEFAULT_FRAME_HEIGHT), text)
        display_img = self.__create_display_img(text_frame)

        input_key = self.__wait_for_input(display_img, ACCEPTABLE_KEYS_FOR_INITIALIZE_STAGE)

        if input_key in (ESC_KEY, Q_KEY):
            self.quit()

        if input_key == Y_KEY:
            return True

        return False

    def run(self) -> None:
        """The main entry point of this stage."""
        is_connector_height_greater_than_width = self.__get_connector_height_greater_than_width_config()
        do_threshold_with_bg = self.__get_do_threshold_with_bg_config()

        detector_config = DetectorConfig(is_connector_height_greater_than_width, do_threshold_with_bg)

        print(detector_config)
