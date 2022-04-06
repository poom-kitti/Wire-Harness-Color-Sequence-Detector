"""This module contains the class to the initialize stage where user can
configure the detector.
"""
from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np

from ..tasks import display
from . import (
    DEFAULT_FRAME_HEIGHT,
    DEFAULT_FRAME_WIDTH,
    WINDOW_NAME,
    BaseStageConfig,
    Stage,
    UserSettingConfig,
)
from .bg_capture_stage import BgCaptureStage, BgCaptureStageConfig
from .keys import ACCEPTABLE_KEYS_FOR_INITIALIZE_STAGE, QUIT_KEYS, Y_KEY


@dataclass
class InitializeStageConfig(BaseStageConfig):
    pass


class InitializeStage(Stage):
    def __create_display_img(self, frame: np.ndarray) -> np.ndarray:
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

        if input_key in QUIT_KEYS:
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

        if input_key in QUIT_KEYS:
            self.quit()

        if input_key == Y_KEY:
            return True

        return False

    def __run_bg_capture_stage(self, user_config: UserSettingConfig) -> None:
        """Run the next stage, which is background capture stage."""
        bg_capture_stage = BgCaptureStage()
        bg_capture_stage_config = BgCaptureStageConfig(self._camera)

        bg_capture_stage.set_config(bg_capture_stage_config, user_config)

        bg_capture_stage.run()

    def set_config(self, stage_config: InitializeStageConfig, user_config: Optional[UserSettingConfig] = None):
        self._camera = stage_config.camera

    def run(self) -> None:
        """The main entry point of this stage."""
        # Get user inputs
        is_connector_height_greater_than_width = self.__get_connector_height_greater_than_width_config()
        do_threshold_with_bg = self.__get_do_threshold_with_bg_config()

        # Initialize detector configurations
        detector_config = UserSettingConfig(is_connector_height_greater_than_width, do_threshold_with_bg)

        # Run next stage
        self.__run_bg_capture_stage(detector_config)
