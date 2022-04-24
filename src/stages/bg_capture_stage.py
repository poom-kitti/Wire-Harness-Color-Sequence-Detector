"""This module contains the class to the background capture stage where user can
capture the background image.
"""
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from ..tasks import display
from . import WINDOW_NAME, BaseStageConfig, Stage, UserSettingConfig
from .keys import (
    ACCEPTABLE_KEYS_FOR_ACCEPT_CAPTURED_BG_FRAME,
    DEFAULT_WAIT_KEY_TIME,
    ENTER_KEY,
    QUIT_KEYS,
    Y_KEY,
)
from .reference_capture_stage import ReferenceCaptureStage, ReferenceCaptureStageConfig


@dataclass
class BgCaptureStageConfig(BaseStageConfig):
    pass


class BgCaptureStage(Stage):
    _has_chosen_bg_img: bool

    def __initialize(self):
        """Initialize the variables for the stage."""
        self._has_chosen_bg_img = False

    def __create_capture_bg_display_img(self, frame: np.ndarray) -> np.ndarray:
        """Add in title and commands promting user to capture bg img."""
        title = "Capture Background Image"
        commands = ["[ENTER] Capture", "[q / ESC] Quit"]
        display_img = display.draw_title_and_command(frame, title, *commands)

        return display_img

    def __create_accept_bg_display_img(self, capture_bg_frame: np.ndarray) -> np.ndarray:
        """Add in title and commands promting user whether to accept the captured
        background image.
        """
        title = "Accept Captured Background Image?"
        commands = ["[y] Yes", "[n] No", "[q / ESC] Quit"]
        display_img = display.draw_title_and_command(capture_bg_frame, title, *commands)

        return display_img

    def __get_capture_frame(self) -> np.ndarray:
        """Get the user captured frame.

        User can capture a frame by pressing the `ENTER` key.
        """
        while True:
            frame = self._camera.read()
            display_img = self.__create_capture_bg_display_img(frame)

            cv2.imshow(WINDOW_NAME, display_img)
            input_key = cv2.waitKey(DEFAULT_WAIT_KEY_TIME)

            if input_key in QUIT_KEYS:
                self.quit()

            if input_key == ENTER_KEY:
                break

        return frame

    def __get_accept_captured_frame(self, captured_bg_frame) -> bool:
        """Get whether the user accepts the captured frame as valid background image."""
        display_img = self.__create_accept_bg_display_img(captured_bg_frame)

        input_key = None
        while input_key not in ACCEPTABLE_KEYS_FOR_ACCEPT_CAPTURED_BG_FRAME:
            cv2.imshow(WINDOW_NAME, display_img)
            input_key = cv2.waitKey(0)

        if input_key in QUIT_KEYS:
            self.quit()

        if input_key == Y_KEY:
            return True

        return False

    def __run_reference_capture_stage(self, bg_img: Optional[np.ndarray] = None) -> None:
        """Run the next stage, which is background capture stage."""
        reference_capture_stage = ReferenceCaptureStage()
        reference_capture_stage_config = ReferenceCaptureStageConfig(self._camera, bg_img)

        reference_capture_stage.set_config(reference_capture_stage_config, self._user_config)

        reference_capture_stage.run()

    def set_config(self, stage_config: BgCaptureStageConfig, user_config: UserSettingConfig) -> None:
        self._camera = stage_config.camera
        self._user_config = user_config

    def run(self) -> None:
        """The main entry point of this stage."""
        # Go to next stage if does not threshold with background
        self.__initialize()

        # Capture background
        captured_frame = None
        while self._user_config.do_threshold_with_bg and not self._has_chosen_bg_img:
            captured_frame = self.__get_capture_frame()
            self._has_chosen_bg_img = self.__get_accept_captured_frame(captured_frame)

        # Run next stage
        self.__run_reference_capture_stage(captured_frame)
