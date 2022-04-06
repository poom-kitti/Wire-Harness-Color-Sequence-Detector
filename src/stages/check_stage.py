"""This module contains the class to the check stage where user can
check the wire housing against the reference image.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

import src.stages.initialize_stage as initialize
import src.stages.reference_capture_stage as reference_capture

from ..tasks import connector, display, preprocess, wire_color, wire_roi, wires
from . import WINDOW_NAME, BaseStageConfig, Stage, UserSettingConfig
from .keys import (
    ACCEPTABLE_KEYS_FOR_CHECK_CAPTURE_STAGE,
    DEFAULT_WAIT_KEY_TIME,
    N_KEY,
    QUIT_KEYS,
    R_KEY,
)

RESULT_SHOWN_TIME = 1250  # ms


@dataclass
class CheckCaptureStageConfig(BaseStageConfig):
    bg_img: Optional[np.ndarray]
    reference_color_sequence: List[Tuple]


class CheckCaptureStage(Stage):
    _bg_img: Optional[np.ndarray]
    _reference_color_sequence: List[Tuple]

    def __create_display_img(self, display_frmae: np.ndarray) -> np.ndarray:
        """Add in title and commands to the frame to be displayed."""
        title = "Check Color Sequence"
        commands = ["[r] Retake Reference", "[n] New Configs", "[q / ESC] Quit"]
        display_img = display.draw_title_and_command(display_frmae, title, *commands)

        return display_img

    def __get_frame_threshold(self, frame_blur: np.ndarray) -> np.ndarray:
        """Perform thresholding on the frame with the technique of thresholding
        depending on the user setting.

        User setting:
            - do_threshold_with_bg is True -> Perform inRange threshold (HSV)
            - do_threshold_with_bg is False -> Perform Otsu threshold
        """
        if self._user_config.do_threshold_with_bg:
            # Prepare background img
            bg_img = cv2.GaussianBlur(self._bg_img, (5, 5), 0)
            bg_img_hsv = cv2.cvtColor(bg_img, cv2.COLOR_BGR2HSV)

            frame_hsv: np.ndarray = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2HSV)

            frame_threshold = preprocess.threshold_with_inRange(frame_hsv, bg_img_hsv)
        else:
            frame_threshold = preprocess.threshold_with_otsu(frame_blur)

        return frame_threshold

    def __check_color_sequence(self):
        """Get the captured wire housing color sequence against the reference."""
        input_key = None
        while input_key not in ACCEPTABLE_KEYS_FOR_CHECK_CAPTURE_STAGE:
            frame = self._camera.read()
            frame_blur: np.ndarray = cv2.GaussianBlur(frame, (5, 5), 0)
            frame_threshold = self.__get_frame_threshold(frame_blur)

            # Check middle section of the frame to see if a wire connector candidate is present
            has_wire_connector_candidate = preprocess.has_wire_connector_candidate(frame_threshold)

            # Perform checking the color sequence if has wire connector candidate
            if has_wire_connector_candidate:
                # Fill bg as white
                frame_white_bg = preprocess.fill_bg_as_white(frame_blur, frame_threshold)

                # Find connector contour
                connector_contour = connector.find_connector_contour(frame_threshold)

                wire_roi_img, display_frame = wire_roi.find_wire_roi(
                    frame, frame_white_bg, connector_contour, self._user_config.is_connector_height_greater_than_width
                )

                cropped_wires = wires.find_wires(wire_roi_img, do_display_wires_thresh=True)
                captured_color_sequence = [wire_color.find_wire_lab_color(wire) for wire in cropped_wires]

                is_same_color_sequence_as_reference = wire_color.is_same_color_sequence(
                    self._reference_color_sequence,
                    captured_color_sequence,
                    wire_color.DEFAULT_ACCEPTABLE_DELTA_E_THRESHOLD,
                )
                display_frame = display.draw_color_sequence_result(display_frame, is_same_color_sequence_as_reference)
            else:
                display_frame = preprocess.get_check_section_display(frame.copy())

            display_img = self.__create_display_img(display_frame)
            # Choose frame display time depending on whether wire connector candidate is found
            wait_key_time = RESULT_SHOWN_TIME if has_wire_connector_candidate else DEFAULT_WAIT_KEY_TIME

            cv2.imshow(WINDOW_NAME, display_img)
            input_key = cv2.waitKey(wait_key_time)

        # Choose stage to transition to based on user input
        if input_key in QUIT_KEYS:
            self.quit()

        # Destroy the window showing the wires threshold of found wire housing
        cv2.destroyWindow(wires.DISPLAY_WIRES_WINDOW_NAME)

        if input_key == R_KEY:
            reference_capture_stage = reference_capture.ReferenceCaptureStage()
            reference_capture_stage.run()

        if input_key == N_KEY:
            initialize_stage = initialize.InitializeStage()
            initialize_stage.run()

    def set_config(self, stage_config: CheckCaptureStageConfig, user_config: UserSettingConfig) -> None:
        self._camera = stage_config.camera
        self._bg_img = stage_config.bg_img
        self._reference_color_sequence = stage_config.reference_color_sequence
        self._user_config = user_config

    def run(self) -> None:
        self.__check_color_sequence()
