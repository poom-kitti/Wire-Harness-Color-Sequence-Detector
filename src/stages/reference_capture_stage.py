"""This module contains the class to the reference capture stage where user can
capture the reference image.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..tasks import connector, display, preprocess, wire_color, wire_roi, wires
from . import WINDOW_NAME, BaseStageConfig, Stage, UserSettingConfig
from .check_stage import CheckCaptureStage, CheckCaptureStageConfig
from .keys import (
    ACCEPTABLE_KEYS_FOR_ACCEPT_CAPTURED_REFERENCE_WIRE_HOUSING,
    DEFAULT_WAIT_KEY_TIME,
    QUIT_KEYS,
    Y_KEY,
)


@dataclass
class ReferenceCaptureStageConfig(BaseStageConfig):
    bg_img: Optional[np.ndarray]


class ReferenceCaptureStage(Stage):
    _bg_img: Optional[np.ndarray]
    _has_reference_color_sequence: bool

    def __initialize(self):
        """Initialize the variables for the stage."""
        self._has_reference_color_sequence = False

    def __create_capture_reference_display_img(self, frame: np.ndarray) -> np.ndarray:
        """Add in title and commands telling user to insert reference wire assy."""
        title = "Insert Reference Wire Assy"
        commands = ["[q / ESC] Quit"]
        display_img = display.draw_title_and_command(frame, title, *commands)

        return display_img

    def __create_accept_reference_display_img(self, captured_wire_housing_frame: np.ndarray) -> np.ndarray:
        """Add in title and commands promting user whether to accept the captured
        wire assy frame.
        """
        title = "Accept Captured Reference Wire Assy?"
        commands = ["[y] Yes", "[n] No", "[q / ESC] Quit"]
        display_img = display.draw_title_and_command(captured_wire_housing_frame, title, *commands)

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

    def __get_reference_color_sequence(self) -> Tuple[List[Tuple], np.ndarray]:
        """Get the color sequence of the reference wire assy.

        User will be prompted to inser the wire assy.
        """
        while True:
            frame = self._camera.read()
            frame_blur: np.ndarray = cv2.GaussianBlur(frame, (5, 5), 0)
            frame_threshold = self.__get_frame_threshold(frame_blur)

            # Check middle section of the frame to see if a wire connector candidate is present
            has_wire_connector_candidate = preprocess.has_wire_connector_candidate(frame_threshold)

            # Perform finding the color sequence if has wire connector candidate
            if has_wire_connector_candidate:
                # Fill bg as white
                frame_white_bg = preprocess.fill_bg_as_white(frame_blur, frame_threshold)

                # Find connector contour
                connector_contour = connector.find_connector_contour(frame_threshold)

                wire_roi_img, display_frame = wire_roi.find_wire_roi(
                    frame, frame_white_bg, connector_contour, self._user_config.is_connector_height_greater_than_width
                )

                cropped_wires = wires.find_wires(wire_roi_img, do_display_wires_thresh=True)

                reference_color_sequence = [wire_color.find_wire_lab_color(wire) for wire in cropped_wires]

                break

            # Show current frame captured if no wire connector candidate found
            display_frame = preprocess.get_check_section_display(frame.copy())
            display_img = self.__create_capture_reference_display_img(display_frame)
            cv2.imshow(WINDOW_NAME, display_img)

            input_key = cv2.waitKey(DEFAULT_WAIT_KEY_TIME)

            if input_key in QUIT_KEYS:
                self.quit()

        return reference_color_sequence, display_frame

    def __get_accept_reference_color_sequence(self, captured_wire_housing_frame: np.ndarray) -> bool:
        """Get whether the user accepts the captured reference wire assy."""
        display_img = self.__create_accept_reference_display_img(captured_wire_housing_frame)

        input_key = None
        while input_key not in ACCEPTABLE_KEYS_FOR_ACCEPT_CAPTURED_REFERENCE_WIRE_HOUSING:
            cv2.imshow(WINDOW_NAME, display_img)
            input_key = cv2.waitKey(0)

        if input_key in QUIT_KEYS:
            self.quit()

        if input_key == Y_KEY:
            return True

        return False

    def __run_check_capture_stage(self, reference_color_sequence: List[Tuple]) -> None:
        """Run the next stage, which is background capture stage."""
        # Destroy the window showing the wires threshold of found wire assy
        cv2.destroyWindow(wires.DISPLAY_WIRES_WINDOW_NAME)

        check_capture_stage = CheckCaptureStage()
        check_capture_stage_config = CheckCaptureStageConfig(self._camera, self._bg_img, reference_color_sequence)

        check_capture_stage.set_config(check_capture_stage_config, self._user_config)

        check_capture_stage.run()

    def set_config(self, stage_config: ReferenceCaptureStageConfig, user_config: UserSettingConfig):
        self._camera = stage_config.camera
        self._bg_img = stage_config.bg_img
        self._user_config = user_config

    def run(self):
        self.__initialize()

        # Capture wire assy reference color sequence
        reference_color_sequence = None
        while not self._has_reference_color_sequence:
            reference_color_sequence, captured_wire_housing_frame = self.__get_reference_color_sequence()
            self._has_reference_color_sequence = self.__get_accept_reference_color_sequence(captured_wire_housing_frame)

        # Run next stage
        if not reference_color_sequence:
            raise Exception("Should not be here as reference color sequence is not found.")
        self.__run_check_capture_stage(reference_color_sequence)
