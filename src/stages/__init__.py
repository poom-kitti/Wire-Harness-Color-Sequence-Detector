import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from ..tools.video_stream import PiCameraStream

WINDOW_NAME = "Wire Harness Color Detector"

DEFAULT_FRAME_WIDTH = 1280
DEFAULT_FRAME_HEIGHT = 720


@dataclass
class UserSettingConfig:
    is_connector_height_greater_than_width: bool
    do_threshold_with_bg: bool


@dataclass
class BaseStageConfig:
    camera: PiCameraStream


class Stage(ABC):
    """A singleton object as base class for the other stage classes."""

    _instance: "Stage"
    _camera: PiCameraStream
    _user_config: UserSettingConfig

    def __new__(cls):
        if not hasattr(cls, "_instance"):
            cls._instance = super(Stage, cls).__new__(cls)

        return cls._instance

    @abstractmethod
    def run(self):
        ...

    @abstractmethod
    def set_config(self, stage_config: BaseStageConfig, user_config: Optional[UserSettingConfig]):
        ...

    def quit(self):
        """Exit from the detector program."""
        sys.exit(1)
