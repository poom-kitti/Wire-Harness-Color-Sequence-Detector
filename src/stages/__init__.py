import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass

WINDOW_NAME = "Wire Harness Color Detector"

DEFAULT_FRAME_WIDTH = 1280
DEFAULT_FRAME_HEIGHT = 720


class Stage(ABC):
    """A singleton object as base class for the other stage classes."""

    _instance: "Stage"

    def __new__(cls):
        if not hasattr(cls, "_instance"):
            cls._instance = super(Stage, cls).__new__(cls)

        return cls._instance

    @abstractmethod
    def run(self):
        ...

    def quit(self):
        """Exit from the detector program."""
        sys.exit(1)


@dataclass
class DetectorConfig:
    is_connector_height_greater_than_width: bool
    do_threshold_with_bg: bool
