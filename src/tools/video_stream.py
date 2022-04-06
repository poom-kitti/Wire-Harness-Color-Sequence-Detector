"""This module contains a class for picamera video stream."""
import queue
import time
from dataclasses import dataclass
from queue import Queue
from threading import Thread
from typing import Iterator, Tuple

from numpy import ndarray
from picamera import PiCamera
from picamera.array import PiRGBArray


@dataclass
class PiCameraConfig:
    framerate: int
    resolution: Tuple[int, int]


class PiCameraStream:
    """A class for Picamera to continously capture frame.

    Attributes:
        camera: The Picamera.
        raw_capture: The raw captured PiRGBArray by the camera.
        capture_stream: An iterator that continuously capture new frames.
        is_running: A flag whether the current camera stream is running.
    """

    is_running: bool

    def __init__(self, camera_config: PiCameraConfig):
        self.camera = self.__set_up_camera(camera_config)
        self.raw_capture = PiRGBArray(self.camera, size=camera_config.resolution)
        self.capture_stream: Iterator[PiRGBArray] = self.camera.capture_continuous(
            self.raw_capture, "bgr", use_video_port=True
        )

        self.__frame_queue: Queue[ndarray] = queue.Queue(maxsize=1)

    def __set_up_camera(self, camera_config: PiCameraConfig) -> PiCamera:
        """Set up the Picamera configurations."""
        camera = PiCamera()
        camera.framerate = camera_config.framerate
        camera.resolution = camera_config.resolution

        # Allow camera to warm up
        time.sleep(2)

        return camera

    def start(self) -> None:
        """Begin a thread to run the capture stream."""
        # Do nothing if already running
        if self.is_running:
            return

        self.is_running = True
        Thread(target=self.__run).start()

    def __shutdown(self) -> None:
        """Close the camera and raw capture."""
        self.raw_capture.close()
        self.camera.close()

    def __run(self) -> None:
        """Replace the current frame with the new frame read by the
        capture stream."""
        for frame in self.capture_stream:
            # Remove unread frame from queue
            try:
                self.__frame_queue.get_nowait()
            except queue.Empty:
                pass
            finally:
                # Add frame to queue
                self.__frame_queue.put(frame.array)

            self.raw_capture.truncate(0)

            if not self.is_running:
                self.__shutdown()
                break

    def stop(self) -> None:
        """Stop and send flag to shutdown the camera stream."""
        self.is_running = False

    def read(self) -> ndarray:
        """Read the most current frame captured by the stream."""
        if not self.is_running:
            raise AttributeError("Camera stream has not started yet.")

        return self.__frame_queue.get()
