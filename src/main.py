import cv2

from .stages.initialize_stage import InitializeStage, InitializeStageConfig
from .tools.video_stream import PiCameraConfig, PiCameraStream

camera_config = PiCameraConfig(60, (1280, 720))


def main() -> None:
    camera = None
    try:
        camera = PiCameraStream(camera_config)
        camera.start()

        detector = InitializeStage()
        detector_config = InitializeStageConfig(camera)
        detector.set_config(detector_config, None)
        detector.run()
    finally:
        cv2.destroyAllWindows()

        if camera is not None:
            camera.stop()


if __name__ == "__main__":
    main()
