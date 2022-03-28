import cv2
import numpy as np

from .tasks import connector, display, preprocess, wire_color, wire_roi, wires

DO_THRESHOLD_WITH_BG_IMG = True
IS_HEIGHT_GREATER_THAN_WIDTH = True


def main():
    # Prepare frame
    frame: np.ndarray = cv2.imread("asset/image2.jpg")
    frame_blur: np.ndarray = cv2.GaussianBlur(frame, (5, 5), 0)

    # Perform thresholding
    if DO_THRESHOLD_WITH_BG_IMG:
        # Prepare img of background only
        bg_img: np.ndarray = cv2.imread("asset/image-bg.jpg")
        bg_img = cv2.GaussianBlur(bg_img, (5, 5), 0)
        bg_img_hsv = cv2.cvtColor(bg_img, cv2.COLOR_BGR2HSV)

        frame_hsv: np.ndarray = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2HSV)

        frame_threshold = preprocess.threshold_with_inRange(frame_hsv, bg_img_hsv)
    else:
        frame_threshold = preprocess.threshold_with_otsu(frame_blur)

    # Fill bg as white
    frame_white_bg = preprocess.fill_bg_as_white(frame_blur, frame_threshold)

    # Find connector contour
    connector_contour = connector.find_connector_contour(frame_threshold)

    wire_roi_img, display_img = wire_roi.find_wire_roi(
        frame, frame_white_bg, connector_contour, IS_HEIGHT_GREATER_THAN_WIDTH
    )

    cropped_wires = wires.find_wires(wire_roi_img)

    for i, wire in enumerate(cropped_wires):
        print(f"wire_{i} lab:", wire_color.find_wire_lab_color(wire))
        cv2.imshow(f"wire_{i}", wire)

    display_img = display.draw_color_sequence_result(display_img, True)
    display_img = display.draw_title_and_command(
        display_img, "Color Sequence Checking", "[Q / ESC] Exit", "[R] Retake Reference Image", "[B] Back"
    )

    cv2.imshow("wire_roi", wire_roi_img)
    cv2.imshow("display", display_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
