import cv2
import numpy as np

from .tasks import connector, preprocess

DO_THRESHOLD_WITH_BG_IMG = True


def main():
    # Prepare frame
    frame: np.ndarray = cv2.imread("asset/image1.jpg")
    frame_blur: np.ndarray = cv2.GaussianBlur(frame, (5, 5), 0)
    frame_hsv: np.ndarray = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2HSV)

    # Prepare img of background only
    bg_img: np.ndarray = cv2.imread("asset/image-bg.jpg")
    bg_img = cv2.GaussianBlur(bg_img, (5, 5), 0)
    bg_img_hsv = cv2.cvtColor(bg_img, cv2.COLOR_BGR2HSV)

    # Display image
    display_img = frame.copy()

    # Perform thresholding
    if DO_THRESHOLD_WITH_BG_IMG:
        threshold_img = preprocess.threshold_with_inRange(frame_hsv, bg_img_hsv)
    else:
        threshold_img = preprocess.threshold_with_otsu(frame_blur)

    # Fill bg as white
    frame_white_bg = preprocess.fill_bg_as_white(frame_blur, threshold_img)

    # Find connector contour
    connector_contour, connector_tresh = connector.find_connector_contour(threshold_img)

    cv2.drawContours(display_img, [connector_contour], -1, (255, 0, 0), 5)

    cv2.imshow("tresh", threshold_img)
    cv2.imshow("connector", display_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
