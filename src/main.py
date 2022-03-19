import cv2
import numpy as np

from .tasks import preprocess

DO_THRESHOLD_WITH_BG_IMG = True


def main():
    # Prepare frame
    frame: np.ndarray = cv2.imread(
        "asset/image1.jpg"
    )  # Use only when need to show to user
    frame_blur: np.ndarray = cv2.GaussianBlur(frame, (5, 5), 0)
    frame_hsv: np.ndarray = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2HSV)

    # Prepare img of background only
    bg_img: np.ndarray = cv2.imread("asset/image-bg.jpg")
    bg_img = cv2.GaussianBlur(bg_img, (5, 5), 0)
    bg_img_hsv = cv2.cvtColor(bg_img, cv2.COLOR_BGR2HSV)

    # Perform thresholding
    if DO_THRESHOLD_WITH_BG_IMG:
        threshold_img = preprocess.threshold_with_inRange(frame_hsv, bg_img_hsv)
    else:
        threshold_img = preprocess.threshold_with_otsu(frame_blur)

    # Fill bg as white
    frame_white_bg = preprocess.fill_bg_as_white(frame_blur, threshold_img)

    cv2.imshow("img", frame)
    cv2.imshow("thresh", threshold_img)
    cv2.imshow("img_white", frame_white_bg)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
