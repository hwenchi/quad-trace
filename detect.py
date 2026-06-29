import sys

import cv2
import numpy as np

MOG2_HISTORY = 500
MOG2_VAR_THRESHOLD = 50
MOG2_LEARNING_RATE = 0.05 # -1 = automatic
MIN_AREA = 10
MAX_AREA = 20000

_mog2 = cv2.createBackgroundSubtractorMOG2(
    history=MOG2_HISTORY,
    varThreshold=MOG2_VAR_THRESHOLD,
    detectShadows=True,
)


def get_foreground_mask(frame: np.ndarray, roi: np.ndarray | None = None) -> np.ndarray:
    if roi is not None:
        frame = cv2.bitwise_and(frame, frame, mask=roi)
    mask = _mog2.apply(frame, learningRate=MOG2_LEARNING_RATE)
    # MOG2 marks shadows as 127 and foreground as 255 — keep only foreground
    return (mask == 255).astype(np.uint8) * 255


def extract_feet(mask: np.ndarray) -> np.ndarray:
    """Return (N, 2) array of bottom-edge centers for each detected blob."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    feet = []
    for c in contours:
        area = cv2.contourArea(c)
        if not (MIN_AREA <= area <= MAX_AREA):
            continue
        x, y, w, h = cv2.boundingRect(c)
        feet.append((x + w // 2, y + h))
    return np.array(feet, dtype=np.float64)


def draw_detections(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = frame.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv2.contourArea(c)
        if not (MIN_AREA <= area <= MAX_AREA):
            continue
        cv2.drawContours(out, [c], -1, (0, 255, 0), 1)
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 0, 0), 1)
        foot = (x + w // 2, y + h)
        cv2.circle(out, foot, 4, (0, 0, 255), -1)
    return out


if __name__ == "__main__":
    frame = cv2.imread(sys.argv[1])
    mask = get_foreground_mask(frame)
    cv2.imshow("mask", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
