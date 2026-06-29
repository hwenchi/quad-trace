import cv2
import numpy as np

DIFF_THRESHOLD = 25
MIN_AREA = 2
MAX_AREA = 50000


def detect(frame: np.ndarray, prev_frame: np.ndarray,
           roi: np.ndarray | None = None) -> np.ndarray:
    """Return (N, 4) array of [x1, y1, x2, y2] for moving blobs."""
    diff = cv2.absdiff(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                       cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY))
    _, mask = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    if roi is not None:
        mask = cv2.bitwise_and(mask, roi)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if MIN_AREA <= area <= MAX_AREA:
            x, y, w, h = cv2.boundingRect(c)
            boxes.append((x, y, x + w, y + h))
    return np.array(boxes, dtype=np.float64) if boxes else np.empty((0, 4), dtype=np.float64)


def draw_detections(frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    out = frame.copy()
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)
        cv2.circle(out, (int((x1 + x2) / 2), int(y2)), 4, (0, 0, 255), -1)
    return out