import cv2
import numpy as np

DIFF_THRESHOLD = 25
MIN_AREA = 2
MAX_AREA = 50000
MAX_CHANGED_FRACTION = 0.2
PLATEAU_KERNEL = 30
PLATEAU_MIN_AREA = 5000


def find_plateaus(mask: np.ndarray):
    """Return (plateau_mask, contours) of large diffuse regions."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (PLATEAU_KERNEL, PLATEAU_KERNEL))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    plateau_mask = np.zeros_like(mask)
    for c in contours:
        if cv2.contourArea(c) > PLATEAU_MIN_AREA:
            cv2.drawContours(plateau_mask, [c], -1, 255, -1)
    return plateau_mask, contours


def detect_motion(frame: np.ndarray, prev_frame: np.ndarray,
                  roi: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (boxes, color_mask). color_mask: red=outside ROI, orange=plateau, green=detected."""
    diff = cv2.absdiff(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                       cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY))
    _, mask = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

    if mask.sum() / 255 > mask.size * MAX_CHANGED_FRACTION:
        return np.empty((0, 4), dtype=np.float64), np.zeros((*mask.shape, 3), dtype=np.uint8)

    plateaus, plateau_contours = find_plateaus(mask)
    after_roi = cv2.bitwise_and(mask, roi) if roi is not None else mask
    filtered = cv2.bitwise_and(after_roi, cv2.bitwise_not(plateaus))

    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    color_mask[mask > 0] = (0, 0, 255)         # red: outside ROI
    color_mask[after_roi > 0] = (0, 165, 255)  # orange: plateau (shadow/lighting)
    color_mask[filtered > 0] = (0, 255, 0)     # green: passed all filters
    cv2.drawContours(color_mask, plateau_contours, -1, (255, 255, 255), 2)

    contours, _ = cv2.findContours(filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if MIN_AREA <= area <= MAX_AREA:
            x, y, w, h = cv2.boundingRect(c)
            boxes.append((x, y, x + w, y + h))
            cv2.rectangle(color_mask, (x, y), (x + w, y + h), (0, 255, 255), 1)
    return np.array(boxes, dtype=np.float64) if boxes else np.empty((0, 4), dtype=np.float64), color_mask


def draw_detections(frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    out = frame.copy()
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)
        cv2.circle(out, (int((x1 + x2) / 2), int(y2)), 4, (0, 0, 255), -1)
    return out