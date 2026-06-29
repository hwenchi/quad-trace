import cv2
import numpy as np
from ultralytics import YOLO

PERSON_CLASS = 0
CONF_THRESHOLD = 0.1

_model = YOLO("yolov8n.pt")


def detect(frame: np.ndarray) -> np.ndarray:
    """Return (N, 5) array of [x1, y1, x2, y2, conf] for all person detections."""
    h, w = frame.shape[:2]
    imgsz = (round(h / 32) * 32, round(w / 32) * 32)
    results = _model(frame, classes=[PERSON_CLASS], conf=0.01, imgsz=imgsz, verbose=False)[0]
    if not results.boxes:
        return np.empty((0, 5), dtype=np.float64)
    return np.column_stack([results.boxes.xyxy.cpu().numpy(),
                             results.boxes.conf.cpu().numpy()])


def draw_detections(frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    out = frame.copy()
    for x1, y1, x2, y2, conf in boxes:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)
        cv2.circle(out, (int((x1 + x2) / 2), int(y2)), 4, (0, 0, 255), -1)
        cv2.putText(out, f"{conf:.2f}", (int(x2) + 4, int(y1) + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    return out
