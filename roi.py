from pathlib import Path

import cv2
import numpy as np

WINDOW_NAME = "ROI"


def redraw(base: np.ndarray, pts: list[tuple[int, int]]) -> None:
    frame = base.copy()
    for i, pt in enumerate(pts):
        cv2.circle(frame, pt, 5, (0, 255, 0), -1)
        if i > 0:
            cv2.line(frame, pts[i - 1], pt, (0, 255, 0), 2)
    if len(pts) > 2:
        cv2.line(frame, pts[-1], pts[0], (0, 255, 0), 1)
    cv2.imshow(WINDOW_NAME, frame)


if __name__ == "__main__":
    frames = sorted(Path("frames").glob("*.jpg"))
    if not frames:
        raise FileNotFoundError("No frames found — run ingest.py first")

    base = cv2.imread(str(frames[0]))
    pts: list[tuple[int, int]] = []

    def on_click(event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
            redraw(base, pts)

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, on_click)
    redraw(base, pts)
    print("Click polygon vertices. Delete to undo. Enter to save.")

    while True:
        key = cv2.waitKey(0)
        if key in (13, 10):  # Enter
            break
        if key == 127 and pts:  # Delete
            pts.pop()
            redraw(base, pts)

    cv2.destroyAllWindows()

    mask = np.zeros(base.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(pts)], 255)
    np.save("roi_mask.npy", mask)
    print(f"Saved roi_mask.npy ({len(pts)} vertices)")