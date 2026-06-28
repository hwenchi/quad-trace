from pathlib import Path

import cv2
import numpy as np

GRID_FILE = Path("gcps_latlon.npy")
WINDOW_NAME = "Calibration"


def collect_pixel_coords(frame_path: Path, n_points: int) -> np.ndarray:
    original = cv2.imread(str(frame_path))
    if original is None:
        raise FileNotFoundError(f"Could not load frame: {frame_path}")

    clicks: list[tuple[int, int]] = []

    def redraw() -> None:
        frame = original.copy()
        for i, (x, y) in enumerate(clicks):
            cv2.circle(frame, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(frame, str(i + 1), (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow(WINDOW_NAME, frame)

    def on_click(event, x, y, flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN or len(clicks) >= n_points:
            return
        clicks.append((x, y))
        redraw()
        print(f"  Point {len(clicks)} pixel: ({x}, {y})")
        if len(clicks) < n_points:
            print(f"Click point {len(clicks) + 1}  (Delete to undo)")
        else:
            print("All points collected. Press Enter to save.")

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, on_click)
    redraw()
    print("Click point 1")

    while True:
        key = cv2.waitKey(0)
        if key in (13, 10):  # Enter
            break
        if key == 127:  # Delete (macOS)
            if clicks:
                clicks.pop()
                redraw()
                print(f"Removed last point. Now at {len(clicks)}/{n_points}. Click point {len(clicks) + 1}")

    cv2.destroyAllWindows()
    return np.array(clicks, dtype=np.float64)


if __name__ == "__main__":
    if not GRID_FILE.exists():
        raise FileNotFoundError("gcps_latlon.npy not found — run gen_gcps.py first")

    frames = sorted(Path("frames").glob("*.jpg"))
    if not frames:
        raise FileNotFoundError("No frames found — run ingest.py first")

    grid_latlon = np.load(GRID_FILE)
    pixels = collect_pixel_coords(frames[0], len(grid_latlon))

    if len(pixels) < len(grid_latlon):
        print(f"Only {len(pixels)}/{len(grid_latlon)} points collected, not saving.")
    else:
        np.save("gcps_pixels.npy", pixels)
        print(f"Saved gcps_pixels.npy ({len(pixels)} points)")
