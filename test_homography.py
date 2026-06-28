import webbrowser
from pathlib import Path

import cv2
import folium
import numpy as np

from geo import utm_to_latlon
from homography import pixels_to_utm

WINDOW_NAME = "Test Homography"


def collect_trajectory(frame_path: Path) -> np.ndarray:
    frame = cv2.imread(str(frame_path))
    if frame is None:
        raise FileNotFoundError(f"Could not load frame: {frame_path}")

    clicks: list[tuple[int, int]] = []

    def on_click(event, x, y, flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        clicks.append((x, y))
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        if len(clicks) > 1:
            cv2.line(frame, clicks[-2], clicks[-1], (0, 255, 0), 2)
        cv2.imshow(WINDOW_NAME, frame)

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, on_click)
    cv2.imshow(WINDOW_NAME, frame)
    print("Click points to draw a trajectory. Press any key when done.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return np.array(clicks, dtype=np.float64)


def show_on_map(latlon: np.ndarray) -> None:
    center = latlon.mean(axis=0).tolist()
    m = folium.Map(location=center, zoom_start=19, max_zoom=21, tiles="Esri.WorldImagery")
    folium.PolyLine(latlon.tolist(), color="yellow", weight=3).add_to(m)
    for i, (lat, lon) in enumerate(latlon):
        folium.CircleMarker(location=[lat, lon], radius=4, color="yellow", fill=True, fill_color="yellow").add_to(m)
    out = Path("test_trajectory.html")
    m.save(str(out))
    webbrowser.open(out.resolve().as_uri())
    print(f"Map saved to {out}")


if __name__ == "__main__":
    frames = sorted(Path("frames").glob("*.jpg"))
    if not frames:
        raise FileNotFoundError("No frames found — run ingest.py first")

    H = np.load("homography.npy")
    pixels = collect_trajectory(frames[0])
    if len(pixels) < 2:
        print("Need at least 2 points.")
    else:
        utm = pixels_to_utm(pixels, H)
        latlon = utm_to_latlon(utm)
        show_on_map(latlon)