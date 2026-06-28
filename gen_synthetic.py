from pathlib import Path

import cv2
import numpy as np

from geo import latlon_to_utm
from homography import utm_to_pixels

FRAMES_DIR = Path("frames")
OUTPUT_DIR = Path("frames_synthetic")
N_FRAMES = 50
N_BLOBS = 3
STEP_M = 3  # meters per frame

CORNERS_LATLON = np.array([
    [40.108806, -88.227611],
    [40.108806, -88.226917],
    [40.106417, -88.227556],
    [40.106417, -88.226861],
])



def random_walk_utm(pos: np.ndarray, step: float, utm_min: np.ndarray, utm_max: np.ndarray) -> np.ndarray:
    angle = np.random.uniform(0, 2 * np.pi)
    pos = pos + step * np.array([np.cos(angle), np.sin(angle)])
    return np.clip(pos, utm_min, utm_max)


if __name__ == "__main__":
    frames = sorted(FRAMES_DIR.glob("*.jpg"))
    if not frames:
        raise FileNotFoundError("No frames found in frames/ — run ingest.py first")

    H = np.load("homography.npy")
    H_inv = np.linalg.inv(H)

    corners_utm = latlon_to_utm(CORNERS_LATLON)
    utm_min = corners_utm.min(axis=0)
    utm_max = corners_utm.max(axis=0)

    background = cv2.imread(str(frames[0]))
    OUTPUT_DIR.mkdir(exist_ok=True)

    positions = np.array([
        utm_min + np.random.rand(2) * (utm_max - utm_min)
        for _ in range(N_BLOBS)
    ])

    for i in range(N_FRAMES):
        frame = background.copy()
        for j in range(N_BLOBS):
            positions[j] = random_walk_utm(positions[j], STEP_M, utm_min, utm_max)

        pixels = utm_to_pixels(positions, H_inv)
        for x, y in pixels:
            cv2.rectangle(frame, (x - 15, y - 40), (x + 15, y), (0, 100, 200), -1)

        path = OUTPUT_DIR / f"{i:05d}.jpg"
        cv2.imwrite(str(path), frame)

    print(f"Generated {N_FRAMES} synthetic frames in {OUTPUT_DIR}/")
