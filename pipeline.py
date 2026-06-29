import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import piexif

from detect import draw_detections, extract_feet, get_foreground_mask
from geo import latlon_to_utm
from homography import pixels_to_utm, utm_to_pixels

FRAMES_DIR = Path("frames")
MASKS_DIR = Path("masks")
DETECTIONS_VIZ_DIR = Path("detections_viz")
SYNTHETIC_DIR = Path("frames_synthetic_overlay")
DETECTIONS_DIR = Path("detections")
POLL_INTERVAL = 1.0  # seconds
PARTITION_SIZE = 100
WARMUP_FRAMES = 100

N_BLOBS = 3
STEP_M = 3

CORNERS_LATLON = np.array([
    [40.108806, -88.227611],
    [40.108778, -88.226917],
    [40.106417, -88.227556],
    [40.106417, -88.226861],
])


def read_timestamp(path: Path) -> int:
    try:
        exif = piexif.load(str(path))
        dt_str = exif["Exif"][piexif.ExifIFD.DateTimeOriginal].decode()
        return int(time.mktime(time.strptime(dt_str, "%Y:%m:%d %H:%M:%S")))
    except (KeyError, Exception):
        return int(path.stat().st_mtime)


class SyntheticBlobs:
    def __init__(self, H: np.ndarray) -> None:
        self.H_inv = np.linalg.inv(H)
        corners_utm = latlon_to_utm(CORNERS_LATLON)
        self.utm_min = corners_utm.min(axis=0)
        self.utm_max = corners_utm.max(axis=0)
        self.positions = self.utm_min + np.random.rand(N_BLOBS, 2) * (self.utm_max - self.utm_min)

    def step(self) -> None:
        angles = np.random.uniform(0, 2 * np.pi, N_BLOBS)
        delta = STEP_M * np.column_stack([np.cos(angles), np.sin(angles)])
        self.positions = np.clip(self.positions + delta, self.utm_min, self.utm_max)

    def overlay(self, frame: np.ndarray) -> np.ndarray:
        self.step()
        out = frame.copy()
        pixels = utm_to_pixels(self.positions, self.H_inv)
        for x, y in pixels:
            cv2.rectangle(out, (x - 15, y - 40), (x + 15, y), (0, 100, 200), -1)
        return out


def process_frame(frame_path: Path, H: np.ndarray,
                  roi: np.ndarray | None, blobs: SyntheticBlobs | None,
                  mask_path: Path | None = None,
                  viz_path: Path | None = None) -> list[dict]:
    frame = cv2.imread(str(frame_path))
    if frame is None:
        return []
    timestamp = read_timestamp(frame_path)

    if blobs is not None:
        frame = blobs.overlay(frame)
        if viz_path is not None:
            cv2.imwrite(str(SYNTHETIC_DIR / frame_path.name), frame)

    mask = get_foreground_mask(frame, roi)
    feet = extract_feet(mask)

    if mask_path is not None:
        mask_viz = draw_detections(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), mask)
        cv2.imwrite(str(mask_path), mask_viz)
    if viz_path is not None:
        cv2.imwrite(str(viz_path), draw_detections(frame, mask))
    if len(feet) == 0:
        return []
    utm = pixels_to_utm(feet, H)
    return [{"timestamp": timestamp, "easting": e, "northing": n} for e, n in utm]


def warmup(frames_dir: Path, roi: np.ndarray | None = None) -> None:
    frames = sorted(frames_dir.glob("*.jpg"))[:WARMUP_FRAMES]
    for frame_path in frames:
        frame = cv2.imread(str(frame_path))
        if frame is not None:
            get_foreground_mask(frame, roi)
    print(f"Warmed up MOG2 on {len(frames)} frames")


def flush(rows: list[dict], partition: int) -> None:
    DETECTIONS_DIR.mkdir(exist_ok=True)
    path = DETECTIONS_DIR / f"{partition:05d}.parquet"
    pd.DataFrame(rows, columns=["timestamp", "easting", "northing"]).to_parquet(path, index=False)
    print(f"Wrote partition {path} ({len(rows)} rows)")


ROI_FILE = Path("roi_mask.npy")


def run(H: np.ndarray, synthetic: bool, debug: bool) -> None:
    roi = np.load(ROI_FILE) if ROI_FILE.exists() else None
    if roi is not None:
        print(f"Loaded ROI mask from {ROI_FILE}")
    else:
        print("No ROI mask found, processing full frame")
    if debug:
        MASKS_DIR.mkdir(exist_ok=True)
        DETECTIONS_VIZ_DIR.mkdir(exist_ok=True)
        if synthetic:
            SYNTHETIC_DIR.mkdir(exist_ok=True)

    blobs = SyntheticBlobs(H) if synthetic else None
    warmup(FRAMES_DIR, roi)
    idx = 0
    rows = []
    partition = 0
    print(f"Watching {FRAMES_DIR} for new frames...")

    while True:
        next_frame = FRAMES_DIR / f"{idx:05d}.jpg"
        if not next_frame.exists():
            time.sleep(POLL_INTERVAL)
            continue

        rows.extend(process_frame(
            next_frame, H, roi, blobs,
            mask_path=MASKS_DIR / next_frame.name if debug else None,
            viz_path=DETECTIONS_VIZ_DIR / next_frame.name if debug else None,
        ))
        idx += 1

        if len(rows) >= PARTITION_SIZE:
            flush(rows, partition)
            rows = []
            partition += 1

        print(f"Processed {idx} frames, {len(rows)} buffered rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true", help="Overlay synthetic moving blobs")
    parser.add_argument("--debug", action="store_true", help="Save masks and overlays to disk")
    args = parser.parse_args()

    H = np.load("homography.npy")
    debug = args.debug or args.synthetic
    run(H, args.synthetic, debug)
