import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import piexif

from detect import detect_motion, draw_detections
from homography import pixels_to_utm

FRAMES_DIR = Path("frames")
DETECTIONS_VIZ_DIR = Path("detections_viz")
DETECTIONS_DIR = Path("detections")
POLL_INTERVAL = 1.0
PARTITION_SIZE = 100


def read_timestamp(path: Path) -> int:
    try:
        exif = piexif.load(str(path))
        dt_str = exif["Exif"][piexif.ExifIFD.DateTimeOriginal].decode()
        return int(time.mktime(time.strptime(dt_str, "%Y:%m:%d %H:%M:%S")))
    except (KeyError, Exception):
        return int(path.stat().st_mtime)


ROI_FILE = Path("roi_mask.npy")


def process_frame(frame: np.ndarray, prev_frame: np.ndarray, timestamp: int,
                  H: np.ndarray, roi: np.ndarray | None = None,
                  viz_path: Path | None = None) -> pd.DataFrame:
    boxes, mask = detect_motion(frame, prev_frame, roi)
    if viz_path is not None:
        viz = np.hstack([draw_detections(frame, boxes), mask])
        cv2.imwrite(str(viz_path), viz)
    if not len(boxes):
        return pd.DataFrame()
    feet = np.column_stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]])
    utm = pixels_to_utm(feet, H)
    return pd.DataFrame({
        "timestamp": timestamp,
        "px": feet[:, 0],
        "py": feet[:, 1],
        "easting": utm[:, 0],
        "northing": utm[:, 1],
    })


def flush(df: pd.DataFrame, partition: int) -> None:
    DETECTIONS_DIR.mkdir(exist_ok=True)
    path = DETECTIONS_DIR / f"{partition:05d}.parquet"
    df.to_parquet(path, index=False)
    print(f"Wrote partition {path} ({len(df)} rows)")


def run(H: np.ndarray, debug: bool) -> None:
    roi = np.load(ROI_FILE) if ROI_FILE.exists() else None
    if roi is not None:
        print(f"Loaded ROI mask from {ROI_FILE}")
    if debug:
        DETECTIONS_VIZ_DIR.mkdir(exist_ok=True)

    idx = 0
    prev_frame = None
    frames: list[pd.DataFrame] = []
    buffered = 0
    partition = 0
    print(f"Watching {FRAMES_DIR} for new frames...")

    while True:
        next_path = FRAMES_DIR / f"{idx:05d}.jpg"
        if not next_path.exists():
            time.sleep(POLL_INTERVAL)
            continue

        frame = cv2.imread(str(next_path))
        timestamp = read_timestamp(next_path)

        if prev_frame is not None:
            df = process_frame(
                frame, prev_frame, timestamp, H, roi,
                viz_path=DETECTIONS_VIZ_DIR / next_path.name if debug else None,
            )
            if not df.empty:
                frames.append(df)
                buffered += len(df)

            if buffered >= PARTITION_SIZE:
                flush(pd.concat(frames, ignore_index=True), partition)
                frames = []
                buffered = 0
                partition += 1

        prev_frame = frame
        idx += 1
        print(f"Processed {idx} frames, {buffered} buffered rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Save detection overlays to disk")
    args = parser.parse_args()

    H = np.load("homography.npy")
    run(H, args.debug)
