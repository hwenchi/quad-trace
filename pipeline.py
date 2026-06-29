import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import piexif

from detect import CONF_THRESHOLD, detect, draw_detections
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


def process_frame(frame_path: Path, H: np.ndarray,
                  viz_path: Path | None = None) -> pd.DataFrame:
    frame = cv2.imread(str(frame_path))
    if frame is None:
        return pd.DataFrame()
    timestamp = read_timestamp(frame_path)
    boxes = detect(frame)
    if viz_path is not None:
        cv2.imwrite(str(viz_path), draw_detections(frame, boxes[boxes[:, 4] >= CONF_THRESHOLD]))
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
        "conf": boxes[:, 4],
    })


def flush(df: pd.DataFrame, partition: int) -> None:
    DETECTIONS_DIR.mkdir(exist_ok=True)
    path = DETECTIONS_DIR / f"{partition:05d}.parquet"
    df.to_parquet(path, index=False)
    print(f"Wrote partition {path} ({len(df)} rows)")


def run(H: np.ndarray, debug: bool) -> None:
    if debug:
        DETECTIONS_VIZ_DIR.mkdir(exist_ok=True)

    idx = 0
    frames: list[pd.DataFrame] = []
    partition = 0
    buffered = 0
    print(f"Watching {FRAMES_DIR} for new frames...")

    while True:
        next_frame = FRAMES_DIR / f"{idx:05d}.jpg"
        if not next_frame.exists():
            time.sleep(POLL_INTERVAL)
            continue

        df = process_frame(
            next_frame, H,
            viz_path=DETECTIONS_VIZ_DIR / next_frame.name if debug else None,
        )
        if not df.empty:
            frames.append(df)
            buffered += len(df)
        idx += 1

        if buffered >= PARTITION_SIZE:
            flush(pd.concat(frames, ignore_index=True), partition)
            frames = []
            buffered = 0
            partition += 1

        print(f"Processed {idx} frames, {buffered} buffered rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Save detection overlays to disk")
    args = parser.parse_args()

    H = np.load("homography.npy")
    run(H, args.debug)
