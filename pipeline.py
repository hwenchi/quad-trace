import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import piexif

from detect import extract_feet, get_foreground_mask
from homography import pixels_to_utm

FRAMES_DIR = Path("frames")
MASKS_DIR = Path("masks")
OUTPUT = Path("detections.parquet")
POLL_INTERVAL = 1.0  # seconds


def read_timestamp(path: Path) -> int:
    try:
        exif = piexif.load(str(path))
        dt_str = exif["Exif"][piexif.ExifIFD.DateTimeOriginal].decode()
        return int(time.mktime(time.strptime(dt_str, "%Y:%m:%d %H:%M:%S")))
    except (KeyError, Exception):
        return int(path.stat().st_mtime)


def process_frame(frame_path: Path, mask_path: Path, H: np.ndarray) -> list[dict]:
    frame = cv2.imread(str(frame_path))
    if frame is None:
        return []
    timestamp = read_timestamp(frame_path)
    mask = get_foreground_mask(frame)
    cv2.imwrite(str(mask_path), mask)
    feet = extract_feet(mask)
    if len(feet) == 0:
        return []
    utm = pixels_to_utm(feet, H)
    return [{"timestamp": timestamp, "easting": e, "northing": n} for e, n in utm]


WARMUP_FRAMES = 20


def warmup(frames_dir: Path) -> int:
    frames = sorted(frames_dir.glob("*.jpg"))[:WARMUP_FRAMES]
    for frame_path in frames:
        frame = cv2.imread(str(frame_path))
        if frame is not None:
            get_foreground_mask(frame)
    print(f"Warmed up MOG2 on {len(frames)} frames")
    return len(frames)


def run(frames_dir: Path, H: np.ndarray) -> None:
    MASKS_DIR.mkdir(exist_ok=True)
    warmup(frames_dir)
    idx = 0
    rows = []
    print(f"Watching {frames_dir} for new frames...")

    while True:
        next_frame = frames_dir / f"{idx:05d}.jpg"
        if not next_frame.exists():
            time.sleep(POLL_INTERVAL)
            continue

        mask_path = MASKS_DIR / next_frame.name
        rows.extend(process_frame(next_frame, mask_path, H))
        idx += 1

        df = pd.DataFrame(rows, columns=["timestamp", "easting", "northing"])
        df.to_parquet(OUTPUT, index=False)
        print(f"Processed {idx} frames, {len(rows)} detections total")


if __name__ == "__main__":
    frames_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else FRAMES_DIR
    H = np.load("homography.npy")
    run(frames_dir, H)
