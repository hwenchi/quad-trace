from pathlib import Path

import numpy as np
import pandas as pd

from homography import pixels_to_utm

DETECTIONS_DIR = Path("detections")


if __name__ == "__main__":
    paths = sorted(DETECTIONS_DIR.glob("*.parquet"))
    if not paths:
        print("No parquet files found in detections/")
        raise SystemExit(1)

    H = np.load("homography.npy")
    for path in paths:
        df = pd.read_parquet(path)
        utm = pixels_to_utm(df[["px", "py"]].to_numpy(), H)
        df["easting"] = utm[:, 0]
        df["northing"] = utm[:, 1]
        df.to_parquet(path, index=False)
        print(f"Reprojected {path} ({len(df)} rows)")