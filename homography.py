from pathlib import Path

import cv2
import numpy as np

from geo import latlon_to_utm


def load_gcps() -> tuple[np.ndarray, np.ndarray]:
    pixel_pts = np.load("gcps_pixels.npy")
    latlon_pts = np.load("gcps_latlon.npy")
    utm_pts = latlon_to_utm(latlon_pts)
    return pixel_pts, utm_pts


def pixels_to_utm(pixels: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Transform (N, 2) pixel coords to (N, 2) UTM coords using homography H."""
    ones = np.ones((len(pixels), 1))
    ph = np.hstack([pixels, ones])
    qh = (H @ ph.T).T
    return qh[:, :2] / qh[:, 2:]


def utm_to_pixels(utm: np.ndarray, H_inv: np.ndarray) -> np.ndarray:
    """Transform (N, 2) UTM coords to (N, 2) pixel coords using inverse homography."""
    ones = np.ones((len(utm), 1))
    ph = np.hstack([utm, ones])
    qh = (H_inv @ ph.T).T
    return (qh[:, :2] / qh[:, 2:]).astype(int)


def compute_homography(pixel_pts: np.ndarray, utm_pts: np.ndarray) -> np.ndarray:
    H, _ = cv2.findHomography(pixel_pts, utm_pts)
    return H


if __name__ == "__main__":
    pixels, utm = load_gcps()
    H = compute_homography(pixels, utm)
    np.save("homography.npy", H)
    print("Saved homography.npy")
    print("H:\n", H)