import cv2
import numpy as np


def compute_homography(pixel_pts: np.ndarray, utm_pts: np.ndarray) -> np.ndarray:
    H, _ = cv2.findHomography(pixel_pts, utm_pts)
    return H


def pixels_to_utm(pixels: np.ndarray, H: np.ndarray) -> np.ndarray:
    ones = np.ones((len(pixels), 1))
    ph = np.hstack([pixels, ones])
    qh = (H @ ph.T).T
    return qh[:, :2] / qh[:, 2:]


def utm_to_pixels(utm: np.ndarray, H_inv: np.ndarray) -> np.ndarray:
    ones = np.ones((len(utm), 1))
    ph = np.hstack([utm, ones])
    qh = (H_inv @ ph.T).T
    return (qh[:, :2] / qh[:, 2:]).astype(int)