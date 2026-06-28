import numpy as np
from pyproj import Transformer

_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32616", always_xy=True)
_to_latlon = Transformer.from_crs("EPSG:32616", "EPSG:4326", always_xy=True)


def latlon_to_utm(latlon: np.ndarray) -> np.ndarray:
    """Transform (N, 2) lat/lon coords to (N, 2) UTM (easting, northing) coords."""
    easting, northing = _to_utm.transform(latlon[:, 1], latlon[:, 0])
    return np.column_stack([easting, northing])


def utm_to_latlon(utm: np.ndarray) -> np.ndarray:
    """Transform (N, 2) UTM (easting, northing) coords to (N, 2) lat/lon coords."""
    lon, lat = _to_latlon.transform(utm[:, 0], utm[:, 1])
    return np.column_stack([lat, lon])