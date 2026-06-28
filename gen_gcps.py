import webbrowser
from pathlib import Path

import folium
import numpy as np

from geo import latlon_to_utm, utm_to_latlon

N_COLS = 3
N_ROWS = 9
MAP_FILE = Path("grid_reference.html")

CORNERS_LATLON = np.array([
    [40.108806, -88.227611],  # 40°06'31.7"N 88°13'39.4"W (NW)
    [40.108778, -88.226917],  # 40°06'31.6"N 88°13'36.9"W (NE)
    [40.106417, -88.227556],  # 40°06'23.1"N 88°13'39.2"W (SW)
    [40.106417, -88.226861],  # 40°06'23.1"N 88°13'36.7"W (SE)
])


def generate_grid(corners_utm: np.ndarray, n_cols: int, n_rows: int) -> np.ndarray:
    e_min, n_min = corners_utm.min(axis=0)
    e_max, n_max = corners_utm.max(axis=0)
    eastings = np.linspace(e_min, e_max, n_cols)
    northings = np.linspace(n_min, n_max, n_rows)
    ee, nn = np.meshgrid(eastings, northings)
    return np.column_stack([ee.ravel(), nn.ravel()])


def build_grid_map(grid_latlon: np.ndarray) -> None:
    center = grid_latlon.mean(axis=0).tolist()
    m = folium.Map(location=center, zoom_start=19, max_zoom=21, tiles="Esri.WorldImagery")
    for i, (lat, lon) in enumerate(grid_latlon):
        folium.CircleMarker(location=[lat, lon], radius=3, color="yellow",
                            fill=True, fill_color="yellow").add_to(m)
        folium.Marker(
            location=[lat, lon],
            tooltip=f"Point {i + 1}",
            icon=folium.DivIcon(html=f'<div style="font-size:14px;font-weight:bold;color:yellow;margin-left:16px">{i + 1}</div>'),
        ).add_to(m)
    m.save(str(MAP_FILE))
    webbrowser.open(MAP_FILE.resolve().as_uri())
    print(f"Grid map opened: {MAP_FILE} ({len(grid_latlon)} points)")


if __name__ == "__main__":
    corners_utm = latlon_to_utm(CORNERS_LATLON)
    grid_utm = generate_grid(corners_utm, N_COLS, N_ROWS)
    grid_latlon = utm_to_latlon(grid_utm)
    build_grid_map(grid_latlon)
    np.save("gcps_latlon.npy", grid_latlon)
    print(f"Saved gcps_latlon.npy ({len(grid_latlon)} points)")