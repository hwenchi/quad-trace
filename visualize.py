import webbrowser
from pathlib import Path

import folium
import folium.plugins
import pandas as pd

from geo import utm_to_latlon

INPUT = Path("detections")
OUTPUT = Path("heatmap.html")

CENTER = [40.107542, -88.227222]
HEATMAP_RADIUS = 10
HEATMAP_BLUR = 10


def render(input_path: Path, output_path: Path) -> None:
    df = pd.read_parquet(input_path)
    if df.empty:
        print("No detections found.")
        return

    utm = df[["easting", "northing"]].to_numpy()
    latlon = utm_to_latlon(utm)

    m = folium.Map(location=CENTER, zoom_start=19, max_zoom=21, tiles="Esri.WorldImagery")
    folium.plugins.HeatMap(latlon.tolist(), radius=HEATMAP_RADIUS, blur=HEATMAP_BLUR).add_to(m)

    m.save(str(output_path))
    webbrowser.open(output_path.resolve().as_uri())
    print(f"Saved {output_path} ({len(df)} detections)")


if __name__ == "__main__":
    render(INPUT, OUTPUT)
