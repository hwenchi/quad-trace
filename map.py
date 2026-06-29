import argparse
import webbrowser
from pathlib import Path

import folium
from branca.element import MacroElement
from jinja2 import Template

from calibrate import GCPS_LATLON

MAP_FILE = Path("gcps_reference.html")

CLICK_HANDLER = MacroElement()
CLICK_HANDLER._template = Template("""
    {% macro script(this, kwargs) %}
    {{ this._parent.get_name() }}.on('click', function(e) {
        L.popup()
            .setLatLng(e.latlng)
            .setContent('Lat: ' + e.latlng.lat.toFixed(6) +
                        '<br>Lon: ' + e.latlng.lng.toFixed(6))
            .openOn({{ this._parent.get_name() }});
    });
    {% endmacro %}
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pick", action="store_true", help="Pick mode: click to get coordinates")
    args = parser.parse_args()

    center = GCPS_LATLON.mean(axis=0).tolist()
    m = folium.Map(location=center, zoom_start=19, max_zoom=21, tiles="Esri.WorldImagery")

    if args.pick:
        m.add_child(CLICK_HANDLER)
        print("Pick mode: click on the map to get coordinates.")
    else:
        for i, (lat, lon) in enumerate(GCPS_LATLON):
            folium.CircleMarker(location=[lat, lon], radius=3, color="yellow",
                                fill=True, fill_color="yellow").add_to(m)
            folium.Marker(
                location=[lat, lon],
                tooltip=f"Point {i + 1}",
                icon=folium.DivIcon(html=f'<div style="font-size:14px;font-weight:bold;color:yellow;margin-left:16px">{i + 1}</div>'),
            ).add_to(m)

    m.save(str(MAP_FILE))
    webbrowser.open(MAP_FILE.resolve().as_uri())