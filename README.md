# QuadTrace

Pedestrian detection and geographic heatmap for the UIUC Main Quad, using the [Quad Cam YouTube live stream](https://www.youtube.com/watch?v=WyrkXojtRXk).

## Pipeline

### Calibration (offline, one-time)

| Script | Purpose |
|--------|---------|
| `map.py` | Display GCPs on satellite map (`--pick` to click and get coordinates) |
| `calibrate.py` | Click pixel coordinates for each GCP, save to `gcps_pixels.npy` |
| `homography.py` | Compute pixel → UTM homography from GCPs, save to `homography.npy` |

### Detection (online, continuous)

Two processes communicate via the `frames/` directory as a queue.

| Script | Purpose |
|--------|---------|
| `ingest.py` | Capture frames from the live stream via yt-dlp + OpenCV, write to `frames/` |
| `pipeline.py` | Poll `frames/` for new frames: frame differencing → UTM → `detections/` parquet |

Use `run.sh` to clean up and start both processes together:

```bash
bash run.sh                  # normal mode
bash run.sh --debug          # save detection overlays to disk for inspection
```

### Visualization (offline, on-demand)

| Script | Purpose |
|--------|---------|
| `visualize.py` | Render accumulated detections as a KDE heatmap in folium |
| `reproject.py` | Recompute UTM coordinates from saved pixel positions after recalibration |
