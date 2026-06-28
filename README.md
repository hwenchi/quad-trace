# QuadTrace

Pedestrian detection and geographic heatmap for the UIUC Main Quad, using the [quadcam YouTube live stream](https://www.youtube.com/watch?v=WyrkXojtRXk).

## Pipeline

### Calibration (offline, one-time)

| Script | Purpose |
|--------|---------|
| `gen_gcps.py` | Generate a reference grid of ground control points, open in browser |
| `calibrate.py` | Click pixel coordinates for each GCP, save to `gcps_pixels.npy` |
| `homography.py` | Compute pixel → UTM homography from GCPs, save to `homography.npy` |

### Detection (online, continuous)

Two processes communicate via the `frames/` directory as a queue.

| Script | Purpose |
|--------|---------|
| `ingest.py` | Capture frames from the live stream via yt-dlp + OpenCV, write to `frames/` |
| `pipeline.py` | Poll `frames/` for new frames: MOG2 detection → UTM → `detections.parquet` |

Use `run.sh` to clean up and start both processes together:

```bash
bash run.sh                  # normal mode
bash run.sh --synthetic      # overlay synthetic moving blobs
bash run.sh --synthetic --debug  # also save frames, masks, overlays to disk
```

### Visualization (offline, on-demand)

| Script | Purpose |
|--------|---------|
| `visualize.py` | Render accumulated detections as a KDE heatmap in folium |

## Install

```bash
uv sync
```