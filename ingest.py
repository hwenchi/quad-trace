import subprocess
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import piexif

STREAM_URL = "https://www.youtube.com/watch?v=WyrkXojtRXk"
CACHE_FILE = Path(".stream_url_cache")


def resolve_stream_url(youtube_url: str) -> str:
    result = subprocess.run(
        ["yt-dlp", "-g", "--no-playlist", "--format", "best", youtube_url],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def get_stream_url(youtube_url: str) -> str:
    if CACHE_FILE.exists():
        cached = CACHE_FILE.read_text().strip()
        expire = int(parse_qs(urlparse(cached).query).get("expire", [0])[0])
        if expire > time.time() + 60:  # at least 1 min left
            return cached

    print("Resolving stream URL...")
    fresh_url = resolve_stream_url(youtube_url)
    CACHE_FILE.write_text(fresh_url)
    return fresh_url


def write_frame(frame: cv2.typing.MatLike, path: Path, timestamp: int) -> None:
    cv2.imwrite(str(path), frame)
    dt = time.strftime("%Y:%m:%d %H:%M:%S", time.localtime(timestamp))
    exif = piexif.dump({"Exif": {piexif.ExifIFD.DateTimeOriginal: dt}})
    piexif.insert(exif, str(path))


def capture_frames(stream_url: str, output_dir: Path, sample_every: int = 30) -> None:
    """Read frames from an HLS stream and save every sample_every-th frame to disk."""
    output_dir.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        raise RuntimeError("Failed to open stream")

    frame_idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream ended or read error.")
            break

        if frame_idx % sample_every == 0:
            path = output_dir / f"{saved:05d}.jpg"
            write_frame(frame, path, int(time.time()))
            print(f"Saved {path}")
            saved += 1

        frame_idx += 1

    cap.release()


if __name__ == "__main__":
    url = get_stream_url(STREAM_URL)
    capture_frames(url, Path("frames"))
