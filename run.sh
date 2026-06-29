#!/bin/bash
set -e

echo "Cleaning up..."
rm -rf frames/ detections/
mkdir -p frames

echo "Starting pipeline..."
uv run python ingest.py &
INGEST_PID=$!

uv run python pipeline.py "$@" &
PIPELINE_PID=$!

trap "kill $INGEST_PID $PIPELINE_PID 2>/dev/null; echo 'Shutdown complete.'" EXIT
wait
