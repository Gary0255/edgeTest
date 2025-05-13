#!/bin/bash
set -e

# Script to build and run the Edge Testing Docker image with GPU support
# Usage: ./docker-run-gpu.sh [extra args]
# Example: ./docker-run-gpu.sh --max-instances 4

# Build the Docker image with GPU tag
echo "Building Docker image 'edgetest-gpu'..."
docker build -t edgetest-gpu -f Dockerfile.gpu .

# Run the container with GPU support and appropriate volume mapping
echo "Running Edge Testing container with GPU support..."
docker run -it --rm \
  --gpus all \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -v "$(pwd)/output:/app/output" \
  edgetest-gpu \
  uv run --extra gpu main.py "$@"

echo "Done."
