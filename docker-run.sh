#!/bin/bash
set -e

# Script to build and run the Edge Testing Docker image
# Usage: ./docker-run.sh [extra args]
# Example: ./docker-run.sh --max-instances 8

# Build the Docker image
echo "Building Docker image 'edgetest-cpu'..."
docker build -t edgetest-cpu .

# Run the container with the appropriate volume mapping for output directory
echo "Running Edge Testing container..."
docker run -it --rm \
  -v "$(pwd)/output:/app/output" \
  edgetest-cpu \
  uv run --extra cpu main.py "$@"

echo "Done."
