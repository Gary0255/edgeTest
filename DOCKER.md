# Docker Setup for Edge Testing

This document explains how to use Docker with this Edge Testing project.

## Overview

The Docker setup allows you to run the Edge Testing framework in an isolated container with all required dependencies. The configuration is optimized for CPU-based testing and uses `uv` for efficient Python dependency management.

## Files

- `Dockerfile`: Defines the container configuration
- `.dockerignore`: Specifies which files should be excluded from the Docker build
- `docker-run.sh`: Helper script to build and run the container

## Requirements

- Docker installed on your system
- `uv.lock` and `pyproject.toml` files (already part of the project)

## Running with Docker

### Option 1: Using the helper script

The simplest way to run the edge tests in Docker is to use the provided script:

```bash
./docker-run.sh
```

This will:
1. Build the Docker image (named 'edgetest-cpu')
2. Run the container with the appropriate volume mounting for logs
3. Execute the main.py script with CPU extras

You can pass additional arguments to the script, which will be forwarded to main.py:

```bash
./docker-run.sh --max-instances 8 --log-level DEBUG
```

### Option 2: Manual Docker commands

If you prefer to run Docker commands directly:

```bash
# Build the image
docker build -t edgetest-cpu .

# Run the container
docker run -it --rm \
  -v "$(pwd)/output:/app/output" \
  edgetest-cpu \
  uv run --extra cpu main.py
```

## Understanding the Docker Setup

The Docker configuration:

1. Uses a base image with Python 3.12 and `uv` pre-installed
2. Installs system dependencies required for OpenCV and computer vision functionality (libgl1, etc.)
3. Installs only CPU-specific dependencies (no GPU/XPU packages)
4. Mounts the output directory to persist log files and results
5. Compiles Python bytecode for better performance
6. Uses Docker's caching mechanisms for faster builds

## Customizing the Docker Configuration

If you need to modify the Docker setup:

- Adjust environment variables in the Dockerfile
- Add additional dependencies to pyproject.toml
- Modify the docker-run.sh script for custom run options

## Accessing Results and Logs

After running the Docker container, all results and logs are automatically saved to your local `output` directory due to the volume mounting configured in the `docker-run.sh` script.

### Log Files:
- Main log file: `output/edge_test_main_[run_id].log`
- Parallel stress test logs: `output/parallel_stress_[run_id].log`
- Individual YOLO tracking logs: `output/yolo_track_[run_id].log`
- Where `[run_id]` is a unique identifier for each test run

### CSV Results:
- Batch CSV files: `output/batch_[n]_[i].csv` (where n=batch size, i=instance number)
- Performance stats: `output/stress_stats.csv`

You can view these files using standard Linux tools or process them with data analysis software:

```bash
# View the most recent log
cat output/edge_test_main_*.log | less

# Search for specific information in logs
grep "Sustainable" output/parallel_stress_*.log

# View CSV contents
cat output/stress_stats.csv

# Process CSV data with Python
python -c "import pandas as pd; df = pd.read_csv('output/stress_stats.csv'); print(df.head())"
```

## Troubleshooting

If you encounter Docker-related issues:

1. Make sure Docker is running on your system
2. Check if there are permission issues with the output directory
3. Verify that uv.lock and pyproject.toml files are up to date
4. For more detailed logs, use `--log-level DEBUG` when running the container
