# Edge Testing Docker Setup

This project supports both CPU-only and GPU-accelerated testing through Docker containers.

## Quick Start

### CPU Version (Default)

For general testing and development:

```bash
./docker-run.sh
```

### GPU Version

For accelerated performance with NVIDIA GPUs:

```bash
./docker-run-gpu.sh
```

## Configuration Files

This repository includes several Docker-related files:

- `Dockerfile` - CPU-optimized container (uses Python slim image)
- `Dockerfile.gpu` - GPU-accelerated container (uses NVIDIA CUDA image)
- `docker-run.sh` - Helper script to build and run the CPU container
- `docker-run-gpu.sh` - Helper script to build and run the GPU container
- `.dockerignore` - Specifies files to exclude from Docker builds
- `DOCKER.md` - Detailed documentation for CPU setup
- `GPU.md` - Detailed documentation for GPU setup

## Choosing Between CPU and GPU

| Feature | CPU Version | GPU Version |
|---------|------------|-------------|
| **Base Image** | Python 3.12 slim | NVIDIA CUDA 12.6 |
| **Dependencies** | PyTorch CPU | PyTorch CUDA, TensorRT |
| **Performance** | Good for testing | Excellent for production |
| **Setup Complexity** | Simple | Requires NVIDIA drivers |
| **Memory Usage** | Lower | Higher (uses VRAM) |
| **Docker Size** | ~1GB | ~4GB |

### When to use CPU version

- For development and testing
- When no GPU is available
- For lightweight deployments
- For compatibility testing

### When to use GPU version

- For maximum performance
- For production workloads
- When processing many instances in parallel
- For benchmarking and stress testing

## Passing Arguments

Both scripts accept additional arguments that are passed to the main application:

```bash
# CPU version with custom arguments
./docker-run.sh --max-instances 8 --log-level DEBUG

# GPU version with custom arguments
./docker-run-gpu.sh --max-instances 4 --log-level DEBUG
```

## Accessing Results

Both versions save results to the `output/` directory, which is mounted as a volume in the container. After running either version, you can find:

- Log files with timestamps
- CSV result files
- Performance metrics

## Further Documentation

- For detailed CPU setup information, see [DOCKER.md](DOCKER.md)
- For detailed GPU setup information, see [GPU.md](GPU.md)
