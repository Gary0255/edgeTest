# NVIDIA GPU Support for Edge Testing

This document explains how to use the Edge Testing framework with NVIDIA GPU acceleration.

## Prerequisites

Before you can run the GPU-enabled Docker container, you need to have the following:

1. **NVIDIA GPU** with compute capability 3.5 or higher
2. **NVIDIA Driver** (version 525 or higher recommended)
3. **Docker** with NVIDIA Container Toolkit installed
4. **uv.lock** and **pyproject.toml** files (already part of the project)

## Installing the NVIDIA Container Toolkit

The NVIDIA Container Toolkit allows Docker containers to access NVIDIA GPUs. If you haven't installed it yet, follow these steps:

### Ubuntu/Debian

```bash
# Add the package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install the NVIDIA Container Toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Verify Installation

After installation, verify that the NVIDIA Container Toolkit is working:

```bash
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
```

If successful, you should see output similar to:

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.104.05   Driver Version: 535.104.05   CUDA Version: 12.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0  On |                  N/A |
|  0%   45C    P8    12W / 200W |    456MiB /  8192MiB |      1%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
```

## Running Edge Testing with GPU Support

### Option 1: Using the helper script

The simplest way to run the GPU-accelerated edge tests is to use the provided script:

```bash
./docker-run-gpu.sh
```

This will:
1. Build the Docker image (named 'edgetest-gpu') using Dockerfile.gpu
2. Run the container with NVIDIA GPU access
3. Execute the main.py script with GPU extras

You can pass additional arguments to the script, which will be forwarded to main.py:

```bash
./docker-run-gpu.sh --max-instances 4 --log-level DEBUG
```

### Option 2: Manual Docker commands

If you prefer to run Docker commands directly:

```bash
# Build the image
docker build -t edgetest-gpu -f Dockerfile.gpu .

# Run the container with GPU support
docker run -it --rm \
  --gpus all \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -v "$(pwd)/output:/app/output" \
  edgetest-gpu \
  uv run --extra gpu main.py
```

## GPU Configuration

The GPU setup includes:

1. **Base Image**: NVIDIA CUDA 12.6 Runtime on Ubuntu 22.04
2. **Dependencies**:
   - PyTorch with CUDA 12.6 support
   - TensorRT for accelerated inference
   - onnxruntime-gpu for optimized model execution
3. **Environment Variables**:
   - CUDA_HOME: Set to the CUDA installation directory
   - Path and LD_LIBRARY_PATH: Updated to include CUDA binaries and libraries

## Performance Considerations

### Expected Speedup

GPU acceleration can significantly improve performance, especially for YOLO inference:
- Model loading: 1.5-2x faster
- Inference speed: 3-10x faster depending on GPU
- Batch processing: 5-20x more efficient

### Memory Requirements

When running on GPUs, be aware of the following:
- VRAM usage scales with model size and batch size
- For YOLOv8, expect ~2-4GB VRAM usage per instance
- Memory requirements increase with parallel instances
- Consider monitoring GPU memory with `nvidia-smi` during testing

## Accessing Results and Logs

The logging system works the same as with CPU testing. All results and logs are automatically saved to your local `output` directory due to the volume mounting configured in the `docker-run-gpu.sh` script.

## Troubleshooting

### Common Issues

1. **"Error: driver failed programming external connectivity..."**
   - Try restarting the Docker service: `sudo systemctl restart docker`

2. **"ERROR: Unable to load GPU drivers with current configuration..."**
   - Make sure your NVIDIA drivers are compatible with the CUDA version in the container
   - Check that the NVIDIA Container Toolkit is properly installed

3. **"Failed to initialize NVML: Driver/library version mismatch"**
   - This indicates a mismatch between the driver version on your host and what's expected in the container
   - Consider using a different CUDA base image that matches your driver version

4. **Out of memory errors**
   - Reduce the number of parallel instances
   - Use a smaller model
   - Configure CUDA to use less memory per process

### GPU Visibility Check

If you suspect the container can't see the GPU, run:

```bash
docker run --rm --gpus all edgetest-gpu nvidia-smi
```

### TensorRT Issues

If TensorRT fails to initialize:
- Try reinstalling it within the container: `pip install -U tensorrt`
- Check compatibility between TensorRT and your GPU
- Verify your NVIDIA driver supports the required CUDA version

## Comparison with CPU Version

|                | CPU Version | GPU Version |
|----------------|-------------|-------------|
| Base Image     | Python 3.12 slim | NVIDIA CUDA 12.6 |
| Dependencies   | PyTorch CPU | PyTorch CUDA, TensorRT |
| Performance    | Limited by CPU cores | Limited by GPU architecture |
| Memory Usage   | System RAM | System RAM + VRAM |
| Docker Flags   | Standard | `--gpus all` |
| Best For       | Testing, development | Production, benchmarking |

## Further Resources

- [NVIDIA Container Toolkit Documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html)
- [Docker GPU Documentation](https://docs.docker.com/config/containers/resource_constraints/#gpu)
- [TensorRT Documentation](https://developer.nvidia.com/tensorrt)
