# EdgeTest

## Overview
EdgeTest is a benchmarking toolkit for YOLO object detection models with a focus on edge device deployment. It automatically detects available hardware acceleration, exports models to optimized formats, and performs parallel stress testing to determine the maximum sustainable inference throughput.

## Features

- **Automatic Hardware Detection**: 
  - NVIDIA GPUs with TensorRT acceleration
  - Intel CPUs with OpenVINO acceleration
  - Fallback to standard PyTorch inference

- **Model Optimization**:
  - PyTorch → TensorRT engine export for NVIDIA GPUs
  - PyTorch → OpenVINO IR export for Intel CPUs

- **Comprehensive Testing**:
  - Parallel instance stress testing
  - Progressive scaling until system limits are reached
  - Real-time performance monitoring
  - Detailed CSV logging of metrics

- **System Monitoring**:
  - CPU utilization tracking
  - Memory consumption analysis
  - GPU usage and temperature monitoring
  - Real-time FPS calculation

## Requirements

- Python 3.12+
- NVIDIA GPU with CUDA/cuDNN (optional, for TensorRT)
- Intel CPU (optional, for OpenVINO)
- Dependencies:
  ```
  gdown>=5.2.0
  onnx>=1.17.0
  onnxruntime-gpu>=1.21.1
  onnxslim>=0.1.51
  tensorrt>=10.10.0.31
  ultralytics>=8.3.129
  ```

## Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -e .
   ```

## Usage

### Basic Usage

Run stress testing with automatic hardware detection and optimization:

```bash
python main.py --source /path/to/video.mp4 --model-pt /path/to/model.pt
```

If you use the default test video path (test_video.mp4) without providing a file, the system will automatically download a sample video from Google Drive.

### Advanced Options

```bash
python main.py \
  --source /path/to/video.mp4 \
  --model-pt yolo11x.pt \
  --duration 300 \
  --interval 5 \
  --max-instances 16 \
  --cpu-threshold 95.0 \
  --mem-threshold 90.0 \
  --fps-threshold 5.0
```

### Default Test Video

When using the default test video path (`test_video.mp4` or `./test_video.mp4`), the system will:
1. Check if the file exists locally
2. If not found, automatically download it from Google Drive
3. Use the downloaded file for testing

The test video source URL: https://drive.google.com/file/d/15Zjw5MAceckgasf3iYeEifcoPe8jcdRB/view?usp=sharing

### Component Scripts

The toolkit consists of three main components that can also be run independently:

#### 1. Main Script (`main.py`)

Detects hardware, exports to optimized formats, and runs stress tests:

```bash
python main.py --source /path/to/video.mp4 --model-pt /path/to/model.pt
```

#### 2. Parallel Stress Testing (`parallel_stress.py`)

Tests multiple parallel instances of the model:

```bash
python parallel_stress.py \
  --source /path/to/video.mp4 \
  --model /path/to/model.engine \
  --max-instances 16
```

#### 3. Single Instance Testing (`stress_test_yolo_track.py`)

Runs a single instance with detailed monitoring:

```bash
python stress_test_yolo_track.py \
  --source /path/to/video.mp4 \
  --model /path/to/model.engine \
  --duration 300 \
  --log-file stats.csv
```

## Command-line Arguments

### Main Script (`main.py`)

| Argument | Description | Default |
|----------|-------------|---------|
| `--source`, `-s` | Video file, camera index, etc. | (required) |
| `--model-pt`, `-p` | YOLO .pt checkpoint path | (required) |
| `--duration`, `-d` | Test duration in seconds | 300 |
| `--interval`, `-i` | Sampling interval in seconds | 5 |
| `--max-instances`, `-n` | Maximum parallel instances to try | 16 |
| `--cpu-threshold` | Average CPU% threshold | 95.0 |
| `--mem-threshold` | Average Memory% threshold | 90.0 |
| `--fps-threshold` | Average FPS threshold | 5.0 |

### Parallel Stress Test (`parallel_stress.py`)

| Argument | Description | Default |
|----------|-------------|---------|
| `--source`, `-s` | YOLO source | ./test_video.mp4 |
| `--model-pt`, `-p` | Path to YOLO .pt checkpoint | yolo11x.pt |
| `--duration`, `-d` | Seconds to run each batch | 200 |
| `--interval`, `-i` | Sampling interval in seconds | 10 |
| `--max-instances`, `-n` | Maximum parallel instances | 16 |
| `--cpu-threshold` | CPU% threshold | 90.0 |
| `--mem-threshold` | Memory% threshold | 90.0 |
| `--fps-threshold` | FPS threshold | 3.0 |

### Stress Test (`stress_test_yolo_track.py`)

| Argument | Description | Default |
|----------|-------------|---------|
| `--source`, `-s` | Input source (video/camera) | test_video.mp4 |
| `--model`, `-m` | YOLO model path | yolo11x.onnx |
| `--duration`, `-d` | Test duration in seconds | 200 |
| `--log-interval`, `-i` | Logging interval in seconds | 10 |
| `--log-file`, `-o` | Output CSV file | stress_stats.csv |

## Output

The tool produces:

1. **CSV log files** with detailed metrics including:
   - Timestamps
   - Average FPS
   - CPU utilization %
   - Memory usage %
   - GPU utilization % (if NVIDIA GPU present)
   - GPU temperature (if NVIDIA GPU present)

2. **Console output** showing:
   - Hardware detection results
   - Model export progress
   - Per-batch test results
   - Maximum sustainable parallel instances

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
