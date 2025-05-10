"""
main.py

1) Detects NVIDIA GPU support; if found, exports PT→TensorRT engine.
2) Else if on Intel CPU, exports PT→OpenVINO IR.
3) Calls parallel_stress.py with the exported model path.
"""

import subprocess
import sys
import os
import argparse
from pathlib import Path
from ultralytics import YOLO

def has_nvidia_gpu():
    """Check for NVIDIA GPU support using PyTorch's CUDA API"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

def is_intel_gpu():
    """Check for Intel GPU support using PyTorch's XPU API"""
    try:
        import torch
        return torch.xpu.is_available() if hasattr(torch, 'xpu') else False
    except ImportError:
        return False

def export_model(pt_path: Path, fmt: str, device: str = None):
    """
    Export a .pt checkpoint to the desired format via Ultraly­tics YOLO.export().
      fmt: "engine" for TensorRT, or "openvino".
      device: e.g. "0" for GPU, or None (uses CPU)
    Returns the Path to the exported model.
    """
    if fmt == "openvino":
        out = Path(str(pt_path.with_suffix("")) + "_openvino_model/")
    else:
        out = pt_path.with_suffix(f".{fmt}")
    if out.exists():
        print(f"[INFO] Found existing {out.name}")
        return out

    print(f"[INFO] Exporting {pt_path.name} → {out.name} (format={fmt})…")
    model = YOLO(str(pt_path))
    kwargs = {"format": fmt}
    if device is not None:
        kwargs["device"] = device
    model.export(**kwargs)
    if not out.exists():
        raise RuntimeError(f"Failed to export to {out}")
    print(f"[INFO] Exported to {out.name}")
    return out

def main():
    p = argparse.ArgumentParser(
        description="Detect hardware, export model, then run parallel stress test"
    )
    p.add_argument("--source",      "-s", default="test_video.mp4",
                   help="YOLO source (video file, camera index, etc.)")
    p.add_argument("--model-pt",    "-p", default="yolo11x.pt",
                   help="Path to your YOLO .pt checkpoint")
    p.add_argument("--duration",    "-d", type=int, default=200,
                   help="Seconds to run each batch")
    p.add_argument("--interval",    "-i", type=int, default=10,
                   help="Sampling interval in seconds")
    p.add_argument("--max-instances","-n", type=int, default=16,
                   help="Maximum parallel instances to try")
    p.add_argument("--cpu-threshold", type=float, default=90.0,
                   help="Avg CPU% threshold")
    p.add_argument("--mem-threshold", type=float, default=90.0,
                   help="Avg Memory% threshold")
    p.add_argument("--fps-threshold", type=float, default=3.0,
                   help="Avg FPS threshold")
    args = p.parse_args()

    pt_path = Path(args.model_pt)
    if has_nvidia_gpu():
        print("[INFO] NVIDIA GPU detected.")
        model_file = export_model(pt_path, fmt="engine", device="0")
    elif is_intel_gpu():
        print("[INFO] Intel GPU detected.")
        model_file = export_model(pt_path, fmt="openvino")
    else:
        print("[INFO] No supported GPU detected — using .pt directly.")
        model_file = pt_path

    # Build subprocess call to parallel_stress.py
    venv_root = os.environ.get("VIRTUAL_ENV")
    if venv_root:
        # Windows path:
        python = os.path.join(venv_root, "Scripts", "python.exe")
    else:
        python = sys.executable
        
    cmd = [
        python, "parallel_stress.py",
        "--source",       args.source,
        "--test-script",  "stress_test_yolo_track.py",
        "--model",        str(model_file),
        "--duration",     str(args.duration),
        "--interval",     str(args.interval),
        "--max-instances",str(args.max_instances),
        "--cpu-threshold",str(args.cpu_threshold),
        "--mem-threshold",str(args.mem_threshold),
        "--fps-threshold",str(args.fps_threshold),
    ]
    print(f"[INFO] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
