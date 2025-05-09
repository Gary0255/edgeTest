"""
parallel_stress.py

1) Detects NVIDIA GPU support.
2) If present, exports yolo11x.pt → yolo11x.engine via YOLO.export().
3) Spawns parallel stress-test subprocesses using the .engine (or .pt if no GPU).
4) Monitors CPU%, Memory%, and average FPS against thresholds.
"""

import subprocess
import time
import psutil
import argparse
import csv
from pathlib import Path
from ultralytics import YOLO

from utils import ensure_test_video

def parse_args():
    p = argparse.ArgumentParser(
        description="Parallel YOLO Stress Test with optional TensorRT export"
    )
    p.add_argument("--source",        "-s", "./test_video.mp4",
                   help="YOLO source (video file, image glob or camera index)")
    p.add_argument("--model-pt",      "-p", default="yolo11x.pt",
                   help="Path to your YOLO .pt checkpoint")
    p.add_argument("--duration",      "-d", type=int, default=200,
                   help="Seconds to run each batch (default: 200s)")
    p.add_argument("--interval",      "-i", type=int, default=10,
                   help="Sampling interval in seconds (default: 10s)")
    p.add_argument("--max-instances", "-n", type=int, default=16,
                   help="Maximum parallel instances to try (default: 16)")
    p.add_argument("--cpu-threshold", type=float, default=90.0,
                   help="Avg CPU%% threshold (default: 90.0)")
    p.add_argument("--mem-threshold", type=float, default=90.0,
                   help="Avg Memory%% threshold (default: 90.0)")
    p.add_argument("--fps-threshold", type=float, default=3.0,
                   help="Avg FPS threshold (default: 3.0)")
    return p.parse_args()

def has_nvidia_gpu():
    """Return True if nvidia-smi is callable and a GPU is present."""
    try:
        res = subprocess.run(
            ["nvidia-smi","--query-gpu=name","--format=csv,noheader"],
            capture_output=True, text=True, check=True
        )
        return bool(res.stdout.strip())
    except Exception:
        return False

def ensure_engine(pt_path: Path):
    """
    If <stem>.engine doesn't exist, export the .pt checkpoint
    to TensorRT engine using Ultraly­tics YOLO.export().
    Returns the Path to the engine file.
    """
    engine_path = pt_path.with_suffix('.engine')
    if engine_path.exists():
        return engine_path

    print(f"[INFO] Exporting {pt_path.name} → {engine_path.name} (TensorRT)…")
    model = YOLO(str(pt_path))
    model.export(format='engine', device=0)  # device=0 for first GPU
    if not engine_path.exists():
        raise RuntimeError(f"Failed to create engine at {engine_path}")
    print(f"[INFO] Export complete.")
    return engine_path

def launch_instances(n, model_path, args):
    procs = []
    for i in range(n):
        logfile = Path(f"batch_{n}_{i}.csv")
        cmd = [
            "python3", "stress_test_yolo_track.py",
            "--source", args.source,
            "--model", str(model_path),
            "--duration", str(args.duration),
            "--log-file", str(logfile),
            "--log-interval", str(args.interval)
        ]
        p = subprocess.Popen(cmd)
        procs.append((p, logfile))
    return procs

def monitor_system(duration, interval):
    cpu_samples, mem_samples = [], []
    start = time.time()
    while time.time() - start < duration:
        cpu_samples.append(psutil.cpu_percent(interval=None))
        mem_samples.append(psutil.virtual_memory().percent)
        time.sleep(interval)
    return (sum(cpu_samples) / len(cpu_samples),
            sum(mem_samples) / len(mem_samples))

def parse_fps_from_csv(logfile: Path):
    last_fps = None
    try:
        with logfile.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                last_fps = float(row["avg_fps"])
    except Exception:
        pass
    return last_fps or 0.0

def main():
    args = parse_args()

    # 0) Handle default test video if needed
    if args.source == "./test_video.mp4":
        args.source = ensure_test_video("test_video.mp4")

    # 1) GPU detection & optional engine export
    model_path = Path(args.model_pt)
    if has_nvidia_gpu():
        print("[INFO] NVIDIA GPU detected.")
        model_path = ensure_engine(model_path)
    else:
        print("[INFO] No NVIDIA GPU found — using .pt directly.")

    sustainable = 0
    for n in range(1, args.max_instances + 1):
        print(f"\n→ Testing {n} parallel instances with model '{model_path.name}' …")
        procs = launch_instances(n, model_path, args)
        time.sleep(10)  # warm-up

        avg_cpu, avg_mem = monitor_system(args.duration, args.interval)
        print(f"   Avg CPU%: {avg_cpu:.1f}, Avg Mem%: {avg_mem:.1f}")

        # terminate and count survivors
        alive = sum(1 for p,_ in procs if p.poll() is None)
        for p,_ in procs: p.terminate()
        print(f"   Processes alive: {alive}/{n}")

        # collect FPS
        fps_vals = [parse_fps_from_csv(log) for _,log in procs]
        avg_fps   = sum(fps_vals)/len(fps_vals) if fps_vals else 0.0
        print(f"   Avg FPS: {avg_fps:.1f}")

        # check thresholds
        if (alive < n or
            avg_cpu > args.cpu_threshold or
            avg_mem > args.mem_threshold or
            avg_fps < args.fps_threshold):
            print(f"   ✗ Unsustainable at N={n}")
            break
        else:
            print(f"   ✓ Sustainable at N={n}")
            sustainable = n

    print(f"\nMax sustainable parallel instances ≈ {sustainable}")

if __name__ == "__main__":
    main()
