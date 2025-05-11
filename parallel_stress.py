#!/usr/bin/env python3
"""
parallel_stress.py

Spawns increasing numbers of YOLO stress-test subprocesses,
monitors overall CPU%, Memory%, and average FPS, and
reports the max sustainable count.
"""

import subprocess
import time
import psutil
import argparse
import csv
import os
import sys
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser(
        description="Find max parallel YOLO instances before CPU, RAM or FPS limits hit"
    )
    p.add_argument("--test-script", "-t", default="stress_test_yolo_track.py",
                   help="Path to your YOLO stress-test script")
    p.add_argument("--source", "-s", default="test_video.mp4",
                   help="YOLO source (video file, image glob or camera index)")
    p.add_argument("--model", "-m", default="yolo11x.pt",
                   help="YOLO model checkpoint")
    p.add_argument("--duration", "-d", type=int, default=200,
                   help="Seconds to run each batch (default: 200s)")
    p.add_argument("--interval", "-i", type=int, default=10,
                   help="Sampling interval in seconds (default: 10s)")
    p.add_argument("--max-instances", "-n", type=int, default=16,
                   help="Maximum parallel instances to try (default: 16)")
    p.add_argument("--cpu-threshold",   type=float, default=90.0,
                   help="Avg CPU%% threshold (default: 90.0)")
    p.add_argument("--mem-threshold",   type=float, default=90.0,
                   help="Avg Memory%% threshold (default: 90.0)")
    p.add_argument("--fps-threshold",   type=float, default=3.0,
                   help="Avg FPS threshold (default: 3.0)")
    return p.parse_args()

def launch_instances(n, args):
    procs = []
    for i in range(n):
        logfile = f"batch_{n}_{i}.csv"
        
        # Use virtual environment python if available
        venv_root = os.environ.get("VIRTUAL_ENV")
        if venv_root:
            if sys.platform == "win32":
                # Windows path
                python = os.path.join(venv_root, "Scripts", "python.exe")
            else:
                # Unix-like path (Linux/macOS)
                python = os.path.join(venv_root, "bin", "python")
        else:
            python = sys.executable
            
        cmd = [
            python, args.test_script,
            "--source", args.source,
            "--model", args.model,
            "--duration", str(args.duration),
            "--log-file", logfile,
            "--log-interval", str(args.interval)
        ]
        p = subprocess.Popen(cmd)
        procs.append((p, Path(logfile)))
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
    """Read the last 'avg_fps' value from a CSV log file."""
    last_fps = None
    try:
        with logfile.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                last_fps = float(row["avg_fps"])
    except Exception:
        pass
    return last_fps

def main():
    args = parse_args()
    sustainable = 0

    for n in range(1, args.max_instances + 1):
        print(f"\n→ Testing {n} parallel instances…")
        # Launch
        procs = launch_instances(n, args)
        time.sleep(10)  # let them spin up

        # Monitor CPU & Memory
        avg_cpu, avg_mem = monitor_system(args.duration, args.interval)
        print(f"   Avg CPU%: {avg_cpu:.1f}, Avg Mem%: {avg_mem:.1f}")

        # Terminate all
        alive = 0
        for p, _ in procs:
            if p.poll() is None:
                alive += 1
            else:
                # exited on its own → inspect p.returncode
                if p.returncode == 0:
                    alive += 1
            p.terminate()
        print(f"   Processes alive at end: {alive}/{n}")

        # Parse FPS logs
        fps_vals = []
        for _, logfile in procs:
            fps = parse_fps_from_csv(logfile)
            if fps is not None:
                fps_vals.append(fps)
        avg_fps = (sum(fps_vals) / len(fps_vals)) if fps_vals else 0.0
        print(f"   Avg FPS across instances: {avg_fps:.1f}")

        # Check thresholds
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
