"""
stress_test_yolo_track.py

Continuously runs Ultraly­tics YOLO tracking on a video (or camera) 
and logs system stats to a CSV for later analysis.
"""

import time
import threading
import subprocess
import csv
import argparse
from pathlib import Path

from utils import ensure_test_video

import psutil
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO Tracking Stress Test with System Metrics Logging"
    )
    parser.add_argument(
        "--source", "-s", type=str, default="test_video.mp4",
        help="Input source (video file, image glob or camera index)"
    )
    parser.add_argument(
        "--model", "-m", type=str, default="yolo11x.pt",
        help="YOLO model checkpoint path"
    )
    parser.add_argument(
        "--duration", "-d", type=int, default=200,
        help="Test duration in seconds (default: 200)"
    )
    parser.add_argument(
        "--log-interval", "-i", type=int, default=10,
        help="Seconds between system‐stats logs (default: 10)"
    )
    parser.add_argument(
        "--log-file", "-o", type=str, default="stress_stats.csv",
        help="Output CSV file for logging (default: stress_stats.csv)"
    )
    return parser.parse_args()

def init_csv(log_file):
    # Write header
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "avg_fps",
            "cpu_pct",
            "mem_pct",
            "gpu_pct",
            "gpu_temp"
        ])

def log_system_stats(log_file, log_interval, stop_event, current_fps):
    """
    Background thread: logs CPU%, RAM%, GPU% and GPU temp every log_interval seconds.
    """
    while not stop_event.is_set():
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory().percent

        # Query NVIDIA GPU stats, if available
        try:
            gpu_query = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=utilization.gpu,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=True
            )
            gpu_pct, gpu_temp = gpu_query.stdout.strip().split(", ")
        except Exception:
            gpu_pct, gpu_temp = "N/A", "N/A"

        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                round(time.time(), 3),
                round(current_fps[0], 2),
                cpu,
                mem,
                gpu_pct,
                gpu_temp
            ])

        time.sleep(log_interval)

def main():
    args = parse_args()

    # Handle default test video if needed
    if args.source == "test_video.mp4":
        args.source = ensure_test_video(args.source)
    # Validate source
    elif not (args.source.isnumeric() or Path(args.source).exists()):
        print(f"[ERROR] Source '{args.source}' not found.")
        return

    init_csv(args.log_file)

    # Shared state for logging thread
    stop_event  = threading.Event()
    current_fps = [0.0]

    # Start logging thread
    logger = threading.Thread(
        target=log_system_stats,
        args=(args.log_file, args.log_interval, stop_event, current_fps),
        daemon=True
    )
    logger.start()

    # Load YOLO model once
    model = YOLO(args.model)

    print(f"Starting stress test for {args.duration}s "
          f"on source '{args.source}' with model '{args.model}'")
    start_time  = time.time()
    frame_count = 0

    # Run tracking loop
    for result in model.track(
        source=args.source,
        stream=True,
        persist=True,
        show=False,
        save=False,
        tracker='botsort.yaml',
        verbose=False,
    ):
        frame_count += 1

        # Update FPS estimate every 100 frames
        if frame_count % 100 == 0:
            elapsed        = time.time() - start_time
            current_fps[0] = frame_count / elapsed
            print(f"[{elapsed:.1f}s] avg FPS: {current_fps[0]:.1f}")

        # Stop after desired duration
        if time.time() - start_time >= args.duration:
            break

    # Clean up
    stop_event.set()
    logger.join()
    print("Stress test complete. Stats saved to:", args.log_file)

if __name__ == "__main__":
    main()
