"""
stress_test_yolo_track.py

Continuously runs Ultraly­tics YOLO tracking on a video (or camera) 
and logs system stats to a CSV for later analysis.
"""

import time
import subprocess
import csv
import argparse
import logging
from pathlib import Path
import uuid

from utils import ensure_test_video
from logger_setup import setup_logger

import psutil
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO Tracking Stress Test with System Metrics Logging"
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)"
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
        "--log-file", "-o", type=str, default="output/stress_stats.csv",
        help="Output CSV file for logging (default: output/stress_stats.csv)"
    )
    return parser.parse_args()

def init_csv(log_file):
    # Ensure the output directory exists
    log_path = Path(log_file)
    log_path.parent.mkdir(exist_ok=True)
    
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


def main():
    args = parse_args()
    
    # Generate a unique run ID
    run_id = str(uuid.uuid4())[:8]
    
    # Set up logger with the specified level
    log_level = getattr(logging, args.log_level)
    log_file = Path("output") / f"yolo_track_{run_id}.log"
    logger = setup_logger(__name__, log_file, level=log_level)
    
    logger.info(f"Starting YOLO tracking stress test (Run ID: {run_id})")
    logger.info(f"Arguments: {vars(args)}")

    # Handle default test video if needed
    if args.source == "test_video.mp4":
        args.source = ensure_test_video(args.source)
        logger.debug(f"Using test video: {args.source}")
    # Validate source
    elif not (args.source.isnumeric() or Path(args.source).exists()):
        logger.error(f"Source '{args.source}' not found.")
        return

    # Initialize CSV for metrics
    logger.debug(f"Initializing CSV metrics file: {args.log_file}")
    init_csv(args.log_file)

    # Load YOLO model once
    logger.info(f"Loading YOLO model: {args.model}")
    model = YOLO(args.model, task="detect")

    logger.info(f"Starting stress test for {args.duration}s "
                f"on source '{args.source}' with model '{args.model}'")
    start_time = time.time()
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

        # Update FPS estimate and log system stats every 100 frames
        if frame_count % 100 == 0:
            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed
            
            # Collect system stats
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
                logger.debug(f"NVIDIA GPU stats: {gpu_pct}%, {gpu_temp}°C")
            except Exception as e:
                gpu_pct, gpu_temp = "N/A", "N/A"
                logger.debug(f"No NVIDIA GPU stats available: {e}")
                
            # Log metrics
            stats_msg = f"[{elapsed:.1f}s] Stats: FPS={current_fps:.1f}, CPU={cpu}%, MEM={mem}%, GPU={gpu_pct}%, GPU_temp={gpu_temp}°C"
            logger.info(stats_msg)
                
            # Write metrics to CSV
            with open(args.log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    round(time.time(), 3),
                    round(current_fps, 2),
                    cpu,
                    mem,
                    gpu_pct,
                    gpu_temp
                ])

        # Stop after desired duration
        if time.time() - start_time >= args.duration:
            break

    # Clean up
    logger.info(f"Stress test complete. Stats saved to: {args.log_file}")

if __name__ == "__main__":
    main()
