"""
Utils module for EdgeTest with helper functions.
"""

from pathlib import Path
import gdown
from logger_setup import setup_logger

# Set up module logger
logger = setup_logger(__name__, "output/edge_test_utils.log")

def ensure_test_video(video_path="test_video.mp4"):
    """
    Ensures the test video exists, downloading it if necessary.
    
    Args:
        video_path: Local path where the video should be stored
        
    Returns:
        Path to the video file
    """
    video_file = Path(video_path)
    
    if not video_file.exists():
        logger.info(f"Test video not found at {video_path}, downloading from Google Drive...")
        
        # Google Drive file ID from the link
        file_id = "15Zjw5MAceckgasf3iYeEifcoPe8jcdRB"
        
        # Use gdown to download from Google Drive
        logger.debug(f"Using file ID: {file_id}")
        gdown.download(
            f"https://drive.google.com/uc?id={file_id}", 
            output=str(video_file), 
            quiet=False
        )
        
        if not video_file.exists():
            logger.error(f"Failed to download test video to {video_path}")
            raise RuntimeError(f"Failed to download test video to {video_path}")
            
        logger.info(f"Download complete: {video_path}")
    else:
        logger.debug(f"Test video already exists at {video_path}")
    
    return str(video_file)
