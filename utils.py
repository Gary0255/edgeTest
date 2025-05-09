"""
Utils module for EdgeTest with helper functions.
"""

from pathlib import Path
import gdown

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
        print(f"Test video not found at {video_path}, downloading from Google Drive...")
        
        # Google Drive file ID from the link
        file_id = "15Zjw5MAceckgasf3iYeEifcoPe8jcdRB"
        
        # Use gdown to download from Google Drive
        gdown.download(
            f"https://drive.google.com/uc?id={file_id}", 
            output=str(video_file), 
            quiet=False
        )
        
        if not video_file.exists():
            raise RuntimeError(f"Failed to download test video to {video_path}")
            
        print(f"Download complete: {video_path}")
    
    return str(video_file)
