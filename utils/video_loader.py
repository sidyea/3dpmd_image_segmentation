# imports

import os
import cv2
import numpy as np
from typing import Generator

def load_video(video_path: str) -> Generator[np.ndarray, None, None]:
    """
    Load a video file and yield frames as numpy arrays.

    Args:
        video_path (str): Path to the video file.

    Yields:
        np.ndarray: Frames of the video as numpy arrays.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a valid video.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame

    cap.release()
