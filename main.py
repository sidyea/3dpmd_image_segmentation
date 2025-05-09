# imports
from utils import video_loader
import os
import cv2
import numpy as np


def main():
    video_path = "C:/Users/shark/VSCodeProjects/FSCM/data/video/sideview/20250122-142515671393.webm" 
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Load the video using the custom loader
    for frame in video_loader.load_video(video_path):
        # Process the frame (for demonstration, we just show it)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    #cv2.destroyAllWindows()

    for i in range(1, 5):
        cv2.destroyAllWindows()
        cv2.waitKey(1)
