# imports
from utils import video_loader
from processing.preprocessing import crop_to_roi
import os
import cv2
import numpy as np


def main():
    print("Now running main.py")

    video_path = "C:/Users/shark/VSCodeProjects/FSCM/data/video/sideview/20250122-142515671393.webm" 

    for frame in video_loader.load_video(video_path):
        # Process the frame
        # Example processing: crop to a region of interest (ROI)
        frame = crop_to_roi(frame)

        # Show the frame
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    for i in range(1, 5):
        cv2.destroyAllWindows()
        cv2.waitKey(1)


main()