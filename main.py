# imports
from utils import video_loader
from processing.preprocessing import crop_to_roi, calculate_intensity_metrics
import cv2
import numpy as np


def main():
    print("Now running main.py")

    results = []

    # open source
    video_path = "C:/Users/shark/VSCodeProjects/FSCM/data/video/sideview/20250122-142515671393.webm" 

    for i, frame in enumerate(video_loader.load_video(video_path)):
        # Crop / Transform
        frame = crop_to_roi(frame)      # crop (values in settings.py)
        frame = frame.astype("float32")/255.0     # global normalization to [0, 1]

        # Intensity metrics
        metrics = calculate_intensity_metrics(frame)
        
        # save results



        # Show the frame
        # Draw metrics on the frame
        cv2.putText(frame, f"Mean: {metrics['mean']:.3f}", (400, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        
    
    # Clean up
    for i in range(1, 5):
        cv2.destroyAllWindows()
        cv2.waitKey(1)


main()