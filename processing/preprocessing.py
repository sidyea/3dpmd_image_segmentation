import numpy as np
import cv2
from config.settings import roi_x_start, roi_x_end, roi_y_start, roi_y_end

def crop_to_roi(frame: np.ndarray) -> np.ndarray:
    """
    Crop the frame to the specified region of interest (ROI).

    Args:
        frame (np.ndarray): The input frame.

    Returns:
        np.ndarray: The cropped frame.
    """
    roi = (roi_x_start, roi_x_end, roi_y_start, roi_y_end)
    x_start, x_end, y_start, y_end = roi
    return frame[y_start:y_end, x_start:x_end]

def calculate_intensity_metrics(frame: np.ndarray) -> dict:
    """
    Calculate intensity metrics for a given frame.

    Args:
        frame (np.ndarray): The input frame.

    Returns:
        dict: A dictionary containing the calculated intensity metrics.
    """
    # Convert to grayscale if the frame is in color
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray_frame = frame


    # Calculate mean intensity
    mean_intensity = np.mean(gray_frame)

    # Calculate standard deviation of intensity
    std_intensity = np.std(gray_frame)

    # Calculate maximum intensity
    max_intensity = np.max(gray_frame)

    # Calculate minimum intensity
    min_intensity = np.min(gray_frame)

    return {
        "mean": mean_intensity,
        "std": std_intensity,
        "max": max_intensity,
        "min": min_intensity
    }