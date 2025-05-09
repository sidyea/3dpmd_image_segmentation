import numpy as np
from config.settings import roi_x_start, roi_x_end, roi_y_start, roi_y_end

def crop_to_roi(frame: np.ndarray, roi: tuple) -> np.ndarray:
    """
    Crop the frame to the specified region of interest (ROI).

    Args:
        frame (np.ndarray): The input frame.
        roi (tuple): A tuple containing the coordinates of the ROI in the format (x_start, x_end, y_start, y_end).

    Returns:
        np.ndarray: The cropped frame.
    """
    roi = (roi_x_start, roi_x_end, roi_y_start, roi_y_end)
    x_start, x_end, y_start, y_end = roi
    return frame[y_start:y_end, x_start:x_end]