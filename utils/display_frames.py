import cv2
from config.settings import slice_start, slice_end

def draw_results(frame, metrics, left, right, edge_locations = True, show_metrics = True):
    """
    Display the results on the frame.

    Args:
        frame (numpy.ndarray): The frame to display results on.
        metrics (dict): Dictionary containing intensity metrics.
        left (float): Left edge location.
        right (float): Right edge location.
        edge_locations (bool): Whether to show edge locations.
        show_metrics (bool): Whether to show metrics.
    """
    if edge_locations:
        cv2.rectangle(frame, (int(left), slice_start), (int(right), slice_end), (0, 255, 0), 2)
    if show_metrics:
        cv2.putText(frame, f"Mean: {metrics['mean']:.3f}", (450, 650), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(frame, f"Left: {left:.3f}", (450, 670), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(frame, f"Right: {right:.3f}", (450, 690), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return frame