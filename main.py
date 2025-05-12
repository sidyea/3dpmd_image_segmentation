# functions and variables imports
from utils import video_loader
from processing.preprocessing import crop_to_roi, calculate_intensity_metrics
from config.settings import video_path, model_path, save_path, save_results
from processing.width_finding import load_model, find_edges
from utils.display_frames import draw_results
from utils.save_data import append_results, save_results_to_csv

# packages imports
import cv2
import torch


def main(save_results = save_results):
    """
    Main function to run the width finding process. 
    It loads the video and model, processes the video frames, and displays the results.
    It also saves the results to a CSV file if specified.

    Args:
        save_results (bool): Whether to save the results to a CSV file.
    """

    print("Now running main.py")

    # Cuda check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
 
    video = video_loader.load_video(video_path)         # Load the video
    width_predictor = load_model(model_path, device)    # Load the model
    results = []                                        # Initialize results list

    print("Running video processing...")
    #for i, frame in enumerate(video_loader.load_video(video_path)):
    for i, frame in enumerate(video):

        frame = crop_to_roi(frame)                                  # Crop to ROI (values in settings.py)
        metrics = calculate_intensity_metrics(frame)                # Calculate intensity metrics
        left, right = find_edges(width_predictor, device, frame)    # Predict edges
        results = append_results(results, metrics, i, left, right)  # Save results
        frame = draw_results(frame, metrics, left, right, 
                             edge_locations=True, show_metrics=True) # Draw results on frame
        cv2.imshow("Frame", frame)                                  # Display the frame

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Video processing interrupted by user.")
            break

    # Close all OpenCV windows
    for i in range(1, 5):
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    if save_results:
        save_results_to_csv(results, save_path, "results") # Save results to CSV

    print("Video processing completed.")



main()