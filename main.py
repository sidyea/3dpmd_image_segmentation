# imports
from utils import video_loader
from processing.preprocessing import crop_to_roi, calculate_intensity_metrics
from config.settings import model_path, save_path, save_results
from processing.width_finding import load_model, find_edges
from utils.display_frames import draw_results
from utils.save_data import append_results, save_results_to_csv
import cv2
import torch


def main(save_results = save_results):
    """
    Main function to run the width finding process. 
    It loads the model, processes the video frames, and displays the results.
    It also saves the results to a CSV file if specified.

    Args:
        save_results (bool): Whether to save the results to a CSV file.
    """

    print("Now running main.py")

    # Cuda check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    width_predictor = load_model(model_path, device)

    # Results list
    results = []

    # open video source
    video_path = "C:/Users/shark/VSCodeProjects/FSCM/data/video/topview/20250110-132256244035.webm" 

    for i, frame in enumerate(video_loader.load_video(video_path)):
        # Crop / Transform
        frame = crop_to_roi(frame)      # crop (values in settings.py)

        # Intensity metrics
        metrics = calculate_intensity_metrics(frame)
        
        # Neural Network
        left, right = find_edges(width_predictor, device, frame)

        # Save results
        results = append_results(results, metrics, i, left, right)

        # Draw results on the frame
        frame = draw_results(frame, metrics, left, right, edge_locations=True, show_metrics=True)
        
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    
    # Clean up
    for i in range(1, 5):
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    # Save results to CSV
    if save_results:
        save_results_to_csv(results, save_path, "results")
        #save_results_to_json(results, save_path, "results")
        print("Results save at path:", save_path)
    else:
        print("Results not saved to CSV.")



main()