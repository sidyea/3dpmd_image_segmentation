# imports
from utils import video_loader
from processing.preprocessing import crop_to_roi, calculate_intensity_metrics
from processing.width_finding import load_model, find_width
import cv2
import torch


def main():
    print("Now running main.py")

    # Cuda check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model_path = "model/slice_z_model_win20_start500_lr1e-4_bs20_ep10.pth"
    width_predictor = load_model(model_path, device)

    # Results list
    results = []

    # open video source
    video_path = "C:/Users/shark/VSCodeProjects/FSCM/data/video/topview/20250110-132256244035.webm" 

    for i, frame in enumerate(video_loader.load_video(video_path)):
        # Crop / Transform
        frame = crop_to_roi(frame)      # crop (values in settings.py)
        frame = frame.astype("float32")/255.0     # global normalization to [0, 1]

        # Intensity metrics
        metrics = calculate_intensity_metrics(frame)
        
        # Neural Network
        width = find_width(width_predictor, device, frame)

        # save results


        # Show the frame
        cv2.putText(frame, f"Mean: {metrics['mean']:.3f}", (450, 650), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(frame, f"Width: {width:.3f}", (450, 670), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        
    
    # Clean up
    for i in range(1, 5):
        cv2.destroyAllWindows()
        cv2.waitKey(1)


main()