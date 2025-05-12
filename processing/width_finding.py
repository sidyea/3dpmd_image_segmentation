import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from config.settings import output_features
from config.settings import slice_start, slice_end
from config.settings import l_edge_mean, l_edge_std, r_edge_mean, r_edge_std

def std_to_px(left, right):
    """
    Convert standardized values to pixel values.

    Args:
        left (float): Standardized left edge value.
        right (float): Standardized right edge value.

    Returns:
        tuple: Pixel values for left and right edges.
    """
    left = left * l_edge_std + l_edge_mean
    right = right * r_edge_std + r_edge_mean

    return left, right

def load_model(model_path, device):
    """
    Load a pre-trained model from the specified path.

    Args:
        model_path (str): Path to the model file.
        device (torch.device): The device to load the model on.

    Returns:
        nn.Module: The loaded model.
    """
    # Model architecture
    class Width_ResNet(nn.Module):
        def __init__(self, output_features=2):
            super(Width_ResNet, self).__init__()
            self.resnet = models.resnet18()
            self.resnet.fc = nn.Linear(self.resnet.fc.in_features, output_features)  # final layer mod

        def forward(self, x):
            return self.resnet(x)
        
    # Load the model
    model = Width_ResNet(output_features=output_features)  # Initialize the model
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)  # Move the model to the specified device

    print(f"Model loaded from {model_path} and moved to {device}")

    # Set the model to evaluation mode
    model.eval() 
    print("Model set to evaluation mode")
    
    return model


def find_edges(model, device, image):
    """
    Find the width of the object in the image using the model.

    Args:
        model (nn.Module): The neural network model.
        device (torch.device): The device to run the model on.
        image (torch.Tensor): The input image.

    Returns:
        float: The width of the object in pixels.
    """
    # Preprocess the image
    # Assuming the image is a numpy array with shape (height, width, channels)
    image = image.astype(np.float32)  # Ensure the image is in float format
    image = image[slice_start:slice_end, :, :]  # Crop the image to the specified slice
    image = image / 255.0  # Normalize to [0, 1]

    image = torch.tensor(image) # Convert to tensor
    image = image.permute(2, 0, 1)  # Change to (channels, height, width)
    image = image.unsqueeze(0)  # Add batch dimension to make it (1, channels, height, width) here [1, 3, 700, 600]
    image = image.to(device)  # Move to the specified device

    image = image.float()  # Ensure the image is in float format


    # Forward pass through the model
    with torch.no_grad():
        output = model(image) # returns standardized values for left and right edges

    output = output.squeeze()  # Remove batch dimension
    output = output.cpu()   # Move to CPU
    output = output.numpy()  # Convert to numpy array

    left = output[0]  # Left side of the object
    right = output[1]  # Right side of the object

    # Post-process the output to get the width
    left, right = std_to_px(left, right)  # Convert standardized values to pixel values


    #width = right - left  # Calculate width in pixels

    return left, right