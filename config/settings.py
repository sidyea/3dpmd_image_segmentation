import pandas as pd

# ROI crop variables

roi_x_start = 400   # Starting x-coordinate (from left side)
roi_x_end = 1000     # Ending x-coordinate (from left side)
roi_y_start = 250   # Starting y-coordinate (from top)
roi_y_end = 950     # Ending y-coordinate (from top)

# Slice variables

slice_start = 500
slice_end = 520

# Neural network variables
model_path = "model/slice_z_model_win20_start500_lr1e-4_bs20_ep10.pth"  # Path to the model
output_features = 2  # Number of output features from the model

# Standardization variables
standard_data_path = "model/slice_data_z__win20_start500.csv"  # Path to the standardization data
std_df = pd.read_csv(standard_data_path)
l_edge_mean = std_df.l_edge.mean()
l_edge_std = std_df.l_edge.std()
r_edge_mean = std_df.r_edge.mean()
r_edge_std = std_df.r_edge.std()