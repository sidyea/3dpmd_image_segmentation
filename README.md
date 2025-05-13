# Image Processing and Segmentation for the 3DPMD process

An implementation of edge detection for the 3DPMD based on the ResNet architecture.

## Explanation of the Process
### Input data

The 3DPMD print process is monitored using a welding camera (Cavitar C300). The configuration of the camera is shown below.

![Top view scanning configuration](/config/images/topview1.png)

The resulting image is of dimension 1440 pixels x 1080 pixels, as 3 channel RGB.


## Processing Flow-Chart

## Implementation details

The settings file under `/config/settings.py` contains all the variables in the process. Here, one can change the video file that is to be processed (variable `video_path`). One can also set the coordinates of the region of interest of the print.

The implementation calculates the measures of central tendency of the region of interest along with predicting the location of the left and right edges of the print. The results can be saved directly to CSV or JSON files by setting the flag `save_results` to `True` under `settings.py`. The results are saved at the path specified by the user in the variable `save_path`.

The model takes in a 600 x 20 pixel (W x H) slice of the input image and generates predictions for the left and right edge in normalized values. The model's weights are provided under `/model/___.pth`. The de-normalization is carried out based on the ground truth values of the edge locations used for model training, and provided under `/model/___.csv`. 
