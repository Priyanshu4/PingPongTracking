# PingPongTracking

Final project for Computer Vision, CSE 5095/4830. 
We built a prototype of a system that can track the 3D position of a ping pong ball using a single smartphone camera. 
We also attempted to track spin by adapting the models and code from https://github.com/cogsys-tuebingen/spindoe.

## Project Structure
The `src` directory contains the code for the main components of our project. It includes our Python modules for object detection, position estimation, kalman filtering, visualization and more.

The `notebooks` directory contains some of our experiments. It has scripts in `.ipynb` format used to test on real data and generate visualizations.

The `data` directory contains data we collected to test our approach. It includes 3 videos with associated configuration files.

## Installation and Usage
To use our code, clone the repository and create a conda environment from the `environment.yaml` file. 
Then, you can run the `.ipynb` files in our `notebooks` folder. 
The main notebook that runs the object detection, position estimation and kalman filtering over an entire video is in `process_video.ipynb`. You can change the video by changing the config file used at the beginning of the script.

