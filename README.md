Beetle Tracker
##### beetle-tracker

## Requirements
This is a python3 project. While it was developed in an Anaconda env all necessary libraries are listed below.
### Tagging Requirements
* cv2
* numpy
### Tracking Requirements
* cv2
* numpy
* idtrackerai
Note: idtrackerai can be difficult to install properly. Even if installed be sure the utility is properly utilizing all GPU resources

## Usage
Beetle tracker runs in two phases. First a video must be tagged manually by a user. This generates a file which is used in the track phase (usually on a high powered GPU machine) to track the beetles through the video.
### Tag
From the root of this repo
1. Run the script. `python tag/preprocessing.py path/to/your/video.mp4` (ex. `python tag/preprocessing.py sample_videos/video2.mp4`)
2. Follow the instructions in the terminal (TODO: improve instructions)
3. Upon completion a file will be generated in the same directory as the video specified.
### Track
(From anaconda prompt with an idtrackerai env)
1. python track/postprocessing.py path/to/the/video.mp4 (note that the generated txt file must be next to the video)
This portion requires no user input but can take several (~10) minutes depending on your configuration.

## Upcoming Features
* InqScribe file generation
* Batch Tagging
* Batch Processing
* Improved UI
