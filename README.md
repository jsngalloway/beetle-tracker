
# Beetle Tracker
Beetle Tracker is a small lightweight pair of scripts used to *tag* beetle videos and then *track* them using [idtrackerai](https://idtrackerai.readthedocs.io/en/latest/). Analysis is completed after tracking and various files are generated

## Usage
Beetle tracker runs in two phases. First a video must be tagged manually by a user. This generates a file which is used in the track phase (usually on a high powered GPU machine) to track the beetles through the video.
### Tag
#### Install
1. Anaconda is recommended for running beetle-tracker, [download it here](https://www.anaconda.com/)
2. Clone OR Download this repo
  a. If OSX open a terminal, or Windows open Git Bash, navigate to a folder where you'd like to install and run `git clone https://github.com/jsngalloway/beetle-tracker.git`
  b.  At the [github page](https://github.com/jsngalloway/beetle-tracker) Click the green code button and `download zip` Unzip where you want
 2. Open the folder beetle-tracker, this will be referred to as the *root* of this repo.
 3. Install the requirements
	 * If using Anaconda: open "Anaconda Prompt" and run `conda install -c conda-forge opencv`
	 * If not using Anaconda: in terminal install cv2 (`pip install opencv-python`) and numpy (`pip install numpy`)
You're all set!
Run the tagging script by making sure you're in the root of the repo and run `python tag/GUI_preprocessing.py` If you see a dialog appear, you're set. Practice tagging a video in `/sample_videos`

### Track
(From anaconda prompt with an idtrackerai env [install link](https://idtrackerai.readthedocs.io/en/latest/how_to_install.html))
1. Install using the above steps
2. Tag a video, or use the sample provided in `/track_sample`
3. Run `python track/postprocessing.py path/to/the/file.json` (note that the generated .json file must be next to the video)
This portion requires no user input but can take several (~10) minutes depending on your configuration.

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

## Upcoming Features
* Batch Tagging
* Batch Processing
* Suggestions? Let me know
