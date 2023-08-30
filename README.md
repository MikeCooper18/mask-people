# Installation
- `git clone https://github.com/MikeCooper18/mask-people.git ./mask_people`
- `cd mask_people`
- `python -m venv venv`
- `venv/Scripts/activate`
- `pip install -r requirements.txt`


# Usage
main.py [-h] [-m {box,split}] [-b BLUR_SIZE] [-l FRAME_LIMIT] [-s] [-v] [-c CONFIDENCE_THRESHOLD] [-i INTERSECTION_THRESHOLD] video output

Mask out sections of a video so that only one person is visible at a time. Outputs a video file for each person detected in the video.

Simple usage: python main.py video.mp4 output_directory

positional arguments:
  video                 The video or folder of videos to process.
  output                The output directory for the processed video(s). Creates this directory if it doesn't exist.

options:
  -h, --help            show this help message and exit
  -m {box,split}, --mask-type {box,split}
                        The type of mask to use. Default is 'box'. 'box' masks using the bounding boxes of each person, 'split' splits the frame vertically between each person.
  -b BLUR_SIZE, --blur-size BLUR_SIZE
                        The size of the blur kernel to use when blurring the mask (only used when mask_type is 'box'). Larger values will increase the size of the masked area. Increase this if the mask doesn't cover all of the movement/props the person has.
  -l FRAME_LIMIT, --frame-limit FRAME_LIMIT
                        The number of frames to process. Useful for testing. If not specified, the entire video will be processed.
  -s, --show-frame-count
                        Whether or not to display the frame count on the output video(s).
  -v, --visualise       Whether or not to display the annotated frame. Useful for testing.
  -c CONFIDENCE_THRESHOLD, --confidence-threshold CONFIDENCE_THRESHOLD
                        The confidence threshold for the YOLO model. Default is 0.1. Increase this if the model is detecting too many false positives. Decrease this if the model is missing people.
  -i INTERSECTION_THRESHOLD, --intersection-threshold INTERSECTION_THRESHOLD
                        The intersection threshold for the YOLO model. Default is 0.8. Increase this if the model is missing overlapping people.
