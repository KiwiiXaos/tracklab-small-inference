Inference Module for object detection, pose detection and tracking algorithms.


## Installation guide[^1]


Download github repo:
```bash
git clone git@github.com:KiwiiXaos/tracklab-small-inference.git
cd inference/
```
Use poetry to install dependencies. Python ">=3.8.0,<4.0" torch "=2.1.1"
```bash
poetry shell
poetry install

poe post-install
```

## Current Available Models
Work in progress

###Pose and Object detections
*Openpifpaf
*Yolov8
*Yolov5

###Trackers
*oc_sort
*strong_sort
*deep_oc_sort
*bot_sort
*bytetrack_sort

## How to use with Python

```bash

import pb-track.inference as inference

#Initialise Inference with mp4 path 

inference = VideoInference('/path/video.mp4')

#models are defined with add_model, it can be a list of several models #TODO: models plugins...
#Available models: openpifpaf, yolov5, yolov8, strong_sort, oc_sort, deep_oc_sort, bot_sort, bytetrack_sort

inference.add_model(['yolov8','oc_sort'])


#you can also load previous results from a json file

inference.read_from_json('/home/celine/pb-dart2/pb-track/inference/test2.json')

#offline inference

inference.process_video('/home/celine/pb-small/pb-track/inference/video_files/baby.json')

#online inference

inference.process_video_online()


```

## How to use with CLI

```bash
python -m inference.run

```


