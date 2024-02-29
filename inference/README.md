Inference Module for object detection, pose detection and tracking algorithms.


## Installation guide[^1]



```bash
git clone git@github.com:KiwiiXaos/tracklab-small-inference.git

cd inference/

poetry shell

poetry install

poe post-install


```

## How to use[^1]

```bash

import pb-track.inference as inference

#Initialise Inference with mp4 path 

inference = VideoInference('/home/celine/Downloads/celine_test12.mp4')

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



