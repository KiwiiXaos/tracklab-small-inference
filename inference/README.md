# Inference Module for object detection, pose detection and tracking algorithms.


## Installation guide


Download github repo:
```bash
git clone git@github.com:KiwiiXaos/tracklab-small-inference.git
cd inference/
```
Use poetry to install dependencies. Python ">=3.8.0,<4.0" torch "=2.1.1"

**Post-install**: among the Yolov5 dependencies, roboflow installs opencv-python-headless. This is a problem because opencv-python-headless doesn't support cv2.imshow, which breaks deployment. Opencv-python-headless must be uninstalled and opencv-python reinstalled.

```bash
poetry shell
poetry install

poe post-install
```

## Current Available Models
Work in progress

### Pose and Object detections
* Openpifpaf
* Yolov8
* Yolov5

### Trackers
* oc_sort
* strong_sort
* deep_oc_sort
* bot_sort
* bytetrack_sort

### Post-process
* af_link 


## Hydra Configuration

1. **Pipeline Settings**: Can be configured and are accessible in the `/inference/configs/` directory.

2. **Main Configuration File**: The main configuration file, `config.yaml`, is located in `/inference/configs/`. This file loads other configuration files and serves as the central configuration point.

3. **Checkpoints Directory**: The `checkpoints_dir` setting in `config.yaml` specifies the path used to load and download model checkpoints.

4. **CLI Usage**: Models are instantiated using Hydra. The pipeline configuration can be defined in the `default` section of `config.yaml`.

5. **Video Path**: `video_path` is a parameter used in CLI commands, specifying the path to the video file.

6. **Python Usage**: When using Python, models are not instantiated using Hydra. Instead, configuration files are read during initialization. The `config_models` parameter specifies the path of default config files for each model.

7. **Work in progress**: Implementing functionality to dynamically load configuration files from custom paths specified by the user.


## How to use with Python

Python examples are available in `tracklab/inference/examples/` folder.

**Import Module**
```bash
import tracklab.inference as inference
```

**Initialise Inference with mp4 file**
```bash
inference = VideoInference('/path/video.mp4')

```
models are defined with add_model, it can be a list of several models
```bash
#Available models: openpifpaf, yolov5, yolov8, strong_sort, oc_sort, deep_oc_sort, bot_sort, bytetrack_sort

inference.add_model(['yolov8','oc_sort'])

```
you can also load previous results from a json file
```bash
inference.read_from_json('/home/celine/pb-dart2/pb-track/inference/test2.json')

```

**Process Multiple files**
```bash
file_list = ['path/video1.mp4', 'path/video2.mp4']
inference.read_from_json('/home/celine/pb-dart2/pb-track/inference/test2.json')

```

**Offline inference**
```bash
inference.process_video('/home/celine/pb-small/pb-track/inference/video_files/baby.json')
```
**Online inference**
Online inference doesn't take any arguments

```bash

inference.process_video_online()


```

## How to use with CLI

Configs files are in `inference/configs`. Configs can be overwritten with CLI using Hydra. 

```bash
python -m inference.run

```

## Framework overview

1. **VideoInference**: Abstract Class instantiated for the pipeline.  
    - *pipeline*: An array containing different models defined for the inference pipeline. This likely includes various stages of processing such as detection and tracking.
    - *previous_results*: Dictionary storing detection results. If two detectors, or two tracker are used the data will be overwriten. Dataformat is:
    ```bash
    annotations:[
                    {
                        keypoints: [x,y,score,x,y,..],
                        score : float,
                        category_id: int,
                        track_id: int,
                        bbox: [x,y,w,h]

                    },
                    {
                        ...
                    }
                ]
    ```
    - *device*: The attribute for defining GPU usage. It automatically detects whether CUDA (NVIDIA's parallel computing platform) is available and sets GPU usage accordingly. 

    - *json output dataformat:* 
        ```bash
        {
            frame: i,
            predictions:[
                {
                    annotations:[
                        {
                            keypoints: [x,y,score,x,y,..],
                            score : float,
                            category_id: int,
                            track_id: int,
                            bbox: [x,y,w,h]

                        },
                        {
                            ...
                        }
                    ]
                }
            ]

        }
        {
            frame: i+1,
            predictions: ...
        }

        ```
2. **Plugins**: Python tracking model files are stored either in the /plugins/ folder or installed via pip. The list of tracking that can be found in the plugins folder.
    - aflink
    - bot_sort
    - deep_oc_sort
    - oc_sort
    - strong_sort
    - byte_track

3. **inference.wrappers**: `wrappers.py` contains model and tracker APIs. It standardizes the input and output format for each model.

4. **Model Checkpoints**: Checkpoints are automatically downloaded if not available in the checkpoint directory. They are configurable in the configuration files.
