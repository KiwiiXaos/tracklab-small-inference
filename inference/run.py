
import logging
import cv2
from cv2 import VideoCapture
import torch
import numpy as np
import json
import alive_progress
import time
from hydra.utils import instantiate
from collections import defaultdict
from matplotlib import pyplot as plt

#from mmdet.apis import DetInferencer
#from mmdet.apis import init_detector, inference_detector, inference_bottomup
#from plugins.reid.bpbreid.scripts.default_config import get_default_config
#import plugins.track.bpbreid_strong_sort.strong_sort as strong_sort
#from plugins.reid.bpbreid.torchreid.data.masks_transforms import (
#    CocoToSixBodyMasks,
#    masks_preprocess_transforms,
#)
#from plugins.reid.bpbreid.torchreid.utils.tools import extract_test_embeddings
#import torchreid
#from plugins.reid.bpbreid.torchreid.utils.imagetools import (
#    build_gaussian_heatmaps,
#    build_gaussian_body_part_heatmaps,
#    keypoints_to_body_part_visibility_scores,
#)
#from plugins.reid.bpbreid.tools.feature_extractor import FeatureExtractor
import logging
import pandas as pd
import logging
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from tracklab.core.visualization_engine import print_count_frame, final_patch, VisualizationEngine
from inference.wrappers import *

#TODO GET MODEL LIST FOR EACH WRAPPERS...

#TODO: A WAY TO DOWNLOAD EVERY MODELS..
# TODO: CONFIG FILE WITH HYDRA, CAN BE CHANGED WITH COMMAND LINE Maybe make a fonction for that ..?



log = logging.getLogger(__name__)

MODEL_MAP = {
    'openpifpaf':OpenpifpafWrapper,
    'yolov8': Yolov8Wrapper,
    'yolov5': Yolov5Wrapper,
    'strong_sort': StrongSortWrapper,
    'oc_sort': OCSortWrapper,
    'deep_oc_sort': DeepOCSortWrapper,
    'bot_sort': BotSortWrapper,
    'bytetrack_sort': BytetrackWrapper,
    'af_link': AFLinkWrapper
}

MODEL_CLASS_MAP = {
    'openpifpaf':'detector',
    'yolov8': 'detector',
    'yolov5': 'detector',
    'strong_sort': 'track',
    'oc_sort': 'track',
    'deep_oc_sort': 'track',
    'bot_sort': 'track',
    'bytetrack_sort': 'track',
    'mmdet': 'detector',
    'mmpose_bottomup': 'detector',
    'mmpose_topdown': 'detector',
    'bdreid': 'reid',
    'bpreid_strong_sort': 'track',
    'af_link':'track'


}



def get_torch_checkpoints_dir():
    base_dir = None
    if hasattr(torch, 'hub') and hasattr(torch.hub, 'get_dir'):
        # new in pytorch 1.6.0
        base_dir = torch.hub.get_dir()
    elif os.getenv('TORCH_HOME'):
        base_dir = os.getenv('TORCH_HOME')
    elif os.getenv('XDG_CACHE_HOME'):
        base_dir = os.path.join(os.getenv('XDG_CACHE_HOME'), 'torch')
    else:
        base_dir = os.path.expanduser(os.path.join('~', '.cache', 'torch'))
    return os.path.join(base_dir, 'checkpoints')


#TODO: Print every settings for command line/api
def Configs_Wrappers(cfg, cli_args, name):
    config = {}
    #args, _ = parser.parse_known_args()
    for setting in cfg:
        if setting in cli_args:
            config[setting.key()] = cli_args.setting.value()
        else:
            config[setting.key()] = cfg.setting.value()
    return config
            
os.environ["HYDRA_FULL_ERROR"] = "1"
log = logging.getLogger(__name__)         


def merge_dataframes(main_df, appended_piece):
    """
    merge all detections into one one Pandas dataframe for visualization tool.
 
    Parameters:
    - main_df pandas.core.frame.DataFrame: The main detections array.
    - appended_piece: List[pandas.core.series.Series]: The detections to add
 
    Returns:
    - pandas.core.frame.DataFrame: all detections merged into one dataframe.
    """
    # Convert appended_piece to a DataFrame if it's not already
    if isinstance(appended_piece, pd.Series):
        appended_piece = pd.DataFrame(appended_piece).T
    elif isinstance(appended_piece, list):  # list of Series or DataFrames
        if len(appended_piece) > 0:
            appended_piece = pd.concat(
                [s.to_frame().T if type(s) is pd.Series else s for s in appended_piece]
            )
        else:
            appended_piece = pd.DataFrame()

    # Append the columns of the df
    new_columns = appended_piece.columns.difference(main_df.columns)
    main_df.loc[:, new_columns] = np.nan

    # Append the rows of the df
    new_index = set(appended_piece.index).difference(main_df.index)
    for index in new_index:
        main_df.loc[index] = np.nan

    # Update all the values (appended_piece overrides)
    main_df.update(appended_piece)
    return main_df


def new_model(one_model, device):
    """
    check if model is available and load it with hydra configs. Used for scripts.
 
    Parameters:
    - one_model String: Model name.
    - device: String: device ('cpu' or 'gpu')
    Returns:
    - ModelWrapper or TrackWrapper
    """
    hydra.initialize(version_base=None, config_path="configs/", job_name="inference")
    cfg = hydra.compose(config_name="config")
    
    if one_model in MODEL_MAP:
                cfg_default = hydra.compose(config_name="config")
                cfg_file_name = cfg.config_models[one_model]
                cfg_model = hydra.compose(config_name=cfg_file_name)[MODEL_CLASS_MAP[one_model]]['cfg']
                cfg_model['path_to_checkpoint'] = cfg_default.checkpoints_dir
                model = MODEL_MAP[one_model](device, cfg_model)
                return model
    raise Exception("model not available")



#im = Image.open('/home/celine/scene_graph/SGG/openpifpaf/image_test_openpifpaf.png')
class VideoInference():
    """
    Inference pipeline
 
    Parameters:
    - video_path String: path of the video to process.

    It is possible to process multiple files or an entire folder. Refer to self.video_batch() and self.process_folder()
    self.pipeline needs to be defined after initialisation

    """
    def __init__(self,  video_path= None) -> None:
        self.path = video_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  # TODO support Mac chips
        log.info(f"Using device: '{self.device}'.")
        self.pipeline = [] 
        self.online = False
        self.json = False
        self.previous_results = []

    def video_batch(self, video_list: list, visualization = False):
        """
        Inference pipeline

        Parameters:
        - video_path String: path of the video to process.

        It is possible to process multiple files or an entire folder. Refer to self.video_batch() and self.process_folder()
        self.pipeline needs to be defined after initialisation

        """

        for video_name in video_list:
            output_json = video_name[:-4] + '_result.json'
            self.path = video_name
            self.previous_results = []
            self.process_video(output_json)
            if visualization:
                video_output = video_name[:-4] + '_output.mp4'
                self.visualization(video_output)
            

    def process_folder(self, directory_path: str, visualization = False):
        """
        process all the mp4 files from one folder.
    
        Parameters:
        - directory_path String: path of the directory.
        - visualization: Bool: Video output or not.
       
        """
        is_directory = os.path.isdir(directory_path)
        if is_directory:
            all_items = os.listdir(directory_path)
            mp4_files = [item for item in all_items if item.endswith('.mp4') and os.path.isfile(os.path.join(directory_path, item))]
            self.video_batch(video_list = mp4_files, visualization =visualization )
        else:
            if directory_path.endswith('.mp4') and os.path.isfile(directory_path):
                self.path = directory_path
                output_json = directory_path[:-4] + '_result.json'
                self.process_video(output_json)
                self.previous_results = []
                if visualization:
                    video_output = directory_path[:-4] + '_output.mp4'
                    self.visualization(video_output)

            else:
                log.exception("Please give a folder as input")


    

    def add_model(self, models):
        """
        add model to pipeline
    
        Parameters:
        - directory_path String: path of the directory.
        - visualization: Bool: Video output or not.
       
        """
        #TODO: Test List and string input
        hydra.initialize(version_base=None, config_path="configs/", job_name="inference")
        cfg = hydra.compose(config_name="config")
        
        def add_single(one_model):
            assert isinstance(one_model, str)
            if one_model in MODEL_MAP:
                cfg_default = hydra.compose(config_name="config")
                cfg_file_name = cfg.config_models[model]
                cfg_model = hydra.compose(config_name=cfg_file_name)[MODEL_CLASS_MAP[model]]['cfg']
                cfg_model['path_to_checkpoint'] = cfg_default.checkpoints_dir

                self.pipeline.append(MODEL_MAP[one_model](self.device, cfg_model))
            else: 
                raise Exception("model not available")
            return self.pipeline

        if isinstance(models, list):
            for model in models:
                
                self.pipeline = add_single(model)
        else:
            self.pipeline = add_single(models)

    
    def process_frame(self,
        visEngine, patch, frame_i, detections_pred, ground_truths, video_name, nframes
    ):
        """
        add detections to frame
    
        Parameters:
        - directory_path String: path of the directory.
        - visualization: Bool: Video output or not.
        Returns:
        - ModelWrapper or TrackWrapper
       
        """
        #TODO: Why reverse colors
        
        # print count of frame
        print_count_frame(patch, frame_i, nframes)

        # draw detections_pred
        if detections_pred is None:
            return patch
        
        for _, detection_pred in detections_pred.iterrows():
            print
            visEngine._draw_detection(patch, detection_pred, is_prediction=True)

        # draw ground truths
        #if ground_truths is not None:
        #    for _, ground_truth in ground_truths.iterrows():
        #        self._draw_detection(patch, ground_truth, is_prediction=False)

        # postprocess image
        patch = final_patch(patch)

        # save files
        
        #if visEngine.cfg.save_images:
        #    filepath = (
        #        visEngine.save_dir
        #        / "images"
        #        / str(video_name)
        #        / Path(image_metadata.file_path).name
        #    )
        #    filepath.parent.mkdir(parents=True, exist_ok=True)
        #    assert cv2.imwrite(str(filepath), patch)
        #if visEngine.cfg.save_videos:
        #visEngine._update_video(patch, '/home/celine/pb-dart2/pb-track/inference/test2')
        return patch
    
    def visualization_dataframe(self, frame_i, previous_results):
        """
        Convert detection results format into dataframe to use pb-track visualisation tool.

        Parameters:
        - frame_i int: frame number.
        - previous_results: list(dict()): detection results.
        Returns:
        - pandas.core.frame.DataFrame: detection results in a Dataframe
        """
        detections = []
        final_detections = pd.DataFrame()

        for idx, annot in enumerate(previous_results['annotations']):

                    extension_factor = [ 0.1, 0.03, 0.1 ]
                    if 'keypoints' in annot:
                        keypoints = np.reshape(np.array(annot['keypoints']), (-1,3))

                    else:
                        keypoints =[[]]
                    bbox = np.array(annot['bbox'])
                    
                    
                    if('track_id' in annot):
                        track_id = annot['track_id']
                    else:
                        track_id = -1


                    detections.append(
                    pd.Series(
                        dict(
                            image_id=frame_i,
                            keypoints_xyc=keypoints,
                            keypoints_conf=annot['score'],
                            bbox_ltwh=bbox,
                            bbox_conf=annot['score'],
                            video_id='video',
                            category_id= annot['category_id'],  # `person` class in posetrack
                            track_id = track_id,
                        ),
                        name=idx,
                        )
                    )

        final_detections = merge_dataframes(final_detections, detections)  
        return final_detections
        
            
            # Break the loop if 'q' is pressed
    
    def visualization_cli(self, cfg, video_path):
        '''
        Entry point for generating video output with CLI.

        Parameters:
        - cfg: Hydra configs
        - video_path: String: Video output name and path.
        '''
        cfg = cfg.cfg
        track_history = defaultdict(lambda: [])
        test = VisualizationEngine(cfg)
        stream = VideoCapture(self.path)
        
        
        video_output = cv2.VideoWriter(video_path,
                                cv2.VideoWriter_fourcc(*"mp4v"),
                                float(30),
                                self.previous_results[0]['width_height'],)
        frame_i = -1

        if(len(self.previous_results) == 0):
            raise Exception('There is no loaded prediction. Read a json file or process a video first')
                
        if (stream.isOpened()== False): 
            raise Exception("Error opening video stream or file")

        while(stream.isOpened()):
            frame_i+= 1
            ret, frame = stream.read()
            detections = []
            final_detections = pd.DataFrame()
            
            #is_openpifpaf = self.previous_results[0]["model"] =='openpifpaf'
            nb_frame = len(self.previous_results)
            if ret and frame_i < len(self.previous_results):
                for idx, annot in enumerate(self.previous_results[frame_i]['annotations']):
                    #TODO: change config file depending of outputs..

                    extension_factor = [ 0.1, 0.03, 0.1 ]
                    if 'keypoints' in annot:
                        keypoints = np.reshape(np.array(annot['keypoints']), (-1,3))

                    else:
                        keypoints =[[]]
                    bbox = np.array(annot['bbox'])
                    
                    
                    if('track_id' in annot):
                        track_id = annot['track_id']
                    else:
                        track_id = -1


                    detections.append(
                    pd.Series(
                        dict(
                            image_id=frame_i,
                            keypoints_xyc=keypoints,
                            keypoints_conf=annot['score'],
                            bbox_ltwh=bbox,
                            bbox_conf=annot['score'],
                            video_id='video',
                            category_id= annot['category_id'],  # `person` class in posetrack
                            track_id = track_id,
                        ),
                        name=idx,
                        )
                    )

                
                final_detections = merge_dataframes(final_detections, detections)  
                result = self.process_frame(test, frame,frame_i, final_detections, False, 'img', nb_frame)
                video_output.write(result)

                if cfg.imshow == True:
                    plt.imshow(frame)
                    plt.title("Tracking")
                    plt.show()
                    #cv2.imshow("tracking ", result)
                    # Break the loop if 'q' is pressed
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            else:
                # Break the loop if the end of the video is reached
                break

        
        stream.release()
        video_output.release()
        if cfg.imshow == True:
            cv2.destroyAllWindows()
                
    def visualization(self, video_path):
        """
        Entry point for generating video output with scripts.

        Parameters:
        - video_path: String: Video output name and path.
        
        """
        cfg = hydra.compose(config_name="visualization/visualization")
        cfg = cfg.visualization
        self.visualization_cli(cfg, video_path)

    def read_from_json(self, filename):
        """
        Read json and save it as result in pipeline.

        Parameters:
        - filename: String: file path
        
        """
        f = open(filename, 'r')
        results = []
        for i, line in enumerate(f):
            results.append(json.loads(line.strip())["predictions"])
        self.previous_results = results
        return self.previous_results
    
    def write_json(self, filename):
        """
        Save detections result in a json file.

        Parameters:
        - filename: String: file path
        
        """
        try:
            with open(filename, "r") as t:
                for i, result in enumerate(self.previous_results):
                    json.dump({
                                                'frame': i,
                                                'predictions': result,
                                        }, t, separators=(',', ':'))
                    t.write('\n')
        except FileNotFoundError:
            log.warning("The file does not exist. Please check the file path.")

    # If video capture is webcam, force online ?
    def process_video(self, output_json=None):
        """
        Process video offline.

        Parameters:
        - output_json: String: output json file path
        
        """
        if output_json != None:
            self.json = True
        
        for model in self.pipeline:
            results = []
            
            if self.json:
                t =open(output_json, 'w')
            log.info('Running', model.name)
            start = time.time()
            if model.read_video:
                stream = VideoCapture(self.path)
                frame_i = -1
                total=stream.get(cv2.CAP_PROP_FRAME_COUNT)
                #pbar = tqdm(total= total, ascii=True, position=0, leave=True)
                if (stream.isOpened()== False): 
                    raise Exception("Error opening video stream or file")
                
                with alive_progress.alive_bar(int(total)+ 1) as pbar:
                    while(stream.isOpened()):
                        frame_i+= 1
                        pbar()
                        ret, frame = stream.read()
                        if ret:

                            if model.initial == True:
                                    frame = model.preprocess(frame)
                                    result = model.process(frame)
                                    results.append(result)
                            else:
                                result = model.process(frame, self.previous_results[frame_i])
                                results.append(result)

                            if self.json:
                                json.dump({
                                    'frame': frame_i,
                                    'predictions': result,
                                }, t, separators=(',', ':'))
                                t.write('\n')
                        else: 
                            self.previous_results = results
                            break
                if self.json:
                    t.close()
                log.info('time', time.time() - start, 's')
            else:
                frame_i = -1
                if len(self.previous_results) ==0 :
                    log.info('could not find outputs, trying to retrieve it from json')
                    self.previous_results = self.read_from_json(output_json)
                if model.postprocess is False:
                    with alive_progress.alive_bar(len(self.previous_results)+ 1) as pbar:
                        for result in self.previous_results:
                            pbar()
                            frame_i += 1
                            new_result = model.process(result)
                            results.append(new_result)
                            if self.json:
                                json.dump({
                                        'frame': frame_i,
                                        'predictions': result,
                                }, t, separators=(',', ':'))
                                t.write('\n')
                        log.info('time',  time.time() - start, 's')
                        if self.json:
                            t.close()
                        self.previous_results = results
                else:
                    new_result = model.process(self.previous_results)
                    self.previous_results = new_result
                    if self.json:
                        self.write_json(output_json)                          
                    
    def process_video_online(self):
        """
        Process video online.
        
        """
        #TODO: Test with videos... 
        #Visual tool..
        cfg = hydra.compose(config_name="visualization/visualization")
        cfg = cfg.visualization.cfg
        visu = VisualizationEngine(cfg)

        #TODO: Buffer
        buffer = 6000

        stream = VideoCapture(self.path)
        frame_i = -1
        if (stream.isOpened()== False): 
            raise Exception("Error opening video stream or file")
        total=stream.get(cv2.CAP_PROP_FRAME_COUNT)    
        while(stream.isOpened()):
            

            ret, frame = stream.read()
            if ret:
                frame_i += 1
                result =[]
                for model in self.pipeline:
                    result = model.process_bloc_stream(frame, result)
                


                final_detections = self.visualization_dataframe(frame_i, result)
                frameshow = self.process_frame(visu, frame, frame_i, final_detections, False, 'img', total)
                
                self.previous_results.append(result)
                if cfg.imshow == True:
                    cv2.imshow("tracking ", frame)
                    #print('v time', time.time() - start, 's')
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

                if(len(self.previous_results) % buffer == 0):
                    self.previous_results.pop()
                #print('v time', time.time() - start, 's')
            else: 
                break
        stream.release()
        if cfg.imshow == True:
            cv2.destroyAllWindows()

    

@hydra.main(config_path="configs", config_name="config")
def func(cfg: DictConfig):
    """
    Hydra CLI entry point.

    Parameters:
    - cfg: DictConfig: Hydra configs

    """
    #TODO: Test it !!!
    inference = VideoInference(cfg.video_path) #VideoInference, Test pas maj..


    if 'detector' in cfg:
        detector = instantiate(cfg.detector, inference.device)
        inference.pipeline.append(detector)

    if 'track' in cfg:
        track = instantiate(cfg.track, inference.device)
        inference.pipeline.append(track)

    if 'json_input' in cfg:
        inference.read_from_json(filename = cfg.json_input)


    if cfg.online is False:
        inference.process_video(cfg.json_output)

    else:
        inference.process_video_online()

    if cfg.visualize: #visualize american
        inference.visualization_cli(cfg.visualization , cfg.video_path)



                
if __name__ == "__main__":
    func()

    