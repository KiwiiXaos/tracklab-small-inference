from PIL import Image
import torch
import numpy as np
import argparse
import logging as log
import json
import pandas as pd
import wget
import requests
import os
from hydra.utils import instantiate


### Model packages
import yolov5
import openpifpaf
#from plugins.track.AFLink.model import PostLinker
#from plugins.track.AFLink.AppFreeLink import AFLink
#from plugins.track.AFLink.dataset import LinkData, DartfishLinkData

from tracklab.utils.collate import Unbatchable
from tracklab.utils.coordinates import *


from ultralytics import YOLO

from plugins.track.oc_sort.ocsort import OCSort

import plugins.track.deep_oc_sort.ocsort as DeepOCSort
import plugins.track.strong_sort.strong_sort as strong_sort

from plugins.track.byte_track.byte_tracker import BYTETracker
import plugins.track.bot_sort.bot_sort as bot_sort
from pathlib import Path

### Utils
from cv2 import COLOR_BGR2RGB, cvtColor, VideoCapture
from tracklab.utils.coordinates import rescale_keypoints
from os.path import join, exists
from openpifpaf import transforms

from tracklab.utils.coordinates import ltrb_to_ltwh
from tracklab.utils.openmmlab import get_checkpoint


from tracklab.utils.coordinates import sanitize_keypoints, generate_bbox_from_keypoints



##### DETECT MULTIPLE #####
class ModelWrapper:
    def __init__(self, model_name: str) -> None:
        self.read_video = True
        self.name = model_name
        self.postprocess = False
        self.initial = True

    def preprocess(self, frame):
        return frame
    def process(self, frame):
        pass
    def run(self, frame: np.array) -> dict:
        frame = self.preprocess(frame)
        return self.process(frame)
    
    def process_bloc_stream(self, frame: np.array, output = None) -> dict:
        return self.run(frame=frame)
    

    

class TrackWrapper:
    def __init__(self, model_name) -> None:
        self.read_video = False
        self.name = model_name
        self.postprocess = False
        self.initial = False

    def preprocess(self, frame):
        return frame
    def process(self, frame):
        pass
    def run(self, output: dict)-> dict:
        return self.process(output)
    
    def process_bloc_stream(self, frame, output) -> dict:
        result = self.run(output)
        return result
    
class DeepTrackWrapper:
    def __init__(self, model_name) -> None:
        self.read_video = True
        self.name = model_name
        self.postprocess = False
        self.initial = False

    def preprocess(self, frame):
        return frame
    def process(self, frame):
        pass
    def run(self, frame: np.array, input: dict) -> dict:
        return self.process(frame, input)
    def process_bloc_stream(self, frame: np.array, output:dict) -> dict:
        result = self.process(frame, output)
        return result

class PostProcessWrapper:
    def __init__(self, model_name) -> None:
        self.read_video = False
        self.name = model_name
        self.postprocess = True
        self.initial = False
        self.read_video = False
    def process(self, frame):
        pass

    def process_bloc_stream(self, frame: np.array, output: dict) -> dict:
        result = self.process(self, output)
        return result
    



def download_drive(checkpoint_path: str, cfg: dict) -> None:
   
    import gdown
    gdown.download(id=cfg.download_weights, output=cfg.path_to_checkpoint + cfg.checkpoint, quiet=False, use_cookies=False)

def download_website(repo_url: str, target_dir:str):
    log.info('Downloading ' + repo_url)
    response = requests.get(repo_url)

    # Check if the request was successful
    if response.status_code == 200:
        # Open file in binary write mode and save the content to the file
        with open(target_dir, 'wb') as file:
            file.write(response.content)
        log.info("File downloaded successfully")
    else:
        log.info("Failed to download file")


def check_checkpoint(checkpoint_path: str, cfg: dict):

    if os.path.exists(checkpoint_path + cfg.checkpoint):
        return
    else:
        log.info("Checkpoint not found, attempting to download...")
        if cfg.download_type == 'drive':
            download_drive(checkpoint_path + cfg.checkpoint, cfg)
        else:
            download_website(cfg.download_weights,checkpoint_path + cfg.checkpoint)




class Yolov5Wrapper(ModelWrapper):
    def __init__(self, device: str, cfg: dict) -> None:
        super().__init__("yolov5")
        self.cfg = cfg
        check_checkpoint(self.cfg.path_to_checkpoint, self.cfg)
        self.model = yolov5.load(self.cfg.path_to_checkpoint + self.cfg.checkpoint)
        self.model.conf = self.cfg.conf  # NMS confidence threshold
        self.model.iou = self.cfg.iou  # NMS IoU threshold
        self.model.agnostic = self.cfg.agnostic  # NMS class-agnostic
        self.model.multi_label = self.cfg.multi_label  # NMS multiple labels per box
        self.model.max_det = self.cfg.max_det  # maximum number of detections per image
        self.read_video = self.cfg.read_video
    @torch.no_grad()
    def process(self, frame: np.array, results = None) -> dict:
        results_by_image = self.model(frame, size=len(frame[0]))
        
        res = []
        shape = (frame.shape[1], frame.shape[0])

        for results in results_by_image.pred[0]:
            
            bbox = results[:4].cpu().numpy()
            score = results[4].cpu().numpy()
            categories = results[5].cpu().numpy()

            if score >= self.cfg.min_confidence:
                output = {
                            'bbox':ltrb_to_ltwh(bbox, shape).tolist(),
                            'score':score.tolist(),
                            'category_id':1
                            }

                res.append(output)            
        return {
            'annotations': res,
            'width_height': (np.shape(frame)[1],np.shape(frame)[0]),
            'model': self.name
        }


#TODO: POSE OR NOT POSE, only the checkpoint is different.. Check if configs should changes...
class Yolov8Wrapper(ModelWrapper):
    def __init__(self, device: str, cfg: dict) -> None:
        super().__init__("yolo")
        #cfg = hydra.compose(config_name="detector/yolo")
        #self.cfg = cfg.detector.cfg
        self.cfg = cfg
        self.model = YOLO(self.cfg.path_to_checkpoint + self.cfg.checkpoint)
        #self.model = torch.hub.load('ultralytics/yolov5', self.cfg.path_to_checkpoint + self.cfg.checkpoint)
        self.model.to(device)
        self.read_video = True
        self.initial = True

    @torch.no_grad()
    def process(self, frame: np.array, results = None) -> dict:
        results_by_image = self.model(source=frame, augment=False)
        res = []
        shape = (frame.shape[1], frame.shape[0]) 
        for results in zip(results_by_image):
            for idx, bbox in enumerate(results[0].boxes.cpu().numpy()):

                if bbox.cls == 0 and bbox.conf >= self.cfg.min_confidence:
                    output = {
                                'bbox':ltrb_to_ltwh(bbox.xyxy[0], shape).tolist(),
                                'score':bbox.conf[0].tolist(),
                                'category_id':1
                                }
                    if results[0].keypoints is not None:
                        kpts = results[0].keypoints.cpu().numpy()[idx].data[0].tolist()
                        output['keypoints'] = [item for sublist in kpts for item in sublist]

                    res.append(output)
            
        return {
            'annotations': res,
            'width_height': (np.shape(frame)[1],np.shape(frame)[0]),
            'model': self.name
        }
#TODO: Every Openpifpaf plugins/models here
#TODO: ARGS to fix !!!
class OpenpifpafWrapper(ModelWrapper):
    def __init__(self, device: str, cfg: dict) -> None:
        super().__init__('openpifpaf')
        parser = argparse.ArgumentParser(
            prog="pifpaf",
            description=__doc__,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            allow_abbrev=False
        )
        #cfg = hydra.compose(config_name="detector/openpifpaf")
        self.cfg = cfg
        check_checkpoint(self.cfg.path_to_checkpoint, self.cfg)
        openpifpaf.decoder.cli(parser)
        openpifpaf.network.Factory.cli(parser)
        openpifpaf.logger.cli(parser)
        self.name = "openpifpaf"
        self.args, _ = parser.parse_known_args()
        self.args.disable_cuda = self.cfg.disable_cuda
        self.args.batch_size = self.cfg.batch_size
        self.args.fast_rescaling = self.cfg.fast_rescaling
        self.args.quiet = self.cfg.quiet
        self.args.instance_threshold = self.cfg.instance_threshold
        self.args.force_complete_pose = self.cfg.force_complete_pose
        self.read_video = True
        model_cpu, _ = openpifpaf.network.Factory().factory()
        self.model = model_cpu.to(device)
        head_metas = [hn.meta for hn in self.model.head_nets]
        self.processor = openpifpaf.decoder.factory(head_metas)
        self.checkpoint = self.cfg.checkpoint
        self.device = device
        self.build_pifpaf()
        args = self.args
        self.pifpaf_preprocess = self.preprocess_factory(args)

    def build_pifpaf(self):
        self.args.checkpoint = self.cfg.checkpoint #MODEL_MAP[PifPafModel.accurate]
        self.args.force_complete_pose = True
        self.args.decoder = ['cifcaf:0']
        self.args.instance_threshold = None
        self.configure()

    def build_plugin(self, plugin_name):
        import hydra
        cfg = hydra.compose(config_name="configs/detector/openpifpaf/plugins")
        self.args.checkpoint = self.cfg


    def configure(self):
        openpifpaf.decoder.configure(self.args)
        openpifpaf.network.Factory.configure(self.args)

    #TODO: Delete..?
    def collate_images_anns_meta(batch):
        idxs = [b[0] for b in batch]
        batch = [b[1] for b in batch]
        anns = [b[-2] for b in batch]
        metas = [b[-1] for b in batch]

        processed_images = torch.utils.data.dataloader.default_collate(
            [b[0] for b in batch]
        )
        idxs = torch.utils.data.dataloader.default_collate(idxs)
        return idxs, (processed_images, anns, metas)
    
    @torch.no_grad()
    def preprocess_factory(self, args):
        rescale_t = None
        if 'long_edge' in args and args.long_edge:
            rescale_t = transforms.RescaleAbsolute(args.long_edge, fast=args.fast_rescaling)
        pad_t = None
        if args.batch_size > 1:
            assert args.long_edge, '--long-edge must be provided for batch size > 1'
            pad_t = transforms.CenterPad(args.long_edge)
        else:
            pad_t = transforms.CenterPadTight(16)
        return transforms.Compose([
            transforms.NormalizeAnnotations(),
            rescale_t,
            pad_t,
            transforms.EVAL_TRANSFORM,
        ])

    @torch.no_grad()
    def preprocess(self, frame):
        #TODO PREPROCESS OPENPIFPAF WITHOUT PIL CONVERTION..?
        processed_image, _, meta = self.pifpaf_preprocess(Image.fromarray(frame), [], None)
        return [processed_image, _, meta]
    
    @torch.no_grad()
    def process(self, input, results = None):
        # input = [processed_image, anns, meta]
        meta = input[2]
        image_tensors_batch = torch.unsqueeze(input[0].float(), 0)
        pred_anns = self.processor.batch(self.model, image_tensors_batch, device=self.device)[0]
        res = []
        for ann in pred_anns:
            inv = ann.inverse_transform(meta)
            res.append(ann.json_data())
        return {
            'annotations': res,
            'width_height': (int(meta['width_height'][0]), int(meta['width_height'][1])),
            'model': self.name
        }


#TODO: Change params... Maybe Hydra file by default, can me selected on command line..?
#/home/celine/pb-dart2/pb-track/inference/configs/track/oc_sort.yaml
class OCSortWrapper(TrackWrapper):
    def __init__(self, device: str, cfg: dict) -> None:
        super().__init__('oc sort')
        #cfg = hydra.compose(config_name="track/oc_sort")
        self.cfg = cfg
        self.tracker = OCSort(**self.cfg.hyperparams)
        self.read_video = False

    def process(self, output):
        pred_bboxes = []
        new_output = {'annotations':[]}
        for index, annotation in enumerate(output['annotations']):
            if annotation['score'] > self.cfg.min_confidence:
                conf = annotation['score']
                cls = annotation['category_id']
                id_i = index
                pred_bboxes.append([annotation['bbox'][0],annotation['bbox'][1], annotation['bbox'][2] + annotation['bbox'][0], annotation['bbox'][1]+ annotation['bbox'][3], conf, cls, index])
        if(len(pred_bboxes)==0):
            return {'annotations':[], 'width_height': output['width_height'], 'tracker': self.name }
        track_output = self.tracker.update(torch.tensor(pred_bboxes), None)
        #results = np.asarray(track_output)  # N'x8 [l,t,r,b,track_id,class,conf,idx]
        for i in range(len(track_output)):
            
            output['annotations'][int(track_output[i][7])]['track_id'] = int(track_output[i][4])
            new_output['annotations'].append(output['annotations'][int(track_output[i][7])])
            new_output['tracker'] = self.name
        new_output['width_height'] = output['width_height']
        new_output['model'] = output['model']
        new_output['tracker'] = self.name
        return new_output
    
    def hyperparams(self):
        parser = argparse.ArgumentParser(
            prog="ocsort",
            description=__doc__,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            allow_abbrev=False
        )
        self.args, _ = parser.parse_known_args()

class BytetrackWrapper(TrackWrapper):
    def __init__(self, device: str, cfg: dict) -> None:
        super().__init__('bytetrack')
        #cfg = hydra.compose(config_name="track/byte_track")
        self.cfg = cfg#.track.cfg
        self.tracker = BYTETracker(cfg.hyperparams.track_thresh, cfg.hyperparams.track_buffer, cfg.hyperparams.match_thresh, cfg.hyperparams.frame_rate)
        self.read_video = False
        self.name = 'byte track'

    #TODO: Delete non tracked or not ..?
    @torch.no_grad()
    def process(self, output: dict) -> dict:
        pred_bboxes = []
        new_output = {'annotations':[]}
        for index, annotation in enumerate(output['annotations']):
            if annotation['score'] > self.cfg.min_confidence:
                conf = annotation['score']
                cls = annotation['category_id']
                id_i = index
                pred_bboxes.append([annotation['bbox'][0],annotation['bbox'][1], annotation['bbox'][2] + annotation['bbox'][0], annotation['bbox'][1]+ annotation['bbox'][3], conf, cls, index])
        if(len(pred_bboxes)==0):
            return {'annotations':[], 'width_height': output['width_height'], 'tracker': self.name }
        track_output = self.tracker.update(torch.tensor(pred_bboxes), None) # N'x8 [l,t,r,b,track_id,class,conf,idx]
        for i in range(len(track_output)):
            output['annotations'][int(track_output[i][7])]['track_id'] = int(track_output[i][4])
            new_output['annotations'].append(output['annotations'][int(track_output[i][7])])
        new_output['width_height'] = output['width_height']
        new_output['model'] = output['model']
        new_output['tracker'] = self.name
        return new_output
    
class BotSortWrapper(DeepTrackWrapper):
    def __init__(self, device: str, cfg: dict) -> None:
        super().__init__('bot sort')
        #cfg = hydra.compose(config_name="track/bot_sort")
        self.cfg = cfg #.track.cfg
        self.device = device
        self.reset()
        self.read_video = True
        self.initial = False
        self.name = 'bot sort'


    def reset(self) -> None:
        """Reset the tracker state to start tracking in a new video."""
        check_checkpoint(self.cfg.path_to_checkpoint, self.cfg)

        checkpoint_path = Path(self.cfg.path_to_checkpoint + self.cfg.checkpoint)

        self.tracker = bot_sort.BoTSORT(
            checkpoint_path,
            self.device,
            self.cfg.fp16,
            **self.cfg.hyperparams
        )

    def preprocess(self, frame):
        return frame


    @torch.no_grad()
    def process(self, frame, output):
        pred_bboxes = []
        new_output = {'annotations':[]}
        for index, annotation in enumerate(output['annotations']):
            if annotation['score'] > self.cfg.min_confidence:
                conf = annotation['score']
                cls = annotation['category_id']
                id_i = index
                pred_bboxes.append([annotation['bbox'][0],annotation['bbox'][1], annotation['bbox'][2] + annotation['bbox'][0], annotation['bbox'][1]+ annotation['bbox'][3], conf, cls, index])
        if(len(pred_bboxes)==0):
            return {'annotations':[], 'width_height': output['width_height'], 'tracker': self.name }
        track_output = self.tracker.update(torch.tensor(pred_bboxes), frame)
        results = np.asarray(track_output)  # N'x8 [l,t,r,b,track_id,class,conf,idx]
        for i in range(len(track_output)):
            output['annotations'][int(track_output[i][7])]['track_id'] = int(track_output[i][4])
            new_output['annotations'].append(output['annotations'][int(track_output[i][7])])
        new_output['width_height'] = output['width_height']
        new_output['model'] = output['model']
        new_output['tracker'] = self.name
        return new_output
    

class DeepOCSortWrapper(DeepTrackWrapper):
    def __init__(self, device: str, cfg: dict) -> None:
        super().__init__('deep oc sort')
        #cfg = hydra.compose(config_name="track/deep_oc_sort")
        self.cfg = cfg#.track.cfg
        self.device = device
        self.reset()
        self.read_video = True
        self.initial = False
        self.name = 'deep oc sort'

    def reset(self):
        """Reset the tracker state to start tracking in a new video."""
        checkpoint_path = Path(self.cfg.path_to_checkpoint + self.cfg.checkpoint)
        check_checkpoint(self.cfg.path_to_checkpoint, self.cfg)

        self.tracker = DeepOCSort.OCSort(
            Path(checkpoint_path),
            self.device,
            self.cfg.fp16,
            **self.cfg.hyperparams
        )

    def preprocess(self, frame):
        return frame
    
    @torch.no_grad()
    def process(self, frame, output):
        pred_bboxes = []
        new_output = {'annotations':[]}
        for index, annotation in enumerate(output['annotations']):
            if annotation['score'] > self.cfg.min_confidence:
                conf = annotation['score']
                cls = annotation['category_id']
                id_i = index
                pred_bboxes.append([annotation['bbox'][0],annotation['bbox'][1], annotation['bbox'][2] + annotation['bbox'][0], annotation['bbox'][1]+ annotation['bbox'][3], conf, cls, index])
        if(len(pred_bboxes)==0):
            return {'annotations':[], 'width_height': output['width_height'], 'tracker': self.name }
        track_output = self.tracker.update(torch.tensor(pred_bboxes), frame)
        results = np.asarray(track_output)  # N'x8 [l,t,r,b,track_id,class,conf,idx]
        for i in range(len(track_output)):
            output['annotations'][int(track_output[i][7])]['track_id'] = int(track_output[i][4])
            new_output['annotations'].append(output['annotations'][int(track_output[i][7])])
        new_output['width_height'] = output['width_height']
        new_output['model'] = output['model']
        new_output['tracker'] = self.name
        return new_output
    
class StrongSortWrapper(DeepTrackWrapper):
    def __init__(self, device: str, cfg: dict) -> None:
        super().__init__('strong sort')
        #cfg = hydra.compose(config_name="track/strong_sort")
        self.cfg = cfg#.track.cfg
        self.device = device
        self.reset()
        self.read_video = True
        self.initial = False
        self.name = 'strong sort'

    def reset(self):
        """Reset the tracker state to start tracking in a new video."""
        check_checkpoint(self.cfg.path_to_checkpoint, self.cfg)
        checkpoint_path = Path(self.cfg.path_to_checkpoint + self.cfg.checkpoint)
        if checkpoint_path.is_file() is False:
            import gdown
            gdown.download(id=self.cfg.download_id, output=self.cfg.path_to_checkpoint + self.cfg.checkpoint, quiet=False, use_cookies=False)

        self.tracker = strong_sort.StrongSORT(
            Path(self.cfg.path_to_checkpoint + self.cfg.checkpoint),
            self.device,
            self.cfg.fp16,
            **self.cfg.hyperparams
        )
        # For camera compensation
        self.prev_frame = None

    @torch.no_grad()
    def process(self, frame, output):
        if self.cfg.ecc:
            if self.prev_frame is not None:
                self.tracker.tracker.camera_update(self.prev_frame, frame)
            self.prev_frame = frame
        pred_bboxes = []
        new_output = {'annotations':[], 'width_height': output['width_height'], 'tracker': self.name }
        for index, annotation in enumerate(output['annotations']):
            if annotation['score'] > self.cfg.min_confidence:
                conf = annotation['score']
                cls = annotation['category_id']
                id_i = index
                pred_bboxes.append([annotation['bbox'][0],annotation['bbox'][1], annotation['bbox'][2] + annotation['bbox'][0], annotation['bbox'][1]+ annotation['bbox'][3], conf, cls, index])
        if(len(pred_bboxes)==0):
            return {'annotations':[], 'width_height': output['width_height'], 'tracker': self.name }
        track_output = self.tracker.update(torch.tensor(pred_bboxes), frame)
        results = np.asarray(track_output)  # N'x8 [l,t,r,b,track_id,class,conf,idx]
        for i in range(len(results)):
            if(int(results[i][8])< len(output['annotations'])):
                #import ipdb
                #ipdb.set_trace()
                output['annotations'][int(results[i][8])]['track_id'] = int(results[i][4])
                new_output['annotations'].append(output['annotations'][int(track_output[i][8])])
                #new_output['tracker'] = self.name
            new_output['width_height'] = output['width_height']
            new_output['model'] = output['model']
            new_output['tracker'] = self.name

        return new_output
    
class AFLinkWrapper(PostProcessWrapper):
    def __init__(self, device: str, cfg: dict) -> None:
        super().__init__('aflink')
    

        self.cfg = cfg#.track.cfg
        self.device = device
        self.model = PostLinker()
        self.model.load_state_dict(torch.load(join(cfg.model_savedir, 'dartfishmodel_epoch100.pth')))
        self.dataset = LinkData('', '')
    
    def convert_AFLink(self, filename):
        annotdict =[]
        if type(filename) is str:
            width_height = json.loads(line.strip())["predictions"]['width_height']
            try:
                with open("your_file.txt", "r") as f:
                    f = open(filename, 'r')
                    for i, line in enumerate(f):
                        for annot in json.loads(line.strip())["predictions"]['annotations']:
                            if 'track_id' in annot:
                                annotdict.append({"bbox":annot['bbox'], "track_id":annot['track_id'], "score":annot['score'], "image_id":i, "category_id":annot['category_id']})  
            except FileNotFoundError:
                raise Exception("File "+filename+" not found")   
                            
        elif type(filename) is list:
            
            width_height = filename[0]['width_height']
            if len(filename) == 0:
                raise Exception("no input to process")
            for i, frame in enumerate(filename):
                for annot in frame['annotations']:
                    annotdict.append({"bbox": annot['bbox'], "track_id": annot['track_id'], "score": annot['score'], "image_id":i, "category_id":annot['category_id']})
        else:
            raise Exception("Unexpected input type")

        return {'annotations': annotdict, 'width_height': width_height}
    

    def AFLink_to_dict(self, af_annots):
        results = []
        print(af_annots)
        
        afdict = af_annots['annotations']
        width_height = af_annots['width_height']


        nb_frame = max(item["image_id"] for item in afdict)
        for i in range(0,nb_frame):
            annot = []
            selected_items = [item for item in afdict if item["image_id"] == i]
            for item in selected_items:
                annot.append({'bbox': item['bbox'], 'score': item['score'], 'category_id': item['category_id'], 'track_id':item['track_id'] })
            results.append({'annotations': annot, 'width_height': width_height, 'model': 'aflink'})
        return results

    def link(self, af_annot):
        linker = AFLink(
            path_in=self.cfg.path_in,
            #TODO: FIX IT
            path_out='/home/celine/pb-dart2/pb-track/inference/video_files/testaflink.json',
            model=self.model,
            dataset=self.dataset,
            thrT=(self.cfg.thrT_min,self.cfg.thrT_max),  # (0,30) or (-10,30)
            thrS=self.cfg.thrS,  # 75
            thrP=self.cfg.thrP,  # 0.05 or 0.10
        )
        linker.pred_data = af_annot
        linker.track = linker.pred_data['annotations']
        output = linker.link()
        return output

    def process(self, output):
        af_annots = self.convert_AFLink(output)
        output = self.link(af_annots)
        return self.AFLink_to_dict(output)
