import os
import torch
import numpy as np
import pandas as pd
import cv2

from plugins.detect.YOLOv5.models.common import DetectMultiBackend
from plugins.detect.YOLOv5.utils.augmentations import letterbox
from plugins.detect.YOLOv5.utils.general import non_max_suppression, scale_boxes

from pbtrack.pipeline import MultiDetector
from pbtrack.utils.coordinates import ltrb_to_ltwh

import logging

log = logging.getLogger(__name__)


def collate_fn(batch):
    idxs = [b[0] for b in batch]
    images = [b["image"] for _, b in batch]
    shapes = [b["shape"] for _, b in batch]
    return idxs, (images, shapes)


class YOLOv5(MultiDetector):
    collate_fn = collate_fn

    def __init__(self, cfg, device, batch_size):
        super().__init__(cfg, device, batch_size)
        self.img_size = cfg.imgsz
        self.conf_thres = cfg.conf_thres
        self.iou_thres = cfg.iou_thres
        self.max_det = cfg.max_det
        self.model = DetectMultiBackend(cfg.path_to_checkpoint)
        self.model.to(device)
        self.stride = self.model.stride
        self.id = 0

    @torch.no_grad()
    def preprocess(self, metadata: pd.Series):
        image = cv2.imread(metadata.file_path)  # BGR
        image = letterbox(image, self.img_size, stride=self.stride, auto=True)[0]  # padded resize
        image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        image = np.ascontiguousarray(image)  # contiguous
        return {
            "image": image,
            "shape": (image.shape[2], image.shape[1]),
        }

    @torch.no_grad()
    def process(self, batch, metadatas: pd.DataFrame):
        images, shapes = batch
        images = np.stack(images, axis=0)
        images = torch.from_numpy(images)
        images = images.type(torch.FloatTensor).to(self.device)
        images /= 255.0
        results_by_image = self.model(images)
        detections = []
        #import ipdb; ipdb.set_trace()
        results_by_image = non_max_suppression(results_by_image, self.conf_thres, self.iou_thres, max_det=self.max_det)
        #detection = scale_boxes(im.shape[2:], preds, im.shape).round()
        for results, shape, (_, metadata) in zip(
                results_by_image, shapes, metadatas.iterrows()
        ):
            for bbox in results.cpu().numpy():
                #import ipdb; ipdb.set_trace()
                if bbox[4] >= self.cfg.min_confidence:
                    detections.append(
                        pd.Series(
                            dict(
                                image_id=metadata.name,
                                bbox_ltwh=ltrb_to_ltwh(bbox[0:4], shape),
                                bbox_conf=bbox[4],
                                video_id=metadata.video_id,
                                category_id=1,  # `person` class in posetrack
                            ),
                            name=self.id,
                        )
                    )
                    self.id += 1
        return detections