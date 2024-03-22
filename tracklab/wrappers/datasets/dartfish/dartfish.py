import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

from tracklab.datastruct import TrackingDataset, TrackingSet


class Dartfish(TrackingDataset):
    """
    Train set: 120000 images
    Val set: 40000 images
    Test set: 40000 images
    """
    def __init__(
        self,
        dataset_path: str,
        annotation_path: str,
        *args,
        **kwargs
    ):
        self.dataset_path = Path(dataset_path)
        assert self.dataset_path.exists(), "Dataset path does not exist in '{}'".format(
            self.dataset_path
        )

        self.annotation_path = Path(annotation_path)
        assert (
            self.annotation_path.exists()
        ), "Annotations path does not exist in '{}'".format(self.annotation_path)

        train_set = load_tracking_set(
            self.annotation_path, self.dataset_path, "train")
        val_set = load_tracking_set(self.annotation_path, self.dataset_path, "val")
        test_set = load_tracking_set(self.annotation_path, self.dataset_path, "test")

        super().__init__(dataset_path, train_set, val_set, test_set, *args, **kwargs)


def load_tracking_set(anns_path, dataset_path, split):
    # Load annotations into Pandas dataframes
    video_metadatas, image_metadatas, detections_gt = load_annotations(anns_path, split)

    # Fix formatting of dataframes to be compatible with pbtrack
    video_metadatas, image_metadatas, detections_gt = fix_formatting(
        video_metadatas, image_metadatas, detections_gt, dataset_path
    )
    return TrackingSet(
        split,
        video_metadatas,
        image_metadatas,
        detections_gt,
    )


def load_annotations(anns_path, split):
    anns_path = anns_path / split
    anns_files_list = list(anns_path.glob("*.json"))
    assert len(anns_files_list) > 0, "No annotations files found in {}".format(
        anns_path
    )
    detections_gt = []
    image_metadatas = []
    video_metadatas = []
    for path in anns_files_list:
        with open(path) as json_file:
            data_dict = json.load(json_file)
            detections_gt.extend(data_dict["annotations"])
            image_metadatas.extend(data_dict["images"])
            video_metadatas.append(
                {
                    "id": data_dict["images"][0]["vid_id"],
                    "nframes": len(data_dict["images"]),
                    "name": path.stem,
                    "categories": data_dict["categories"],
                }
            )

    return (
        pd.DataFrame(video_metadatas),
        pd.DataFrame(image_metadatas),
        pd.DataFrame(detections_gt),
    )


def fix_formatting(video_metadatas, image_metadatas, detections_gt, dataset_path):
    # Videos
    video_metadatas.set_index("id", drop=False, inplace=True)

    # Images
    image_metadatas.drop(["frame_id", "nframes"], axis=1, inplace=True)
    image_metadatas["file_name"] = image_metadatas["file_name"].apply(
        lambda x: os.path.join(dataset_path, x)
    )
    image_metadatas["frame"] = image_metadatas["file_name"].apply(
        lambda x: int(float(os.path.basename(x)[:6])) // 3
    )
    image_metadatas.rename(
        columns={"vid_id": "video_id", "file_name": "file_path"},
        inplace=True,
    )
    image_metadatas.set_index("id", drop=False, inplace=True)


    # Detections
    detections_gt.rename(columns={"bbox": "bbox_ltwh"}, inplace=True)
    detections_gt.bbox_ltwh = detections_gt.bbox_ltwh.apply(lambda x: np.array(x))
    detections_gt.rename(columns={"keypoints": "keypoints_xyc"}, inplace=True)
    detections_gt.keypoints_xyc = detections_gt.keypoints_xyc.apply(
        lambda x: np.reshape(np.array(x), (-1, 3))
    )
    detections_gt.set_index("id", drop=False, inplace=True)
    # compute detection visiblity as average keypoints visibility
    detections_gt["visibility"] = detections_gt.keypoints_xyc.apply(
        lambda x: x[:, 2].mean())
    # add video_id to detections, will be used for bpbreid 'camid' parameter
    detections_gt = detections_gt.merge(
        image_metadatas[["video_id"]], how="left", left_on="image_id", right_index=True
    )

    return video_metadatas, image_metadatas, detections_gt