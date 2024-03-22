import os
import json
import torch
import numpy as np
import pandas as pd
from tabulate import tabulate
import pathlib

from tracklab.core import Evaluator as EvaluatorBase
from tracklab.utils import wandb

from plugins.eval.Dartfish.eval.posetrack21 import posetrack21 as posetrack21
from plugins.eval.Dartfish.eval import mot
from poseval.eval_helpers import (
    load_data_dir,
    Joint,
    mapmetrics2dict,
    precmetrics2dict,
    recallmetrics2dict,
    motmetrics2dict,
)
from poseval.evaluateAP import evaluateAP
from poseval.evaluateTracking import evaluateTracking
from torchmetrics.detection.mean_ap import MeanAveragePrecision

import logging

log = logging.getLogger(__name__)


# FIXME some parts can be cleaned but works for now
class DartfishEvaluator(EvaluatorBase):
    def __init__(self, cfg, *args, **kwargs):
        self.cfg = cfg
        # CHANGED HERE FOR DARTFISH
        self.cfg.posetrack_trackers_folder = os.path.join(pathlib.Path().resolve(), self.cfg.posetrack_trackers_folder)
        self.cfg.mot_trackers_folder = os.path.join(pathlib.Path().resolve(), self.cfg.mot_trackers_folder)

    def run(self, tracker_state):
        log.info("Starting evaluation on PoseTrack21")
        image_metadatas = (
            tracker_state.image_metadatas.merge(
                tracker_state.video_metadatas["name"],
                left_on="video_id",
                right_on="id",
            )
            .set_index(tracker_state.image_metadatas.index)
            .rename(columns={"name": "video_name"})
        )
        images = self._images(image_metadatas)
        category = self._category(tracker_state.video_metadatas)
        seqs = list(tracker_state.video_metadatas.name)
        bbox_column = self.cfg.bbox_column_for_eval
        eval_pose_on_all = self.cfg.eval_pose_on_all
        if self.cfg.eval_pose_estimation:
            annotations = self._annotations_pose_estimation_eval(
                tracker_state.detections_pred,
                image_metadatas,
                bbox_column,
                eval_pose_on_all,
            )
            trackers_folder = os.path.join(
                self.cfg.posetrack_trackers_folder, "pose_estimation"
            )
            self._save_json(images, annotations, category, trackers_folder)

            # Keypoint evaluation
            argv = ["", self.cfg.posetrack_gt_folder, trackers_folder]
            gtFramesAll, prFramesAll = load_data_dir(argv, seqs)
            apAll, preAll, recAll = evaluateAP(
                gtFramesAll, prFramesAll, "", False, False
            )
            res_combined = mapmetrics2dict(apAll)
            self._print_results(
                res_combined,
                scale_factor=1.0,
                title="Pose estimation - keypoints average precision",
                print_by_video=self.cfg.print_by_video,
            )
            wandb.log(res_combined, "PoseTrack21/kp/AP")
            res_combined = precmetrics2dict(preAll)
            self._print_results(
                res_combined,
                scale_factor=1.0,
                title="Pose estimation - keypoints precision",
                print_by_video=self.cfg.print_by_video,
            )
            wandb.log(res_combined, "PoseTrack21/kp/precision")
            res_combined = recallmetrics2dict(recAll)
            self._print_results(
                res_combined,
                scale_factor=1.0,
                title="Pose estimation - keypoints recall",
                print_by_video=self.cfg.print_by_video,
            )
            wandb.log(res_combined, "PoseTrack21/kp/recall")

        if self.cfg.eval_pose_tracking:
            annotations = self._annotations_tracking_eval(
                tracker_state.detections_pred, image_metadatas, bbox_column
            )
            trackers_folder = os.path.join(
                self.cfg.posetrack_trackers_folder, "pose_tracking"
            )
            self._save_json(images, annotations, category, trackers_folder)

            # Keypoint tracking evaluation
            evaluator = posetrack21.api.get_api(
                trackers_folder=trackers_folder,
                gt_folder=self.cfg.posetrack_gt_folder,
                eval_type="pose_tracking",
                use_parallel=self.cfg.use_parallel,
                num_parallel_cores=self.cfg.num_parallel_cores,
                SEQS=seqs,
                similarity_fn="OKS"
            )
            res_combined, res_by_video = evaluator.eval()
            self._print_results(
                res_combined,
                res_by_video,
                scale_factor=100,
                title="Pose tracking - keypoints HOTA",
                print_by_video=self.cfg.print_by_video,
            )
            wandb.log(res_combined, "PoseTrack21/kp/HOTA", res_by_video)

            argv = ["", self.cfg.posetrack_gt_folder, trackers_folder]
            gtFramesAll, prFramesAll = load_data_dir(argv, seqs)
            metricsAll = evaluateTracking(gtFramesAll, prFramesAll, "", False, False)
            metrics = np.zeros([Joint().count + 4, 1])
            for i in range(Joint().count + 1):
                metrics[i, 0] = metricsAll["mota"][0, i]
            metrics[Joint().count + 1, 0] = metricsAll["motp"][0, Joint().count]
            metrics[Joint().count + 2, 0] = metricsAll["pre"][0, Joint().count]
            metrics[Joint().count + 3, 0] = metricsAll["rec"][0, Joint().count]
            res_combined = motmetrics2dict(metrics)
            self._print_results(
                res_combined,
                scale_factor=1.0,
                title="Pose tracking - keypoints MOTA",
                print_by_video=self.cfg.print_by_video,
            )
            wandb.log(res_combined, "PoseTrack21/kp/MOTA")

        if self.cfg.eval_reid_pose_tracking:
            annotations = self._annotations_reid_pose_tracking_eval(
                tracker_state.detections_pred, image_metadatas, bbox_column
            )
            trackers_folder = os.path.join(
                self.cfg.posetrack_trackers_folder, "reid_pose_tracking"
            )
            self._save_json(images, annotations, category, trackers_folder)
            evaluator = posetrack21.api.get_api(
                trackers_folder=trackers_folder,
                gt_folder=self.cfg.posetrack_gt_folder,
                eval_type="reid_tracking",
                use_parallel=self.cfg.use_parallel,
                num_parallel_cores=self.cfg.num_parallel_cores,
                SEQS=seqs,
            )
            res_combined, res_by_video = evaluator.eval()
            self._print_results(
                res_combined,
                res_by_video,
                scale_factor=100,
                title="Pose tracking cross-video - keypoints HOTA",
                print_by_video=self.cfg.print_by_video,
            )
            wandb.log(res_combined, "PoseTrack21/kp/ReID", res_by_video)

        if self.cfg.eval_mot:
            # Bounding box evaluation
            # AP
            bbox_map = self.compute_bbox_map(
                tracker_state.detections_pred,
                tracker_state.detections_gt,
                tracker_state.image_metadatas,
            )
            map(bbox_map.pop, ["map_per_class", "mar_100_per_class"])
            headers = bbox_map.keys()
            data = [np.round(100 * bbox_map[name].item(), 2) for name in headers]
            log.info(
                "MOT - bbox mAP\n" + tabulate([data], headers=headers, tablefmt="plain")
            )
            wandb.log(bbox_map, "PoseTrack21/bbox/AP")

            # HOTA
            trackers_folder = self.cfg.mot_trackers_folder
            mot_df = self._mot_encoding(
                tracker_state.detections_pred, image_metadatas, bbox_column
            )
            self._save_mot(mot_df, trackers_folder)

            # Adding saving JSON files
            if self.cfg.eval_reid_pose_tracking:
                annotations = self._annotations_reid_pose_tracking_eval(
                tracker_state.detections_pred, image_metadatas, bbox_column
            )
            else:
                annotations = self._annotations_reid_tracking_eval(
                    tracker_state.detections_pred, image_metadatas, bbox_column
                )
            self._save_json(images, annotations, category, trackers_folder)

            evaluator = posetrack21.api.get_api(
                trackers_folder=trackers_folder,
                gt_folder=self.cfg.mot_gt_folder,
                eval_type="posetrack_mot",
                use_parallel=self.cfg.use_parallel,
                num_parallel_cores=self.cfg.num_parallel_cores,
                SEQS=seqs,
            )
            res_combined, res_by_video = evaluator.eval()
            self._print_results(
                res_combined,
                res_by_video,
                100,
                title="MOT - bbox HOTA",
                print_by_video=self.cfg.print_by_video,
            )
            wandb.log(res_combined, "PoseTrack21/bbox/HOTA", res_by_video)

            # # MOTA
            # dataset = mot.PTWrapper(
            #     self.cfg.mot_gt_folder,
            #     self.cfg.mot.dataset_path,
            #     seqs,
            #     vis_threshold=self.cfg.mot.vis_threshold,
            # )
            # mot_accums = []
            # for seq in dataset:
            #     results = seq.load_results(os.path.join(trackers_folder, "results"))
            #     mot_accums.append(
            #         mot.get_mot_accum(
            #             results,
            #             seq,
            #             use_ignore_regions=self.cfg.mot.use_ignore_regions,
            #             ignore_iou_thres=self.cfg.mot.ignore_iou_thres,
            #         )
            #     )
            # if mot_accums:
            #     log.info("MOT - bbox MOTA")
            #     str_summary, summary = mot.evaluate_mot_accums(
            #         mot_accums,
            #         [str(s) for s in dataset if not s.no_gt],
            #         generate_overall=True,
            #     )
            #     results_mot_bbox = summary.to_dict(orient="index")
            #     wandb.log(
            #         results_mot_bbox["OVERALL"],
            #         "PoseTrack21/bbox/MOTA",
            #         results_mot_bbox,
            #     )

    # PoseTrack helper functions
    @staticmethod
    def _images(image_metadatas):
        len_before_drop = len(image_metadatas)
        image_metadatas["id"] = image_metadatas.index
        image_metadatas.dropna(
            subset=[
                "file_path",
                "id",
                "frame",
            ],
            how="any",
            inplace=True,
        )
        if len_before_drop != len(image_metadatas):
            log.warning(
                "Dropped {} rows with NA values from image metadatas".format(
                    len_before_drop - len(image_metadatas)
                )
            )
        image_metadatas.rename(columns={"file_path": "file_name"}, inplace=True)
        image_metadatas["frame_id"] = image_metadatas["id"]

        images = {}
        videos_names = image_metadatas["video_name"].unique()
        for video_name in videos_names:
            images_by_video = image_metadatas[
                image_metadatas["video_name"] == video_name
            ]
            images[video_name] = images_by_video[
                ["file_name", "id", "frame_id"]
            ].to_dict("records")
        return images

    @staticmethod
    def _category(video_metadatas):
        return video_metadatas.categories[0]

    # FIXME fuse different annotations functions
    @staticmethod
    def _annotations_pose_estimation_eval(
        detections_pred, image_metadatas, bbox_column, eval_pose_on_all
    ):
        detections_pred = detections_pred.copy()
        na_col_to_drop = [
            "keypoints_xyc",
            bbox_column,
            "image_id",
        ]
        if not eval_pose_on_all:
            # If set to false, will evaluate pose estimation only on tracked detections (i.e. detections with a
            # defined 'track_id')
            na_col_to_drop.append("track_id")
        len_before_drop = len(detections_pred)
        detections_pred.dropna(
            subset=na_col_to_drop,
            how="any",
            inplace=True,
        )
        # drop detections that are in ignored regions
        detections_pred.drop(
            detections_pred[detections_pred.ignored].index, inplace=True
        )
        if len_before_drop != len(detections_pred):
            log.warning(
                "Dropped {} rows with NA values".format(
                    len_before_drop - len(detections_pred)
                )
            )
        detections_pred.rename(
            columns={"keypoints_xyc": "keypoints", bbox_column: "bbox"},
            inplace=True,
        )
        if "scores" not in detections_pred.columns:
            # 'scores' can already be present if loaded from a json file with external detections_pred
            # for PoseTrack21 author baselines, not using their provided score induces a big drop in performance
            detections_pred["scores"] = detections_pred["keypoints"].apply(
                lambda x: x[:, 2]
            )
        detections_pred["track_id"] = detections_pred.index

        annotations = {}
        videos_names = image_metadatas["video_name"].unique()
        for video_name in videos_names:
            image_ids = image_metadatas[
                image_metadatas["video_name"] == video_name
            ].index
            predictions_by_video = detections_pred[
                detections_pred["image_id"].isin(image_ids)
            ]
            annotations[video_name] = predictions_by_video[
                ["bbox", "image_id", "keypoints", "scores", "track_id"]
            ].to_dict("records")
        return annotations

    # FIXME fuse different annotations functions
    @staticmethod
    def _annotations_tracking_eval(detections_pred, image_metadatas, bbox_column):
        detections_pred = detections_pred.copy()
        detections_pred["id"] = detections_pred.index
        col_to_drop = [
            "keypoints_xyc",
            bbox_column,
            "image_id",
            "track_id",
        ]
        col_to_drop = [col for col in col_to_drop if col in detections_pred.columns]
        len_before_drop = len(detections_pred)
        detections_pred.dropna(
            subset=col_to_drop,
            how="any",
            inplace=True,
        )
        # drop detections that are in ignored regions
        detections_pred.drop(
            detections_pred[detections_pred.ignored].index, inplace=True
        )
        if len_before_drop != len(detections_pred):
            log.warning(
                "Dropped {} rows with NA values".format(
                    len_before_drop - len(detections_pred)
                )
            )
        detections_pred.rename(
            columns={"keypoints_xyc": "keypoints", bbox_column: "bbox"},
            inplace=True,
        )
        detections_pred["scores"] = detections_pred["keypoints"].apply(
            lambda x: x[:, 2]
        )
        detections_pred["track_id"] = detections_pred.index

        annotations = {}
        videos_names = image_metadatas["video_name"].unique()
        for video_name in videos_names:
            image_ids = image_metadatas[
                image_metadatas["video_name"] == video_name
            ].index
            predictions_by_video = detections_pred[
                detections_pred["image_id"].isin(image_ids)
            ]
            annotations[video_name] = predictions_by_video[
                ["bbox", "image_id", "keypoints", "scores", "track_id"]
            ].to_dict("records")

    # FIXME fuse different annotations functions
    @staticmethod
    def _annotations_reid_pose_tracking_eval(
        detections_pred, image_metadatas, bbox_column
    ):
        detections_pred = detections_pred.copy()
        len_before_drop = len(detections_pred)
        detections_pred.dropna(
            subset=[
                "keypoints_xyc",
                bbox_column,
                "image_id",
                "track_id",
            ],
            how="any",
            inplace=True,
        )
        # drop detections that are in ignored regions
        detections_pred.drop(
            detections_pred[detections_pred.ignored].index, inplace=True
        )
        if len_before_drop != len(detections_pred):
            log.warning(
                "Dropped {} rows with NA values".format(
                    len_before_drop - len(detections_pred)
                )
            )
        detections_pred.rename(
            columns={"keypoints_xyc": "keypoints", bbox_column: "bbox"},
            inplace=True,
        )
        detections_pred["scores"] = detections_pred["keypoints"].apply(
            lambda x: x[:, 2]
        )
        detections_pred["track_id"] = detections_pred["track_id"].astype(int)

        annotations = {}
        videos_names = image_metadatas["video_name"].unique()
        for video_name in videos_names:
            image_ids = image_metadatas[
                image_metadatas["video_name"] == video_name
            ].index
            predictions_by_video = detections_pred[
                detections_pred["image_id"].isin(image_ids)
            ]
            annotations[video_name] = predictions_by_video[
                ["bbox", "image_id", "keypoints", "scores", "track_id"]
            ].to_dict("records")
        return annotations

    @staticmethod
    def _annotations_reid_tracking_eval(
        detections_pred, image_metadatas, bbox_column
    ):
        detections_pred = detections_pred.copy()
        len_before_drop = len(detections_pred)
        detections_pred.dropna(
            subset=[
                bbox_column,
                "image_id",
                "track_id",
            ],
            how="any",
            inplace=True,
        )
        # drop detections that are in ignored regions
        detections_pred.drop(
            detections_pred[detections_pred.ignored].index, inplace=True
        )
        if len_before_drop != len(detections_pred):
            log.warning(
                "Dropped {} rows with NA values".format(
                    len_before_drop - len(detections_pred)
                )
            )
        detections_pred.rename(
            columns={bbox_column: "bbox"},
            inplace=True,
        )
        detections_pred["scores"] = detections_pred["bbox_conf"]
        detections_pred["track_id"] = detections_pred["track_id"].astype(int)

        annotations = {}
        videos_names = image_metadatas["video_name"].unique()
        for video_name in videos_names:
            image_ids = image_metadatas[
                image_metadatas["video_name"] == video_name
            ].index
            predictions_by_video = detections_pred[
                detections_pred["image_id"].isin(image_ids)
            ]
            annotations[video_name] = predictions_by_video[
                ["bbox", "image_id", "scores", "track_id"]
            ].to_dict("records")
        return annotations

    @staticmethod
    def _save_json(images, annotations, category, path):
        os.makedirs(path, exist_ok=True)
        for video_name in images.keys():
            file_path = os.path.join(path, f"{video_name}.json")
            with open(file_path, "w") as f:
                json.dump(
                    {
                        "images": images[video_name],
                        "annotations": annotations[video_name],
                        "categories": category,
                    },
                    f,
                    cls=DartfishEvaluator.PoseTrackEncoder,
                    sort_keys=True,
                    indent=4,
                )

    # MOT helper functions
    @staticmethod
    def _mot_encoding(detections_pred, image_metadatas, bbox_column):
        detections_pred = detections_pred.copy()
        image_metadatas["id"] = image_metadatas.index
        df = pd.merge(
            image_metadatas.reset_index(drop=True),
            detections_pred.reset_index(drop=True),
            left_on="id",
            right_on="image_id",
        )
        len_before_drop = len(df)
        df.dropna(
            subset=[
                "video_name",
                "frame",
                "track_id",
                bbox_column,
                #"keypoints_xyc",
            ],
            how="any",
            inplace=True,
        )
        # drop detections that are in ignored regions
        df.drop(df[df.ignored].index, inplace=True)
        if len_before_drop != len(df):
            log.warning(
                "Dropped {} rows with NA values".format(len_before_drop - len(df))
            )
        df["track_id"] = df["track_id"].astype(int)
        df["bb_left"] = df[bbox_column].apply(lambda x: x[0])
        df["bb_top"] = df[bbox_column].apply(lambda x: x[1])
        df["bb_width"] = df[bbox_column].apply(lambda x: x[2])
        df["bb_height"] = df[bbox_column].apply(lambda x: x[3])
        df = df.assign(x=-1, y=-1, z=-1)
        return df

    @staticmethod
    def _save_mot(mot_df, save_path):
        save_path = os.path.join(save_path, "results")
        os.makedirs(save_path, exist_ok=True)
        # <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
        videos_names = mot_df["video_name"].unique()
        for video_name in videos_names:
            file_path = os.path.join(save_path, f"{video_name}.txt")
            file_df = mot_df[mot_df["video_name"] == video_name].copy()
            if not file_df.empty:
                file_df.sort_values(by="frame", inplace=True)
                file_df[
                    [
                        "frame",
                        "track_id",
                        "bb_left",
                        "bb_top",
                        "bb_width",
                        "bb_height",
                        "bbox_conf",
                        "x",
                        "y",
                        "z",
                    ]
                ].to_csv(
                    file_path,
                    header=False,
                    index=False,
                )
            else:
                open(file_path, "w").close()

    @staticmethod
    def format_metric(metric_name, metric_value, scale_factor):
        if (
            "TP(0)" in metric_name
            or "FN(0)" in metric_name
            or "FP(0)" in metric_name
            or "TN(0)" in metric_name
        ):
            return int(metric_value)
        else:
            return np.around(metric_value * scale_factor, 2)

    @staticmethod
    def _print_results(
        res_combined,
        res_by_video=None,
        scale_factor=1.0,
        title="",
        print_by_video=False,
    ):
        headers = res_combined.keys()
        data = [
            DartfishEvaluator.format_metric(name, res_combined[name], scale_factor)
            for name in headers
        ]
        log.info(f"{title}\n" + tabulate([data], headers=headers, tablefmt="plain"))
        if print_by_video and res_by_video:
            data = []
            for video_name, res in res_by_video.items():
                video_data = [video_name] + [
                    DartfishEvaluator.format_metric(name, res[name], scale_factor)
                    for name in headers
                ]
                data.append(video_data)
            headers = ["video"] + list(headers)
            log.info(
                f"{title} by videos\n"
                + tabulate(data, headers=headers, tablefmt="plain")
            )

    @staticmethod
    def compute_bbox_map(detections_pred, detections_gt, metadatas):
        images_ids = metadatas[metadatas.is_labeled].index
        detections_pred = detections_pred[detections_pred.ignored == False]
        metric = MeanAveragePrecision(box_format="xywh", iou_type="bbox", num_classes=1)
        preds = []
        targets = []
        for image_id in images_ids:
            targets_by_image = detections_gt[detections_gt["image_id"] == image_id]
            if not targets_by_image.empty:
                targets.append(
                    {
                        "boxes": torch.tensor(
                            np.vstack(targets_by_image.bbox_ltwh.values).astype(float)
                        ),
                        "labels": torch.tensor(targets_by_image.category_id.values),
                    }
                )
                preds_by_image = detections_pred[
                    detections_pred["image_id"] == image_id
                ]
                if not preds_by_image.empty:
                    preds.append(
                        {
                            "boxes": torch.tensor(
                                np.vstack(preds_by_image.bbox_ltwh.values).astype(float)
                            ),
                            "scores": torch.tensor(
                                preds_by_image.bbox_conf.values.astype(float)
                            ),
                            "labels": torch.tensor(
                                preds_by_image.category_id.values.astype(int)
                            ),
                        }
                    )
                else:
                    preds.append(
                        {
                            "boxes": torch.tensor([]),
                            "scores": torch.tensor([]),
                            "labels": torch.tensor([]),
                        }
                    )
        metric.update(preds, targets)
        return metric.compute()

    class PoseTrackEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.flatten().tolist()
            return json.JSONEncoder.default(self, obj)