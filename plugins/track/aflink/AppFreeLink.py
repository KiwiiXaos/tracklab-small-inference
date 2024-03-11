"""
@Author: Du Yunhao
@Filename: AFLink.py
@Contact: dyh_bupt@163.com
@Time: 2021/12/28 19:55
@Discription: Appearance-Free Post Link
"""
import os
import glob
import json
import argparse
import numpy as np
import torch
from os.path import join, exists
from collections import defaultdict
from sklearn.preprocessing import normalize
from scipy.optimize import linear_sum_assignment
from ultralytics.utils.ops import xyxy2xywh
from . import config as cfg
from train import train
from . import dataset #import LinkData, DartfishLinkData
from . import model as Model
INFINITY = 1e5

class AFLink:
    def __init__(self, path_in, path_out, model, dataset, thrT: tuple, thrS: int, thrP: float):
        self.thrP = thrP          # 预测阈值
        self.thrT = thrT          # 时域阈值
        self.thrS = thrS          # 空域阈值
        self.model = model        # 预测模型
        self.dataset = dataset    # 数据集类
        self.path_out = path_out  # 结果保存路径
        self.pred_data = ''#json.load(open(path_in))
        self.track = ''#self.pred_data['annotations']
        self.model.cuda()
        self.model.eval()

    # 获取轨迹信息
    def gather_info(self):
        id2info = defaultdict(list)
        for row in self.track:

            x, y, w, h = np.array(row['bbox'])
            id2info[row['track_id']].append([row['image_id'], x, y, w, h])
        self.track = np.array(self.track)
        id2info = {k: np.array(v) for k, v in id2info.items()}
        return id2info

    # 损失矩阵压缩
    def compression(self, cost_matrix, ids):
        # 行压缩
        mask_row = cost_matrix.min(axis=1) < self.thrP
        matrix = cost_matrix[mask_row, :]
        ids_row = ids[mask_row]
        # 列压缩
        mask_col = cost_matrix.min(axis=0) < self.thrP
        matrix = matrix[:, mask_col]
        ids_col = ids[mask_col]
        # 矩阵压缩
        return matrix, ids_row, ids_col

    # 连接损失预测
    def predict(self, track1, track2):
        track1, track2 = self.dataset.transform(track1, track2)
        track1, track2 = track1.unsqueeze(0).cuda(), track2.unsqueeze(0).cuda()
        score = self.model(track1, track2)[0, 1].detach().cpu().numpy()
        return 1 - score

    # 去重复: 即去除同一帧同一ID多个框的情况
    @staticmethod
    def deduplicate(tracks):
        seen = set()
        dedup_list = [x for x in tracks if [(x['track_id'], x['image_id']) not in seen, seen.add((x['track_id'], x['image_id']))][0]]
        return dedup_list

    # 主函数
    def link(self):
        id2info = self.gather_info()
        num = len(id2info)  # 目标数量
        ids = np.array(list(id2info))  # 目标ID
        fn_l2 = lambda x, y: np.sqrt(x ** 2 + y ** 2)  # L2距离
        cost_matrix = np.ones((num, num)) * INFINITY  # 损失矩阵
        '''计算损失矩阵'''
        for i, id_i in enumerate(ids):      # 前一轨迹
            for j, id_j in enumerate(ids):  # 后一轨迹
                if id_i == id_j: continue   # 禁止自娱自乐
                info_i, info_j = id2info[id_i], id2info[id_j]
                fi, bi = info_i[-1][0], info_i[-1][1:3]
                fj, bj = info_j[0][0], info_j[0][1:3]
                if not self.thrT[0] <= fj - fi < self.thrT[1]: continue
                if self.thrS < fn_l2(bi[0] - bj[0], bi[1] - bj[1]): continue
                cost = self.predict(info_i, info_j)
                if cost <= self.thrP: cost_matrix[i, j] = cost
        '''二分图最优匹配'''
        id2id = dict()  # 存储临时匹配结果
        ID2ID = dict()  # 存储最终匹配结果
        cost_matrix, ids_row, ids_col = self.compression(cost_matrix, ids)
        indices = linear_sum_assignment(cost_matrix)
        for i, j in zip(indices[0], indices[1]):
            if cost_matrix[i, j] < self.thrP:
                id2id[ids_row[i]] = ids_col[j]
        for k, v in id2id.items():
            if k in ID2ID:
                ID2ID[v] = ID2ID[k]
            else:
                ID2ID[v] = k
        res = self.track.copy()
        print('Fixed tracking ids: ', ID2ID)
        for t in res:
            if t['track_id'] in ID2ID.keys():
                t['track_id'] = ID2ID[t['track_id']]
        res = self.deduplicate(res)
        self.pred_data['annotations'] = res

        def convert(o):
            if isinstance(o, np.int64): return int(o)
            raise TypeError

        with open(self.path_out, 'w', encoding='utf-8') as f:
            json.dump(self.pred_data, f, ensure_ascii=False, indent=10, default=convert)
        return self.pred_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir-in', type=str, default='/work/vita/corbiere/dartfish/krishna_models/WP3/bboxes_tracking/yolov5/runs/train/exp15/weights/ocsort_test', help='JSON annotations files')
    parser.add_argument('--dir-out', type=str, default=None, help='OutputJSON annotations files')
    parser.add_argument('--thrT-min', type=int, default=0, help='Ante Temporal threshold')
    parser.add_argument('--thrT-max', type=int, default=30, help='Post Temporal threshold')
    parser.add_argument('--thrS', type=int, default=200, help='Spatial threshold')
    parser.add_argument('--thrP', type=float, default=0.05, help='Prediction threshold')
    opt = parser.parse_args()
    if not opt.dir_out:
        opt.dir_out = opt.dir_in + '_aflink'
    if not exists(opt.dir_out): os.mkdir(opt.dir_out)

    model = Model.PostLinker()
    model.load_state_dict(torch.load(join(cfg.model_savedir, 'dartfishmodel_epoch100.pth')))
    inf_dataset = dataset.DartfishLinkData(cfg.root_train, 'test')
    for path_in in sorted(glob.glob(opt.dir_in + '/*.json')):
        print('--- Processing the file {}'.format(path_in))
        linker = AFLink(
            path_in=path_in,
            path_out=path_in.replace(opt.dir_in, opt.dir_out),
            model=model,
            dataset=inf_dataset,
            thrT=(opt.thrT_min,opt.thrT_max),  # (0,30) or (-10,30)
            thrS=opt.thrS,  # 75
            thrP=opt.thrP,  # 0.05 or 0.10
        )
        linker.link()

