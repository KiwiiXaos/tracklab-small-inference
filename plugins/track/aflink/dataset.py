"""
@Author: Du Yunhao
@Filename: dataset.py
@Contact: dyh_bupt@163.com
@Time: 2021/12/28 9:24
@Discription: dataset
"""
import torch
import numpy as np
import glob
import json
import pandas as pd
from os.path import join
from random import randint, normalvariate
from torch.utils.data import Dataset, DataLoader
from ultralytics.utils.ops import xyxy2xywh
from . import config as cfg

SEQ = {
    'train': [
        'MOT17-02-FRCNN',
        'MOT17-04-FRCNN',
        'MOT17-05-FRCNN',
        'MOT17-09-FRCNN',
        'MOT17-10-FRCNN',
        'MOT17-11-FRCNN',
        'MOT17-13-FRCNN'
    ],
    'test': [
        'MOT17-01-FRCNN',
        'MOT17-03-FRCNN',
        'MOT17-06-FRCNN',
        'MOT17-07-FRCNN',
        'MOT17-08-FRCNN',
        'MOT17-12-FRCNN',
        'MOT17-14-FRCNN'
    ]
}

SEQ_DARTFISH = {
    "train": [
        "Tracking_Biel_2021.04.07_21.22.20_video81",
        "Tracking_Biel_2021.04.07_21.22.20_video83",
        "Tracking_Biel_2021.04.07_21.22.20_video84",
        "Tracking_Biel_2021.04.30_17.45.07_video81",
        "Tracking_Biel_2021.04.30_17.45.07_video83",
        "Tracking_Biel_2021.04.30_17.45.07_video84",
        "Tracking_Biel_2021.08.18_19.46.52_video81",
        "Tracking_Biel_2021.08.18_19.46.52_video82",
        "Tracking_Biel_2021.08.18_19.46.52_video83",
        "Tracking_Davos_2021.09.11_20.45.20_video111",
        "Tracking_Davos_2021.09.11_20.45.20_video112",
        "Tracking_Davos_2021.09.11_20.45.20_video113",
        "Tracking_Davos_2021.09.17_20.44.28_video110",
        "Tracking_Davos_2021.09.17_20.44.28_video112",
        "Tracking_Davos_2021.09.17_20.44.28_video113",
        "Tracking_Lousanne_2022.11.04_19.42.46_video81",
        "Tracking_Lousanne_2022.11.04_19.42.46_video82",
        "Tracking_Lousanne_2022.11.04_19.42.46_video84",
        "Tracking_OSC_2021.05.23_12.15.23_video211",
        "Tracking_OSC_2021.05.23_12.15.23_video212",
        "Tracking_OSC_2021.05.23_12.15.23_video214",
        "Tracking_OSC_2021.05.27_20.14.57_video211",
        "Tracking_OSC_2021.05.27_20.14.57_video212",
        "Tracking_OSC_2021.05.27_20.14.57_video213"
     ],
     "test": [
        "Tracking_Biel_2021.04.07_21.22.20_video82",
        "Tracking_Biel_2021.04.30_17.45.07_video82",
        "Tracking_Biel_2021.08.18_19.46.52_video84",
        "Tracking_Davos_2021.09.11_20.45.20_video110",
        "Tracking_Davos_2021.09.17_20.44.28_video111",
        "Tracking_Lousanne_2022.11.04_19.42.46_video83",
        "Tracking_OSC_2021.05.23_12.15.23_video213",
        "Tracking_OSC_2021.05.27_20.14.57_video214"
     ]
}

class LinkData(Dataset):
    def __init__(self, root, mode='train', minLen=cfg.model_minLen, inputLen=cfg.model_inputLen):
        """
        :param minLen: 仅考虑长度超过该阈值的GT轨迹
        :param inputLen: 网络输入轨迹长度
        """
        self.minLen = minLen
        self.inputLen = inputLen
        if root:
            assert mode in ('train', 'val')
            self.root = root
            self.mode = mode
            self.id2info = self.initialize()
            self.ids = list(self.id2info.keys())

    def initialize(self):
        id2info = dict()
        for seqid, seq in enumerate(SEQ['train'], start=1):
            path_gt = join(self.root, '{}/gt/gt.txt'.format(seq, self.mode))
            gts = np.loadtxt(path_gt, delimiter=',')
            gts = gts[(gts[:, 6] == 1) * (gts[:, 7] == 1)]  # 仅考虑“considered" & "pedestrian"
            ids = set(gts[:, 1])
            for objid in ids:
                id_ = objid + seqid * 1e5
                track = gts[gts[:, 1] == objid]
                fxywh = [[t[0], t[2], t[3], t[4], t[5]] for t in track]
                if len(fxywh) >= self.minLen:
                    id2info[id_] = np.array(fxywh)
        return id2info

    def fill_or_cut(self, x, former: bool):
        """
        :param x: input
        :param former: True代表该轨迹片段为连接时的前者
        """
        lengthX, widthX = x.shape
        if lengthX >= self.inputLen:
            if former:
                x = x[-self.inputLen:]
            else:
                x = x[:self.inputLen]
        else:
            zeros = np.zeros((self.inputLen - lengthX, widthX))
            if former:
                x = np.concatenate((zeros, x), axis=0)
            else:
                x = np.concatenate((x, zeros), axis=0)
        return x

    def transform(self, x1, x2):
        # fill or cut
        x1 = self.fill_or_cut(x1, True)
        x2 = self.fill_or_cut(x2, False)
        # min-max normalization
        min_ = np.concatenate((x1, x2), axis=0).min(axis=0)
        max_ = np.concatenate((x1, x2), axis=0).max(axis=0)
        subtractor = (max_ + min_) / 2
        divisor = (max_ - min_) / 2 + 1e-5
        x1 = (x1 - subtractor) / divisor
        x2 = (x2 - subtractor) / divisor
        # numpy to torch
        x1 = torch.tensor(x1, dtype=torch.float)
        x2 = torch.tensor(x2, dtype=torch.float)
        # unsqueeze channel=1
        x1 = x1.unsqueeze(dim=0)
        x2 = x2.unsqueeze(dim=0)
        return x1, x2

    def __getitem__(self, item):
        info = self.id2info[self.ids[item]]
        numFrames = info.shape[0]
        if self.mode == 'train':
            idxCut = randint(self.minLen//3, numFrames - self.minLen//3)  # 随机裁剪点
            # 样本对儿
            info1 = info[:idxCut + int(normalvariate(-5, 3))]  # 为索引添加随机偏差
            info2 = info[idxCut + int(normalvariate(5, 3)):]   # 为索引添加随机偏差
            # 时域扰动
            info2_t = info2.copy()
            info2_t[:, 0] += (-1) ** randint(1, 2) * randint(30, 100)
            # 空间扰动
            info2_s = info2.copy()
            info2_s[:, 1] += (-1) ** randint(1, 2) * randint(100, 500)
            info2_s[:, 2] += (-1) ** randint(1, 2) * randint(100, 500)
        else:
            idxCut = numFrames // 2
            # 样本对儿
            info1 = info[:idxCut]
            info2 = info[idxCut:]
            # 时域扰动
            info2_t = info2.copy()
            info2_t[:, 0] += (-1) ** idxCut * 50
            # 空间扰动
            info2_s = info2.copy()
            info2_s[:, 1] += (-1) ** idxCut * 300
            info2_s[:, 2] += (-1) ** idxCut * 300
        # 返回正负样本对儿
        return self.transform(info1, info2), \
               self.transform(info2, info1), \
               self.transform(info1, info2_t), \
               self.transform(info1, info2_s), \
               (1, 0, 0, 0)

    def __len__(self):
        return len(self.ids)


class DartfishLinkData(LinkData):
    def __init__(self, root, mode='train', minLen=cfg.model_minLen, inputLen=cfg.model_inputLen):
        self.minLen = minLen
        self.inputLen = inputLen
        if root:
            assert mode in ('train', 'test')
            self.root = root
            self.mode = mode
            self.id2info = self.initialize()
            self.ids = list(self.id2info.keys())

    def initialize(self):
        id2info = dict()
        for seqid, seq in enumerate(SEQ_DARTFISH[self.mode], start=1):
            print(f'({seqid}/{len(SEQ_DARTFISH[self.mode])}) Load sequence: {seq}')
            path_gt = join(self.root, f'{seq}.csv')
            df = pd.read_csv(path_gt)
            df = df[df.attribute=='bbox']  # keep only bounding box
            if 'position' in df.columns:
                df.drop(columns=['position'], inplace=True)
            ids = df['tracker_id'].unique()
            for objid in ids:
                id_ = objid + seqid * 1e5
                track = df[df.tracker_id==objid]
                fxywh = []
                for t in track.values:
                    bbox_xywh = xyxy2xywh(t[4:8])
                    fxywh.append([t[0], bbox_xywh[0], bbox_xywh[1], bbox_xywh[2], bbox_xywh[3]])
                    if len(fxywh) >= self.minLen:
                        id2info[id_] = np.array(fxywh)
        return id2info


if __name__ == '__main__':
    dataset = DartfishLinkData(
        root='/work/vita/datasets/Dartfish/Track_Data/annotations_linked_by_video',
        mode='train'
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=2,
        shuffle=True,
        num_workers=1,
        drop_last=False
    )
    print(len(dataset))
    print(len(dataloader))
    for i, (pair1, pair2, pair3, pair4, labels) in enumerate(dataloader):
        pairs_1 = torch.cat((pair1[0], pair2[0], pair3[0], pair4[0]), dim=0)
        pairs_2 = torch.cat((pair1[1], pair2[1], pair3[1], pair4[1]), dim=0)
        label = torch.cat(labels, dim=0)
        print(pairs_1.shape)
        print(label)
        break