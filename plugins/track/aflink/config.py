"""
@Author: Du Yunhao
@Filename: config.py
@Contact: dyh_bupt@163.com
@Time: 2021/12/28 15:06
@Discription: config
"""
root_train = '/work/vita/datasets/Dartfish/Track_Data/annotations_linked_by_video'
train_batch = 16
train_epoch = 100
train_lr = 0.001
#train_lr = 0.0001
train_warm = 0
train_decay = 0.00001
num_workers = 0
val_batch = 32
model_minLen = 30
model_inputLen = 30
model_savedir = '/work/vita/corbiere/dartfish/aflink'