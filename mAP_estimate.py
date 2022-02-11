import torch
from torch.utils.data import DataLoader

import cv2
import numpy as np
from torchvision import transforms
from YOLO_V1_DataSet import YoloV1DataSet
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt

from nms import generate_q_sigmoid, sigmoid_lut, post_process, NMS_max, torch_post_process, torch_NMS_max
from sigmoid import generate_q_sigmoid, sigmoid_lut, q17p14, q_mul, q_div

import sys
import os

PROJECT_ROOT = os.path.join(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "ai8x-training"))

import importlib
mod = importlib.import_module("yolov1_bn_model_noaffine")

import ai8x

from map import calculate_map_main, NMS, gt_std
from YOLO_V1_DataSet import YoloV1DataSet

ai8x.set_device(85, simulate=False, round_avg=False, verbose=True)

dataset_root = "/data/yiwei/VOC2007"
checkpoint_fname = './log/QAT-20220210-174132/scaled224_noaffine_shift0.99_maxim_yolo_qat_ep200.pth'   # batch_size 16 full training set

data_type = 'Test'
data_type = 'Train'
dataSet = YoloV1DataSet(imgs_dir=f"{dataset_root}/{data_type}/JPEGImages",
                        annotations_dir=f"{dataset_root}/{data_type}/Annotations",
                        ClassesFile=f"{dataset_root}/VOC_remain_class.data",
                        train_root=f"{dataset_root}/{data_type}/ImageSets/Main/",
                        )

dataLoader = DataLoader(dataSet, batch_size=256, shuffle=False, num_workers=4)

Yolo = mod.Yolov1_net(num_classes=dataSet.Classes, bias=True)
qat_policy = {'start_epoch': 150,
              'weight_bits': 8,
              'bias_bits': 8,
              'shift_quantile': 0.99}

ai8x.fuse_bn_layers(Yolo)
ai8x.initiate_qat(Yolo, qat_policy)
Yolo.load_state_dict(torch.load(checkpoint_fname, map_location=lambda storage, loc: storage))


pred_results = []
gt_results = []

with torch.no_grad():
    for batch_index, batch_test in enumerate(dataLoader):
        data = batch_test[0].float()
        label_data = batch_test[1].float()
        bb_pred, _ = Yolo(data)
        pred_results.append(bb_pred)
        gt_results.append(label_data)


gt_results = torch.cat(gt_results)
gt_results = gt_results.squeeze(dim=3)
gt_results_std = gt_std(gt_results, S=7, B=2, img_size=224)
for cls_id in range(dataSet.Classes):
    total_bounding_box = sum(len([b for b in boxes if b[-1] == cls_id]) for boxes in gt_results_std)
    print(f'{dataSet.IntToClassName[cls_id]} has {total_bounding_box} bounding boxes.')

pred_results = torch.cat(pred_results)  # N x 7 x 7 x (2 x B + num_class)
bounding_box_pred = NMS(pred_results, img_size=224, confidence_threshold=0.5, iou_threshold=0.5)  # , iou_threshold=0.)
bounding_box_pred = [[[y[0], y[1], y[2], y[3], y[-1], y[4]] for y in z] for z in bounding_box_pred]

calculate_map_main(gt_results_std, bounding_box_pred, iou_gt_thr=0.5, class_num=dataSet.Classes)
