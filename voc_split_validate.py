import torch
from torch.utils.data import DataLoader

import cv2
import numpy as np

import sys, os
sys.path.append("../")

from YOLO_V1_DataSet_V2 import YoloV1DataSet
dataSet = YoloV1DataSet(imgs_dir="../../../../../YOLO_V1_GPU/VOC2007/Train/JPEGImages",
                        annotations_dir="../../../../../YOLO_V1_GPU/VOC2007/Train/Annotations",
                        ClassesFile="../../VOC_remain_class.data",
                        num_per_class=150)

# from YOLO_V1_DataSet_small import YoloV1DataSet
# dataSet = YoloV1DataSet(imgs_dir="../../../../../YOLO_V1_GPU/VOC2007/Train/JPEGImages",
#                         annotations_dir="../../../../../YOLO_V1_GPU/VOC2007/Train/Annotations",
#                         ClassesFile="../../VOC_remain_class.data",
#                         data_path='../../../../../YOLO_V1_GPU/VOC2007/Train/ImageSets/Main')


ClassNameToInt = dataSet.ClassNameToInt
#
# dataLoader = DataLoader(dataSet, batch_size=4, shuffle=True, num_workers=0)
#
print(len(dataSet))
print(len(dataSet.ground_truth))


# valide the mapping from image to groundtruth
img_data, ground_truth = dataSet.__getitem__(100)
from matplotlib import pyplot as plt
img_data_show = img_data.permute(2, 1, 0) + 1.
plt.imshow(img_data_show.numpy())
plt.show()

print("Pixel range: {} {}".format(img_data.max(), img_data.min()))

print(ground_truth.shape)
for i in range(7):
    for j in range(7):
        if not (ground_truth[i][j][0][9] == 0):
            print(i, j, ground_truth[i][j][0][10:])


