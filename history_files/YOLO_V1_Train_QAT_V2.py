import importlib

import os
from matplotlib.image import imread
import numpy as np
import matplotlib.pyplot as plt
import time

import torch
from torch import nn
import torch.optim as optim
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms
#import distiller.apputils as apputils
import cv2

import sys
#sys.path.insert(0, 'yolo/')
#sys.path.insert(1, 'distiller/')
#sys.path.insert(2, '/data/detection/')

from YOLO_V1_DataSet_small1 import YoloV1DataSet
from YOLO_V1_LossFunction import  Yolov1_Loss

mod = importlib.import_module("yolov1_bn_model_noaffine")

import ai8x
#%matplotlib inline

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='Use which gpu to train the model.')
args = parser.parse_args()

def count_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

class Args:
    def __init__(self, act_mode_8bit):
        self.act_mode_8bit = act_mode_8bit
        self.truncate_testset = False

# Data path, modify to match file system layout
#data_path = '/data/detection'

dataSet = YoloV1DataSet(imgs_dir="../YOLO_V1_GPU/VOC2007/Train/JPEGImages", annotations_dir="../YOLO_V1_GPU/VOC2007/Train/Annotations", ClassesFile="../YOLO_V1_GPU/VOC2007/Train/VOC_remain_class.data")

dataLoader = DataLoader(dataSet,batch_size=16,shuffle=True,num_workers=4)

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

ai8x.set_device(device=85, simulate=False, round_avg=False)
Yolo = mod.Yolov1_net(num_classes=dataSet.Classes, bias=True)
Yolo = Yolo.to(device)
print("NUMBER OF PARAMETERS",  sum(p.numel() for p in Yolo.parameters()))


loss_function = Yolov1_Loss().to(device)
optimizer = optim.SGD(Yolo.parameters(),lr=3e-5,momentum=0.9,weight_decay=0.0005)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[50, 100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,20000,30000,40000],gamma=0.8)


num_epochs = 400
qat_policy = {'start_epoch':150,
              'weight_bits':8,
              'bias_bits':8,
              'shift_quantile': 0.99}

# MAIN TRAINING

best_acc = 0
best_qat_acc = 0
for epoch in range(0, num_epochs):
    loss_sum = 0
    loss_coord = 0
    loss_confidence = 0
    loss_classes = 0
    epoch_iou = 0
    epoch_object_num = 0
    scheduler.step()

    if epoch > 0 and epoch == qat_policy['start_epoch']:
        print('QAT is starting!')
        # Fuse the BN parameters into conv layers before Quantization Aware Training (QAT)
        torch.save(Yolo.state_dict(), f'yolo_models/scaled224_noaffine_shift{qat_policy["shift_quantile"]}_maxim_yolo_beforeQAT_ep{epoch:04d}.pth')
        ai8x.fuse_bn_layers(Yolo)

        # Switch model from unquantized to quantized for QAT
        ai8x.initiate_qat(Yolo, qat_policy)

        # Model is re-transferred to GPU in case parameters were added
        Yolo.to(device)


    for batch_index, batch_train in enumerate(dataLoader):

        optimizer.zero_grad()
        train_data = batch_train[0].float().to(device)
        train_data.requires_grad = True

        label_data = batch_train[1].float().to(device)
        label_data[:, :, :, :, 5] = label_data[:, :, :, :, 5] / 224
        label_data[:, :, :, :, 6] = label_data[:, :, :, :, 6] / 224
        label_data[:, :, :, :, 7] = label_data[:, :, :, :, 7] / 224
        label_data[:, :, :, :, 8] = label_data[:, :, :, :, 8] / 224
        label_data[:, :, :, :, 9] = label_data[:, :, :, :, 9] / (224*224)


#         label_data = batch_train[1].float().to(device)
        bb_pred, _ = Yolo(train_data)
        loss = loss_function(bounding_boxes=bb_pred,ground_truth=label_data)
        batch_loss = loss[0]
        loss_coord = loss_coord + loss[1]
        loss_confidence = loss_confidence + loss[2]
        loss_classes = loss_classes + loss[3]
        epoch_iou = epoch_iou + loss[4]
        epoch_object_num = epoch_object_num + loss[5]
        batch_loss.backward()
        optimizer.step()
        batch_loss = batch_loss.item()
        loss_sum = loss_sum + batch_loss

        #print("batch_index : {} ; batch_loss : {}".format(batch_index, batch_loss))


    if epoch % 50 == 0:
        if epoch >= qat_policy['start_epoch']:
            torch.save(Yolo.state_dict(), f'yolo_models/scaled224_noaffine_shift{qat_policy["shift_quantile"]}_maxim_yolo_qat_ep{epoch:04d}.pth')
        else:
            torch.save(Yolo.state_dict(), f'yolo_models/scaled224_noaffine_shift{qat_policy["shift_quantile"]}_maxim_yolo_ep{epoch:04d}.pth')


    avg_loss= loss_sum/batch_index
    print("epoch : {} ; loss : {} ; avg_loss: {}".format(epoch,{loss_sum},{avg_loss}))

    epoch = epoch + 1

torch.save(Yolo.state_dict(), f'yolo_models/scaled224_noaffine_shift{qat_policy["shift_quantile"]}_maxim_yolo_qat_ep{epoch:04d}.pth')
