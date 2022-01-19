#---------------Hyperparameter-----------------
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='.')
parser.add_argument('--qat', type=bool, default=True, help='.')
parser.add_argument('--fuse', type=bool, default=False, help='.')
args = parser.parse_args()

#---------------step1:Dataset-------------------
import torch
from YOLO_V1_DataSet_small import YoloV1DataSet
dataSet = YoloV1DataSet(imgs_dir="../dataset/train/VOC2007/Train/JPEGImages",
                        annotations_dir="../dataset/train/VOC2007/Train/Annotations",
                        ClassesFile="../dataset/train/VOC2007/Train/VOC_remain_class.data")
print("Images in Dataset:",len(dataSet),len(dataSet.ground_truth))
from torch.utils.data import DataLoader
dataLoader = DataLoader(dataSet,batch_size=16,shuffle=True,num_workers=4)

#---------------step2:Model-------------------
from yolov1_bn_model import Yolov1_net 
import sys
sys.path.append("../")
import ai8x
ai8x.set_device(85, simulate=False, round_avg=False, verbose=True)
Yolo = Yolov1_net(num_classes=dataSet.Classes, bias=True).cuda(device=args.gpu)
print(Yolo)
print("NUMBER OF PARAMETERS",  sum(p.numel() for p in Yolo.parameters()))
#Yolo.initialize_weights()
#Yolo.load_state_dict(torch.load('./YOLO_V1_40.pth'))
#---------------step3:LossFunction-------------------
from YOLO_V1_LossFunction import  Yolov1_Loss
loss_function = Yolov1_Loss().cuda(device=args.gpu)

#---------------step4:Optimizer-------------------
import torch.optim as optim
optimizer = optim.SGD(Yolo.parameters(),lr=3e-3,momentum=0.9,weight_decay=0.0005)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,20000,30000,40000],gamma=0.9)

#---------------Quantizing Policy--------------------
import numpy as np
qat_policy = dict()
qat_policy['weight_bits'] = 8
qat_policy['bias_bits'] = 8
fuse_epoch = np.arange(100, (len(list(Yolo.children()))+1)*100, 100)
qat_epoch = np.arange(100, (len(list(Yolo.children()))+1)*100, 100) + np.max(fuse_epoch)
qat_layer = np.arange(0, len(list(Yolo.children()))+1, 1)
qat_epoch_layer = dict(zip(qat_epoch, qat_layer))
fuse_epoch_layer = dict(zip(fuse_epoch, qat_layer))
print(qat_epoch)
print(qat_layer)
print(qat_epoch_layer)
print(fuse_epoch_layer)

#--------------step5:Tensorboard Feature Map------------
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import torch.nn as nn
writer = SummaryWriter('log/wQAT')

def feature_map_visualize(img_data, writer):
    img_data = img_data.unsqueeze(0)
    img_grid = vutils.make_grid(img_data, normalize=True, scale_each=True)
    for i,m in enumerate(Yolo.modules()):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or \
                isinstance(m, nn.ReLU) or isinstance(m, nn.MaxPool2d) or isinstance(m, nn.AdaptiveAvgPool2d):
            img_data = m(img_data)
            img_data = img_data.permute(1, 0, 2, 3)
            img_grid = vutils.make_grid(img_data, normalize=True, scale_each=True)
            img_data = img_data.permute(1, 0, 2, 3)
            writer.add_image('feature_map', img_grid)

#---------------step6:Train-------------------
epoch = 0
while epoch <= 2000*dataSet.Classes:

    loss_sum = 0
    loss_coord = 0
    loss_confidence = 0
    loss_classes = 0
    epoch_iou = 0
    epoch_object_num = 0
    scheduler.step()


    if args.qat and epoch in qat_epoch: # The epoch to quantize the each layer
        Yolo.quantize_layer(layer_index=qat_epoch_layer[epoch], qat_policy=qat_policy)
    if args.fuse and epoch in fuse_epoch:
        Yolo.fuse_bn_layer(layer_index=fuse_epoch_layer[epoch])

    for batch_index, batch_train in enumerate(dataLoader):

        optimizer.zero_grad()
        train_data = batch_train[0].float().cuda(device=args.gpu)
        train_data.requires_grad = True
    
        label_data = batch_train[1].float().cuda(device=args.gpu)
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
    epoch = epoch + 1
    # if epoch % 500 == 0:
    #     torch.save(Yolo.state_dict(), './weights_lwQAT_20210630/YOLO_V1_Z_5_450_Guanchu-BN_bs16_quant1_' + str(epoch) + '.pth')
        #writer.close()
        #writer = SummaryWriter(logdir='log/wQAT',filename_suffix=str(epoch) + '~' + str(epoch + 500))
        
    avg_loss= loss_sum/batch_index
    print("epoch : {} ; loss : {} ; avg_loss: {}".format(epoch,{loss_sum},{avg_loss}))
    """for name, layer in Yolo.named_parameters():
        writer.add_histogram(name + '_grad', layer.grad.cpu().data.numpy(), epoch)
        writer.add_histogram(name + '_data', layer.cpu().data.numpy(), epoch)
    """
    #feature_map_visualize(batch_train[0], writer)
    writer.add_scalar('Train/Loss_sum', loss_sum, epoch)
    #writer.add_scalar('Train/Loss_coord', loss_coord, epoch)
    #writer.add_scalar('Train/Loss_confidenct', loss_confidence, epoch)
    #writer.add_scalar('Train/Loss_classes', loss_classes, epoch)
    #writer.add_scalar('Train/Epoch_iou', epoch_iou / epoch_object_num, epoch)
writer.close()
