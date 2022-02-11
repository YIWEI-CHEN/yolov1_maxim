import os
import sys

# go to the directory of ai8x
PROJECT_ROOT = os.path.join(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "ai8x-training"))

import cv2
import numpy as np
import torch
import importlib

import ai8x
from YOLO_V1_DataSet import YoloV1DataSet
from map import NMS, gt_std, IOU


# loading dataset
dataset_root = "/data/yiwei/VOC2007"
dataSet = YoloV1DataSet(imgs_dir=f"{dataset_root}/Test/JPEGImages",
                        annotations_dir=f"{dataset_root}/Test/Annotations",
                        ClassesFile=f"{dataset_root}/VOC_remain_class.data",
                        train_root=f"{dataset_root}/Test/ImageSets/Main/",
                        )

# loading model
ai8x.set_device(85, simulate=False, round_avg=False, verbose=True)
mod = importlib.import_module("yolov1_bn_model_noaffine")
Yolo = mod.Yolov1_net(num_classes=dataSet.Classes, bias=True)
qat_policy = {'start_epoch': 150,
              'weight_bits': 8,
              'bias_bits': 8,
              'shift_quantile': 0.99}
ai8x.fuse_bn_layers(Yolo)
ai8x.initiate_qat(Yolo, qat_policy)

checkpoint_fname = './log/QAT-20220210-174132/scaled224_noaffine_shift0.99_maxim_yolo_qat_ep200.pth'
Yolo.load_state_dict(torch.load(checkpoint_fname, map_location=lambda storage, loc: storage)) # batch_size 16

# ck_fname = checkpoint_fname.split("/")[-1]
# checkpoint_dir = checkpoint_fname.replace(ck_fname, "")
# print(f"checkpoint_dir {checkpoint_dir}")
# import distiller.apputils as apputils
# apputils.save_checkpoint(checkpoint_dir, "ai85net5", Yolo,
#                             optimizer=None, scheduler=None, extras=None,
#                             is_best=False, name="Yolov1", dir=checkpoint_dir,
#                          )

if __name__ == '__main__':
    dir_name = os.path.join(os.path.dirname(checkpoint_fname), "bad_predictions")
    os.makedirs(dir_name, exist_ok=True)

    # for img_idx in [0, 14, 21, 54, 67, 90, 107, 118, 119, 124, 178]:
    for img_idx in [7, 9, 28, 48, 58, 61, 94, 108, 109, 111, 125, 170]:
        img_data = dataSet.read_img(item=img_idx)
        img_name = os.path.basename(dataSet.img_path[img_idx])
        train_data, ground_truth = dataSet[img_idx]

        ground_truth = torch.unsqueeze(ground_truth, 0)
        ground_truth = ground_truth.squeeze(dim=3)
        ground_truth = gt_std(ground_truth)[0]

        # draw ground truth
        for box in ground_truth:
            img_data = cv2.rectangle(img_data, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            img_data = cv2.putText(img_data, "ground truth", (box[0], box[3] + 15),
                                   cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

        # predict
        train_data = torch.unsqueeze(train_data, 0)
        print(f'Predicting {img_name}')
        pred_results, _ = Yolo(train_data)
        NMS_boxes = NMS(bounding_boxes=pred_results)[0]
        for i, box in enumerate(NMS_boxes):
            for j, gt_box in enumerate(ground_truth):
                iou = IOU(gt_box, box)
                print(f'IOU between {i} pred_box with {j} ground_truth_box: {iou}')
            has_obj_prob = box[4]
            class_index = box[-1]
            # convert box from float to int
            box = np.array(box[0:4]).astype(np.int)
            # draw predicted box
            img_data = cv2.rectangle(img_data, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
            img_data = cv2.putText(img_data, "prob:{:.2f}".format(has_obj_prob), (box[0], box[1] - 4),
                                          cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

        # save result
        pred_img = os.path.join(dir_name, f"{img_name.split('.')[0]}_pred.jpg")
        cv2.imwrite(pred_img, img_data)
