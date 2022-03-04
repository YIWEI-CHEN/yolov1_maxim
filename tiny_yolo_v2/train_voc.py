import logging
import os
import random
import argparse
import sys

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(PROJECT_ROOT, "ai8x-training"))

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from loss import YoloLoss
from tiny_yolov2 import TinyYoloV2
from utils import custom_collate_fn, create_exp_dir, get_logger, get_time_str
from voc_dataset import VOCDataset

import ai8x


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=400, help='Maximum training epoch.')
    parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=16, help='Minibatch size.')
    parser.add_argument('--gpu', type=int, default=1, help='Use which gpu to train the model.')
    parser.add_argument('--exp', type=str, default="tiny-yolo-v2", help='Experiment name.')
    parser.add_argument('--seed', type=int, default=7, help='Random seed.')
    args = parser.parse_args()
    return args


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def log_init():
    import glob

    fdir0 = os.path.join("logs", args.exp + '-' + get_time_str())
    create_exp_dir(fdir0, scripts_to_save=glob.glob('*.py'))
    args.output_dir = fdir0

    logger = get_logger(logdir=fdir0, tag=args.exp, log_level=logging.INFO)
    logger.info("args = %s", args)

    return logger


# Initialize the dataset and dataloader
def dataset_init():
    dataset = VOCDataset(root_path="/data/yiwei/VOCdevkit", image_size=224)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                             collate_fn=custom_collate_fn)
    return dataset, data_loader


# Train
def train(logger):
    dataset, data_loader = dataset_init()

    # Set ai8x device
    ai8x.set_device(device=85, simulate=False, round_avg=False)

    model = TinyYoloV2(num_classes=dataset.num_classes)
    model = model.cuda()
    logger.info("NUMBER OF PARAMETERS {}".format(sum(p.numel() for p in model.parameters())))

    # Initialize the loss function
    criterion = YoloLoss(dataset.num_classes, model.anchors)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[50, 100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,20000,30000,40000],gamma=0.8)

    # Initialize the quantization policy
    num_epochs = args.num_epochs

    # Main training
    num_iter_per_epoch = len(data_loader)
    for epoch in range(0, num_epochs):
        for batch_index, batch_train in enumerate(data_loader):
            train_data = batch_train[0].float().cuda()
            train_data.requires_grad = True
            label_data = batch_train[1]
            optimizer.zero_grad()
            logits = model(train_data)
            loss, loss_coord, loss_conf, loss_cls = criterion(logits, label_data)
            loss.backward()
            optimizer.step()
            logger.info("Epoch: {}/{}, Iteration: {}/{}, Lr: {}, Loss:{:.2f} (Coord:{:.2f} Conf:{:.2f} Cls:{:.2f})".format(
                epoch + 1,
                num_epochs,
                batch_index + 1,
                num_iter_per_epoch,
                optimizer.param_groups[0]['lr'],
                loss,
                loss_coord,
                loss_conf,
                loss_cls))
        if epoch % 50 == 0:
            torch.save(model.state_dict(), os.path.join(args.output_dir, "tiny_yolo_v2_ep{}.pth".format(epoch)))
        scheduler.step()


def main():
    # Set GPU
    setup_seed(args.seed)
    logger = log_init()

    # args.device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    logger.info('Running on device: {}'.format(args.gpu))
    train(logger)


if __name__ == "__main__":
    args = get_args()
    main()
