# yolov1_maxim
 This repo covers the MAX78000 model training and synthesis pipeline for the YOLO v1 model.

---

### Role of each Python script:

* YOLO_V1_Train_QAT.py: Layer-wise QAT; set args.qat = True to quantize layers 1, 2, ..., 24 are quantized in epoch 100, 200, ..., 2400; set args.fuse = True to fuse BN layers 1, 2, ..., 24 are in epoch 2500, 2600, ..., 4800.

* YOLO_V1_Test.py: Fake INT8 test of the model; change the directory of weight file (*.pth) to test different models.

* YOLO_V1_Test_INT8.py: Real INT8 test of the model; no involved in the current stage.

* YOLO_V1_DataSet_small.py: Preprocess the VOC2007 dataset.

* yolov1_bn_model.py: Define the structure of the deep neural network.

* YOLO_V1_LossFunction.py: Define the loss function.

* weights/YOLO_V1_Z_5_450_Guanchu-BN_bs16_quant1_3000.pth: Model parameter after 3000 epoch training, where args.qat = True and args.fuse = False. 

* Weights/YOLO_V1_Z_5_450_Guanchu-BN_bs16_quant1_4000.pth: Model parameter after 4000 epoch training, where args.qat = True and args.fuse = False. 


---

### How to train the YOLO model?

1. Put this folder ('code') and the 'dataset' folder to 'ai8x-training', where ai8x.py is in the same directory.

2. Run 'python3 YOLO_V1_Train_QAT.py --gpu 0 --qat True --fuse False'.

    * You can change the hyperparameter as you want. But there is no need to do this because the current hyperparameters work for our Layer-wise QAT training.


---

### How to test the trained model (Fake INT8 testing)?

1. Open YOLO_V1_Test.py, revise line 27 into the directory of your trained model.

2. Run YOLO_V1_Test.py. (python3 YOLO_V1_Test.py or using Pycharm)


---

### How to do real INT8 testing?

We intend to focus on the real INT8 testing after the model has passed the Fake INT8 testing. Hence, YOLO_V1_Test_INT8.py, nms.py, and sigmoid.py are useless in the current stage.


---

### How to generate the checkpoint file of our model?

1. Open YOLO_V1_Test.py and uncomment lines 14, 15, and 29-36.

2. Run YOLO_V1_Test.py to generate the checkpoint file in directory ./weights/. Then, you can quantize the checkpoint using ai8x-synthesis.


---

### Trained models

The follow links contains previous trained models and logs.

1. [weight_20210711](https://drive.google.com/drive/folders/1vq-7v-ALpb-Rja-A26G-X3aW25lZYPr6?usp=sharing)

2. [yolo_models_test](https://drive.google.com/drive/folders/1i2Wiom7VP05wWcpyN4yMaJavjDQnS49T?usp=sharing) 

3. [logs](https://drive.google.com/drive/folders/1gHSb_aIbfadDJwKjGwqWi9tjVzLNtzx0?usp=sharing)


---
### VOC 2007 Dataset
* You can download [train/validation](http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar) 
  and [test](http://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar) by the above hyperlinks.
* Or if you have an Texas A&M account, you might access the VOC2007 on datalab6.engr.tamu.edu
   * Train: /data/yiwei/VOC2007/Train
   * Test: /data/yiwei/VOC2007/Test

---

### Environment setup
```bash
$ git clone git@github.com:YIWEI-CHEN/yolov1_maxim.git
$ cd yolov1_maxim
# note that ai8x-training and ai8x-synthesis should in the project root (e.g., yolov1_maxim)
$ git clone --recursive https://github.com/MaximIntegratedAI/ai8x-training.git
$ git clone --recursive https://github.com/MaximIntegratedAI/ai8x-synthesis.git

# in your virtual environment
# the following torch installation for NVIDIA RTX A5000
$ pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.htm
$ pip tensorboard matplotlib numpy

$ cd yolov1_maxim/ai8x-training/distiller
# remove lines of numpy, torch, torchvision in requirements.txt
$ pip install -e .
```

---
### Reference
* The repo starts from https://www.dropbox.com/s/pssda2gxrqa51v9/yolov1_maxim.zip?dl=0
* The YOLOv1 train and test framework are from https://github.com/ProgrammerZhujinming/YOLOv1