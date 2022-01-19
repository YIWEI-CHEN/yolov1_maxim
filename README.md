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
