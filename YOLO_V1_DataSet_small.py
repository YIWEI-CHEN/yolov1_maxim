from torch.utils.data import Dataset
import os
import cv2
import xml.etree.ElementTree as ET
import torch
import torchvision.transforms as transforms

class YoloV1DataSet(Dataset):

    def __init__(self, imgs_dir="./VOC2007/Train/JPEGImages",
                 annotations_dir="./VOC2007/Train/Annotations", img_size=224, S=7, B=2,
                 ClassesFile="./VOC2007/Train/VOC_remain_class.data",
                 data_path='..'): # 图片路径、注解文件路径、图片尺寸、每个grid cell预测的box数量、类别文件
        self.five = []
        self.annot = []
        self.paths_five = ["person_train.txt",
                           "car_train.txt",
                           "bicycle_train.txt",
                           "chair_train.txt",
                           "sofa_train.txt"]
        self.paths_five = [os.path.join(data_path, self.paths_five[i]) for i in range(len(self.paths_five))]

        uniq = 0
        for i in range(len(self.paths_five)):
            count = 0
            
            with open(self.paths_five[i], 'r') as f:
                #print(self.paths_five[i])
                for line in f:
                    if line.strip().split(" ")[-1]=="1":
                        #print(line)
                        #line = line.replace('\n','')
                        line = line.strip().split(" ")[0]
                        #print(line,"here")
                    
                        #self.five.append(line+".jpg")
                        if line+".jpg" not in self.five:
                            uniq+=1
                            self.five.append(line+".jpg")
                            self.annot.append(line+".xml")
                            count+=1
                            #print(line+".jpg")
                            if count==100:
                                #print(i)
                                break
        
        print("#Unique images",uniq)
        
        #img_names = os.listdir(imgs_dir)
        img_names = self.five
        img_names.sort()
        #print(len(img_names))
        self.transfrom = transforms.Compose([
            #transforms.ToPILImage(),
            #transforms.Resize((224,224)),
            transforms.ToTensor(), # height * width * channel -> channel * height * width
            transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))
        ])
        self.img_path = []
        for img_name in img_names:
            #print(len(img_names),"KKK")
            #print(img_name[:-4])
            #if img_name in self.five:
            self.img_path.append(os.path.join(imgs_dir,img_name))
        #annotation_names = os.listdir(annotations_dir)
        annotation_names = self.annot
        annotation_names.sort() #图片和文件排序后可以按照相同索引对应
        self.annotation_path = []
        for annotation_name in annotation_names:
            #print(annotation_name,"ann")
            #if annotation_name[:-4] in self.five[:-4]:
            self.annotation_path.append(os.path.join(annotations_dir,annotation_name))
        print(len(self.img_path), len(self.annotation_path))
        self.img_size = img_size
        self.S = S
        self.B = B
        self.grid_cell_size = self.img_size / self.S
        self.ClassNameToInt = {}
        self.IntToClassName = {}
        classIndex = 0
        with open(ClassesFile, 'r') as f:
            for line in f:
                line = line.replace('\n','')
                self.ClassNameToInt[line] = classIndex #根据类别名制作索引
                self.IntToClassName[classIndex] = line
                classIndex = classIndex + 1
        print(self.ClassNameToInt)
        self.Classes = classIndex # 一共的类别个数
        self.getGroundTruth()

    # PyTorch 无法将长短不一的list合并为一个Tensor
    def getGroundTruth(self):
        self.ground_truth = [[[list() for i in range(self.S)] for j in range(self.S)] for k in
                             range(len(self.img_path))]  # 根据标注文件生成ground_truth
        ground_truth_index = 0
        for annotation_file in self.annotation_path:
            ground_truth = [[list() for i in range(self.S)] for j in range(self.S)]
            # 解析xml文件--标注文件
            tree = ET.parse(annotation_file)
            annotation_xml = tree.getroot()
            # 计算 目标尺寸 -> 原图尺寸 self.img_size * self.img_size , x的变化比例
            width = (int)(annotation_xml.find("size").find("width").text)
            scaleX = self.img_size / width
            # 计算 目标尺寸 -> 原图尺寸 self.img_size * self.img_size , y的变化比例
            height = (int)(annotation_xml.find("size").find("height").text)
            scaleY = self.img_size / height
            # 因为两次除法的误差可能比较大 这边采用除一次乘一次的方式
            # 一个注解文件可能有多个object标签，一个object标签内部包含一个bnd标签
            objects_xml = annotation_xml.findall("object")
            for object_xml in objects_xml:
                # 获取目标的名字
                class_name = object_xml.find("name").text
                if class_name not in self.ClassNameToInt: # 不属于我们规定的类
                    continue
                bnd_xml = object_xml.find("bndbox")
                # 目标尺度放缩
                xmin = (int)((int)(bnd_xml.find("xmin").text) * scaleX)
                ymin = (int)((int)(bnd_xml.find("ymin").text) * scaleY)
                xmax = (int)((int)(bnd_xml.find("xmax").text) * scaleX)
                ymax = (int)((int)(bnd_xml.find("ymax").text) * scaleY)
                # 目标中心点
                centerX = (xmin + xmax) / 2
                centerY = (ymin + ymax) / 2
                # 当前物体的中心点落于 第indexI行 第indexJ列的 grid cell内
                indexI = (int)(centerY / self.grid_cell_size)
                indexJ = (int)(centerX / self.grid_cell_size)
                # 真实物体的list
                ClassIndex = self.ClassNameToInt[class_name]
                ClassList = [0 for i in range(self.Classes)]
                ClassList[ClassIndex] = 1
                ground_box = list([centerX / self.grid_cell_size - indexJ,centerY / self.grid_cell_size - indexI,(xmax-xmin)/self.img_size,(ymax-ymin)/self.img_size,1,xmin,ymin,xmax,ymax,(xmax-xmin)*(ymax-ymin)])
                #增加上类别
                ground_box.extend(ClassList)
                ground_truth[indexI][indexJ].append(ground_box)

            #同一个grid cell内的多个groudn_truth，选取面积最大的两个
            for i in range(self.S):
                for j in range(self.S):
                    if len(ground_truth[i][j]) == 0:
                        #print(self.Classes,"class")
                        self.ground_truth[ground_truth_index][i][j].append([0 for i in range(10 + self.Classes)])
                    else:
                        ground_truth[i][j].sort(key = lambda box: box[9], reverse=True)
                        self.ground_truth[ground_truth_index][i][j].append(ground_truth[i][j][0])

            ground_truth_index = ground_truth_index + 1
        self.ground_truth = torch.Tensor(self.ground_truth).float()

        # ground_truth = [normalized_center_x,  0
        #                 normalized_center_y,  1
        #                 normalized_width,     2
        #                 normalized_height,    3
        #                 confidence,           4
        #                 center_x,             5
        #                 center_y,             6
        #                 width,                7
        #                 height,               8
        #                 area,                 9
        #                 class_onehot,         10:30
        #     ]

    def __getitem__(self, item):
        # height * width * channel
        #print(len(self.img_path),"KKKKKKK")
        img_data = cv2.imread(self.img_path[item])
        img_data = cv2.resize(img_data, (224, 224), interpolation=cv2.INTER_AREA)
        
        img_data = self.transfrom(img_data)
        return img_data,self.ground_truth[item]

    def __len__(self):
        return len(self.img_path)
