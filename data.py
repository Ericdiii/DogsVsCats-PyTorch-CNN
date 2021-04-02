import os
import torch.utils.data as data
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms

# Default size of input image
IMAGE_SIZE = 200

# Define a conversion relationship to change image data to Tensor format
dataTransform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),                          # Resize image and zoom to a suitable size
    transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),        # Crop a suitable size from the image center
    transforms.ToTensor()   # Important point: Convert to Tensor format, normalized to [0.0, 1.0], transpose from (H×W×C) to (C×H×W)
]) 


class DogsVSCatsDataset(data.Dataset):      # Create a new dataset class, inherit the data.Dataset parent class in PyTorch
    def __init__(self, mode, dir):          # Default constructor, incoming dataset category (training or testing), and dataset path        
        self.mode = mode
        self.list_img = []                  # Create a new image list, used to store the image path
        self.list_label = []                # Create a new label list to store the label of the cat (0) or dog (1) corresponding to the picture
        self.data_size = 0                  # Record dataset size
        self.transform = dataTransform      # Conversion relationship

        # In the training set mode need to be extract the image path and label
        if self.mode == 'train':
            dir = dir + '/train/'           # The training set path is in "dir"/train/
            for file in os.listdir(dir):    # Traverse the dir folder
                self.list_img.append(dir + file)        # Add the image path and file name to the image list
                self.data_size += 1                     # Data set size increased by 1
                name = file.split(sep='.')              # Split the file name: "cat.0.jpg" will be split into "cat", ".", "jpg"
                
                # label uses One-Hot encoding, "1,0" means cat, "0,1" means dog, in any case only one position is "1"; 
                # In the case of using CrossEntropyLoss() to calculate Loss, label only needs to enter the index of "1", cat is 0, dog is 1
                if name[0] == 'cat':
                    self.list_label.append(0)         # 图片为猫，label为0
                else:
                    self.list_label.append(1)         # 图片为狗，label为1，注意：list_img和list_label中的内容是一一配对的
                    
        # In the test set mode only need to extract the image path            
        elif self.mode == 'test':
            dir = dir + '/test/'            # 测试集路径为"dir"/test/
            for file in os.listdir(dir):
                self.list_img.append(dir + file)    # 添加图片路径至image list
                self.data_size += 1
                self.list_label.append(2)       # 添加2作为label，实际未用到，也无意义
        else:
            print('Undefined Dataset!')

    def __getitem__(self, item):            # 重载data.Dataset父类方法，获取数据集中数据内容
        if self.mode == 'train':                                        # 训练集模式下需要读取数据集的image和label
            img = Image.open(self.list_img[item])                       # 打开图片
            label = self.list_label[item]                               # 获取image对应的label
            return self.transform(img), torch.LongTensor([label])       # 将image和label转换成PyTorch形式并返回
        elif self.mode == 'test':                                       # 测试集只需读取image
            img = Image.open(self.list_img[item])
            return self.transform(img)                                  # 只返回image
        else:
            print('None')

    def __len__(self):
        return self.data_size               # 返回数据集大小

