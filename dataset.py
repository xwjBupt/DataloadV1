"""
适用于检测数据集，一张图片对应一个xml（一幅图不超过num个目标）
Version:18-6-12
edit by :XWJ
"""
from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2
import glob   #用于遍历文件夹
import xml.etree.ElementTree as ET   #读取XML文件常用库
# Ignore warnings
import warnings
from PIL import Image

class Myship(Dataset):

    '''
    root_dir：dataset dir
    num: max num of target in one image
    trasform: image transform
    target_transform: bnding box trasnform
    '''
    def __init__(self,  root_dir,num, transform=None,target_transform = None):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.maxnum = num
        self.img_path = os.path.join(self.root_dir,'img')
        self.ann_path = os.path.join(self.root_dir,'Annotation')
        self.im_name = os.listdir(self.img_path)
        self.ann_name = os.listdir(self.ann_path)
    '''
    return length of datset
    '''
    def __len__(self):
        path = os.path.join(self.root_dir,'img')
        ls = os.listdir(path)
        length = len(ls)
        return length
    '''
    return a pair of data
    img and its bnding box
    '''
    def __getitem__(self,idx):

        im_name = os.path.join(self.img_path,self.im_name[idx])
        ann_name = os.path.join(self.ann_path,self.ann_name[idx])
        tree = ET.parse(ann_name)  # 将对应名的xml文件解析成树的样式
        objs = tree.findall('object')  # 找到所有名称为： object的节点
        boxes = np.zeros((self.maxnum, 5), dtype=np.float)
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')  # 找到object节点下的bndbox子节点
            name = obj.find('name').text.lower().strip()
            x1 = float(bbox.find('xmin').text)
            y1 = float(bbox.find('ymin').text)
            x2 = float(bbox.find('xmax').text)
            y2 = float(bbox.find('ymax').text)
            boxes[ix, 0] = x1
            boxes[ix, 1] = y1
            boxes[ix, 2] = x2
            boxes[ix, 3] = y2
            boxes[ix, 4] = name
        target = boxes.tolist()

        with Image.open(im_name) as img:
            img= img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img,torch.Tensor(target)

if __name__ == '__main__':


    data_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    print ('transform done')
    datadir = '/media/xwj/Data/DataSet/seaship'
    transformed_dataset = Myship(root_dir=datadir,num=10,transform=data_transform)
    print ("loading")
    dataloader = DataLoader(transformed_dataset, batch_size=4,
                            shuffle=True, num_workers=1)
    count = 0
    for im,box in dataloader:
        print ('##'*15)
        print (im.size())
        print (box.size())
        print('##' * 15)
        count +=1
        if count ==5:
            break
