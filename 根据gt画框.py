# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image, ImageDraw
import cv2

# plt.rcParams['font.sans-serif']=['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

sets=[('train'), ('val'), ('test'), ('trainval')]

classes=[ 'airplane', 'baseballdiamond', "basketballcourt", "bridge", "groundtrackfield",
                "harbor", "ship", "storagetank", 'tenniscourt', "vehicle"]




# # 对原始标签绘图，从中抽吧
annopath=r'E:\datasets\DIOR-DCNet\Annotations'
imagepath=r'E:\datasets\DIOR-DCNet\JPEGImages'
outputpath=r'E:\datasets\DIOR-DCNet\gtimg'
allfile=os.listdir(annopath)
for i in allfile:
    imgid=i.split('/')[-1].split('.')[0]
    img=Image.open(imagepath+'/'+imgid+'.jpg')
    a = ImageDraw.ImageDraw(img)
    infile=open(annopath+'/'+i)
    tree=ET.parse(infile)
    root=tree.getroot()
    for obj in root.iter('object'):
        # cls = obj.find('name').text
        # if cls != class_name == 1:
        #     continue
        # cls_id = 0
        xmlbox = obj.find('bndbox')
        cls=obj.find('name').text
        xmin,xmax,ymin,ymax = float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text)
        a.line([(float(xmin),float(ymin)), (float(xmin),float(ymax))],fill='red',width=2)
        a.line([(float(xmin),float(ymax)), (float(xmax),float(ymax))],fill='red',width=2)
        a.line([(float(xmax),float(ymax)), (float(xmax),float(ymin))],fill='red',width=2)
        a.line([(float(xmax),float(ymin)), (float(xmin),float(ymin))],fill='red',width=2)
        a.text((float(xmin)-10, float(ymin)-10),cls,fill='white')
    img.save(outputpath+'/'+imgid+'.jpg')
    print('done: ', imgid)
    
    
