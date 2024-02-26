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

# DIOR
# resultpath=r'E:\datasets\DIOR-DCNet\result_1_30'
# imagepath=r'E:\datasets\DIOR-DCNet\JPEGImages'
# gtpath=r'E:\datasets\DIOR-DCNet\Annotations'
# outputpath=r'E:\datasets\DIOR-DCNet\resultimg'
# filelist=os.listdir(resultpath)
# for i in filelist:
#     abspath=os.path.join(resultpath,i)
#     imgid=i.split('.')[0]
#     img=Image.open(os.path.join(imagepath, imgid+'.jpg'))
#     a = ImageDraw.ImageDraw(img)
#     with open(abspath, 'r') as f:
#         lines=f.readlines()
#         for line in lines:
#             xmin,ymin,xmax,ymax=line.split()
#             a.line([(float(xmin),float(ymin)), (float(xmin),float(ymax))],fill='blue',width=1)
#             a.line([(float(xmin),float(ymax)), (float(xmax),float(ymax))],fill='blue',width=1)
#             a.line([(float(xmax),float(ymax)), (float(xmax),float(ymin))],fill='blue',width=1)
#             a.line([(float(xmax),float(ymin)), (float(xmin),float(ymin))],fill='blue',width=1)
#     gt=os.path.join(gtpath,imgid+'.xml')
#     target=ET.parse(gt).getroot()
#     for obj in target.iter('object'):
#         xmlbox = obj.find('bndbox')
#         xmin,xmax,ymin,ymax = float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text)
#         a.line([(float(xmin),float(ymin)), (float(xmin),float(ymax))],fill='red',width=2)
#         a.line([(float(xmin),float(ymax)), (float(xmax),float(ymax))],fill='red',width=2)
#         a.line([(float(xmax),float(ymax)), (float(xmax),float(ymin))],fill='red',width=2)
#         a.line([(float(xmax),float(ymin)), (float(xmin),float(ymin))],fill='red',width=2)
        
#     img.save(outputpath+'/'+imgid+'.jpg')
#     print(imgid+' done')
    

# NWPU
resultpath=r'E:\datasets\NWPU-VOC\result'
imagepath=r'E:\datasets\NWPU-VOC\JPEGImages'
gtpath=r'E:\datasets\NWPU-VOC\Annotations'
outputpath=r'E:\datasets\NWPU-VOC\resultimg'


filelist=os.listdir(resultpath)
for i in filelist:
    abspath=os.path.join(resultpath,i)
    imgid=i.split('.')[0]
    img=Image.open(os.path.join(imagepath, imgid+'.jpg'))
    #img=img.resize((1800,1200), Image.ANTIALIAS)
    a = ImageDraw.ImageDraw(img)
    with open(abspath, 'r') as f:
        lines=f.readlines()
        for line in lines:
            xmin,ymin,xmax,ymax=line.split()
            a.line([(float(xmin),float(ymin)), (float(xmin),float(ymax))],fill='blue',width=1)
            a.line([(float(xmin),float(ymax)), (float(xmax),float(ymax))],fill='blue',width=1)
            a.line([(float(xmax),float(ymax)), (float(xmax),float(ymin))],fill='blue',width=1)
            a.line([(float(xmax),float(ymin)), (float(xmin),float(ymin))],fill='blue',width=1)
    gt=os.path.join(gtpath,imgid+'.xml')
    target=ET.parse(gt).getroot()
    for obj in target.iter('object'):
        xmlbox = obj.find('bndbox')
        xmin,xmax,ymin,ymax = float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text)
        a.line([(float(xmin),float(ymin)), (float(xmin),float(ymax))],fill='red',width=2)
        a.line([(float(xmin),float(ymax)), (float(xmax),float(ymax))],fill='red',width=2)
        a.line([(float(xmax),float(ymax)), (float(xmax),float(ymin))],fill='red',width=2)
        a.line([(float(xmax),float(ymin)), (float(xmin),float(ymin))],fill='red',width=2)
        
    img.save(outputpath+'/'+imgid+'.jpg')
    print(imgid+' done')

