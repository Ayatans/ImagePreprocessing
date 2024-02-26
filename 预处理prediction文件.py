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
# resultpath=r'E:\datasets\DIOR-DCNet\results'
# outputpath=r'E:\datasets\DIOR-DCNet/result_1_30'
# f=open(os.path.join(resultpath, 'prediction_results_1_30.txt'),'r')
# lines=f.readlines()
# for i in lines:
#     allele=i.split()
#     if len(allele)==2:
#         continue
#     if len(allele)==5:
#         ids, xmin, ymin, xmax, ymax=allele
#     ids=str(int(ids)+11726).zfill(5)
#     with open(os.path.join(outputpath, ids+'.txt'),'a') as ff:
#         ff.write(xmin+' '+ymin+' '+xmax+' '+ymax+'\n')
    
# NWPU
resultpath=r'E:\datasets\NWPU-VOC\prediction_results_nwpu.txt'
testpath=r'E:\datasets\NWPU-VOC\test.txt'
outputpath=r'E:\datasets\NWPU-VOC/result'

with open(testpath, 'r') as f:
    lines=f.readlines()
idss=[]
for line in lines:
    idss.append(line.strip())

f=open(resultpath,'r')
lines=f.readlines()
for i in lines:
    allele=i.split()
    if len(allele)==2:
        continue
    if len(allele)==5:
        ids, xmin, ymin, xmax, ymax=allele
    ids=idss[int(ids)]
    with open(os.path.join(outputpath, ids+'.txt'),'a') as ff:
        ff.write(xmin+' '+ymin+' '+xmax+' '+ymax+'\n')