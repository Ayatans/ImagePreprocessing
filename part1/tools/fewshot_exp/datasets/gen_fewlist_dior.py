import argparse
import random
import os
import numpy as np
from os import path
import sys

seed = sys.argv[1]

classes=[ 'airplane', 'airport','baseballfield', 'basketballcourt', 'bridge', 'chimney',
            'dam', 'Expressway-Service-area', 'Expressway-toll-station',
          'harbor', 'golffield',  'groundtrackfield', 'overpass',
          'ship', 'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill']

few_nums = [1, 2, 3, 5, 10, 20, 30]
DROOT = '/remote-home/yczhang/code/dataset/DIOR-DCNet'
root =  DROOT + '/diorlist' + str(seed) + '/'
rootfile =  DROOT + '/dior_train.txt'   # 11725个


def get_bbox_fewlist(rootfile, shot):
    # harbor+ESarea没有仅含1个的图！报错
    with open(rootfile) as f:
        names = f.readlines()   # 所有训练文件
    random.seed(6)
    cls_lists = [[] for _ in range(len(classes))]   # 20 lists
    cls_counts = [0] * len(classes) # 每类计数器
    while min(cls_counts) < shot:   # 仍有某类数量<shot，就抽数据
        imgpath = random.sample(names, 1)[0]    # 会持续抽将names抽光，最后这句报错。
        labpath = imgpath.strip().replace('images', 'labels') \
                                 .replace('JPEGImages', 'labels') \
                                 .replace('.jpg', '.txt').replace('.png','.txt')    # labpath在labels文件夹里
        # To avoid duplication
        names.remove(imgpath)   # 抽过的去除，防止重复抽取
        if not os.path.getsize(labpath):
            continue
        # Load converted annotations
        bs = np.loadtxt(labpath)
        bs = np.reshape(bs, (-1, 5))

        # if bs.shape[0] > 3: # 这里将大于3个目标的图片全排除了！！why？先注释了。
        #     continue

        # Check total number of bbox per class so far
        overflow = False
        bcls = bs[:,0].astype(np.int).tolist()  # 一张图里所有类别，每个类只取一个目标了
        for ci in set(bcls):    # 遍历各项，每个类只取一个目标了，这里考虑去掉set！
            if cls_counts[ci] + bcls.count(ci) > shot:  # 如果算上这些数据就会超过shot，就不继续了
                overflow = True
                #print(len(names),'break')
                # break
                continue
            # Add current imagepath to the file lists
            cls_counts[ci] += bcls.count(ci)
            cls_lists[ci].append(imgpath)   # 将图片加入对应类别的list中，说明该类别要用到该图


        # if overflow:
        #     continue
        # print(cls_counts)
        # for ci in set(bcls):
        #     cls_counts[ci] += bcls.count(ci)
        #     cls_lists[ci].append(imgpath)  # 将图片加入对应类别的list中，说明该类别要用到该图
        #print(min(cls_counts))
    return cls_lists


def gen_bbox_fewlist():
    print('-----------------------------------------------------------')
    print('----------- Generating fewlist  (bboxes) ------------------')
    for n in few_nums:
        print('===> On {} shot ...'.format(n))
        filelists = get_bbox_fewlist(rootfile, n)
        for i, clsname in enumerate(classes):
            print('   | Processing class: {}'.format(clsname))
            with open(path.join(root, 'box_{}shot_{}_train.txt'.format(n, clsname)), 'w') as f:
                for name in filelists[i]:
                    f.write(name)


def main():
    gen_bbox_fewlist()



if __name__ == '__main__':
    main()
