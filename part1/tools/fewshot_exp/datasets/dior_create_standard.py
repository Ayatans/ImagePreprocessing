import os
from maskrcnn_benchmark.data.datasets.dior import DIORDataset
import sys
seed=int(sys.argv[1])

cls = DIORDataset.CLASSES[1:]
#yolodir = '../Fewshot_Detection'
#for shot in [30, 20, 10, 5, 3, 2, 1]:
# Split 5
for shot in [20, 10, 5, 3]:
    ids = []
    for c in cls:
        with open('/remote-home/yczhang/code/dataset/DIOR-DCNet/diorlist%d/box_%dshot_%s_train.txt'%(seed,shot, c)) as f:
            content = f.readlines()
        content = [i.strip().split('/')[-1][:-4] for i in content]  # 文件名
        ids += content
    ids = list(set(ids))
    with open('/remote-home/yczhang/code/dataset/DIOR-DCNet/ImageSets/Main/trainval_%dshot_novel_standard_seed%d.txt'%(shot,seed), 'w+') as f:
        for i in ids:
            f.write(i + '\n')


