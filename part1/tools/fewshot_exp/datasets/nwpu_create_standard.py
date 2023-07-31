import os
from maskrcnn_benchmark.data.datasets.nwpu import NWPUDataset
import sys
seed=int(sys.argv[1])

cls = NWPUDataset.CLASSES[1:]
#yolodir = '../Fewshot_Detection'
for shot in [10, 5, 3]:
    ids = []
    for c in cls:
        with open('/root/code/zyc/dataset/NWPU-VOC/nwpulist%d/box_%dshot_%s_train.txt'%(seed,shot, c)) as f:
            content = f.readlines()
        content = [i.strip().split('/')[-1][:-4] for i in content]  # 文件名
        ids += content
    ids = list(set(ids))
    with open('/root/code/zyc/dataset/NWPU-VOC/ImageSets/Main/trainval_%dshot_novel_standard_seed%d.txt'%(shot,seed), 'w+') as f:
        for i in ids:
            f.write(i + '\n')


