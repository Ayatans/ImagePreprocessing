# 由于报错找不到maskrcnn benchmark，必须要加个环境变量。
import sys

sys.path.append('/remote-home/yczhang/code/DCNet/')
from maskrcnn_benchmark.data.datasets.dior import DIORDataset
from collections import OrderedDict

# split = int(sys.argv[1])
for split in range(1, 6):
    dataset = DIORDataset('/remote-home/yczhang/code/dataset/DIOR-DCNet', 'trainval')
    keeps = []
    all_cls = DIORDataset.CLASSES
    novel_cls = [DIORDataset.CLASSES_SPLIT1_NOVEL,
                 DIORDataset.CLASSES_SPLIT2_NOVEL,
                 DIORDataset.CLASSES_SPLIT3_NOVEL,
                 DIORDataset.CLASSES_SPLIT4_NOVEL,
                 DIORDataset.CLASSES_SPLIT5_NOVEL][split - 1]
    novel_index = [all_cls.index(c) for c in novel_cls]
    print(novel_index)
    # 图里完全没有novel的才为训练集
    for i in range(len(dataset.ids)):
        anno = dataset.get_groundtruth(i)
        label = anno.get_field('labels')
        # 查询图片中是否有novel类目标
        count = [(label == j).sum().item() for j in novel_index]
        # 没有才保留该图片
        if sum(count) == 0:
            keeps.append(i)

    box_count = [0] * 21
    for i in keeps:
        anno = dataset.get_groundtruth(i)
        label = anno.get_field('labels')
        for j in label:
            box_count[j] += 1   # 对应类别的数据计数器+1
    print("trainval:%d" % (len(keeps)))
    print(dict(zip(all_cls, box_count)))
    with open(
            '/remote-home/yczhang/code/dataset/DIOR-DCNet/ImageSets/Main/trainval_split%d_base.txt' % (split),
            'w+') as f:
        for i in keeps:
            f.write(dataset.ids[i] + '\n')


    # 下面是对测试数据的构建

    dataset = DIORDataset('/remote-home/yczhang/code/dataset/DIOR-DCNet', 'test', use_difficult=True)
    keeps = []
    all_cls = DIORDataset.CLASSES
    novel_cls = [DIORDataset.CLASSES_SPLIT1_NOVEL,
                 DIORDataset.CLASSES_SPLIT2_NOVEL,
                 DIORDataset.CLASSES_SPLIT3_NOVEL,
                 DIORDataset.CLASSES_SPLIT4_NOVEL,
                 DIORDataset.CLASSES_SPLIT5_NOVEL][split - 1]
    novel_index = [all_cls.index(c) for c in novel_cls]
    print(novel_index)
    for i in range(len(dataset.ids)):
        anno = dataset.get_groundtruth(i)
        label = anno.get_field('labels')
        count = [(label == j).sum().item() for j in novel_index]
        if sum(count) == 0:
            keeps.append(i)

    box_count = [0] * 21
    for i in keeps:
        anno = dataset.get_groundtruth(i)
        label = anno.get_field('labels')
        for j in label:
            box_count[j] += 1
    print("test:%d" % (len(keeps)))
    print(dict(zip(all_cls, box_count)))
    with open('/remote-home/yczhang/code/dataset/DIOR-DCNet/ImageSets/Main/test_split%d_base.txt' % (split),
              'w+') as f:
        for i in keeps:
            f.write(dataset.ids[i] + '\n')

