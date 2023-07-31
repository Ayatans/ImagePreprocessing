import torch
import torch.utils.data
from PIL import Image
import sys
import cv2
import os
import os.path
import random
import numpy as np

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

import collections
from maskrcnn_benchmark.structures.bounding_box import BoxList
import albumentations as A


class DIORDataset_Meta(torch.utils.data.Dataset):
    CLASSES = ("__background__ ", 'airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge', 'chimney',
               'dam', 'Expressway-Service-area', 'Expressway-toll-station',
               'harbor', 'golffield', 'groundtrackfield', 'overpass',
               'ship', 'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill')

    CLASSES_SPLIT1_BASE = (
        "__background__ ", 'airplane', 'airport', 'dam', 'Expressway-Service-area',
        'Expressway-toll-station', 'harbor', 'golffield', 'groundtrackfield', 'overpass', 'stadium',
        'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill'
    )

    CLASSES_SPLIT2_BASE = (
        "__background__ ", 'baseballfield', 'basketballcourt', 'bridge', 'chimney', 'dam',
        'Expressway-Service-area', 'golffield', 'overpass', 'ship', 'stadium',
        'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill'
    )

    CLASSES_SPLIT3_BASE = (
        "__background__ ", 'airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
        'chimney', 'Expressway-Service-area', 'Expressway-toll-station', 'harbor', 'groundtrackfield',
        'overpass', 'ship', 'stadium', 'trainstation', 'windmill'
    )

    CLASSES_SPLIT4_BASE = (
        "__background__ ", 'airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
        'chimney', 'dam', 'Expressway-toll-station', 'harbor', 'golffield',
        'groundtrackfield', 'ship', 'storagetank', 'tenniscourt', 'vehicle'
    )

    CLASSES_SPLIT5_BASE = (
        "__background__ ", 'airport', 'basketballcourt', 'bridge', 'chimney',
        'dam', 'Expressway-Service-area', 'Expressway-toll-station', 'harbor', 'golffield',
        'groundtrackfield', 'overpass', 'ship', 'stadium', 'storagetank', 'vehicle'
    )

    CLASSES_SPLIT1_NOVEL = ('baseballfield', 'basketballcourt', 'bridge', 'chimney', 'ship')
    CLASSES_SPLIT2_NOVEL = ('airplane', 'airport', 'Expressway-toll-station', 'harbor', 'groundtrackfield')
    CLASSES_SPLIT3_NOVEL = ('dam', 'golffield', 'storagetank', 'tenniscourt', 'vehicle')
    CLASSES_SPLIT4_NOVEL = ('Expressway-Service-area', 'overpass', 'stadium', 'trainstation', 'windmill')
    CLASSES_SPLIT5_NOVEL = ('airplane', 'baseballfield', 'tenniscourt', 'trainstation', 'windmill')

    def __init__(self, data_dir, split, use_difficult=False, transforms=None, toofew=True, shots=200, size=224, seed=0):

        # data_dir: "DIOR-DCNet"  ,split: "trainval_split1_base"
        self.root = data_dir
        self.image_set = split
        self.keep_difficult = use_difficult
        self.transforms = transforms

        self._annopath = os.path.join(self.root, "Annotations", "%s.xml")
        self._imgpath = os.path.join(self.root, "JPEGImages", "%s.jpg")
        self._imgsetpath = os.path.join(self.root, "ImageSets", "Main", "%s.txt")

        # 旋转增强
        self.rotate_transform1 = A.Rotate(limit=180, p=1)
        self.horizon_flip=A.HorizontalFlip(p=0.5)

        phase = 1
        if 'split1_base' in split:
            cls = DIORDataset_Meta.CLASSES_SPLIT1_BASE
        elif 'split2_base' in split:
            cls = DIORDataset_Meta.CLASSES_SPLIT2_BASE
        elif 'split3_base' in split:
            cls = DIORDataset_Meta.CLASSES_SPLIT3_BASE
        elif 'split4_base' in split:
            cls = DIORDataset_Meta.CLASSES_SPLIT4_BASE
        elif 'split5_base' in split:
            cls = DIORDataset_Meta.CLASSES_SPLIT5_BASE
        else:
            phase = 2
            cls = DIORDataset_Meta.CLASSES
        self.cls = cls[1:]
        self.class_to_ind = dict(zip(cls, range(len(cls))))
        self.categories = dict(zip(range(len(cls)), cls))
        self.list_root = os.path.join('/root/code/zyc/DCNet', 'fs_list')
        if 'base' in split:
            fname = 'dior_traindict_full.txt'
            metafile = os.path.join(self.list_root, fname)
        else:
            fname = 'dior_traindict_bbox_' + str(shots) + 'shot.txt'
            metafile = os.path.join(self.list_root, fname)
        if 'standard' in split and seed > 0:
            fname = 'dior_traindict_bbox_' + str(shots) + 'shot_seed' + str(seed) + '.txt'
            metafile = os.path.join(self.list_root, fname)
        metainds = [[]] * len(self.cls)
        with open(metafile, 'r') as f:
            metafiles = []
            for line in f.readlines():
                pair = line.rstrip().split()
                if len(pair) == 2:
                    pass
                elif len(pair) == 4:
                    pair = [pair[0] + ' ' + pair[1], pair[2] + ' ' + pair[3]]
                else:
                    raise NotImplementedError('{} not recognized'.format(pair))
                metafiles.append(pair)
            metafiles = {k: v for k, v in metafiles}
            self.metalines = [[]] * len(self.cls)
            for i, clsname in enumerate(self.cls):
                with open(metafiles[clsname], 'r') as imgf:     # 打开的是diorlist1中不同类别在不同shot下用到的训练图片id
                    lines = [l for l in imgf.readlines()]
                    self.metalines[i] = lines
                    if (shots > 100):   # metalines包含了该类所有训练图像，从该类的图片id中随机抽shots个。shots<100就全用？
                        self.metalines[i] = random.sample(self.metalines[i], shots) # 应该是微调时就不采样，全用。

        self.ids = []   # 最终是64*shots长度
        self.img_size = size
        if (phase == 2):
            for j in range(len(self.cls)):
                # 从各shots文件中随机取出shots*64个元素，可重复
                self.metalines[j] = np.random.choice(self.metalines[j], shots).tolist()
            for i in range(shots):  # 64倍？
                metaid = []
                for j in range(len(self.cls)):
                    metaid.append([j, self.metalines[j][i].rstrip()])   # j是类别，i是数量
                self.ids.append(metaid)
        else:   # phase 1 每类抽shots个数据
            for i in range(shots):
                metaid = []
                for j in range(len(self.cls)):
                    metaid.append([j, self.metalines[j][i].rstrip()])
                self.ids.append(metaid)

    def __getitem__(self, index):
        # 原来的DCNet用的
        # img_ids = self.ids[index]
        # data = []
        # for cls_id, img_id in img_ids:
        #     img = cv2.imread(img_id, cv2.IMREAD_COLOR)
        #     img = img.astype(np.float32, copy=False)
        #     img -= np.array([[[102.9801, 115.9465, 122.7717]]])
        #     height, width, _ = img.shape    # 原图尺寸
        #     img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        #     # img = Image.open(img_id).convert("RGB")
        #
        #     mask = self.get_mask(img_id, cls_id, height, width)
        #
        #     img = torch.from_numpy(img).unsqueeze(0)
        #     mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(3)
        #     imgmask = torch.cat([img, mask], dim=3)     # 加了mask，不是切好的那种，所以目标如果小，resize之后就更小了。
        #     imgmask = imgmask.permute(0, 3, 1, 2).contiguous()  # torch.FloatTensor
        #     data.append(imgmask)
        #
        # res = torch.cat(data, dim=0)

        # # # 重新构建，模仿PCNN
        img_ids = self.ids[index]   # index：每次bs per GPU个int值
        data = []
        for cls_id, img_id in img_ids:
            img = cv2.imread(img_id, cv2.IMREAD_COLOR)
            img = img.astype(np.float32, copy=False)
            img -= np.array([[[102.9801, 115.9465, 122.7717]]])
            height, width, _ = img.shape  # 原图尺寸

            # 制作切好的目标patch
            path = img_id.split('JPEG')[0] + 'Annotations/' + img_id.split('/')[-1].split('.jpg')[0] + '.xml'
            target = ET.parse(path).getroot()
            # img = img[:, :, ::-1]  # 调整图片的通道为反序，例如RGB为BGR
            image_patch = np.zeros((self.img_size, self.img_size), dtype=np.float32)
            mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
            mask[:, :] = 0
            mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(3)
            for obj in target.iter('object'):  # 遍历个某张图内所有的<object>元素。也就是1个标签
                name = obj.find('name').text.strip()  # 读取<name>元素，获得类别名
                if name != self.cls[cls_id]:
                    continue
                bbox = obj.find('bndbox')  # 获得该图像的bbox信息
                pts = ['xmin', 'ymin', 'xmax', 'ymax']
                bndbox = []
                for i, pt in enumerate(pts):
                    cur_pt = int(float(bbox.find(pt).text)) - 1  # 获取bbox中这些键的值，-1干嘛的？
                    if i % 2 == 0:  # i是偶数，读取到的是xmin，xmax
                        cur_pt = int(cur_pt)  # 这里为什么没有除以x比例，像上面一样？因为要根据bbox裁剪？
                        bndbox.append(cur_pt)
                    elif i % 2 == 1:  # i是奇数，读取到的是ymin，ymax
                        cur_pt = int(cur_pt)
                        bndbox.append(cur_pt)  # 相当于也是按pts的顺序加入bndbox了
                image_patch = self.crop(img, bndbox, size=self.img_size)  # 根据bbox切图
                break
            xd = bndbox[2] - bndbox[0]
            yd = bndbox[3] - bndbox[1]
            if xd > yd:
                ratio = yd / xd
                xd = 256
                yd = int(256 * ratio)
                yup = 128 - int(yd / 2)
                ydown = 128 + int(yd / 2)
                mask[yup:ydown, :] = 1
            else:
                ratio = xd / yd
                yd = 256
                xd = int(256 * ratio)
                xup = 128 - int(xd / 2)
                xdown = 128 + int(xd / 2)
                mask[:, xup:xdown] = 1
                # 现在image patch 和5个旋转patch大小都是256*256
            # image_patch1 = self.rotate_transform1(image=image_patch)['image']
            image_patch1 = self.horizon_flip(image=image_patch)['image']
            image_patch = torch.from_numpy(image_patch).unsqueeze(0)
            image_patch1 = torch.from_numpy(image_patch1).unsqueeze(0)
            image_patch = torch.cat([image_patch, mask], dim=3)
            image_patch1 = torch.cat([image_patch1, mask], dim=3)
            image_patch = image_patch.permute(0, 3, 1, 2).contiguous()
            image_patch1 = image_patch1.permute(0, 3, 1, 2).contiguous()
            # data.append(image_patch)    # torch.FloatTensor [1, 4, 256, 256]
            data.append(image_patch1)   # torch.FloatTensor
        res = torch.cat(data, dim=0)    # torch.FloatTensor  res.shape=[20, 4, 256, 256]
        return res

    # 来自PCNN
    def crop(self, image, purpose, size):
        # 裁剪purpose区域内的图片并resize到size，保持bbox宽高比不变，其余部分填0
        # 不太理解下面这些变换，好像并没有很大的作用？变来变去都是一样的。
        # 提取图片在purpose即bbox内的部分
        cut_image = image[int(purpose[1]):int(purpose[3]), int(purpose[0]):int(purpose[2]), :]
        # 裁剪后图片的高宽
        height, width = cut_image.shape[0:2]
        # 宽高中的最大值
        max_hw = max(height, width)
        # cty，ctx为一半的高宽
        cty, ctx = [height // 2, width // 2]
        # ci=正方形
        cropped_image = np.zeros((max_hw, max_hw, 3), dtype=cut_image.dtype)
        # 不管max是宽还是高，x0 y0都是0，x1 y1都是width height，只是可能抛弃了一两个点因为触发
        x0, x1 = max(0, ctx - max_hw // 2), min(ctx + max_hw // 2, width)
        y0, y1 = max(0, cty - max_hw // 2), min(cty + max_hw // 2, height)
        # 若x0 y0不是0 则报异常
        assert x0 == 0
        assert y0 == 0
        # left=width//2，right=width//2
        # top=height//2， bottom=height//2
        left, right = ctx - x0, x1 - ctx
        top, bottom = cty - y0, y1 - cty

        cropped_cty, cropped_ctx = max_hw // 2, max_hw // 2
        y_slice = slice(cropped_cty - top, cropped_cty + bottom)
        x_slice = slice(cropped_ctx - left, cropped_ctx + right)
        cropped_image[y_slice, x_slice, :] = cut_image[y0:y1, x0:x1, :]

        return cv2.resize(cropped_image, dsize=(size, size), interpolation=cv2.INTER_LINEAR)

    def get_img_info(self, index):
        cls_id, img_id = self.ids[index][0]
        path = img_id.split('JPEG')[0] + 'Annotations/' + img_id.split('/')[-1].split('.jpg')[0] + '.xml'
        anno = ET.parse(path).getroot()
        size = anno.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))
        return {"height": im_info[0], "width": im_info[1]}

    def __len__(self):
        return len(self.ids)

    def get_mask(self, img_id, cls_id, height, width):
        mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        y_ration = float(height) / self.img_size
        x_ration = float(width) / self.img_size

        path = img_id.split('JPEG')[0] + 'Annotations/' + img_id.split('/')[-1].split('.jpg')[0] + '.xml'
        target = ET.parse(path).getroot()
        for obj in target.iter('object'):
            difficult = False  # int(obj.find("difficult").text) == 1
            if difficult:
                continue
            name = obj.find('name').text.strip()
            if (name != self.cls[cls_id]):
                continue
            bbox = obj.find('bndbox')
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(float(bbox.find(pt).text)) - 1
                if i % 2 == 0:
                    cur_pt = int(cur_pt / x_ration)
                    bndbox.append(cur_pt)
                elif i % 2 == 1:
                    cur_pt = int(cur_pt / y_ration)
                    bndbox.append(cur_pt)
            mask[bndbox[1]:bndbox[3], bndbox[0]:bndbox[2]] = 1
            break
        return mask

    def get_groundtruth(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        anno = self._preprocess_annotation(anno)

        height, width = anno["im_info"]
        target = BoxList(anno["boxes"], (width, height), mode="xyxy")
        target.add_field("labels", anno["labels"])
        # target.add_field("difficult", anno["difficult"])
        return target

    def _preprocess_annotation(self, target):
        boxes = []
        gt_classes = []
        difficult_boxes = []
        TO_REMOVE = 1

        for obj in target.iter("object"):
            difficult = False  # int(obj.find("difficult").text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find("name").text.lower().strip()
            bb = obj.find("bndbox")
            # Make pixel indexes 0-based
            # Refer to "https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/pascal_voc.py#L208-L211"
            box = [
                bb.find("xmin").text,
                bb.find("ymin").text,
                bb.find("xmax").text,
                bb.find("ymax").text,
            ]
            bndbox = tuple(
                map(lambda x: x - TO_REMOVE, list(map(int, box)))
            )

            boxes.append(bndbox)
            gt_classes.append(self.class_to_ind[name])
            difficult_boxes.append(difficult)

        size = target.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))

        res = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(gt_classes),
            "difficult": torch.tensor(difficult_boxes),
            "im_info": im_info,
        }
        return res

    def map_class_id_to_class_name(self, class_id):
        # return PascalVOCDataset.CLASSES[class_id]
        return self.cls[class_id]
