# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Centralized catalog of paths."""

import os


class DatasetCatalog(object):
    DATA_DIR = "/remote-home/yczhang/code/dataset/VOC/VOCdevkit/"
    DIOR_DIR = '/remote-home/yczhang/code/dataset/DIOR-DCNet/'
    NWPU_DIR = '/remote-home/yczhang/code/dataset/NWPU-VOC/'
    DATASETS = {
        "coco_2017_train": {
            "img_dir": "coco/train2017",
            "ann_file": "coco/annotations/instances_train2017.json"
        },
        "coco_2017_val": {
            "img_dir": "coco/val2017",
            "ann_file": "coco/annotations/instances_val2017.json"
        },
        "coco_2014_train": {
            "img_dir": "coco/train2014",
            "ann_file": "coco/annotations/instances_train2014.json"
        },
        "coco_2014_val": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/instances_val2014.json"
        },
        "coco_2014_minival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/instances_minival2014.json"
        },
        "coco_2014_valminusminival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/instances_valminusminival2014.json"
        },
        "coco_2014_train_base": {
            "img_dir": "coco/train2014",
            "ann_file": "coco/annotations/instances_train2014_base.json"
        },
        "coco_meta_2014_train_base": {
            "img_dir": "coco/train2014",
            "ann_file": "coco/annotations/instances_train2014.json"
        },
        "coco_2014_train_30shot_standard": {
            "img_dir": "coco/train2014",
            "ann_file": "coco/annotations/instances_train2014_30shot_novel_standard.json"
        },
        "coco_meta_2014_train_30shot_standard": {
            "img_dir": "coco/train2014",
            "ann_file": "coco/annotations/instances_train2014_30shot_novel_standard.json"
        },
        "coco_2014_val_30shot_standard": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/instances_val2014_30shot_novel_standard.json"
        },
        "coco_2014_train_10shot_standard": {
            "img_dir": "coco/train2014",
            "ann_file": "coco/annotations/instances_train2014_10shot_novel_standard.json"
        },
        "coco_meta_2014_train_10shot_standard": {
            "img_dir": "coco/train2014",
            "ann_file": "coco/annotations/instances_train2014_10shot_novel_standard.json"
        },
        "coco_2014_val_10shot_standard": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/instances_val2014_10shot_novel_standard.json"
        },
        "coco_2014_valminusminival_base": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/instances_valminusminival2014_base.json"
        },
        "coco_2014_minival_base": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/instances_minival2014_base.json"
        },
        "keypoints_coco_2014_train": {
            "img_dir": "coco/train2014",
            "ann_file": "coco/annotations/person_keypoints_train2014.json",
        },
        "keypoints_coco_2014_val": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/person_keypoints_val2014.json"
        },
        "keypoints_coco_2014_minival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/person_keypoints_minival2014.json",
        },
        "keypoints_coco_2014_valminusminival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/person_keypoints_valminusminival2014.json",
        },
        "voc_2007_train": {
            "data_dir": "VOC2007",
            "split": "train"
        },
        "voc_2007_trainval": {
            "data_dir": "VOC2007",
            "split": "trainval"
        },
        "voc_2007_trainval_split1_base": {
            "data_dir": "VOC2007",
            "split": "trainval_split1_base"
        },
        "voc_2012_trainval_split1_base": {
            "data_dir": "VOC2012",
            "split": "trainval_split1_base"
        },
        "voc_2007_test_split1_base": {
            "data_dir": "VOC2007",
            "split": "test_split1_base"
        },
        "voc_2007_trainval_split2_base": {
            "data_dir": "VOC2007",
            "split": "trainval_split2_base"
        },
        "voc_2012_trainval_split2_base": {
            "data_dir": "VOC2012",
            "split": "trainval_split2_base"
        },
        "voc_2007_test_split2_base": {
            "data_dir": "VOC2007",
            "split": "test_split2_base"
        },
        "voc_2007_trainval_split3_base": {
            "data_dir": "VOC2007",
            "split": "trainval_split3_base"
        },
        "voc_2012_trainval_split3_base": {
            "data_dir": "VOC2012",
            "split": "trainval_split3_base"
        },
        "voc_2007_test_split3_base": {
            "data_dir": "VOC2007",
            "split": "test_split3_base"
        },
        "voc_2007_val": {
            "data_dir": "VOC2007",
            "split": "val"
        },
        "voc_2007_test": {
            "data_dir": "VOC2007",
            "split": "test"
        },
        "voc_2012_train": {
            "data_dir": "VOC2012",
            "split": "train"
        },
        "voc_2012_val": {
            "data_dir": "VOC2012",
            "split": "val"
        },
        "voc_2012_test": {
            "data_dir": "VOC2012",
            "split": "test"
            # PASCAL VOC2012 doesn't made the test annotations available, so there's no json annotation
        },
        "voc_2007_trainval_10shot_extreme": {
            "data_dir": "VOC2007",
            "split": "trainval_10shot_novel_extreme"
        },
        "voc_2012_trainval_10shot_extreme": {
            "data_dir": "VOC2012",
            "split": "trainval_10shot_novel_extreme"
        },
        "voc_2007_trainval_10shot_standard": {
            "data_dir": "VOC2007",
            "split": "trainval_10shot_novel_standard"
        },
        "voc_2012_trainval_10shot_standard": {
            "data_dir": "VOC2012",
            "split": "trainval_10shot_novel_standard"
        },
        "voc_2007_trainval_5shot_standard": {
            "data_dir": "VOC2007",
            "split": "trainval_5shot_novel_standard"
        },
        "voc_2012_trainval_5shot_standard": {
            "data_dir": "VOC2012",
            "split": "trainval_5shot_novel_standard"
        },
        "voc_2007_trainval_3shot_standard": {
            "data_dir": "VOC2007",
            "split": "trainval_3shot_novel_standard"
        },
        "voc_2012_trainval_3shot_standard": {
            "data_dir": "VOC2012",
            "split": "trainval_3shot_novel_standard"
        },
        "voc_2007_trainval_2shot_standard": {
            "data_dir": "VOC2007",
            "split": "trainval_2shot_novel_standard"
        },
        "voc_2012_trainval_2shot_standard": {
            "data_dir": "VOC2012",
            "split": "trainval_2shot_novel_standard"
        },
        "voc_2007_trainval_1shot_standard": {
            "data_dir": "VOC2007",
            "split": "trainval_1shot_novel_standard"
        },
        "voc_2012_trainval_1shot_standard": {
            "data_dir": "VOC2012",
            "split": "trainval_1shot_novel_standard"
        },
        "voc_meta_2007_trainval_10shot_standard": {
            "data_dir": "VOC2007",
            "split": "trainval_10shot_novel_standard"
        },
        "voc_meta_2012_trainval_10shot_standard": {
            "data_dir": "VOC2012",
            "split": "trainval_10shot_novel_standard"
        },
        "voc_meta_2007_trainval_5shot_standard": {
            "data_dir": "VOC2007",
            "split": "trainval_5shot_novel_standard"
        },
        "voc_meta_2012_trainval_5shot_standard": {
            "data_dir": "VOC2012",
            "split": "trainval_5shot_novel_standard"
        },
        "voc_meta_2007_trainval_3shot_standard": {
            "data_dir": "VOC2007",
            "split": "trainval_3shot_novel_standard"
        },
        "voc_meta_2012_trainval_3shot_standard": {
            "data_dir": "VOC2012",
            "split": "trainval_3shot_novel_standard"
        },
        "voc_meta_2007_trainval_2shot_standard": {
            "data_dir": "VOC2007",
            "split": "trainval_2shot_novel_standard"
        },
        "voc_meta_2012_trainval_2shot_standard": {
            "data_dir": "VOC2012",
            "split": "trainval_2shot_novel_standard"
        },
        "voc_meta_2007_trainval_1shot_standard": {
            "data_dir": "VOC2007",
            "split": "trainval_1shot_novel_standard"
        },
        "voc_meta_2012_trainval_1shot_standard": {
            "data_dir": "VOC2012",
            "split": "trainval_1shot_novel_standard"
        },
        "voc_meta_trainval_split1_base": {
            "data_dir": "VOC2007",
            "split": "trainval_split1_base"
        },     
        "voc_meta_trainval_split2_base": {
            "data_dir": "VOC2007",
            "split": "trainval_split2_base"
        },     
        "voc_meta_trainval_split3_base": {
            "data_dir": "VOC2007",
            "split": "trainval_split3_base"
        },
        "dior_trainval": {
            "data_dir": "",
            "split": "trainval"
        },
        "dior_trainval_split1_base": {
            "data_dir": "",
            "split": "trainval_split1_base"
        },
        "dior_test_split1_base": {
            "data_dir": "",
            "split": "test_split1_base"
        },
        "dior_trainval_split2_base": {
            "data_dir": "",
            "split": "trainval_split2_base"
        },
        "dior_test_split2_base": {
            "data_dir": "",
            "split": "test_split2_base"
        },
        "dior_trainval_split3_base": {
            "data_dir": "",
            "split": "trainval_split3_base"
        },
        "dior_test_split3_base": {
            "data_dir": "",
            "split": "test_split3_base"
        },
        "dior_trainval_split4_base": {
            "data_dir": "",
            "split": "trainval_split4_base"
        },
        "dior_test_split4_base": {
            "data_dir": "",
            "split": "test_split4_base"
        },
        "dior_trainval_split5_base": {
            "data_dir": "",
            "split": "trainval_split5_base"
        },
        "dior_test_split5_base": {
            "data_dir": "",
            "split": "test_split5_base"
        },
        "dior_train": {
            "data_dir": "",
            "split": "train"
        },
        "dior_val": {
            "data_dir": "",
            "split": "val"
        },
        "dior_test": {
            "data_dir": "",
            "split": "test"
        },
        "dior_trainval_30shot_standard": {
            "data_dir": "",
            "split": "trainval_30shot_novel_standard"
        },
        "dior_trainval_20shot_extreme": {
            "data_dir": "",
            "split": "trainval_20shot_novel_extreme"
        },
        "dior_trainval_20shot_standard": {
            "data_dir": "",
            "split": "trainval_20shot_novel_standard"
        },
        "dior_trainval_10shot_extreme": {
            "data_dir": "",
            "split": "trainval_10shot_novel_extreme"
        },
        "dior_trainval_10shot_standard": {
            "data_dir": "",
            "split": "trainval_10shot_novel_standard"
        },
        "dior_trainval_5shot_standard": {
            "data_dir": "",
            "split": "trainval_5shot_novel_standard"
        },
        "dior_trainval_3shot_standard": {
            "data_dir": "",
            "split": "trainval_3shot_novel_standard"
        },
        "dior_trainval_2shot_standard": {
            "data_dir": "",
            "split": "trainval_2shot_novel_standard"
        },
        "dior_trainval_1shot_standard": {
            "data_dir": "",
            "split": "trainval_1shot_novel_standard"
        },
        "dior_meta_trainval_30shot_standard": {
            "data_dir": "",
            "split": "trainval_30shot_novel_standard"
        },
        "dior_meta_trainval_20shot_standard": {
            "data_dir": "",
            "split": "trainval_20shot_novel_standard"
        },
        "dior_meta_trainval_10shot_standard": {
            "data_dir": "",
            "split": "trainval_10shot_novel_standard"
        },
        "dior_meta_trainval_5shot_standard": {
            "data_dir": "",
            "split": "trainval_5shot_novel_standard"
        },
        "dior_meta_trainval_3shot_standard": {
            "data_dir": "",
            "split": "trainval_3shot_novel_standard"
        },
        "dior_meta_trainval_2shot_standard": {
            "data_dir": "",
            "split": "trainval_2shot_novel_standard"
        },
        "dior_meta_trainval_1shot_standard": {
            "data_dir": "",
            "split": "trainval_1shot_novel_standard"
        },
        "dior_meta_trainval_split1_base": {
            "data_dir": "",
            "split": "trainval_split1_base"
        },
        "dior_meta_trainval_split2_base": {
            "data_dir": "",
            "split": "trainval_split2_base"
        },
        "dior_meta_trainval_split3_base": {
            "data_dir": "",
            "split": "trainval_split3_base"
        },
        "dior_meta_trainval_split4_base": {
            "data_dir": "",
            "split": "trainval_split4_base"
        },
        "dior_meta_trainval_split5_base": {
            "data_dir": "",
            "split": "trainval_split5_base"
        },
        "nwpu_train": {
            "data_dir": "",
            "split": "train"
        },
        "nwpu_trainval": {
            "data_dir": "",
            "split": "trainval"
        },
        "nwpu_trainval_split1_base": {
            "data_dir": "",
            "split": "trainval_split1_base"
        },
        "nwpu_test_split1_base": {
            "data_dir": "",
            "split": "test_split1_base"
        },
        "nwpu_val": {
            "data_dir": "",
            "split": "val"
        },
        "nwpu_test": {
            "data_dir": "",
            "split": "test"
        },
        "nwpu_trainval_10shot_standard": {
            "data_dir": "",
            "split": "trainval_10shot_novel_standard"
        },
        "nwpu_trainval_5shot_standard": {
            "data_dir": "",
            "split": "trainval_5shot_novel_standard"
        },
        "nwpu_trainval_3shot_standard": {
            "data_dir": "",
            "split": "trainval_3shot_novel_standard"
        },
        "nwpu_meta_trainval_10shot_standard": {
            "data_dir": "",
            "split": "trainval_10shot_novel_standard"
        },
        "nwpu_meta_trainval_5shot_standard": {
            "data_dir": "",
            "split": "trainval_5shot_novel_standard"
        },
        "nwpu_meta_trainval_3shot_standard": {
            "data_dir": "",
            "split": "trainval_3shot_novel_standard"
        },
        "nwpu_meta_trainval_split1_base": {
            "data_dir": "",
            "split": "trainval_split1_base"
        },

    }

    @staticmethod
    def get(name):
        if "voc_meta" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(data_dir, attrs["data_dir"]),
                split=attrs["split"],
            )
            return dict(
                factory="PascalVOCDataset_Meta",
                args=args,
            )
        if "dior_meta" in name:
            data_dir = DatasetCatalog.DIOR_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                # data_dir=os.path.join(data_dir, attrs["data_dir"]),
                data_dir=data_dir,
                split=attrs["split"],
            )
            return dict(
                factory="DIORDataset_Meta",
                args=args,
            )
        if "nwpu_meta" in name:
            data_dir = DatasetCatalog.NWPU_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                # data_dir=os.path.join(data_dir, attrs["data_dir"]),
                data_dir=data_dir,
                split=attrs["split"],
            )
            return dict(
                factory="NWPUDataset_Meta",
                args=args,
            )
        if "coco_meta" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="COCODataset_Meta",
                args=args,
            )

        if "coco" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="COCODataset",
                args=args,
            )
        if "dior" in name:
            data_dir = DatasetCatalog.DIOR_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                # data_dir=os.path.join(data_dir, attrs["data_dir"]),
                data_dir=data_dir,
                split=attrs["split"],
            )
            return dict(
                factory="DIORDataset",
                args=args,
            )
        if "nwpu" in name:
            data_dir = DatasetCatalog.NWPU_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                # data_dir=os.path.join(data_dir, attrs["data_dir"]),
                data_dir=data_dir,
                split=attrs["split"],
            )
            return dict(
                factory="NWPUDataset",
                args=args,
            )
        elif "voc" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(data_dir, attrs["data_dir"]),
                split=attrs["split"],
            )
            return dict(
                factory="PascalVOCDataset",
                args=args,
            )
        raise RuntimeError("Dataset not available: {}".format(name))


class ModelCatalog(object):
    S3_C2_DETECTRON_URL = "https://dl.fbaipublicfiles.com/detectron"
    C2_IMAGENET_MODELS = {
        "MSRA/R-50": "ImageNetPretrained/MSRA/R-50.pkl",
        "MSRA/R-50-GN": "ImageNetPretrained/47261647/R-50-GN.pkl",
        "MSRA/R-101": "ImageNetPretrained/MSRA/R-101.pkl",
        "MSRA/R-101-GN": "ImageNetPretrained/47592356/R-101-GN.pkl",
        "FAIR/20171220/X-101-32x8d": "ImageNetPretrained/20171220/X-101-32x8d.pkl",
    }

    C2_DETECTRON_SUFFIX = "output/train/{}coco_2014_train%3A{}coco_2014_valminusminival/generalized_rcnn/model_final.pkl"
    C2_DETECTRON_MODELS = {
        "35857197/e2e_faster_rcnn_R-50-C4_1x": "01_33_49.iAX0mXvW",
        "35857345/e2e_faster_rcnn_R-50-FPN_1x": "01_36_30.cUF7QR7I",
        "35857890/e2e_faster_rcnn_R-101-FPN_1x": "01_38_50.sNxI7sX7",
        "36761737/e2e_faster_rcnn_X-101-32x8d-FPN_1x": "06_31_39.5MIHi1fZ",
        "35858791/e2e_mask_rcnn_R-50-C4_1x": "01_45_57.ZgkA7hPB",
        "35858933/e2e_mask_rcnn_R-50-FPN_1x": "01_48_14.DzEQe4wC",
        "35861795/e2e_mask_rcnn_R-101-FPN_1x": "02_31_37.KqyEK4tT",
        "36761843/e2e_mask_rcnn_X-101-32x8d-FPN_1x": "06_35_59.RZotkLKI",
        "37129812/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x": "09_35_36.8pzTQKYK",
        # keypoints
        "37697547/e2e_keypoint_rcnn_R-50-FPN_1x": "08_42_54.kdzV35ao"
    }

    @staticmethod
    def get(name):
        if name.startswith("Caffe2Detectron/COCO"):
            return ModelCatalog.get_c2_detectron_12_2017_baselines(name)
        if name.startswith("ImageNetPretrained"):
            return ModelCatalog.get_c2_imagenet_pretrained(name)
        raise RuntimeError("model not present in the catalog {}".format(name))

    @staticmethod
    def get_c2_imagenet_pretrained(name):
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        name = name[len("ImageNetPretrained/"):]
        name = ModelCatalog.C2_IMAGENET_MODELS[name]
        url = "/".join([prefix, name])
        return url

    @staticmethod
    def get_c2_detectron_12_2017_baselines(name):
        # Detectron C2 models are stored following the structure
        # prefix/<model_id>/2012_2017_baselines/<model_name>.yaml.<signature>/suffix
        # we use as identifiers in the catalog Caffe2Detectron/COCO/<model_id>/<model_name>
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        dataset_tag = "keypoints_" if "keypoint" in name else ""
        suffix = ModelCatalog.C2_DETECTRON_SUFFIX.format(dataset_tag, dataset_tag)
        # remove identification prefix
        name = name[len("Caffe2Detectron/COCO/"):]
        # split in <model_id> and <model_name>
        model_id, model_name = name.split("/")
        # parsing to make it match the url address from the Caffe2 models
        model_name = "{}.yaml".format(model_name)
        signature = ModelCatalog.C2_DETECTRON_MODELS[name]
        unique_name = ".".join([model_name, signature])
        url = "/".join([prefix, model_id, "12_2017_baselines", unique_name, suffix])
        return url
