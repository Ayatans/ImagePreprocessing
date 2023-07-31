#!/bin/bash
cd /root/code/zyc/dataset
cp voc_per_class.py /root/code/zyc/datasets/voc
python voc_per_class.py  # 生成diorlist，label_1c两个文件夹内文件
wget http://pjreddie.com/media/files/voc_label.py
python voc_label.py       # 生成labels文件夹内文件
cat 2007_train.txt 2007_val.txt 2012_*.txt > dior_train.txt

mkdir diorlist1 # for random seed 1

cd /root/code/zyc/DCNet

python tools/fewshot_exp/datasets/gen_fewlist_dior.py 1 # for random seed 1   生成diorlist1内文件
python tools/fewshot_exp/datasets/gen_fewlist_nwpu.py 1 # for random seed 1   生成diorlist1内文件

cd fs_list

python random_dict.py   # 生成DCNet/fs_list内文件

cd ..

#init base/novel sets for fewshot exps
python tools/fewshot_exp/datasets/dior_create_base.py   # 根据4个split，生成DIOR-DCNet/ImageSets/Main/内一堆trainval test开头 base结尾的文件 包含不同split训或测时用的图片id
python tools/fewshot_exp/datasets/dior_create_standard.py 1 # for random seed 1  根据shots，生成/ImageSets/Main/内一堆trainval开头 standard结尾文件
python tools/fewshot_exp/datasets/nwpu_create_standard.py 1 # for random seed 1  根据shots，生成/ImageSets/Main/内一堆trainval开头 standard结尾文件

mkdir fs_exp
