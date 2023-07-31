#!/bin/bash
ROOT=../..
export PYTHONPATH=$ROOT:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=4,5,6,7
export NGPUS=4
SHOT=(10) #(10 5 3)  #(20 10 5 3) #(10 5 3)
mkdir fs_exp/nwpu_standard_results
for shot in ${SHOT[*]}
do
    configfile=configs/noveltest/e2e_nwpu_split1_${shot}shot_finetune.yaml
    python -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port 26527 $ROOT/tools/test.py --config-file ${configfile} 2>&1 | tee logs/log_${split}_${shot}_dior_finetune.txt
    mv inference/nwpu_test/result.txt fs_exp/nwpu_standard_results/result_split1_${shot}shot.txt
done

python $ROOT/tools/fewshot_exp/cal_novel_nwpu.py


