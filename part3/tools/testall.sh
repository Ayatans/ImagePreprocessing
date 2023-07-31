gap2500=(0012500 0010000 0007500 0005000 0002500)
gap1250=(0013750 0012500 0011250 0010000 0008750 0007500 0006250 0005000 0003750 0002500 0001250)
for a in ${gap1250[*]}
#for a in ${gap2500[*]}
do
  ck=output/archived_2022-12-20_17:37:41/model_${a}.pth
  python -m torch.distributed.launch --nproc_per_node=4 tools/test_net.py --ckpt ${ck}
done
