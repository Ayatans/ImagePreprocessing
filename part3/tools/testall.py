import os

maxfile=11
gap=1250
for i in range(1, maxfile):
    os.system('python -m torch.distributed.launch --nproc_per_node=4 tools/test_net.py ')