import torch
import maskrcnn_benchmark


nwpupath=r'E:\datasets\NWPU-VOC/predictions.pth'
pred=torch.load(nwpupath)
with open( r'E:\datasets\NWPU-VOC\prediction_results_nwpu.txt', 'w') as f:
    for ind, pre in enumerate(pred):
        thisbbox = pre.bbox
        if len(thisbbox):
            for j in thisbbox:
                if len(j) == 4:
                    xmin, ymin, xmax, ymax = j.numpy()
                    s = str(ind) + ' ' + str(xmin) + ' ' + str(ymin) + ' ' + str(xmax) + ' ' + str(ymax) + '\n'
                else:
                    xmin, ymin, xmax, ymax = 0, 0, 0, 0
                    s = str(ind) + ' 0\n'
                f.write(s)
        else:
            f.write(str(ind) + ' 0\n')