from maskrcnn_benchmark.data.datasets.nwpu import NWPUDataset
import sys
# arg 1: pathdir;
for shot in [3, 5, 10]:
    all_cls = NWPUDataset.CLASSES
    novel_cls = [NWPUDataset.CLASSES_SPLIT1_NOVEL,][0]
    novel_index = [all_cls.index(c) for c in novel_cls]
    AP = [0] * 5
    try:
        with open(sys.argv[1] + '/result_split1_%dshot.txt'%(shot), 'r') as f:
            content = f.readlines()
        for k, j in enumerate(novel_index):
            AP[k] += float(content[j][18 : 24])
        print("NWPU split1  %2dshot:novel map:%.4f"%(shot, sum(AP) / 5))
    except Exception as e:
        continue
