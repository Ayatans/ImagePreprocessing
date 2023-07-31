import glob
# 此文件有什么实际作用？后面用上了吗？
#vlists = glob.glob('voc*shot.txt')
#vlists = glob.glob('nwpu*shot.txt')
vlists = glob.glob('dior*shot.txt')

for seed in range(1,2):
    for v in vlists:
        d=open(v).readlines()
        with open(v.split('.txt')[0]+'_seed'+str(seed)+'.txt','a') as f:
            res = []
            for line in d:
                res.append(line.replace('diorlist','diorlist'+str(seed)))
            for i in res:
                f.write(i)
