import pickle

# SS的pkl也是dict，indexes， boxes，scores，其中scores全是1.0

pkl_path = '/remote-home/yczhang/code/odwscl/proposal/SS/voc/SS-voc07_trainval.pkl'
pkl_path = '/remote-home/yczhang/code/DIORproposals/SS/SSdior-trainval-fast-11726-23463.pkl'
data_list=[]
with open(pkl_path, 'rb') as f:
    while True:
        try:
            data = pickle.load(f)
        except EOFError:
            break
print(type(data))
ans=[]
for k,v in data.items():
    print(k)
    print(len(v))
    if k=='indexes':
        print(v)
    # if k=='scores':
    #     print(type(v[0]), len(v[0]),type(v[0][0]))
    if k=='boxes':
        #print(type(v[0]), len(v[0]), type(v[0][0]), type(v[0][0][0]))
        for i in v:
            ans.append(len(i))
            # print(len(i))
print(sum(ans)/len(ans))
