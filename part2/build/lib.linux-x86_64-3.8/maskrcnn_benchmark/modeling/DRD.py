import torch
import torch.nn.functional as F
import torch.nn as nn


class DenseRelationDistill(nn.Module):

    def __init__(self, indim, keydim, valdim, dense_sum=False):  # 256 32 128 True
        super(DenseRelationDistill, self).__init__()
        # self.key_q = nn.Conv2d(indim, keydim, kernel_size=(3,3), padding=(1,1), stride=1)
        # self.value_q = nn.Conv2d(indim, valdim, kernel_size=(3,3), padding=(1,1), stride=1)
        self.key_t = nn.Conv2d(indim, keydim, kernel_size=(3, 3), padding=(1, 1),
                               stride=1)  # c=256，c/2=value128 c/8=key32
        self.value_t = nn.Conv2d(indim, valdim, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.abla_val_t = nn.AdaptiveAvgPool2d((1, 1))
        self.abla_val_q = nn.AdaptiveAvgPool2d((1, 1))
        self.abla2_feat = nn.AdaptiveAvgPool2d((128, 128))
        self.sum = dense_sum
        if self.sum:
            self.key_q0 = nn.Conv2d(indim, keydim, kernel_size=(3, 3), padding=(1, 1), stride=1)
            self.value_q0 = nn.Conv2d(indim, valdim, kernel_size=(3, 3), padding=(1, 1), stride=1)
            self.key_q1 = nn.Conv2d(indim, keydim, kernel_size=(3, 3), padding=(1, 1), stride=1)
            self.value_q1 = nn.Conv2d(indim, valdim, kernel_size=(3, 3), padding=(1, 1), stride=1)
            self.key_q2 = nn.Conv2d(indim, keydim, kernel_size=(3, 3), padding=(1, 1), stride=1)
            self.value_q2 = nn.Conv2d(indim, valdim, kernel_size=(3, 3), padding=(1, 1), stride=1)
            self.key_q3 = nn.Conv2d(indim, keydim, kernel_size=(3, 3), padding=(1, 1), stride=1)
            self.value_q3 = nn.Conv2d(indim, valdim, kernel_size=(3, 3), padding=(1, 1), stride=1)
            self.key_q4 = nn.Conv2d(indim, keydim, kernel_size=(3, 3), padding=(1, 1), stride=1)
            self.value_q4 = nn.Conv2d(indim, valdim, kernel_size=(3, 3), padding=(1, 1), stride=1)
            self.bnn0 = nn.BatchNorm2d(256)
            self.bnn1 = nn.BatchNorm2d(256)
            self.bnn2 = nn.BatchNorm2d(256)
            self.bnn3 = nn.BatchNorm2d(256)
            self.bnn4 = nn.BatchNorm2d(256)
            # self.bnn0 = nn.BatchNorm2d(512)
            # self.bnn1 = nn.BatchNorm2d(512)
            # self.bnn2 = nn.BatchNorm2d(512)
            # self.bnn3 = nn.BatchNorm2d(512)
            # self.bnn4 = nn.BatchNorm2d(512)
            self.combine = nn.Conv2d(512, 256, kernel_size=1, padding=0, stride=1)
            # self.combine = nn.Conv2d(768, 256, kernel_size=1, padding=0, stride=1)
            self.foregap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, features, attentions):
        features = list(features)  # len=5
        # 有传入dict形式的attentions，则将各attention concat到一起。
        device = torch.device("cuda")
        if isinstance(attentions, dict):
            for i in range(len(attentions)):
                if i == 0:
                    atten = attentions[i].unsqueeze(0)
                    atten=atten.to(attentions[i].unsqueeze(0).device)
                else:
                    atten = torch.cat((atten, attentions[i].unsqueeze(0)), dim=0)
                    atten = atten.to(attentions[i].unsqueeze(0).device)
            # attentions = atten.cuda()     # 11.21 252上报错，tensor不在同一个gpu上
            attentions = atten.cuda()

        output = []
        h, w = attentions.shape[2:]
        ncls = attentions.shape[0]  # attentions.shape=[15, 256, 16, 16] ncls, channel, h, w

        val_t = self.value_t(attentions)  # 3*3conv，卷积+降维    support [15, 128, 16, 16] ncls, channel,h, w
        key_t = self.key_t(attentions)  # 3*3conv，卷积+降维    support [15, 32, 16, 16]  ncls, channel,h, w

        # 消融：key是向量
        # key_t=self.abla_val_t(val_t).view(ncls, 128,-1) # [15, 128, 1] 这里怎么写的val_t？不应该对整个feature求池化吗？整个feature第2通道不对应


        # 消融2，feat as value，key vec，novalue
        # key_t = self.abla_val_t(attentions).view(ncls, 256, -1) # [15, 256, 1]

        # 消融3，keyvec，novalue单向
        # key_t = self.abla_val_t(attentions).view(ncls, 256, -1) # [15, 256, 1]


        for idx in range(len(features)):  # 所以idx in range(5) 01234，5个尺度各进行一遍
            feature = features[idx]  # feature.shape [bs per GPU, 256, 13/25/50/100/200, 13/25/50/100/200] 所以是有多尺度的

            # # 前景增强 Foreground Augment
            featuregap=self.foregap(feature)    # featuregap.shape=[bs per GPU, 256, 1, 1] 全局信息
            feature=feature-featuregap

            bs = feature.shape[0]
            H, W = feature.shape[2:]
            # 这里注意，先把多尺度的F统一变成小的，再提取的k和v，不知道多尺度的信息损失多不多。可以尝试先对原尺度的feature提k，v
            # 然后将kv变成attention尺寸。
            feature = F.interpolate(feature, size=(h, w), mode='bilinear', align_corners=True)  # 将feature改为attention尺寸

            val_q = eval('self.value_q' + str(idx))(feature)  # shape [bs per GPU, 128, 16, 16]
            key_q = eval('self.key_q' + str(idx))(feature).view(bs, 32, -1)  #  [bs per GPU, 32, 256]

            # 消融，key是向量
            # key_q=self.abla_val_q(val_q).view(bs,128,-1)

            # 消融2 消融3
            # key_q = self.abla_val_q(feature).view(bs, 256, -1)  # [1, 256, 1]

            for i in range(bs):  # 叠加每类1个的vs*W
                kq = key_q[i].unsqueeze(0).permute(0, 2, 1)  # [1, 256, 32]
                vq = val_q[i].unsqueeze(0)  # [1, 128, 16, 16]

                # kq[1:]=1. * kq[1:] / (torch.norm(kq[1:], 2, keepdim=True).expand_as(x) + 1e-12)
                featurei=feature[i].unsqueeze(0)
                # 文中线性变换的意思就是将最后两维展平,但是说线性变换是可以学习的啊，并没有出现

                # 一维key的消融：
                # kqq=key_q[i]  # [1, 128, 1]
                # p=torch.mul(key_t, kqq)    # [15,128,1]
                # p = F.softmax(p, dim=1)  # softmax归一化    [15, 128, 1]

                # 原：
                p = torch.matmul(kq, key_t.view(ncls, 32, -1))  # key的相似度 1*256*32 * 15*32*256
                p = F.softmax(p, dim=1)  # softmax归一化    [15, 256, 256]

                # support的value和相似度相乘，和value通道数相同 val_t先转为[ncls, 128, 256] 再乘积，
                # 原：
                val_t_out = torch.matmul(val_t.view(ncls, 128, -1), p).view(ncls, 128, h, w)  # [15 128 h w]
                val_q_out = torch.matmul(vq.view(1, 128, -1), p).view(ncls, 128, h, w)  # 1*128*256 * 15*256*256 = 15*128*256 Vq 和 加权图相乘

                # 消融：S仅单向对query的value加权，support的value不变。
                # val_t_out=val_t   # p不和support的key加权，onlyvalue
                # val_q_out = torch.matmul(vq.view(1, 128, -1), p).view(ncls, 128, h, w)  # 1*128*256 * 15*256*256 = 15*128*256 Vq 和 加权图相乘

                # 消融：用原feature替换value
                # val_t_out=torch.matmul(attentions.view(ncls,256,-1),p).view(ncls, 256, h, w)
                # val_q_out = torch.matmul(featurei.view(1, 256, -1), p).view(ncls, 256, h, w)

                # 消融：key为向量，一维S=[15,128,1]
                # val_t_out=torch.mul(val_t.view(ncls,128,-1),p).view(ncls, 128, h,w)   # [15, 128, 16, 16]
                # val_q_out=torch.mul(vq.view(1,128,-1), p).view(ncls,128,h,w)  # [15, 128, 16, 16]

                # 消融2，abla2-2，feat as value，key vec，novalue。最终版
                # kq=key_q[i].unsqueeze(0)    # [1, 256, 1]
                # p=torch.matmul(kq, key_t.view(ncls, 1, -1)) # 1*256*1 * 15*1*256 = 15*256*256
                # p = F.softmax(p, dim=1)  #[15, 256, 256]
                # val_t_out=attentions  # feat as support value   # [15, 256, 16, 16]
                # val_q_out=torch.mul(featurei.view(1, 256, -1), p).view(ncls, 256, h, w)    # [15, 256, 16, 16]

                # # 消融2，abla2-2-new，feat as value，key vec，novalue，不好，放弃！
                # kqq=key_q[i]  # [256, 1]
                # p=torch.mul(key_t, kqq)    # [15,256,1]
                # p = F.softmax(p, dim=1)  # softmax归一化    [15, 256, 1]
                # val_t_out=attentions   # [15, 256, 16, 16]
                # val_q_out=torch.mul(featurei.view(1,256,-1), p).view(ncls,256,h,w)  # [15, 256, 16, 16]

                # # # 消融3：keyvec，novalue单向。
                # kq = key_q[i].unsqueeze(0)  # [1, 256, 1]
                # p = torch.matmul(kq, key_t.view(ncls, 1, -1))  # 1*256*1 * 15*1*256 = 15*256*256
                # p = F.softmax(p, dim=1)  # [15, 256, 256]
                # val_t_out = val_t  # support value   # [15, 128, 16, 16]
                # val_q_out = torch.matmul(vq.view(1, 128, -1), p).view(ncls, 128, h, w)  # [15, 128, 16, 16]

                # # # 消融4：keyvec。
                # kq = key_q[i].unsqueeze(0)  # [1, 256, 1]
                # p = torch.matmul(kq, key_t.view(ncls, 1, -1))  # 1*256*1 * 15*1*256 = 15*256*256
                # p = F.softmax(p, dim=1)  # [15, 256, 256]
                # val_t_out = torch.matmul(val_t.view(ncls, 128, -1), p).view(ncls, 128, h, w)  # support value   # [15, 128, 16, 16]
                # val_q_out = torch.matmul(vq.view(1, 128, -1), p).view(ncls, 128, h, w)  # [15, 128, 16, 16]

                for j in range(ncls):  # 对N个N类的support图像求和
                    if (j == 0):
                        # final_2 = torch.cat((vq,val_t_out[j].unsqueeze(0)),dim=1)   # 仅对support的value加权
                        final_2 = torch.cat((val_q_out[j].unsqueeze(0), val_t_out[j].unsqueeze(0)), dim=1)  # 双向加权
                    else:
                        # final_2 += torch.cat((vq,val_t_out[j].unsqueeze(0)),dim=1)  # 然后叠加，是维度不变只加值吗？那vq岂不是加了N次？还是说类似list的添加操作？
                        final_2 += torch.cat((val_q_out[j].unsqueeze(0), val_t_out[j].unsqueeze(0)),dim=1)  # 然后叠加，维度不变只加值,vq加了N次
                # final_2.shape=[1, 256, 16, 16]
                if (i == 0):  # bs的第一个数据
                    final_1 = final_2
                else:
                    final_1 = torch.cat((final_1, final_2), dim=0)  # 其余数据叠加
            # 插值前，final_1.shape=[1, 256, 16, 16] 插值后 后两维会有：13 25 50 100 200
            final_1 = F.interpolate(final_1, size=(H, W), mode='bilinear', align_corners=True)  # 还原回feature原尺寸
            if self.sum:
                final_1 = eval('self.bnn' + str(idx))(final_1)

            output.append(final_1)

        if self.sum:
            for i in range(len(output)):
                output[i] = self.combine(torch.cat((features[i], output[i]), dim=1))
        output = tuple(output)

        return output
