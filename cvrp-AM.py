import random
import torch as t
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
import time
import numpy as np
import os
from scipy.stats import ttest_rel

if t.cuda.is_available():
    DEVICE = t.device('cuda')
else:
    DEVICE = t.device('cpu')

t.manual_seed(111)
random.seed(111)
np.random.seed(111)
output_dir = 'cvrp-AM-model'
save_dir = os.path.join(os.getcwd(), output_dir)

embedding_size = 128
city_size = 21  # 节点总数
batch = 512  # 每个batch的算例数
times = 2500  # 训练中每个epoch所需的训练batch数
epochs = 100  # 训练的epoch总数
test2save_times = 20  # 训练过程中每次保存模型所需的测试batch数
min = 100  # 当前已保存的所有模型中测试路径长度的最小值
l = 30  # 车辆的初始容量
M = 8  # 多头注意力中的头数
dk = embedding_size / M  # 多头注意力中每一头的维度
C = 10  # 做softmax得到选取每个点概率前，clip the result所使用的参数
bl_alpha = 0.05  # 做t-检验更新baseline时所设置的阈值
mask_size = t.LongTensor(batch).to(DEVICE)  # 用于标号，方便后面两点间的距离计算
for i in range(batch):
    mask_size[i] = city_size * i

is_train = False  # 是否训练

# 测试
test_times = 100  # 测试时所需的batch总数
test_is_sample = False  # 测试时是否要使用sampling方法，否则使用greedy方法


class act_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(3, embedding_size)  # 用于客户点的坐标加容量需求的embedding
        self.embedding_p = nn.Linear(2, embedding_size)  # 用于仓库点的坐标embedding

        self.wq1 = nn.Linear(embedding_size, embedding_size)
        self.wk1 = nn.Linear(embedding_size, embedding_size)
        self.wv1 = nn.Linear(embedding_size, embedding_size)

        self.w1 = nn.Linear(embedding_size, embedding_size)

        self.wq2 = nn.Linear(embedding_size, embedding_size)
        self.wk2 = nn.Linear(embedding_size, embedding_size)
        self.wv2 = nn.Linear(embedding_size, embedding_size)

        self.w2 = nn.Linear(embedding_size, embedding_size)

        self.wq3 = nn.Linear(embedding_size, embedding_size)
        self.wk3 = nn.Linear(embedding_size, embedding_size)
        self.wv3 = nn.Linear(embedding_size, embedding_size)
        self.w3 = nn.Linear(embedding_size, embedding_size)

        self.wq = nn.Linear(embedding_size * 2 + 1, embedding_size)
        self.wk = nn.Linear(embedding_size, embedding_size)
        self.wv = nn.Linear(embedding_size, embedding_size)
        self.w = nn.Linear(embedding_size, embedding_size)

        self.q = nn.Linear(embedding_size, embedding_size)
        self.k = nn.Linear(embedding_size, embedding_size)

        self.fw1 = nn.Linear(embedding_size, embedding_size * 4)
        self.fb1 = nn.Linear(embedding_size * 4, embedding_size)

        self.fw2 = nn.Linear(embedding_size, embedding_size * 4)
        self.fb2 = nn.Linear(embedding_size * 4, embedding_size)

        self.fw3 = nn.Linear(embedding_size, embedding_size * 4)
        self.fb3 = nn.Linear(embedding_size * 4, embedding_size)

        # Batch Normalization(BN)
        self.BN11 = nn.BatchNorm1d(embedding_size)
        self.BN12 = nn.BatchNorm1d(embedding_size)
        self.BN21 = nn.BatchNorm1d(embedding_size)
        self.BN22 = nn.BatchNorm1d(embedding_size)
        self.BN31 = nn.BatchNorm1d(embedding_size)
        self.BN32 = nn.BatchNorm1d(embedding_size)

    def forward(self, s, d, l, train):  # s坐标，d需求，l初始容量, train==0表示无需计算梯度且使用greedy方法，train>0表示需要计算梯度且使用sampling方法
        # s :[batch x seq_len x 2]
        s1 = t.unsqueeze(s, dim=1)
        s1 = s1.expand(batch, city_size, city_size, 2)
        s2 = t.unsqueeze(s, dim=2)
        s2 = s2.expand(batch, city_size, city_size, 2)
        ss = s1 - s2
        dis = t.norm(ss, 2, dim=3, keepdim=True)  # dis表示任意两点间的距离 (batch, city_size, city_size, 1)

        pro = t.FloatTensor(batch, city_size * 2).to(DEVICE)  # 每个点被选取时的选取概率,将其连乘可得到选取整个路径的概率
        seq = t.LongTensor(batch, city_size * 2).to(DEVICE)  # 选取的序列
        index = t.LongTensor(batch).to(DEVICE)  # 当前车辆所在的点
        tag = t.ones(batch * city_size).to(DEVICE)
        distance = t.zeros(batch).to(DEVICE)  # 总距离
        rest = t.LongTensor(batch, 1, 1).to(DEVICE)  # 车的剩余容量
        dd = t.LongTensor(batch, city_size).to(DEVICE)  # 客户需求
        rest[:, 0, 0] = l
        dd[:, :] = d[:, :, 0]  # 需求
        index[:] = 0
        sss = t.cat([s, d.float()], dim=2)  # [batch x seq_len x 3] 坐标与容量需求拼接

        # node为所有点的的初始embedding
        node = self.embedding(sss)  # 客户点embedding坐标加容量需求
        node[:, 0, :] = self.embedding_p(s[:, 0, :])  # 仓库点只embedding坐标
        x111 = node  # [batch x seq_len x embedding_dim]

        ##################################################################################
        # encoder部分
        #####################################################
        # 第一层MHA
        query1 = self.wq1(node)
        query1 = t.unsqueeze(query1, dim=2)
        query1 = query1.expand(batch, city_size, city_size, embedding_size)
        key1 = self.wk1(node)
        key1 = t.unsqueeze(key1, dim=1)
        key1 = key1.expand(batch, city_size, city_size, embedding_size)
        value1 = self.wv1(node)
        value1 = t.unsqueeze(value1, dim=1)
        value1 = value1.expand(batch, city_size, city_size, embedding_size)
        x = query1 * key1
        x = x.view(batch, city_size, city_size, M, -1)
        x = t.sum(x, dim=4)  # u=q^T x k
        x = x / (dk ** 0.5)
        x = F.softmax(x, dim=2)
        x = t.unsqueeze(x, dim=4)
        x = x.expand(batch, city_size, city_size, M, 16)
        x = x.contiguous()
        x = x.view(batch, city_size, city_size, -1)
        x = x * value1
        x = t.sum(x, dim=2)  # MHA :(batch, city_size, embedding_size)
        x = self.w1(x)  # 得到一层MHA的结果

        x = x + x111
        #####################
        # 第一层第一个BN
        x = x.permute(0, 2, 1)
        x = self.BN11(x)
        x = x.permute(0, 2, 1)
        # x = t.tanh(x)
        #####################
        # 第一层FF
        x1 = self.fw1(x)
        x1 = F.relu(x1)
        x1 = self.fb1(x1)

        x = x + x1
        #####################
        # 第一层第二个BN
        x = x.permute(0, 2, 1)
        x = self.BN12(x)
        x = x.permute(0, 2, 1)
        x1 = x  # h_i^(l) n=1
        #####################################################
        # 第二层MHA
        query2 = self.wq2(x)
        query2 = t.unsqueeze(query2, dim=2)
        query2 = query2.expand(batch, city_size, city_size, embedding_size)
        key2 = self.wk2(x)
        key2 = t.unsqueeze(key2, dim=1)
        key2 = key2.expand(batch, city_size, city_size, embedding_size)
        value2 = self.wv2(x)
        value2 = t.unsqueeze(value2, dim=1)
        value2 = value2.expand(batch, city_size, city_size, embedding_size)
        x = query2 * key2
        x = x.view(batch, city_size, city_size, M, -1)
        x = t.sum(x, dim=4)
        x = x / (dk ** 0.5)
        x = F.softmax(x, dim=2)
        x = t.unsqueeze(x, dim=4)
        x = x.expand(batch, city_size, city_size, M, 16)
        x = x.contiguous()
        x = x.view(batch, city_size, city_size, -1)
        x = x * value2
        x = t.sum(x, dim=2)
        x = self.w2(x)

        x = x + x1
        #####################
        # 第二层第一个BN
        x = x.permute(0, 2, 1)
        x = self.BN21(x)
        x = x.permute(0, 2, 1)

        #####################
        # 第二层FF
        x1 = self.fw2(x)
        x1 = F.relu(x1)
        x1 = self.fb2(x1)
        x = x + x1
        #####################
        # 第二层第二个BN
        x = x.permute(0, 2, 1)
        x = self.BN22(x)
        x = x.permute(0, 2, 1)

        x1 = x  # h_i^(l) n=2
        #####################################################
        # 第三层MHA
        query3 = self.wq3(x)
        query3 = t.unsqueeze(query3, dim=2)
        query3 = query3.expand(batch, city_size, city_size, embedding_size)
        key3 = self.wk3(x)
        key3 = t.unsqueeze(key3, dim=1)
        key3 = key3.expand(batch, city_size, city_size, embedding_size)
        value3 = self.wv3(x)
        value3 = t.unsqueeze(value3, dim=1)
        value3 = value3.expand(batch, city_size, city_size, embedding_size)
        x = query3 * key3
        x = x.view(batch, city_size, city_size, M, -1)
        x = t.sum(x, dim=4)
        x = x / (dk ** 0.5)
        x = F.softmax(x, dim=2)
        x = t.unsqueeze(x, dim=4)
        x = x.expand(batch, city_size, city_size, M, 16)
        x = x.contiguous()
        x = x.view(batch, city_size, city_size, -1)
        x = x * value3
        x = t.sum(x, dim=2)
        x = self.w3(x)

        x = x + x1
        #####################
        # 第三层第一个BN
        x = x.permute(0, 2, 1)
        x = self.BN31(x)
        x = x.permute(0, 2, 1)
        #####################
        # 第三层FF
        x1 = self.fw3(x)
        x1 = F.relu(x1)
        x1 = self.fb3(x1)
        x = x + x1
        #####################
        # 第三层第二个BN
        x = x.permute(0, 2, 1)
        x = self.BN32(x)
        x = x.permute(0, 2, 1)  # h_i^(l) n=3 (batch, city_size, embedding_size)
        x = x.contiguous()
        avg = t.mean(x, dim=1)  # 最后将所有节点的嵌入信息取平均得到整个图的嵌入信息，(batch, embedding_size)

        ##################################################################################
        # decoder部分
        for i in range(city_size * 2):  # decoder输出序列的长度不超过city_size * 2
            flag = t.sum(dd, dim=1)  # dd:(batch, city_size)
            f1 = t.nonzero(flag > 0).view(-1)  # 取得需求不全为0的batch号
            f2 = t.nonzero(flag == 0).view(-1)  # 取得需求全为0的batch号

            if f1.size()[0] == 0:  # batch所有需求均为0
                pro[:, i:] = 1  # pro:(batch, city_size*2)
                seq[:, i:] = 0  # swq:(batch, city_size*2)
                temp = dis.view(-1, city_size, 1)[
                    index + mask_size]  # dis:任意两点间的距离 (batch, city_size, city_size, 1) temp:(batch, city_size,1)
                distance = distance + temp.view(-1)[mask_size]  # 加上当前点到仓库的距离
                break

            ind = index + mask_size
            tag[ind] = 0  # tag:(batch*city_size)
            start = x.view(-1, embedding_size)[ind]  # (batch, embedding_size)，每个batch中选出一个节点

            end = rest[:, :, 0]  # (batch, 1)
            end = end.float()  # 车上剩余容量

            graph = t.cat([avg, start, end], dim=1)  # 结合图embedding，当前点embedding，车剩余容量: (batch,embedding_size*2 + 1)_
            query = self.wq(graph)  # (batch, embedding_size)
            query = t.unsqueeze(query, dim=1)
            query = query.expand(batch, city_size, embedding_size)
            key = self.wk(x)
            value = self.wv(x)
            temp = query * key
            temp = temp.view(batch, city_size, M, -1)
            temp = t.sum(temp, dim=3)  # (batch, city_size, M)
            temp = temp / (dk ** 0.5)

            mask = tag.view(batch, -1, 1) < 0.5  # 访问过的点tag=0
            mask1 = dd.view(batch, city_size, 1) > rest.expand(batch, city_size, 1)  # 客户需求大于车剩余容量的点

            flag = t.nonzero(index).view(-1)  # 在batch中取得当前车不在仓库点的batch号
            mask = mask + mask1  # mask:(batch x city_size x 1)
            mask = mask > 0
            mask[f2, 0, 0] = 0  # 需求全为0则使车一直在仓库
            if flag.size()[0] > 0:  # 将有车不在仓库的batch的仓库点开放
                mask[flag, 0, 0] = 0

            mask = mask.expand(batch, city_size, M)
            temp.masked_fill_(mask, -float('inf'))
            temp = F.softmax(temp, dim=1)
            temp = t.unsqueeze(temp, dim=3)
            temp = temp.expand(batch, city_size, M, 16)
            temp = temp.contiguous()
            temp = temp.view(batch, city_size, -1)
            temp = temp * value
            temp = t.sum(temp, dim=1)
            temp = self.w(temp)  # hc,(batch,embedding_size)

            query = self.q(temp)
            key = self.k(x)  # (batch, city_size, embedding_size)
            query = t.unsqueeze(query, dim=1)  # (batch, 1 ,embedding_size)
            query = query.expand(batch, city_size, embedding_size)  # (batch, city_size, embedding_size)
            temp = query * key
            temp = t.sum(temp, dim=2)
            temp = temp / (dk ** 0.5)
            temp = t.tanh(temp) * C  # (batch, city_size)

            mask = mask[:, :, 0]
            temp.masked_fill_(mask, -float('inf'))
            p = F.softmax(temp, dim=1)  # 得到选取每个点时所有点可能被选择的概率

            indexx = t.LongTensor(batch).to(DEVICE)
            if train != 0:
                indexx[f1] = t.multinomial(p[f1], 1)[:, 0]  # 按sampling策略选点
            else:
                indexx[f1] = (t.max(p[f1], dim=1)[1])  # 按greedy策略选点

            indexx[f2] = 0
            p = p.view(-1)
            pro[:, i] = p[indexx + mask_size]
            pro[f2, i] = 1
            rest = rest - (dd.view(-1)[indexx + mask_size]).view(batch, 1, 1)  # 车的剩余容量
            dd = dd.view(-1)
            dd[indexx + mask_size] = 0
            dd = dd.view(batch, city_size)

            temp = dis.view(-1, city_size, 1)[index + mask_size]
            distance = distance + temp.view(-1)[indexx + mask_size]

            mask3 = indexx == 0
            mask3 = mask3.view(batch, 1, 1)
            rest.masked_fill_(mask3, l)  # 车回到仓库将容量设为初始值

            index = indexx
            seq[:, i] = index[:]

        if train == 0:
            seq = seq.detach()
            pro = pro.detach()
            distance = distance.detach()

        return seq, pro, distance  # 被选取的点序列,每个点被选取时的选取概率,这些序列的总路径长度


net1 = act_net()
net1 = net1.to(DEVICE)
net1.load_state_dict(t.load('cvrp-AM-model/AM_VRP20.pt'))
net2 = act_net()
net2 = net2.to(DEVICE)
net2.load_state_dict(net1.state_dict())

# 训练部分
if is_train is True:
    opt = optim.Adam(net1.parameters(), 0.0001)

    tS = t.rand(batch * test2save_times, city_size, 2)  # 坐标0~1之间
    tD = np.random.randint(1, 10, size=(batch * test2save_times, city_size, 1))  # 所有客户的需求
    tD = t.LongTensor(tD)
    tD[:, 0, 0] = 0  # 仓库点的需求为0

    S = t.rand(batch * times, city_size, 2)
    D = np.random.randint(1, 10, size=(batch * times, city_size, 1))  # 所有客户的需求
    D = t.LongTensor(D)
    D[:, 0, 0] = 0  # 仓库点的需求为0

    for epoch in range(epochs):
        for i in range(times):
            t.cuda.empty_cache()
            s = S[i * batch: (i + 1) * batch]  # [batch x seq_len x 2]
            d = D[i * batch: (i + 1) * batch]  # [batch x seq_len x 1]
            s = s.to(DEVICE)
            d = d.to(DEVICE)

            t1 = time.time()
            seq2, pro2, dis2 = net2(s, d, l, 0)  # baseline return seq, pro, distance
            seq1, pro1, dis1 = net1(s, d, l, 2)
            t2 = time.time()
            # print('nn_output_time={}'.format(t2 - t1))
            ###################################################################
            # 带baseline的策略梯度训练算法,dis2作为baseline
            pro = t.log(pro1)
            loss = t.sum(pro, dim=1)
            score = dis1 - dis2  # advantage reward(优势函数)

            score = score.detach()
            loss = score * loss
            loss = t.sum(loss) / batch  # 最终损失函数

            opt.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(net1.parameters(), 1)
            opt.step()
            print('epoch={},i={},mean_dis1={},mean_dis2={}'.format(epoch, i, t.mean(dis1), t.mean(
                dis2)))  # ,'disloss:',t.mean((dis1-dis2)*(dis1-dis2)), t.mean(t.abs(dis1-dis2)), nan)

            # OneSidedPairedTTest(做t-检验看当前Sampling的解效果是否显著好于greedy的解效果,如果是则更新使用greedy策略作为baseline的net2参数)
            if (dis1.mean() - dis2.mean()) < 0:
                tt, pp = ttest_rel(dis1.cpu().numpy(), dis2.cpu().numpy())
                p_val = pp / 2
                assert tt < 0, "T-statistic should be negative"
                if p_val < bl_alpha:
                    print('Update baseline')
                    net2.load_state_dict(net1.state_dict())

            # 每隔xxx步做测试判断结果有没有改进，如果改进了则把当前模型保存下来
            if (i + 1) % 100 == 0:
                length = t.zeros(1).to(DEVICE)
                for j in range(test2save_times):
                    t.cuda.empty_cache()
                    s = tS[j * batch: (j + 1) * batch]
                    d = tD[j * batch: (j + 1) * batch]
                    s = s.to(DEVICE)
                    d = d.to(DEVICE)
                    seq, pro, dis = net1(s, d, l, 0)
                    length = length + t.mean(dis)
                length = length / test2save_times
                if length < min:
                    t.save(net1.state_dict(), os.path.join(save_dir,
                                                           'epoch{}-i{}-dis_{:.5f}.pt'.format(
                                                               epoch, i, length.item())))
                    min = length
                print('min=', min.item(), 'length=', length.item())
# 测试部分
else:
    # 按照greedy策略测试
    if test_is_sample is False:
        tS = t.rand(batch * test_times, city_size, 2)  # 坐标0~1之间
        tD = np.random.randint(1, 10, size=(batch * test_times, city_size, 1))  # 所有客户的需求
        tD = t.LongTensor(tD)
        tD[:, 0, 0] = 0  # 仓库点的需求为0
        sum_dis = t.zeros(1).to(DEVICE)

        sum_clock = 0  # 记录生成解的总时间
        for i in range(test_times):
            t.cuda.empty_cache()
            s = tS[i * batch: (i + 1) * batch]
            d = tD[i * batch: (i + 1) * batch]
            s = s.to(DEVICE)
            d = d.to(DEVICE)
            clock1 = time.time()
            seq, pro, dis = net1(s, d, l, 0)
            clock2 = time.time()
            deta_clock = clock2 - clock1
            sum_clock = sum_clock + deta_clock

            print("i:{},dis:{},deta_clock:{}".format(i, t.mean(dis), deta_clock))
            sum_dis = sum_dis + t.mean(dis)
        mean_dis = sum_dis / test_times
        mean_clock = sum_clock / test_times
        print("mean_dis:{},mean_clock:{}".format(mean_dis, mean_clock))

    # 按照sampling策略测试
    else:
        tS = t.rand(test_times, city_size, 2)  # 坐标0~1之间
        tD = np.random.randint(1, 10, size=(test_times, city_size, 1))  # 所有客户的需求
        tD = t.LongTensor(tD)
        tD[:, 0, 0] = 0  # 仓库点的需求为0

        all_repeat_size = 1280
        num_batch_repeat = all_repeat_size // batch

        sum_dis = t.zeros(1).to(DEVICE)
        sum_clock = 0  # 记录生成解的总时间
        for i in range(test_times):
            t.cuda.empty_cache()
            available_seq = []
            available_dis = []
            deta_clock = 0
            for _ in range(num_batch_repeat):
                s = tS[i].repeat(batch, 1, 1).to(DEVICE)
                d = tD[i].repeat(batch, 1, 1).to(DEVICE)

                clock1 = time.time()
                seq, pro, dis = net1(s, d, l, 1)
                clock2 = time.time()

                mini_deta_clock = clock2 - clock1
                deta_clock = deta_clock + mini_deta_clock
                for j in range(batch):
                    available_seq.append(seq[j])
                    available_dis.append(dis[j])

            available_seq = t.stack(available_seq)
            available_dis = t.stack(available_dis)

            mindis, mindis_index = t.min(available_dis, 0)
            sum_dis = sum_dis + mindis
            sum_clock = sum_clock + deta_clock
            print("i:{},mindis:{},deta_clock:{}".format(i, mindis, deta_clock))
        mean_dis = sum_dis / test_times
        mean_clock = sum_clock / test_times
        print("mean_dis:{},mean_clock:{}".format(mean_dis.item(), mean_clock))
