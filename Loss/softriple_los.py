# Implementation of SoftTriple Loss
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init


class SoftTripleLoss(nn.Module):
    """
    SoftTriple Loss 类，用于度量学习任务。
    """
    def __init__(self, la, gamma, tau, margin, dim, cN, K):
        super(SoftTripleLoss, self).__init__()
        """
        初始化 SoftTriple 损失函数。
        
        参数:
        la -- 分类损失的权重。
        gamma -- 相似度缩放因子。
        tau -- 正则化项的权重。
        margin -- 分类损失的边界值。
        dim -- 输入特征的维度。
        cN -- 类别的数量。
        K -- 每个类别的原型数量。
        """
        self.la = la
        self.gamma = 1./gamma
        self.tau = tau
        self.margin = margin
        self.cN = cN
        self.K = K
        # 定义类别中心参数
        self.fc = Parameter(torch.Tensor(dim, cN*K))
        # 初始化权重矩阵，用于定义类别内部的原型之间的关系
        self.weight = torch.zeros(cN*K, cN*K, dtype=torch.bool).cuda()
        for i in range(0, cN):
            for j in range(0, K):
                self.weight[i*K+j, i*K+j+1:(i+1)*K] = 1
        # 使用 Kaiming 初始化方法初始化类别中心参数
        init.kaiming_uniform_(self.fc, a=math.sqrt(5))
        return


    def forward(self, input, target):
        """
        参数:
        input -- 输入特征。
        target -- 目标标签。
        """
        # 将类别中心向量归一化
        centers = F.normalize(self.fc, p=2, dim=0)
        # 计算输入特征与类别中心之间的相似度
        simInd = input.matmul(centers)
        # 重塑相似度矩阵，以适应后续计算
        simStruc = simInd.reshape(-1, self.cN, self.K)
        # 计算每个输入样本属于每个类别的每个原型的概率
        prob = F.softmax(simStruc*self.gamma, dim=2)
        # 计算最终的类别相似度
        simClass = torch.sum(prob*simStruc, dim=2)
        # 为分类损失创建一个边界矩阵
        marginM = torch.zeros(simClass.shape).cuda()
        marginM[torch.arange(0, marginM.shape[0]), target] = self.margin
        # 计算分类损失
        lossClassify = F.cross_entropy(self.la*(simClass-marginM), target)
        # 如果启用正则化，计算正则化项并将其加到分类损失上
        if self.tau > 0 and self.K > 1:
            simCenter = centers.t().matmul(centers)
            reg = torch.sum(torch.sqrt(2.0+1e-5-2.*simCenter[self.weight]))/(self.cN*self.K*(self.K-1.))
            return lossClassify+self.tau*reg
        else:
            return lossClassify
        
if __name__ == "__main__":
    s = SoftTripleLoss(20, 0.1, 0.2, 0.01, 1, 98, 10).to('cuda')
    x = torch.randn(32,1).to('cuda').long().float()
    y = torch.randn(32,1).to('cuda').long().float()
    m = s(x,y)