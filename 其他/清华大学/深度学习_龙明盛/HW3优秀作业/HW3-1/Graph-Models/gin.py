import os.path as osp
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GINConv

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0]


class ApplyNodeFunc(nn.Module):
    """Update the node feature hv with MLP, BN and ReLU."""

    def __init__(self, mlp):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp
        self.bn = nn.BatchNorm1d(self.mlp.output_dim)

    def forward(self, h):
        h = self.mlp(h)
        h = self.bn(h)
        h = F.relu(h)
        return h


class MLP(nn.Module):
    """MLP with linear output"""

    # num_layers:共有多少层
    # input_dim：输入维度
    # hidden_dim：隐藏层维度，所有隐藏层维度都一样
    # hidden_dim：输出维度
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """MLP layers construction
        Paramters
        ---------
        num_layers: int
            The number of linear layers
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction
        """
        super(MLP, self).__init__()
        self.linear_or_not = True  # default is linear model这个时候只有一层MLP
        self.num_layers = num_layers
        self.output_dim = output_dim

        # 层数合法性判断
        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:  # 只有一层则按线性变换来玩，输入就是输出
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:  # 有多层则按下面代码处理
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))  # 第一层比较特殊，输入维度到隐藏层维度
            for layer in range(num_layers - 2):  # 中间隐藏层可以循环来玩，隐藏层维度到隐藏层维度
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))  # 最后一层，隐藏层维度到输出维度

            for layer in range(num_layers - 1):  # 除了最后一层都加BN
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x):  # 前向传播
        if self.linear_or_not:  # 只有单层MLP
            # If linear model
            return self.linear(x)
        else:  # 多层MLP
            # If MLP
            h = x
            for i in range(self.num_layers - 1):  # 除最后一层外都加一个relu
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
            return self.linears[-1](h)  # 最后一层用线性变换把维度转到输出维度


class GIN(nn.Module):

    def __init__(self, input_dim, output_dim, num_layers=4, num_mlp_layers=2, hidden_dim=64,
                 final_dropout=0.5, learn_eps=True):
        """model parameters setting
        Paramters
        ---------
        num_layers: int这个是GIN的层数
            The number of linear layers in the neural network
        num_mlp_layers: intMLP的层数
            The number of linear layers in mlps
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction
        final_dropout: float最后一层的抓爆率
            dropout ratio on the final linear layer
        learn_eps: boolean在学习epsilon参数时是否区分节点本身和邻居节点
            If True, learn epsilon to distinguish center nodes from neighbors
            If False, aggregate neighbors and center nodes altogether.
        """
        super(GIN, self).__init__()
        self.num_layers = num_layers
        self.learn_eps = learn_eps

        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):  # GIN有几层，除了最后一层每层都定义一个MLP（num_mlp_layers层）来进行COMBINE
            if layer == 0:  # 第一层GIN，注意输入维度，
                mlp = MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim)

            # 更新特征的方式是ApplyNodeFunc，邻居汇聚方式为neighbor_pooling_type
            # 具体参考：https://docs.dgl.ai/api/python/nn.pytorch.html#ginconv
            self.ginlayers.append(
                GINConv(ApplyNodeFunc(mlp), 0, self.learn_eps))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Linear function for graph poolings of output of each layer
        # which maps the output of different layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()

        # 以下代码是将每一层点的表征保存下来，然后作为最后的图的表征计算
        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(
                    nn.Linear(input_dim, output_dim))
            else:
                self.linears_prediction.append(
                    nn.Linear(hidden_dim, output_dim))

        self.drop = nn.Dropout(final_dropout)
        self.pool = torch.nn.AdaptiveAvgPool1d(1)


    def forward(self, h, g):  # 前向传播
        # list of hidden representation at each layer (including input)
        hidden_rep = [h]

        for i in range(self.num_layers - 1):  # 根据GIN层数做循环
            h = self.ginlayers[i](h, g)  # 做原文公式4.1的操作
            h = self.batch_norms[i](h)  # 接BN
            h = F.relu(h)  # 接RELU
            hidden_rep.append(h)  # 保存每一层的输出，作为最后图表征的计算

        score_over_layer = 0

        # 根据hidden_rep计算图表征
        # perform pooling over all nodes in each graph in every layer
        for i, h in enumerate(hidden_rep):
            score_over_layer += self.drop(self.linears_prediction[i](h))

        return score_over_layer


device = torch.device('cpu')
model = GIN(dataset.num_features, dataset.num_classes).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)


def train(data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()


@torch.no_grad()
def test(data):
    model.eval()
    out, accs = model(data.x, data.edge_index), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = out[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


acc_curve = []
for epoch in range(1, 101):
    train(data)
    train_acc, val_acc, test_acc = test(data)
    acc_curve.append([train_acc, val_acc, test_acc])
    print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
          f'Test: {test_acc:.4f}')

# show learning curve
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(range(1, 101), acc_curve)
plt.legend(labels=['Train', 'Validation', 'Test'], loc='best')
plt.savefig('./checkpoints/GIN.png')
plt.close()
