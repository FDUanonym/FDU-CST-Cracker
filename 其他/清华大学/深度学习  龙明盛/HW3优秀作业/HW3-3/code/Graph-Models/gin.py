import os.path as osp

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GINConv, MLP
import wandb

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0]

wandb.init(project='dl-hw3', name='GIN')

class GIN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mlp1 = MLP([in_channels, 32, 32], norm='batch_norm')
        mlp2 = MLP([32, out_channels, out_channels], norm='batch_norm')
        self.conv1 = GINConv(mlp1)
        self.conv2 = GINConv(mlp2)

    def forward(self, x, edge_index, edge_weight):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x


device = torch.device('cuda')
model = GIN(dataset.num_features, dataset.num_classes).to(device)
data = data.to(device)
# optimizer = torch.optim.Adam([
    # dict(params=model.conv1.parameters(), weight_decay=1e-3),
    # dict(params=model.conv2.parameters(), weight_decay=1e-3)
# ], lr=0.1)

optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=5e-4)

def train(data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_attr)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test(data):
    model.eval()
    out, accs = model(data.x, data.edge_index, data.edge_attr), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = out[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


for epoch in range(1, 201):
    loss = train(data)
    train_acc, val_acc, test_acc = test(data)
    print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
          f'Test: {test_acc:.4f}')

    wandb.log({'epoch': epoch, 'loss': loss, 'train_acc': train_acc, 'val_acc': val_acc, 
          'test_acc': test_acc})