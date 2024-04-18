import os.path as osp

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.nn import ReLU
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import DeepGCNLayer, GENConv, LayerNorm, Linear

dataset = "Cora"
path = osp.join(osp.dirname(osp.realpath(__file__)), "data", dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0]


class DeeperGCN(torch.nn.Module):
    """Adapted from [an official example](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/ogbn_proteins_deepgcn.py#L32)."""

    def __init__(self, hidden_channels, num_layers, dropout=0.1):
        super().__init__()

        self.node_encoder = Linear(data.x.size(-1), hidden_channels)

        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = GENConv(
                hidden_channels,
                hidden_channels,
                aggr="softmax",
                t=1.0,
                learn_t=True,
                msg_norm=True,
                learn_msg_scale=True,
                num_layers=2,
                norm="layer",
            )
            norm = LayerNorm(hidden_channels, affine=True, mode="node")
            act = ReLU(inplace=True)

            layer = DeepGCNLayer(
                conv, norm, act, block="res+", dropout=dropout, ckpt_grad=i % 3
            )
            self.layers.append(layer)

        self.lin = Linear(hidden_channels, data.y.size(-1))
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.node_encoder(x)

        x = self.layers[0].conv(x, edge_index)

        for layer in self.layers[1:]:
            x = layer(x, edge_index)

        x = self.layers[0].act(self.layers[0].norm(x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        return self.lin(x)


# device = torch.device('cpu')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeeperGCN(hidden_channels=64, num_layers=14, dropout=0.5).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


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
    for _, mask in data("train_mask", "val_mask", "test_mask"):
        pred = out[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


for epoch in range(1, 201):
    train(data)
    train_acc, val_acc, test_acc = test(data)
    print(
        f"Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, "
        f"Test: {test_acc:.4f}"
    )
