"""开发脚本：torch + PyG 参考实现（与作者 model.py 结构一致，供 numpy 实现对拍）。

只用于开发校验，不随插件分发。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, global_mean_pool

from flavor_feats import EDGE_DIM, NODE_DIM, PHARMA_DIM

GAT_HEADS = (8, 8, 4, 4, 2)


class ReferenceOdorModel(nn.Module):
    """与作者 `IntegratedOdorModel` 完全一致（edge 维度可配置，checkpoint 为 4）。"""

    def __init__(self, hidden_dim=128, pharma_dim=PHARMA_DIM, dropout=0.3, num_classes=138, edge_dim=EDGE_DIM):
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(NODE_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.gnn = nn.ModuleList(
            [
                GATv2Conv(hidden_dim, hidden_dim, edge_dim=hidden_dim, heads=h, concat=False, dropout=dropout)
                for h in GAT_HEADS
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in GAT_HEADS])
        self.pooling = global_mean_pool
        self.pharma_encoder = nn.Sequential(
            nn.Linear(pharma_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, data):
        x = self.node_encoder(data.x)
        edge_attr = self.edge_encoder(data.edge_attr)
        for i, conv in enumerate(self.gnn):
            x_res = conv(x, data.edge_index, edge_attr)
            x = x + x_res
            x = self.norms[i](x)
            x = F.relu(x)
        graph_repr = self.pooling(x, data.batch)
        pharma = data.pharma_features.view(graph_repr.size(0), -1)
        pharma_repr = self.pharma_encoder(pharma)
        return self.classifier(torch.cat([graph_repr, pharma_repr], dim=-1))


def load_reference(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]
    num_classes = state["classifier.12.weight"].shape[0]
    edge_dim = state["edge_encoder.0.weight"].shape[1]
    model = ReferenceOdorModel(num_classes=num_classes, edge_dim=edge_dim)
    model.load_state_dict(state)
    model.eval()
    return model


def torch_logits(model, x, edge_index, edge_attr, pharma):
    data = Data(
        x=torch.as_tensor(x, dtype=torch.float32),
        edge_index=torch.as_tensor(edge_index, dtype=torch.long),
        edge_attr=torch.as_tensor(edge_attr, dtype=torch.float32),
        pharma_features=torch.as_tensor(pharma, dtype=torch.float32),
        batch=torch.zeros(x.shape[0], dtype=torch.long),
    )
    with torch.no_grad():
        return model(data).numpy().reshape(-1)
