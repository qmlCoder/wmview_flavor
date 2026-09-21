"""PharmaGNN 推理的纯 numpy 实现（运行时不需要 PyTorch / PyG）。

逐算子复刻训练时使用的 `IntegratedOdorModel`：
node/edge 编码器 → 5 层 GATv2Conv（8/8/4/4/2 头，残差 + LayerNorm + ReLU）
→ 全局平均池化 → 药效团 2 层 MLP → 拼接 → 4 层 MLP 分类头 → sigmoid。

GATv2Conv 的注意力、自环（edge_attr 以“入边均值”填充）与聚合顺序与
`torch_geometric.nn.GATv2Conv(concat=False, add_self_loops=True)` 一致，
数值一致性由 `validate_numpy.py` 与 torch 参考实现对拍验证。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

NEGATIVE_SLOPE = 0.2
LAYER_NORM_EPS = 1e-5
GAT_HEADS = (8, 8, 4, 4, 2)
GAT_LAYERS = len(GAT_HEADS)


# ============================ 基础算子 ============================


def linear(x, weight, bias=None):
    y = x @ weight.T
    return y if bias is None else y + bias


def layer_norm(x, weight, bias):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + LAYER_NORM_EPS) * weight + bias


def relu(x):
    return np.maximum(x, 0.0)


def leaky_relu(x):
    return np.where(x > 0, x, x * NEGATIVE_SLOPE)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def scatter_sum(src, index, num_nodes):
    out = np.zeros((num_nodes, *src.shape[1:]), dtype=src.dtype)
    if index.size:
        np.add.at(out, index, src)
    return out


def scatter_mean(src, index, num_nodes):
    """按 index 求均值；空组返回 0（与 PyG scatter(reduce='mean') 一致）。"""
    out = scatter_sum(src, index, num_nodes)
    count = np.zeros(num_nodes, dtype=np.int64)
    if index.size:
        np.add.at(count, index, 1)
    return out / np.maximum(count, 1).reshape(-1, *([1] * (src.ndim - 1)))


def softmax_by_index(alpha, index, num_nodes):
    """按 index 分组做 softmax（PyG `utils.softmax`）。alpha: (E,) 或 (E, H)。"""
    out = np.empty_like(alpha)
    flat = alpha.reshape(alpha.shape[0], -1)
    res = out.reshape(alpha.shape[0], -1)
    for g in range(flat.shape[1]):
        col = flat[:, g]
        row_max = np.full(num_nodes, -np.inf)
        np.maximum.at(row_max, index, col)
        shifted = np.exp(col - row_max[index])
        denom = np.zeros(num_nodes)
        np.add.at(denom, index, shifted)
        res[:, g] = shifted / denom[index]
    return out


# ============================ GATv2Conv ============================


def gatv2_conv(x, edge_index, edge_attr, params, heads, out_channels):
    """与 PyG GATv2Conv(concat=False, edge_dim=..., add_self_loops=True) 等价。"""
    num_nodes = x.shape[0]
    h, c = heads, out_channels

    x_l = linear(x, params["lin_l.weight"], params.get("lin_l.bias")).reshape(num_nodes, h, c)
    x_r = linear(x, params["lin_r.weight"], params.get("lin_r.bias")).reshape(num_nodes, h, c)

    src, dst = edge_index[0], edge_index[1]
    attr = edge_attr
    if attr.size:  # 去掉自环后按“入边均值”补自环（PyG fill_value='mean'）
        keep = src != dst
        src, dst, attr = src[keep], dst[keep], attr[keep]
    loop_attr = (
        scatter_mean(attr, dst, num_nodes)
        if attr.shape[0]
        else np.zeros((num_nodes, attr.shape[1] if attr.ndim == 2 else 0), dtype=np.float32)
    )
    loop_src = loop_dst = np.arange(num_nodes, dtype=np.int64)
    src = np.concatenate([src, loop_src])
    dst = np.concatenate([dst, loop_dst])
    attr = np.concatenate([attr, loop_attr], axis=0)

    alpha = x_l[src] + x_r[dst]
    if "lin_edge.weight" in params:
        alpha = alpha + linear(attr, params["lin_edge.weight"]).reshape(-1, h, c)
    alpha = leaky_relu(alpha)
    alpha = (alpha * params["att"]).sum(axis=-1)  # (E, H)
    alpha = softmax_by_index(alpha, dst, num_nodes)

    out = scatter_sum(x_l[src] * alpha[..., None], dst, num_nodes)  # (N, H, C)
    out = out.mean(axis=1)  # concat=False
    if params.get("bias") is not None:
        out = out + params["bias"]
    return out


# ============================ 模型 ============================


class FlavorModel:
    """加载 npz 权重的 PharmaGNN 推理器（单分子）。"""

    def __init__(self, weights_path):
        data = np.load(Path(weights_path), allow_pickle=False)
        self.w = {k: data[k] for k in data.files}
        self.num_classes = self.w["classifier.12.weight"].shape[0]

    def _mlp(self, x, prefix, layers):
        """layers: 形如 [(0, 'linear'), (1, 'ln'), (2, 'relu'), (3, 'drop'), ...] 的算子序列。"""
        for idx, kind in layers:
            if kind == "linear":
                x = linear(x, self.w[f"{prefix}.{idx}.weight"], self.w[f"{prefix}.{idx}.bias"])
            elif kind == "ln":
                x = layer_norm(x, self.w[f"{prefix}.{idx}.weight"], self.w[f"{prefix}.{idx}.bias"])
            elif kind == "relu":
                x = relu(x)
            # dropout 在推理时是恒等映射
        return x

    def logits(self, x, edge_index, edge_attr, pharma):
        w = self.w
        x = self._mlp(x, "node_encoder", [(0, "linear"), (1, "ln"), (2, "relu")])
        edge_attr = self._mlp(edge_attr, "edge_encoder", [(0, "linear"), (1, "ln"), (2, "relu")])

        for i, heads in enumerate(GAT_HEADS):
            prefix = f"gnn.{i}"
            params = {
                "lin_l.weight": w[f"{prefix}.lin_l.weight"],
                "lin_l.bias": w.get(f"{prefix}.lin_l.bias"),
                "lin_r.weight": w[f"{prefix}.lin_r.weight"],
                "lin_r.bias": w.get(f"{prefix}.lin_r.bias"),
                "lin_edge.weight": w.get(f"{prefix}.lin_edge.weight"),
                "att": w[f"{prefix}.att"],
                "bias": w.get(f"{prefix}.bias"),
            }
            res = gatv2_conv(x, edge_index, edge_attr, params, heads, x.shape[1])
            x = x + res
            x = relu(layer_norm(x, w[f"norms.{i}.weight"], w[f"norms.{i}.bias"]))

        graph_repr = x.mean(axis=0, keepdims=True)  # global_mean_pool（单分子）
        pharma_repr = self._mlp(
            pharma.reshape(1, -1),
            "pharma_encoder",
            [(0, "linear"), (1, "ln"), (2, "relu"), (4, "linear"), (5, "ln"), (6, "relu")],
        )
        fused = np.concatenate([graph_repr, pharma_repr], axis=-1)
        return self._mlp(
            fused,
            "classifier",
            [
                (0, "linear"),
                (1, "ln"),
                (2, "relu"),
                (4, "linear"),
                (5, "ln"),
                (6, "relu"),
                (8, "linear"),
                (9, "ln"),
                (10, "relu"),
                (12, "linear"),
            ],
        ).reshape(-1)

    def predict_proba(self, x, edge_index, edge_attr, pharma) -> np.ndarray:
        return sigmoid(self.logits(x, edge_index, edge_attr, pharma))
