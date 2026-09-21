"""开发脚本：校验 numpy 推理实现。

1) 与 torch + PyG 参考实现逐分子对拍 logits（应达到 1e-4 以内）；
2) 用数据集算 macro-AUC，与 checkpoint 记录的 val_auc 比对（校验标签顺序与特征口径）。

用法（在装了 torch 的开发环境里执行）：
    python py/validate_numpy.py --split val --n 8          # 对拍 + AUC
    python py/validate_numpy.py --split val --n 0          # 只算 AUC
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from flavor_feats import PharmacophoreExtractor, graph_arrays, graph_mol  # noqa: E402
from flavor_np import FlavorModel  # noqa: E402

SPLITS = {
    "train": ("flavor/data1/train.csv", "flavor/data1/train-sdf"),
    "val": ("flavor/data1/val.csv", "flavor/data1/val-sdf"),
    "test": ("flavor/data1/test.csv", "flavor/data1/test-sdf"),
}


def load_split(split: str):
    csv_path, sdf_dir = SPLITS[split]
    csv_path, sdf_dir = ROOT / csv_path, ROOT / sdf_dir
    with csv_path.open(encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header = rows[0]
    label_names = header[1:]
    samples = []
    for i, row in enumerate(rows[1:]):
        sdf = sdf_dir / f"flavor_{i}.sdf"
        if not sdf.exists():
            continue
        mol = Chem.SDMolSupplier(str(sdf), removeHs=False)[0]
        if mol is None:
            continue
        samples.append((i, row[0], np.asarray([float(v) for v in row[1:]], dtype=np.float32), mol))
    return label_names, samples


def average_ranks(values: np.ndarray) -> np.ndarray:
    """平均秩（并列取平均），等价 scipy.stats.rankdata。"""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_vals = values[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    pos, neg = y_true == 1, y_true == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = average_ranks(y_score)
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val", choices=sorted(SPLITS))
    parser.add_argument("--n", type=int, default=8, help="与 torch 对拍的分子数（0 表示只算 AUC）")
    parser.add_argument("--torch-auc", action="store_true", help="同时用 torch 参考实现算 AUC")
    args = parser.parse_args()

    label_names, samples = load_split(args.split)
    print(f"数据集 {args.split}: {len(samples)} 个分子，{len(label_names)} 个标签")

    extractor = PharmacophoreExtractor()
    model = FlavorModel(ROOT / "flavor" / "pharma33_weights.npz")
    print(f"模型输出维度: {model.num_classes}")

    torch_model = None
    if args.n > 0 or args.torch_auc:
        from reference_torch import load_reference, torch_logits

        torch_model = load_reference(ROOT / "flavor" / "best_model_pharma33.pt")

    # ---------- 1) numpy vs torch 逐分子对拍 ----------
    if args.n > 0:
        worst = 0.0
        for idx, smiles, _, mol in samples[: args.n]:
            pharma = extractor.extract(mol)
            x, ei, ea, ph = graph_arrays(graph_mol(mol), pharma)
            np_logits = model.logits(x, ei, ea, ph)
            th_logits = torch_logits(torch_model, x, ei, ea, ph)
            diff = float(np.max(np.abs(np_logits - th_logits)))
            worst = max(worst, diff)
            print(f"  sample {idx:5d} atoms={x.shape[0]:3d} max|Δlogit|={diff:.3e}  smiles={smiles[:40]}")
        print(f"最大偏差: {worst:.3e}  ->  {'一致' if worst < 1e-4 else '不一致，需排查'}")

    # ---------- 2) AUC ----------
    y_true, y_score = [], []
    for idx, smiles, labels, mol in samples:
        pharma = extractor.extract(mol)
        x, ei, ea, ph = graph_arrays(graph_mol(mol), pharma)
        y_true.append(labels)
        y_score.append(model.predict_proba(x, ei, ea, ph))
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    per_label = [roc_auc(y_true[:, i], y_score[:, i]) for i in range(len(label_names))]
    valid = np.asarray([v for v in per_label if not np.isnan(v)])
    print(f"numpy  macro-AUC = {valid.mean():.4f}   （{len(valid)}/{len(label_names)} 个标签参与）")
    best = np.argsort([-v if not np.isnan(v) else 1.0 for v in per_label])[:15]
    print("  Top15 标签 AUC: " + ", ".join(f"{label_names[i]}={per_label[i]:.3f}" for i in best))

    # macro/micro 汇总（micro：把标签展平后整体算）
    print(f"numpy  micro-AUC = {roc_auc(y_true.reshape(-1), y_score.reshape(-1)):.4f}")

    if torch_model is not None:
        th_score = []
        for idx, smiles, labels, mol in samples:
            pharma = extractor.extract(mol)
            x, ei, ea, ph = graph_arrays(graph_mol(mol), pharma)
            logits = torch_logits(torch_model, x, ei, ea, ph)
            th_score.append(1.0 / (1.0 + np.exp(-logits)))
        th_score = np.asarray(th_score)
        th_per_label = [roc_auc(y_true[:, i], th_score[:, i]) for i in range(len(label_names))]
        th_valid = np.asarray([v for v in th_per_label if not np.isnan(v)])
        print(f"torch  macro-AUC = {th_valid.mean():.4f}")
        print(f"numpy vs torch 概率最大偏差 = {np.max(np.abs(y_score - th_score)):.3e}")


if __name__ == "__main__":
    main()
