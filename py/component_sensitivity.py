"""开发脚本：量化药效团特征里各「逐原子分量」对预测的影响。

药效团特征的 11 维里，有 4 维来自逐原子数值：
    (平均 Gasteiger 电荷, MR 之和, LogP 均值, 局部 TPSA)
wasm 化时这三者（Gasteiger / Crippen / TPSA）没有现成库，需要自己实现。
本脚本把某一分量整体清零后重算预测，用来判断「这分量算不准」会有多严重：

- 若影响与构象噪声（见 sensitivity.py，约 0.03）同量级 → 允许实现有微小偏差
- 若影响远大于噪声 → 必须与 RDKit 逐位对齐

用法: build/venv/Scripts/python.exe py/component_sensitivity.py --n 15
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from flavor_feats import PharmacophoreExtractor, graph_arrays, graph_mol, prepare_mol_3d  # noqa: E402
from flavor_np import FlavorModel  # noqa: E402

RDLogger.DisableLog("rdApp.*")

# 11 维药效团特征里的列索引
COL_TYPE = 0
COL_XYZ = (1, 2, 3)
COL_CHARGE = 4
COL_MR = 5
COL_LOGP = 6
COL_TPSA = 7
COL_DIST = (8, 9, 10)


def feature_matrix(mol, extractor):
    """直接拿逐药效团的原始 11 维矩阵（不池化），便于逐列清零。"""
    rows = []
    for feat in extractor.factory.GetFeaturesForMol(mol, confId=-1):
        if feat.GetFamily() not in extractor.__class__ and False:
            continue
        rows.append((feat, None))
    return rows


def raw_feature_array(mol, extractor) -> np.ndarray:
    """复刻 PharmacophoreExtractor.extract 的前半段，返回 (F, 11)。"""
    from flavor_feats import FAMILIES

    feats = []
    for feat in extractor.factory.GetFeaturesForMol(mol, confId=-1):
        family = feat.GetFamily()
        if family not in FAMILIES:
            continue
        try:
            atom_ids = list(feat.GetAtomIds())
            vec = [
                float(FAMILIES.index(family)),
                *extractor._average_xyz(mol, atom_ids),
                *extractor._atom_props(mol, atom_ids),
                *extractor._center_distances(mol, atom_ids),
            ]
            feats.append(vec)
        except Exception:
            continue
    if not feats:
        return np.zeros((0, 11), dtype=np.float32)
    return np.asarray(feats, dtype=np.float32)


def pool(arr: np.ndarray) -> np.ndarray:
    if arr.shape[0] == 0:
        return np.zeros(33, dtype=np.float32)
    return np.concatenate([arr.max(axis=0), arr.mean(axis=0), arr.sum(axis=0)]).astype(np.float32)


def topk(v: np.ndarray, k: int) -> set:
    return set(np.argsort(v)[::-1][:k].tolist())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--csv", default=str(ROOT / "flavor" / "data1" / "val.csv"))
    ap.add_argument("--out", default=str(ROOT / "build" / "parity" / "component_sensitivity.json"))
    args = ap.parse_args()

    with open(args.csv, encoding="utf-8") as f:
        rows = list(csv.reader(f))
    smiles_list = [r[0] for r in rows[1:] if r and r[0]][: args.n]

    model = FlavorModel(ROOT / "flavor" / "pharma33_weights.npz")
    extractor = PharmacophoreExtractor()

    def probs_with(mol, zero_cols=(), perturb=0.0) -> np.ndarray:
        raw = raw_feature_array(mol, extractor).copy()
        if raw.shape[0] and zero_cols:
            for c in zero_cols:
                raw[:, c] = 0.0
        if raw.shape[0] and perturb:
            rng = np.random.default_rng(0)
            raw += rng.normal(0.0, perturb, raw.shape).astype(np.float32)
        pharma = pool(raw)
        x, ei, ea, pt = graph_arrays(graph_mol(mol), pharma)
        return 1.0 / (1.0 + np.exp(-model.logits(x, ei, ea, pt).reshape(-1)))

    variants = {
        "zero_charge": dict(zero_cols=(COL_CHARGE,)),
        "zero_mr": dict(zero_cols=(COL_MR,)),
        "zero_logp": dict(zero_cols=(COL_LOGP,)),
        "zero_tpsa": dict(zero_cols=(COL_TPSA,)),
        "zero_xyz+dist": dict(zero_cols=COL_XYZ + COL_DIST),
        "noise_1e-3": dict(perturb=1e-3),
        "noise_1e-2": dict(perturb=1e-2),
    }

    stats: dict[str, list] = {k: [] for k in variants}
    print(f"用例 {len(smiles_list)} 个\n")
    for smi in smiles_list:
        mol = prepare_mol_3d(Chem.MolFromSmiles(smi))
        base = probs_with(mol)
        line = []
        for name, kw in variants.items():
            p = probs_with(mol, **kw)
            row = {
                "max_dp": float(np.abs(p - base).max()),
                "mean_dp": float(np.abs(p - base).mean()),
                **{f"top{k}": len(topk(base, k) & topk(p, k)) / k for k in (1, 5, 10)},
            }
            stats[name].append(row)
            line.append(f"{name} 0.0{row['max_dp']*1000:04.1f}")
        print(f"{smi[:38]:38s} " + " | ".join(line))

    print("\n=== 汇总（相对完整特征的基线） ===")
    print(f"{'variant':14s} {'maxΔp(均值)':>12s} {'maxΔp(最坏)':>12s} {'meanΔp':>10s} {'top1':>7s} {'top5':>7s} {'top10':>7s}")
    summary = {}
    for name, r in stats.items():
        s = {
            "max_dp_mean": float(np.mean([x["max_dp"] for x in r])),
            "max_dp_worst": float(np.max([x["max_dp"] for x in r])),
            "mean_dp": float(np.mean([x["mean_dp"] for x in r])),
            **{f"top{k}": float(np.mean([x[f"top{k}"] for x in r])) for k in (1, 5, 10)},
        }
        summary[name] = s
        print(
            f"{name:14s} {s['max_dp_mean']:12.4f} {s['max_dp_worst']:12.4f} {s['mean_dp']:10.5f} "
            f"{s['top1']:7.1%} {s['top5']:7.1%} {s['top10']:7.1%}"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n已写出 {out}")


if __name__ == "__main__":
    main()
