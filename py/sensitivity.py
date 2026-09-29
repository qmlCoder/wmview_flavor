"""开发脚本：量化「构象不确定性」对预测结果的影响。

背景：`prepare_mol_3d` 里用的是 `AllChem.ETKDGv3()`，其 `randomSeed` 默认 -1（随机），
所以**当前 Python 实现对 SMILES 输入本身就是非确定性的**——每次调用可能给出不同结果。
本脚本用它当「噪声地板」，衡量三种情形的差异：

    A. ETKDG 随机种子（当前实现，跑两次）
    B. ETKDG 固定种子（可复现的基线）
    C. 二维平面构象（z=0，作为「不实现 3D 生成」的退路上限）

输出每种情形相对固定种子的 top-K 标签重合率与概率偏差，用来决定
wasm 化时 SMILES 路径要做到什么程度（是否需要复刻 ETKDG）。

用法: build/venv/Scripts/python.exe py/sensitivity.py --n 15
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from flavor_feats import (  # noqa: E402
    PharmacophoreExtractor,
    graph_arrays,
    graph_mol,
)
from flavor_np import FlavorModel  # noqa: E402

RDLogger.DisableLog("rdApp.*")


def build(smiles: str, mode: str, seed: int | None = None):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"无法解析 SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    if mode == "flat2d":
        AllChem.Compute2DCoords(mol)
    else:
        ps = AllChem.ETKDGv3()
        if seed is not None:
            ps.randomSeed = seed
        if AllChem.EmbedMolecule(mol, ps) != 0:
            raise ValueError("嵌入失败")
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except Exception:
            pass
    AllChem.ComputeGasteigerCharges(mol)
    return mol


def topk(labels: np.ndarray, k: int) -> set:
    return set(np.argsort(labels)[::-1][:k].tolist())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--csv", default=str(ROOT / "flavor" / "data1" / "val.csv"))
    ap.add_argument("--out", default=str(ROOT / "build" / "parity" / "sensitivity.json"))
    args = ap.parse_args()

    with open(args.csv, encoding="utf-8") as f:
        rows = list(csv.reader(f))
    smiles_list = [r[0] for r in rows[1:] if r and r[0]][: args.n]

    model = FlavorModel(ROOT / "flavor" / "pharma33_weights.npz")
    extractor = PharmacophoreExtractor()

    def probs_of(mol) -> np.ndarray:
        pharma = extractor.extract(mol)
        x, ei, ea, pt = graph_arrays(graph_mol(mol), pharma)
        return 1.0 / (1.0 + np.exp(-model.logits(x, ei, ea, pt).reshape(-1)))

    variants = {
        "fixed": lambda s: build(s, "etkdg", args.seed),
        "random1": lambda s: build(s, "etkdg", None),
        "random2": lambda s: build(s, "etkdg", None),
        "flat2d": lambda s: build(s, "flat2d"),
    }

    stats: dict[str, list] = {k: [] for k in variants if k != "fixed"}
    print(f"用例 {len(smiles_list)} 个，基线 = ETKDG(seed={args.seed}) + MMFF\n")
    print(f"{'SMILES':42s} " + " ".join(f"{k:>18s}" for k in stats))

    for smi in smiles_list:
        base = probs_of(variants["fixed"](smi))
        base_top = {k: topk(base, k) for k in (1, 5, 10, 20)}
        for name, fn in variants.items():
            if name == "fixed":
                continue
            p = probs_of(fn(smi))
            row = {
                "max_dp": float(np.abs(p - base).max()),
                "mean_dp": float(np.abs(p - base).mean()),
                **{f"top{k}": len(base_top[k] & topk(p, k)) / k for k in (1, 5, 10, 20)},
            }
            stats[name].append(row)
        print(
            f"{smi[:42]:42s} "
            + " ".join(
                f"Δp≤{stats[k][-1]['max_dp']:.3f} t10={stats[k][-1]['top10']:.0%}".rjust(18)
                for k in stats
            )
        )

    print("\n=== 汇总（相对固定种子的基线） ===")
    print(f"{'variant':10s} {'maxΔp(均值)':>12s} {'maxΔp(最坏)':>12s} {'meanΔp':>10s} {'top1':>7s} {'top5':>7s} {'top10':>7s} {'top20':>7s}")
    summary = {}
    for name, rows in stats.items():
        s = {
            "max_dp_mean": float(np.mean([r["max_dp"] for r in rows])),
            "max_dp_worst": float(np.max([r["max_dp"] for r in rows])),
            "mean_dp": float(np.mean([r["mean_dp"] for r in rows])),
            **{f"top{k}": float(np.mean([r[f"top{k}"] for r in rows])) for k in (1, 5, 10, 20)},
        }
        summary[name] = s
        print(
            f"{name:10s} {s['max_dp_mean']:12.4f} {s['max_dp_worst']:12.4f} {s['mean_dp']:10.5f} "
            f"{s['top1']:7.1%} {s['top5']:7.1%} {s['top10']:7.1%} {s['top20']:7.1%}"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    import json

    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n已写出 {out}")


if __name__ == "__main__":
    main()
