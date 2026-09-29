"""开发脚本：导出「前向对拍用例」。

用与插件完全相同的特征口径（`flavor_feats.py`）在 RDKit 上算出图特征与药效团特征，
再跑 numpy 前向（`flavor_np.py`）得到参考 logits / 概率，一并写进 cases.json。
flavor-core（Rust / wasm）吃同样的特征，输出应与参考值一致（阈值 1e-4）。

这样可以把「特征是否正确」与「前向是否正确」分开验证：本步骤只验证前向。

用法: build/venv/Scripts/python.exe py/export_parity.py --n 12
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from flavor_feats import (  # noqa: E402
    PharmacophoreExtractor,
    graph_arrays,
    graph_mol,
    mol_from_smiles,
)
from flavor_np import FlavorModel  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12, help="用例数（取 val.csv 前 N 条）")
    ap.add_argument("--csv", default=str(ROOT / "flavor" / "data1" / "val.csv"))
    ap.add_argument("--out", default=str(ROOT / "build" / "parity" / "cases.json"))
    args = ap.parse_args()

    with open(args.csv, encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header = rows[0]
    body = rows[1:]
    smiles_list = [r[0] for r in body if r and r[0]][: args.n]
    print(f"标签列 {len(header) - 1} 个；取前 {len(smiles_list)} 条 SMILES 生成用例")

    model = FlavorModel(ROOT / "flavor" / "pharma33_weights.npz")
    extractor = PharmacophoreExtractor()

    cases = []
    for i, smi in enumerate(smiles_list, 1):
        mol3d = mol_from_smiles(smi)
        pharma = extractor.extract(mol3d)
        x, edge_index, edge_attr, pharma_t = graph_arrays(graph_mol(mol3d), pharma)
        logits = model.logits(x, edge_index, edge_attr, pharma_t).reshape(-1)
        probs = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
        cases.append(
            {
                "name": smi,
                "nodes": int(x.shape[0]),
                "edges": int(edge_index.shape[1]),
                "x": x.reshape(-1).tolist(),
                "edge_index": edge_index.reshape(-1).astype(int).tolist(),
                "edge_attr": edge_attr.reshape(-1).tolist(),
                "pharma": pharma_t.reshape(-1).tolist(),
                "logits": logits.tolist(),
                "probs": probs.tolist(),
            }
        )
        print(
            f"  [{i}/{len(smiles_list)}] {smi[:40]:42s} 重原子 {x.shape[0]:3d} "
            f"键 {edge_index.shape[1]:3d} 药效团 {extractor.last_count:2d}"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"num_classes": int(model.num_classes), "cases": cases}, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"已写出 {out}（{out.stat().st_size / 1024:.1f} KB）")


if __name__ == "__main__":
    main()
