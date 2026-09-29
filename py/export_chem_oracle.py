"""开发脚本：导出「化学层对拍」基准。

给同一批分子（val 集前 N 条）用 RDKit 算出：

- 带氢 3D 分子的 molblock（作为双侧共同输入，避免构象差异干扰对比）
- 图特征（x / edge_index / edge_attr）与药效团 33 维向量
- 逐原子 Crippen(logP, MR)、TPSA 贡献、Gasteiger 电荷
- 药效团特征清单（家族 + 原子索引，按 SMARTS 查询原子顺序）
- 最终概率（numpy 前向）

前端实现（`src/chem/`）吃同一份 molblock，应能在
图特征 / 逐原子数值 / 药效团清单上完全一致，概率在 1e-4 内一致。

用法: build/venv/Scripts/python.exe py/export_chem_oracle.py --n 12
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Crippen, rdMolDescriptors

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from flavor_feats import (  # noqa: E402
    FAMILIES,
    PharmacophoreExtractor,
    graph_arrays,
    graph_mol,
)
from flavor_np import FlavorModel  # noqa: E402

RDLogger.DisableLog('rdApp.*')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=12)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--csv', default=str(ROOT / 'flavor' / 'data1' / 'val.csv'))
    ap.add_argument('--out', default=str(ROOT / 'build' / 'chem' / 'oracle.json'))
    args = ap.parse_args()

    with open(args.csv, encoding='utf-8') as f:
        rows = list(csv.reader(f))
    smiles_list = [r[0] for r in rows[1:] if r and r[0]][: args.n]

    model = FlavorModel(ROOT / 'flavor' / 'pharma33_weights.npz')
    extractor = PharmacophoreExtractor()

    cases = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol)
        ps = AllChem.ETKDGv3()
        ps.randomSeed = args.seed
        if AllChem.EmbedMolecule(mol, ps) != 0:
            print(f'跳过（嵌入失败）: {smi}')
            continue
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except Exception:
            pass
        AllChem.ComputeGasteigerCharges(mol)

        # --- 特征 ---
        pharma = extractor.extract(mol)
        x, edge_index, edge_attr, pharma_t = graph_arrays(graph_mol(mol), pharma)
        probs = 1.0 / (1.0 + np.exp(-model.logits(x, edge_index, edge_attr, pharma_t).reshape(-1)))

        # --- 逐原子数值（带氢分子上的口径） ---
        crippen = Crippen._GetAtomContribs(mol)
        tpsa = rdMolDescriptors._CalcTPSAContribs(mol)
        charges = [
            float(a.GetDoubleProp('_GasteigerCharge')) if a.HasProp('_GasteigerCharge') else 0.0
            for a in mol.GetAtoms()
        ]

        # --- 药效团清单 ---
        feats = []
        for feat in extractor.factory.GetFeaturesForMol(mol, confId=-1):
            fam = feat.GetFamily()
            if fam not in FAMILIES:
                continue
            feats.append({'family': fam, 'atoms': [int(i) for i in feat.GetAtomIds()]})

        cases.append(
            {
                'name': smi,
                'molblock': Chem.MolToMolBlock(mol),
                'x': x.reshape(-1).tolist(),
                'nodes': int(x.shape[0]),
                'edge_index': edge_index.reshape(-1).astype(int).tolist(),
                'edges': int(edge_index.shape[1]),
                'edge_attr': edge_attr.reshape(-1).tolist(),
                'pharma': pharma_t.reshape(-1).tolist(),
                'probs': probs.tolist(),
                'crippen': [[float(a), float(b)] for a, b in crippen],
                'tpsa': [float(v) for v in tpsa],
                'charges': charges,
                'features': feats,
            }
        )
        print(f'  {smi[:40]:42s} 原子 {x.shape[0]:3d} 键 {edge_index.shape[1]:3d} 药效团 {len(feats):2d}')

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'num_classes': int(model.num_classes), 'cases': cases}, ensure_ascii=False), encoding='utf-8')
    print(f'\n已写出 {out}（{out.stat().st_size / 1024:.1f} KB，{len(cases)} 个用例）')


if __name__ == '__main__':
    main()
