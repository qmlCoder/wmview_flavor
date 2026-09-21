"""wmview.flavor 插件的推理入口（numpy + RDKit，无需 PyTorch）。

主程序通过 `wmapi_cores.run_exe(source, exe, args, input)` 调用：
stdin 收到一段 JSON，stdout 输出若干行日志，最后一行是结果：

    @@FLAVOR_RESULT@@{"ok": true, "results": [...]}

输入 JSON 字段（三选一指定分子来源）：
    smiles    : SMILES 字符串（用 RDKit 生成 3D 构象）
    molblock  : SDF / MOL block 文本（含 3D 坐标时直接使用）
    syms/xyzs/bonds[/charge] : 主程序场景里的原子符号 / 坐标 / 连接表
可选：
    topk       : 只返回前 K 个标签（默认全部 138 个）
    threshold  : 命中阈值（默认 0.5，仅用于标注 hit）
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

RESULT_MARK = "@@FLAVOR_RESULT@@"


def log(msg: str) -> None:
    print(msg, flush=True)


def find_asset(name: str) -> Path | None:
    """按 环境变量 → exe 同目录 → 打包内嵌(frozen) 的顺序找权重/标签文件。"""
    candidates = []
    import os

    home = os.environ.get("WMVIEW_FLAVOR_HOME")
    if home:
        candidates += [Path(home) / "flavor" / name, Path(home) / name]
    here = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
    candidates += [here / "flavor" / name, here.parent / "flavor" / name, here / name]
    if hasattr(sys, "_MEIPASS"):
        candidates += [Path(sys._MEIPASS) / "flavor" / name, Path(sys._MEIPASS) / name]
    for p in candidates:
        if p.is_file():
            return p
    return None


def build_mol(payload: dict):
    """按输入构造带 3D 构象与 Gasteiger 电荷的分子，返回 (分子, 警告文本)。"""
    from flavor_feats import mol_from_atoms, mol_from_block, mol_from_smiles

    if payload.get("smiles"):
        log(f"[flavor] 由 SMILES 生成 3D 构象: {payload['smiles']}")
        return mol_from_smiles(payload["smiles"]), None
    if payload.get("molblock"):
        log("[flavor] 使用分子文件的坐标与键级")
        return mol_from_block(payload["molblock"]), None
    if payload.get("syms") and payload.get("xyzs"):
        bonds = payload.get("bonds") or []
        log(f"[flavor] 使用场景分子: {len(payload['syms'])} 原子 / {len(bonds)} 键，推断键级")
        return mol_from_atoms(payload["syms"], payload["xyzs"], bonds, charge=int(payload.get("charge") or 0))
    raise ValueError("缺少分子输入（smiles / molblock / syms+xyzs 之一）")


def main() -> int:
    parser = argparse.ArgumentParser(description="wmview.flavor 气味预测")
    parser.add_argument("--smiles", help="直接给 SMILES（调试用）")
    parser.add_argument("--mol", help="分子文件路径（SDF/MOL，调试用）")
    parser.add_argument("--topk", type=int, default=0, help="只返回前 K 个标签")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    if args.smiles or args.mol:
        payload = {"topk": args.topk, "threshold": args.threshold}
        if args.smiles:
            payload["smiles"] = args.smiles
        else:
            payload["molblock"] = Path(args.mol).read_text(encoding="utf-8", errors="ignore")
    else:
        raw = sys.stdin.read()
        if not raw.strip():
            log("用法：把 JSON 从 stdin 传入，或用 --smiles / --mol 调试")
            parser.print_help()
            return 2
        payload = json.loads(raw)

    try:
        from flavor_feats import PharmacophoreExtractor, graph_arrays, graph_mol
        from flavor_np import FlavorModel
        from rdkit import Chem

        weights_path = find_asset("pharma33_weights.npz")
        labels_path = find_asset("labels.json")
        if weights_path is None or labels_path is None:
            raise FileNotFoundError("找不到模型权重 pharma33_weights.npz / labels.json")

        model = FlavorModel(weights_path)
        labels = json.loads(Path(labels_path).read_text(encoding="utf-8"))
        if len(labels) != model.num_classes:
            raise ValueError(f"标签数 {len(labels)} 与模型输出 {model.num_classes} 不一致")

        mol, warning = build_mol(payload)
        if warning:
            log(f"[flavor] 警告: {warning}")
        extractor = PharmacophoreExtractor()
        pharma = extractor.extract(mol)
        x, edge_index, edge_attr, pharma_tensor = graph_arrays(graph_mol(mol), pharma)
        probs = model.predict_proba(x, edge_index, edge_attr, pharma_tensor)

        threshold = float(payload.get("threshold") or args.threshold)
        order = probs.argsort()[::-1]
        topk = int(payload.get("topk") or args.topk or 0)
        if topk > 0:
            order = order[:topk]
        results = [
            {"label": labels[i], "prob": round(float(probs[i]), 6), "hit": bool(probs[i] >= threshold)}
            for i in order
        ]

        smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))
        n_pharma = int(extractor.last_count)
        log(f"[flavor] 分子 {x.shape[0]} 个重原子，{n_pharma} 个药效团，输出 {len(results)} 个气味标签")
        print(
            RESULT_MARK
            + json.dumps(
                {
                    "ok": True,
                    "smiles": smiles,
                    "atoms": int(x.shape[0]),
                    "pharmacophores": n_pharma,
                    "threshold": threshold,
                    "warning": warning,
                    "results": results,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 0
    except Exception as exc:  # 任何异常都以 JSON 返回，便于前端提示
        log(traceback.format_exc())
        print(RESULT_MARK + json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), flush=True)
        return 1
if __name__ == "__main__":
    raise SystemExit(main())
