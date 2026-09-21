"""开发脚本：把 PyTorch checkpoint 导出为 numpy 权重（.npz）+ 标签表（.json）。

用法（在装了 torch 的开发环境里执行）：
    python py/export_weights.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "flavor" / "best_model_pharma33.pt"
OUT_NPZ = ROOT / "flavor" / "pharma33_weights.npz"
OUT_LABELS = ROOT / "flavor" / "labels.json"
LABELS_CSV = ROOT / "flavor" / "data1" / "train.csv"


def main() -> None:
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]
    arrays = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in state.items()}

    print(f"checkpoint: epoch={ckpt.get('epoch')} val_auc={ckpt.get('val_auc')}")
    print(f"参数张量 {len(arrays)} 个，输出类别数 = {arrays['classifier.12.weight'].shape[0]}")
    for key in sorted(arrays):
        print(f"  {key:32s} {tuple(arrays[key].shape)}")

    np.savez_compressed(OUT_NPZ, **arrays)
    print(f"已写出权重: {OUT_NPZ} ({OUT_NPZ.stat().st_size / 1024 / 1024:.1f} MB)")

    with LABELS_CSV.open(encoding="utf-8") as f:
        header = next(csv.reader(f))
    labels = header[1:]
    if len(labels) != arrays["classifier.12.weight"].shape[0]:
        print(f"警告：标签数 {len(labels)} 与模型输出维度不一致", file=sys.stderr)
    OUT_LABELS.write_text(json.dumps(labels, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已写出标签: {OUT_LABELS}（{len(labels)} 个，来源 {LABELS_CSV.name}）")


if __name__ == "__main__":
    main()
