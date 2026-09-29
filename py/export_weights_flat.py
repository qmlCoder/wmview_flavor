"""开发脚本：把 `pharma33_weights.npz` 转成扁平 f32，供 flavor-core（Rust / wasm）加载。

产物（随插件分发，放 dist/assets 里由前端 fetch）：
    flavor/pharma33_weights.bin   —— 所有张量按名字排序后顺序拼接的 f32 小端数据
    flavor/pharma33_weights.json  —— 张量索引：name → {shape, offset}

用法: build/venv/Scripts/python.exe py/export_weights_flat.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
NPZ = ROOT / "flavor" / "pharma33_weights.npz"
OUT_BIN = ROOT / "flavor" / "pharma33_weights.bin"
OUT_JSON = ROOT / "flavor" / "pharma33_weights.json"


def main() -> None:
    data = np.load(NPZ, allow_pickle=False)
    names = sorted(data.files)

    tensors: dict[str, dict] = {}
    chunks: list[np.ndarray] = []
    offset = 0
    for name in names:
        arr = np.ascontiguousarray(data[name], dtype="<f4")
        tensors[name] = {"shape": list(arr.shape), "offset": offset}
        chunks.append(arr.reshape(-1))
        offset += int(arr.size)

    flat = np.concatenate(chunks) if chunks else np.zeros(0, dtype="<f4")
    OUT_BIN.write_bytes(flat.tobytes(order="C"))
    OUT_JSON.write_text(
        json.dumps({"tensors": tensors, "numel": int(flat.size)}, ensure_ascii=False, indent=1)
        + "\n",
        encoding="utf-8",
    )

    print(f"张量 {len(names)} 个，元素 {flat.size} 个（f32 {flat.size * 4 / 1024 / 1024:.2f} MB）")
    print(f"已写出 {OUT_BIN.name}（{OUT_BIN.stat().st_size / 1024 / 1024:.2f} MB）")
    print(f"已写出 {OUT_JSON.name}（{OUT_JSON.stat().st_size / 1024:.1f} KB）")


if __name__ == "__main__":
    main()
