"""开发脚本：从本地 RDKit 安装里导出前端要用的常量表 → `src/chem/tables/`。

- `crippen.ts`   ← rdkit/Data/Crippen.txt（Wildman-Crippen 原子类型：SMARTS + logP + MR）
- `fdef.ts`      ← rdkit/Data/BaseFeatures.fdef（药效团家族定义）
- `elements.ts`  ← 元素符号 / 原子量（直接问 RDKit，保证与 `atom.GetMass()` 一致）

只在改动表数据/升级 RDKit 时重跑；产物入库（体积很小）。

用法: build/venv/Scripts/python.exe py/export_tables.py
"""

from __future__ import annotations

import json
from pathlib import Path

from rdkit import Chem

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(Chem.RDConfig.RDDataDir)
OUT = ROOT / "src" / "chem" / "tables"

HEADER = "// 由 py/export_tables.py 生成，请勿手工修改。\n"


def emit(name: str, body: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(HEADER + body, encoding="utf-8")
    print(f"已写出 {path.relative_to(ROOT)}（{path.stat().st_size / 1024:.1f} KB）")


def export_crippen() -> None:
    """Crippen.txt：制表符分隔，行序即优先级（RDKit 取第一个匹配的类型）。

    与 RDKit 解析口径保持一致：`#` 开头是注释，logP/MR 为空串时按 0.0 处理
    （表里 Me2 那几行金属类型的 MR 就是空的），SMARTS 为空的空行直接跳过。
    """

    def num(field: str) -> float:
        field = field.strip()
        if not field:
            return 0.0
        try:
            return float(field)
        except ValueError:
            return 0.0

    text = (DATA / "Crippen.txt").read_text(encoding="utf-8", errors="replace")
    rows = []
    for raw in text.splitlines():
        line = raw.rstrip("\r\n")
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        label, smarts = parts[0].strip(), parts[1].strip()
        if not smarts:
            continue
        rows.append((label, smarts, num(parts[2]), num(parts[3])))
    print(f"Crippen 表：{len(rows)} 条 SMARTS 规则")
    payload = json.dumps(rows, ensure_ascii=False)
    emit(
        "crippen.ts",
        f"/** Crippen.txt 全表，行序即优先级：[标签, SMARTS, logP, MR] */\n"
        f"export const CRIPPEN_TABLE: [string, string, number, number][] = {payload}\n",
    )


def export_fdef() -> None:
    """BaseFeatures.fdef 原文（前端自己解析 AtomType / DefineFeature）。"""
    text = (DATA / "BaseFeatures.fdef").read_text(encoding="utf-8", errors="replace")
    emit(
        "fdef.ts",
        "/** RDKit BaseFeatures.fdef 原文 */\n"
        f"export const BASE_FEATURES_FDEF = {json.dumps(text)}\n",
    )


def export_elements() -> None:
    """元素符号与原子量（RDKit 口径，用于 GetMass）。"""
    table = Chem.GetPeriodicTable()
    symbols = []
    weights = []
    for z in range(1, 119):
        symbols.append(table.GetElementSymbol(z))
        weights.append(round(table.GetAtomicWeight(z), 6))
    payload_sym = json.dumps(symbols)
    payload_w = json.dumps(weights)
    emit(
        "elements.ts",
        "/** 元素符号（下标 = 原子序数 - 1）与 RDKit 口径的原子量 */\n"
        f"export const ELEMENT_SYMBOLS: string[] = {payload_sym}\n"
        f"export const ATOMIC_WEIGHTS: number[] = {payload_w}\n",
    )


def main() -> None:
    print(f"RDKit 数据目录：{DATA}")
    export_crippen()
    export_fdef()
    export_elements()


if __name__ == "__main__":
    main()
