"""把 py/predict.py 打包成单文件 flavor.exe（免安装 python、免安装依赖）。

用法（在装了 numpy / rdkit / pyinstaller 的环境里执行）：
    build/venv/Scripts/python.exe build_exe.py

产物：项目根目录 flavor.exe，随 index.js + assets 一起分发。
运行时依赖只有 numpy + rdkit（模型权重用 numpy 前向实现，不需要 torch）。
权重与标签也会一并打进 exe（同时支持放在 exe 同目录的 flavor/ 下覆盖，见 py/predict.py）。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# 优先用独立打包环境；没有则退回当前解释器
VENV_PYTHON = ROOT / "build" / "venv" / "Scripts" / "python.exe"
INTERP = VENV_PYTHON if VENV_PYTHON.exists() else Path(sys.executable)


def main() -> None:
    onefile = "--onedir" not in sys.argv
    distpath = ROOT if onefile else ROOT / "build" / "onedir-dist"
    cmd = [
        str(INTERP),
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onefile" if onefile else "--onedir",
        "--console",  # 保留 stdout/stderr（主程序 run_exe 以隐藏窗口方式拉起）
        "--name",
        "flavor",
        "--distpath",
        str(distpath),
        "--workpath",
        str(ROOT / "build" / "pyi"),
        "--specpath",
        str(ROOT / "build"),
        # rdkit 需要 BaseFeatures.fdef 等数据文件，整包收集最稳
        "--collect-all",
        "rdkit",
        "--exclude-module",
        "torch",
        "--exclude-module",
        "torch_geometric",
        "--exclude-module",
        "matplotlib",
        "--exclude-module",
        "tkinter",
        # 模型权重与标签表打进 exe（放 exe 同目录 flavor/ 下可覆盖）
        "--add-data",
        f"{ROOT / 'flavor' / 'pharma33_weights.npz'}{';'}flavor",
        "--add-data",
        f"{ROOT / 'flavor' / 'labels.json'}{';'}flavor",
        str(ROOT / "py" / "predict.py"),
    ]
    print("运行:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    if onefile:
        exe = ROOT / "flavor.exe"
        print(f"完成: {exe} ({exe.stat().st_size / 1024 / 1024:.1f} MB)")
    else:
        print(f"完成: {distpath / 'flavor'}")


if __name__ == "__main__":
    main()
