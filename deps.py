"""开发脚本：安装/检查构建依赖（开发环境用，不随插件分发）。

用法（在 build/venv 里执行）：
    python deps.py            # 检查
    python deps.py --install  # 缺失则安装
"""

import argparse
import importlib.util
import subprocess
import sys

MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"

# 运行时依赖（exe 里需要的包）
RUNTIME_PACKAGES = ["numpy", "rdkit"]
# 开发/校验依赖（只有 dev 需要；torch 走 CPU 专用源安装，见 build_exe.py 说明）
DEV_PACKAGES = ["torch-geometric", "pyinstaller"]


def check(packages: list) -> list:
    missing = []
    for pkg in packages:
        module = pkg.split("==")[0].split(">=")[0].replace("-", "_")
        if importlib.util.find_spec(module) is None:
            missing.append(pkg)
    return missing


def install(packages: list) -> None:
    for pkg in packages:
        print(f"[deps] 安装 {pkg} …")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-i", MIRROR])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--install", action="store_true", help="缺失依赖时自动安装")
    args = parser.parse_args()

    for title, packages in (("运行时", RUNTIME_PACKAGES), ("开发", DEV_PACKAGES)):
        missing = check(packages)
        if not missing:
            print(f"[deps] {title}依赖齐全: {', '.join(packages)}")
        else:
            print(f"[deps] {title}缺失: {', '.join(missing)}")
            if args.install:
                install(missing)


if __name__ == "__main__":
    main()
