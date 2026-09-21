"""
打包 wmview.flavor 插件压缩包

产物: wmview.flavor.zip（文件名不带版本号，版本号见包内 info.json）
解压到主程序根目录 plugs/ 下（目录名保持 wmview.flavor），重启 wmview 即可加载。

用法: python pack.py
"""
import json
import os
from zipfile import ZipFile, ZIP_DEFLATED


def compress_specified_items(output_zip: str, items: list, compress_level: int = 6):
    """自定义压缩指定的 文件/文件夹 列表"""
    with ZipFile(output_zip, "w", ZIP_DEFLATED, compresslevel=compress_level) as zipf:
        for item in items:
            if not os.path.exists(item):
                print(f"警告：{item} 不存在，已跳过")
                continue
            if os.path.isfile(item):
                zipf.write(item, arcname=os.path.basename(item))
            elif os.path.isdir(item):
                for root, dirs, files in os.walk(item):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, os.path.dirname(item))
                        zipf.write(file_path, arcname=arcname)


# 版本号单一来源：package.json 的 version，打包时同步进 info.json 随包发布
with open("package.json", encoding="utf-8") as f:
    VERSION = json.load(f)["version"]

with open("info.json", encoding="utf-8") as f:
    _info = json.load(f)
if _info.get("version") != VERSION:
    _info["version"] = VERSION
    with open("info.json", "w", encoding="utf-8") as f:
        json.dump(_info, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"已同步 info.json 的 version → {VERSION}")

# ====================== 【只需修改这里】 ======================
# MF 构建产物: index.js 是入口，assets/ 是运行时 chunk（必须一起打包）。
# flavor.exe 是推理程序（PyInstaller 打包，内置 numpy + rdkit + 模型权重）。
TO_COMPRESS = [
    "dist/index.js",  # MF remote 入口
    "dist/assets",  # 运行时 chunk + css
    "info.json",  # 插件市场元数据（含版本号）
    "icon.png",  # 插件图标 → zip 根目录 icon.png
    "flavor.exe",  # 气味预测推理程序（免 python 运行时）
]

OUTPUT = "wmview.flavor.zip"
compress_specified_items(OUTPUT, TO_COMPRESS, compress_level=6)
print(f"压缩完成: {OUTPUT}")
