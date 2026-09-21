# wmview.flavor

wmview 外部插件：**分子气味预测**。模型是第三方开源的 **PharmaGNN**（药效团特征 + GATv2 图注意力网络，
多标签分类），本仓库把它改造成 wmview 插件：面板里选一个分子 → 输出 138 个气味标签的概率。

## 目录

```
wmview.flavor/
├── index.ts              # 插件入口（exposes './index'）：export plugin + export default
├── Sider.vue             # 插件侧边栏面板（分子来源 / 阈值 / 结果展示 / 导出 CSV / 介绍与教程）
├── plugin.d.ts           # Plugin / Sider / Setting 契约类型
├── wmapi.d.ts            # 宿主 API 类型（wmapi_cores / wmapi_files / wmapi_scene ...）
├── comps/                # 通用面板组件（随主程序同步，勿手改）
├── flavor.exe            # 推理程序（PyInstaller 单文件，内置 numpy + rdkit + 模型权重）
├── py/                   # 推理与校验脚本（打包进 exe 的就是这里的 predict.py）
│   ├── predict.py        # 运行时入口：stdin JSON → stdout @@FLAVOR_RESULT@@{...}
│   ├── flavor_feats.py   # RDKit 特征提取（11 维原子 / 4 维键 / 33 维药效团）
│   ├── flavor_np.py      # numpy 前向实现（GATv2 + MLP，无需 torch）
│   ├── export_weights.py # 开发：best_model_pharma33.pt → flavor/pharma33_weights.npz
│   ├── reference_torch.py# 开发：torch + PyG 参考实现
│   └── validate_numpy.py # 开发：numpy 与 torch 对拍 + 数据集 AUC
├── icon.png              # 侧边栏图标（苯环字形，随包分发）
├── flavor/               # 第三方模型与数据（原作者材料，保持原样）
│   ├── best_model_pharma33.pt    # 原 checkpoint（torch）
│   ├── model.py                  # 原作者训练代码
│   ├── pharma33_weights.npz      # 导出的 numpy 权重（打包进 exe）
│   ├── labels.json               # 138 个气味标签（顺序与模型输出一致）
│   └── data1/ data2/             # 原作者数据集（仅用于校验，不随插件分发）
├── build_exe.py          # PyInstaller 打包 flavor.exe
├── deps.py               # 构建依赖检查/安装
├── pack.py               # 打包 wmview.flavor.zip
└── notes/latest.md       # 修改记录（追加式）
```

## 工作原理

1. 面板取分子：**场景分子**（`wmapi_files.get_moleInfo` 的原子符号/坐标/连接表）或 **SMILES**
   （RDKit 生成 3D 构象）。
2. `flavor.exe` 收到 JSON：场景分子用 `rdDetermineBonds` 依坐标推断键级与芳香性；SMILES 走
   ETKDG + MMFF94 优化。
3. 特征：重原子图（11 维原子 + 4 维键类型 one-hot）+ 3D 药效团特征（11 维/药效团，max/mean/sum
   池化成 33 维，与权重口径一致）。
4. numpy 前向：节点/边编码器 → 5 层 GATv2Conv（8/8/4/4/2 头，残差 + LayerNorm + ReLU）→ 全局平均
   池化 → 药效团 MLP → 拼接 → 4 层 MLP → sigmoid → 138 个气味概率。
5. 面板按阈值/条数展示，可导出 CSV。

> 运行时**不需要 PyTorch**：权重已导出为 `flavor/pharma33_weights.npz`，前向用 numpy 实现
> （`py/flavor_np.py`），与 torch + PyG 参考实现逐分子对拍一致（最大偏差 ~1e-7）。
> 校验集（`flavor/data1/val.csv` + `val-sdf`）macro-AUC 0.8833，与 checkpoint 记录的 val_auc
> 0.8796 吻合，说明标签顺序与特征口径与训练时一致。

## 输入与限制

| 输入 | 键级来源 | 说明 |
| --- | --- | --- |
| SMILES | RDKit 生成 3D 构象（ETKDG + MMFF94） | 推荐，最省事 |
| 场景分子（原文件可读） | 分子文件自带的键级 | SDF/MOL/PDB 等，最准确 |
| 场景分子（只有坐标+连接表） | `rdDetermineBonds` 由 3D 坐标推断 | 分子带显式氢时准确 |
| 场景分子（无显式氢且读不到文件） | 退化：按单键处理 | 结果仅供参考，面板会给出警告；建议改用 SMILES |

> 无氢分子的键级推断是 xyz2mol 类算法的固有限制（缺少氢原子数就无法定键级）；
> 插件在这种情况下会明确提示，而不是静默给出看似正常的结果。

## 安装使用

把 `wmview.flavor.zip` 解压到主程序根目录的 `plugs/wmview.flavor/`（目录名必须是
`wmview.flavor`，remote 名由目录名换算），重启主程序即可在侧边栏看到 **flavor** 面板。

## 开发

```sh
npm install          # 首次
npm run dev          # dev server，固定 3001 端口
npm run type-check   # 类型检查（必须零错误）
npm run build        # 产物 dist/index.js + dist/assets
python pack.py       # 打包 wmview.flavor.zip（含 flavor.exe）

# 推理程序
flavor.exe --smiles "CCCOC(=O)C=Cc1ccco1" --topk 10   # 命令行自测
build_exe.py         # 重新打包 flavor.exe（需要 build/venv 里的 numpy/rdkit/pyinstaller）
py/validate_numpy.py --split val --n 8                # 校验 numpy 实现（需要 torch + PyG）
```

## 约定

- 插件以 **Module Federation remote** 构建，remote 名为 `wmview_flavor`（安装目录
  `plugs/wmview.flavor/` 由主程序换算得到），入口文件名固定 `index.js`。
- 依赖 `vue` / `element-plus` 由主程序以 singleton 提供，插件不得把它们打进产物。
- 与主程序交互**只用 `window.wmapi_*` API**（类型以 `wmapi.d.ts` 为准），不引用主程序内部模块。
- `run_exe` 的 source 统一用 `plugin_flavor`；exe 输出以 `@@FLAVOR_RESULT@@` 打头的一行 JSON 作为结果。

## 宿主 API 版本（重要）

程序自带的 `plugin-deps/wmapi.d.ts` 模板**已过期**，它写的是 `wmapi_files.get_moleInfo(name)`（同步）、
`get_files()`、`get_selects()`，而程序实际暴露的是：

| 模板里（不存在） | 程序实际提供 |
| --- | --- |
| `get_moleInfo(name)` | `get_mole_info(name): Promise<MoleInfo>`（**异步**） |
| `get_files()` | `get_file_list()` |
| `get_selects()` | `get_select()` / `get_option()` / `on_select(cb)` |
| `MoleInfo{ syms, xyzs, bonds, open, obts, eles, atos }` | `MoleInfo{ syms, xyzs, bonds }` |

本仓库的 `wmapi.d.ts` 已按程序的 `src/wmapi.d.ts`、`src/plugin/filelist/wmapi.d.ts`、
`src/plugin/scene/wmapi.d.ts`（含 `wmarg.d.ts`）同步。注意 `wmview-plugin update` 会用模板
覆盖这个文件，更新后需要重新同步。

面板读取分子结构时做了三层兼容：`get_mole_info` → 旧模板 `get_moleInfo` → `wmapi_scene.get_mole_geom`；
工具栏的「接口自检」按钮会把宿主实际暴露的接口清单写进日志，方便比对版本。

### 拿 API 的正确姿势

运行中的 wmview 内置本地 HTTP 服务（端口写在 `%LOCALAPPDATA%\wmview\port.json`），可以直接问程序本身：

```sh
GET  http://127.0.0.1:<port>/get_apidoc      # 核心 + 已加载插件的 API 文档
POST http://127.0.0.1:<port>/run_script      # {"script":"return Object.keys(window.wmapi_files)"}
```

比翻 `plugin-deps/wmapi.d.ts` 模板可靠得多（模板已经漂移过：`get_moleInfo` / `get_files` / `get_selects` 都不存在）。
面板的「接口自检」按钮就是干这件事的。

另外两条实测结论：
- `wmapi_files.get_plugFold()` 返回的是 **plugs 根目录**，安装后的插件在 `<plugs>/wmview.flavor/`，
  所以 `flavor.exe` 的路径是 `<plugs>/wmview.flavor/flavor.exe`；开发联调（dev server）时插件代码不在这里，
  可以把「程序路径」填成项目里的 `flavor.exe`。
- `run_exe` 在**路径不存在时会抛异常**（不是返回 `[false, msg]`），所以面板逐个候选路径 try/catch。

主程序 dev 模式默认连接 `http://localhost:3001/index.js` 加载开发中的插件（面板外层有蓝色环、
名字带 `@dev`）；正式安装版需在 插件 → 开发指南 中打开开关。
