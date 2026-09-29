# wmview.flavor

wmview 外部插件：**分子气味预测**。模型是第三方开源的 **PharmaGNN**（药效团特征 + GATv2
图注意力网络，多标签分类），本仓库把它改造成 wmview 插件：面板里选一个分子 → 输出 138
个气味标签的概率。

实现上是**纯前端 + Rust→wasm**：化学感知用 `@rdkit/rdkit`（MinimalLib），模型前向是
`flavor-core/`（纯 Rust，编成 123 KB 的 wasm）在 WebView 里直接算。没有外部进程、没有
Python、不联网——这也是能覆盖 Android 的原因（`run_exe` 在 Android 沙箱里不允许执行
私有目录里的二进制）。

## 目录

```
wmview.flavor/
├── src/                  # 前端源码（Vue/TS 一律放这里）
│   ├── index.ts          # 插件入口（exposes './index'）：export plugin + export default
│   ├── Sider.vue         # 插件侧边栏面板（分子来源 / 阈值 / 结果展示 / 导出 CSV / 介绍与教程）
│   ├── chem/             # 化学层：rdkit.js 封装 + 逐原子描述符 + 药效团特征 + 端到端管线
│   │   ├── rdkit.ts      # rdkit.js 加载与薄封装（解析、SMARTS、加氢、坐标）
│   │   ├── browser.ts    # 运行时装配：fetch 两个 wasm + 权重，初始化推理核
│   │   ├── pipeline.ts   # 端到端：分子输入 → 特征 → wasm 前向 → 结果
│   │   ├── graph.ts      # get_json() → 分子图（含 aromaticAtoms / atomRings）
│   │   ├── peratom.ts    # 逐原子 TPSA / Crippen / Gasteiger（按 RDKit 源码移植）
│   │   ├── fdef.ts       # BaseFeatures.fdef 解析 + 药效团识别
│   │   ├── features.ts   # 11 维原子 / 4 维键 / 33 维药效团特征
│   │   ├── bondorder.ts  # 场景分子的键级推断（近似 DetermineBonds）
│   │   └── tables/       # 由 py/export_tables.py 导出的表（Crippen / fdef / 元素量）
│   └── comps/            # 通用面板组件（随主程序同步，勿手改）
├── flavor-core/          # Rust 推理核（前向 + wasm 绑定），wasm-pack 产物在 pkg/
│   ├── src/model.rs      # 与 py/flavor_np.py 逐算子等价的 GATv2 + MLP 前向
│   ├── src/weights.rs    # 扁平权重加载（.json 索引 + .bin f32）
│   ├── src/wasm.rs       # wasm-bindgen 导出 load_weights / predict / logits
│   └── verify-wasm.mjs   # 在 node 里验证 wasm 链路（偏差 + 耗时）
├── flavor/               # 第三方模型与数据
│   ├── pharma33_weights.bin/.json   # 扁平 f32 权重（wasm 用，6.30 MB）
│   ├── pharma33_weights.npz         # numpy 版权重（对拍用）
│   ├── labels.json                  # 138 个气味标签（顺序与模型输出一致）
│   ├── best_model_pharma33.pt       # 原 checkpoint（torch）
│   ├── model.py                     # 原作者训练代码
│   └── data1/ data2/                # 原作者数据集（仅用于校验，不随插件分发）
├── py/                   # **仅开发用**：Python 参考实现与校验脚本（不参与运行时）
├── tools/                # 开发用校验脚本（化学层对拍 / 2D vs 3D 差异 / WebView 冒烟）
├── pack.py               # 打包 wmview.flavor.zip
└── notes/latest.md       # 修改记录（追加式）
```

## 工作原理

1. 面板取分子：**场景分子**（`wmapi_files.get_mole_info` 的原子符号/坐标/连接表，或直接读
   原分子文件拿键级）或 **SMILES**。
2. 前端化学层（`src/chem/`）：rdkit.js 解析与 sanitize → 芳香性/成环 → 加氢 → 逐原子
   Crippen / TPSA / Gasteiger → `BaseFeatures.fdef` 药效团识别 → 特征组装。
   场景分子没有显式氢时先按坐标补氢，再由 `bondorder.ts` 依价键规则推断键级。
3. wasm 前向（`flavor-core`）：节点/边编码器 → 5 层 GATv2Conv（8/8/4/4/2 头，残差 +
   LayerNorm + ReLU）→ 全局平均池化 → 药效团 MLP → 拼接 → 4 层 MLP → sigmoid → 138 个概率。
4. 面板按阈值/条数展示，可导出 CSV；命中项写入结果日志。

运行时资源（都随插件包发布，在 `assets/` 下）：

| 资源 | 体积 | 说明 |
| --- | --- | --- |
| `RDKit_minimal-*.wasm` | 7.33 MB | rdkit.js 的 MinimalLib 构建（2026.03.6） |
| `flavor_core_bg-*.wasm` | 123 KB | Rust 前向核 |
| `pharma33_weights-*.bin` | 6.30 MB | 扁平 f32 权重 |
| `pharma33_weights-*.json` | 6.7 KB | 张量索引 |

首次打开面板时后台加载，实测本机整条链路（fetch 14 MB + 两个 wasm 实例化 + 权重装配）
约 240 ms；之后每个分子几十毫秒。

## 精度与校验

三层校验，都能复现（见「验证」一节）：

| 校验 | 比较对象 | 结果 |
| --- | --- | --- |
| 化学层（`tools/validate-chem.ts`） | 本机 RDKit，**同一份 molblock** 双侧输入 | 逐原子 Crippen / TPSA、原子特征 x、edge_index、键特征 **偏差 0**；药效团清单 12/12 完全一致 |
| 化学层同上 | 同上 | 逐原子 Gasteiger 电荷 ≤0.15（杂化靠经验规则推断）；最终概率最大偏差 **1.6e-3**，top-1 12/12 |
| 前向核（`node flavor-core/verify-wasm.mjs`） | `py/flavor_np.py`（与 torch + PyG 逐分子一致，偏差 ~1e-7） | 最大 logit 偏差 **4.77e-7**、概率偏差 8.94e-8，top-1 12/12 |
| 数据集 | `flavor/data1/val.csv` + `val-sdf` | macro-AUC 0.8833，与 checkpoint 记录的 val_auc 0.8796 吻合 |

> 化学层与 RDKit 的对照是**逐项**做的（不是只看最终概率）：Crippen 用 `Data/Crippen.txt`
> 的 110 条 SMARTS，TPSA 按 `MolSurf.cpp` 的查表逻辑，Gasteiger 按 `GasteigerParams.cpp`，
> 药效团由 `BaseFeatures.fdef` 驱动并按 `FeatureParser.cpp` / `MolChemicalFeatureFactory.cpp`
> 复刻（续行、`!` 取反、`$(...)` 合并、同族超集去重、匹配原子顺序）。

## 已知限制

1. **SMILES 路径只有 2D 坐标**（影响最大的一条）。rdkit.js 不提供构象生成（ETKDG/MMFF），
   所以 SMILES 走 RDKit 的 2D 平面坐标，而 11 维药效团特征里含**绝对坐标**与中心距统计，
   模型又是在 3D 构象上训练的。前端（2D）与 `flavor.exe`（ETKDG + MMFF94 的 3D）在 12 个
   val 分子上的差异（`npx tsx tools/compare-3d.ts`）：

   | 指标 | 结果 |
   | --- | --- |
   | 最大 Δp | 0.36 |
   | 平均 Δp | 0.21 |
   | top-1 标签一致 | 9/12 |
   | top-10 标签平均重叠 | 7.1/10 |

   想要准确结果就**用场景分子**（真实 3D 坐标，与 RDKit 的差异就是上表之外的那 1.6e-3）。
   要彻底解决得在前端做构象生成（移植 ETKDG/MMFF，或让宿主提供 3D 坐标），目前没做。
2. **Gasteiger 电荷**依赖杂化推断，与 RDKit 有 ≤0.15 的逐原子偏差（对最终概率影响 ~1e-3）。
3. **场景分子的键级靠推断**（宿主只给原子/坐标/连接表，没有键级）。带显式氢时前端做
   **精确求解**（搜索 + 剪枝，要求每个原子的价被恰好填满），12 个 val 分子全部还原正确、
   概率与参考实现的最大偏差 0.002（见 `tools/test-bondorder.ts`）。**没有显式氢时问题本身
   欠定**（氢数和双键数是同一个未知量，连 Python 版的 `rdDetermineBonds` 也会解错——无氢苯
   会得到 `c1c#cc#cc#1`），前端只按键长挑出「有把握的多重键」（羰基、烯、炔），芳香键
   刻意留成单键交给氢补，同 12 个用例还原 10/12、最大偏差 0.26，并且**始终给出告警**。
   硝基一类电荷分离结构仍会退回贪心并告警。
4. rdkit.js（MinimalLib）的 SMARTS 解析器处理不了 RDKit 原样合并的嵌套 `$()`，已改等价
   扁平写法回退；`get_coords()` 返回 JS 数组而不是 JSON、`get_substruct_matches()` 无匹配
   时返回 `{}`，这两处坑也都已在 `rdkit.ts` 里挡住。

## 输入与限制

| 输入 | 键级来源 | 几何来源 | 说明 |
| --- | --- | --- | --- |
| 场景分子（原文件可读） | 分子文件自带 | 文件里的 3D 坐标 | **最准确**，推荐 |
| 场景分子（只有坐标 + 连接表） | 带显式氢：精确求解；无显式氢：按键长挑明确的多重键 | 场景坐标（3D） | 带氢时与参考实现一致；无氢时芳香环会退化并告警 |
| SMILES | RDKit 解析 | RDKit 的 2D 平面坐标 | 最省事，精度见「已知限制」第 1 条 |
| 场景分子（无显式氢且读不到文件） | 退化：按单键处理 | 场景坐标 | 结果仅供参考，面板会给警告 |

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
npm run build        # type-check + wasm-pack + vite build → dist/
npm run build-only   # 只跑 vite（没改 Rust 时够用）
python pack.py       # 打包 wmview.flavor.zip
```

改了 Rust 必须重编 wasm（`npm run build` 已包含）：

```sh
npm run build:wasm   # 需要 rustup 的 wasm32-unknown-unknown 目标 + wasm-pack
```

首次运行 wasm-pack 会下载 wasm-bindgen / wasm-opt 到 `%LOCALAPPDATA%\.wasm-pack`，之后离线可用。

> 构建需要 `flavor/` 里的三个运行时资源：`pharma33_weights.bin`、`pharma33_weights.json`、
> `labels.json`（`.gitignore` 里 `flavor` 整个目录被忽略，换机器时要把这几个文件带过去，
> 或先跑 `py/export_weights_flat.py` 重新导出）。`flavor-core/pkg/` 同理属于构建产物，
> 需要 `npm run build:wasm` 生成。

`py/` 是**开发用的参考实现**，不参与运行时，主要用途：

```sh
build/venv/Scripts/python.exe py/export_weights_flat.py    # npz → 扁平 f32（给 wasm）
build/venv/Scripts/python.exe py/export_tables.py          # RDKit 数据表 → src/chem/tables
build/venv/Scripts/python.exe py/export_chem_oracle.py     # 化学层对拍基准
build/venv/Scripts/python.exe py/export_parity.py --n 12   # 前向对拍用例
py/validate_numpy.py --split val --n 8                     # numpy 实现校验（需 torch + PyG）
flavor.exe --smiles "CCCOC(=O)C=Cc1ccco1" --topk 10        # 旧的 Python 版本（已不再随包分发）
```

## 验证

```sh
npx tsx tools/validate-chem.ts        # 化学层 vs 本机 RDKit（12 分子逐项对拍）
npx tsx tools/test-bondorder.ts       # 键级推断 vs 真值（只有坐标+连接表时的还原率）
node flavor-core/verify-wasm.mjs      # wasm 前向 vs numpy 参考
npx tsx tools/compare-3d.ts           # SMILES 2D vs Python 3D 的差距（需要 flavor.exe）
```

WebView 冒烟：在**宿主里**跑一次完整推理，验证 wasm 与权重真的能加载。

```sh
npx vite build --config tools/smoke.vite.config.ts   # 产出 build/smoke/
# 把 build/smoke 拷到 <plugs>/wmview.flavor/smoke/，然后在宿主里（端口见 port.json）：
POST http://127.0.0.1:<port>/run_script
{"script":"await import('http://asset.localhost/plugs/wmview.flavor/smoke/smoke.js');\
 return JSON.stringify(await window.smoke_('CCCOC(=O)C=Cc1ccco1'))"}
```

实测输出：

```json
{"runtime":{"rdkit":"2026.03.6","classes":138,"labels":138,"features":27,"loadMs":235},
 "reply":{"ok":true,"smiles":"CCCOC(=O)C=Cc1ccco1","atoms":13,"pharmacophores":10,
          "top":[{"label":"sweet","prob":0.5874},{"label":"fruity","prob":0.5785}]}}
```

## 约定

- 插件以 **Module Federation remote** 构建，remote 名为 `wmview_flavor`（安装目录
  `plugs/wmview.flavor/` 由主程序换算得到），入口文件名固定 `index.js`。
- 依赖 `vue` / `element-plus` 由主程序以 singleton 提供，插件不得把它们打进产物。
- 与主程序交互**只用 `window.wmapi_*` API**（类型以 `wmapi.d.ts` 为准），不引用主程序内部模块。
- 插件资源用相对地址（vite `base: './'`）引用：宿主把插件目录挂在
  `http://asset.localhost/plugs/wmview.flavor/` 下，所以
  `new URL('assets/x.wasm', import.meta.url)` 能直接 fetch 到。
- rdkit.js 的 MinimalLib 构建**不读** `Module.wasmBinary`，又会在宿主 WebView 里把 wasm 路径
  算成宿主页面目录（取到 index.html，报 "expected magic word"）；必须用 `instantiateWasm`
  接管实例化，见 `src/chem/rdkit.ts`。

## 宿主 API 版本（重要）

程序自带的 `plugin-deps/wmapi.d.ts` 模板**已过期**，它写的是 `wmapi_files.get_moleInfo(name)`（同步）、
`get_files()`、`get_selects()`，而程序实际暴露的是：

| 模板里（不存在） | 程序实际提供 |
| --- | --- |
| `get_moleInfo(name)` | `get_mole_info(name): Promise<MoleInfo>`（**异步**） |
| `get_files()` | `get_file_list()` |
| `get_selects()` | `get_select()` / `get_option()` / `on_select(cb)` / `on_option(cb)` |
| `show_file(name)` | `load_file(name, show)`（`show=false` 为静默加载） |

本仓库的 `wmapi.d.ts` 已按程序的声明同步（也可以直接读 `%LOCALAPPDATA%\wmview\wmapi.d.ts`，
那是程序运行时写出来的最新版）。注意 `wmview-plugin update` 会用模板覆盖这个文件，
更新后需要重新同步。

最近一次同步（对应主程序当前声明）：`wmapi_cores` 增加 `add_appdoc` / `show_doc`；
`wmapi_files` 的 `show_file` 改为 `load_file(name, show)`、增加 `on_option`；
`wmapi_scene` 增加 `set_atom_select` / `on_atom_select`，`add_systm` 的 `dirs` 需给 3 个坐标轴单位向量。

面板读取分子结构时做了三层兼容：`get_mole_info` → 旧模板 `get_moleInfo` → `wmapi_scene.get_mole_geom`；
工具栏的「接口自检」按钮会把宿主实际暴露的接口清单与推理核状态写进日志。

### 拿 API 的正确姿势

运行中的 wmview 内置本地 HTTP 服务（端口写在 `%LOCALAPPDATA%\wmview\port.json`），可以直接问程序本身：

```sh
GET  http://127.0.0.1:<port>/get_apidoc      # 核心 + 已加载插件的 API 文档
POST http://127.0.0.1:<port>/run_script      # {"script":"return Object.keys(window.wmapi_files)"}
```

比翻 `plugin-deps/wmapi.d.ts` 模板可靠得多（模板已经漂移过：`get_moleInfo` / `get_files` /
`get_selects` 都不存在）。面板的「接口自检」按钮就是干这件事的。

主程序 dev 模式默认连接 `http://localhost:3001/index.js` 加载开发中的插件（面板外层有蓝色环、
名字带 `@dev`）；正式安装版需在 插件 → 开发指南 中打开开关。
