# 修改记录

> 每次功能/修复完成后在本文件**追加**要点（沿用主程序约定，如 `- 修复:...`）。

- 初始化:插件工程骨架（`wmview-plugin init` 落地 AGENTS.md / index.ts / Sider.vue / comps / wmapi.d.ts）
- 初始化:补齐 package.json、vite.config.ts、tsconfig*、env.d.ts、index.html、plugin.d.ts、info.json、pack.py
- 新增:PharmaGNN 气味预测推理链路（`py/flavor_feats.py` 特征 + `py/flavor_np.py` numpy 前向）
- 新增:权重导出脚本 `py/export_weights.py`，把 `best_model_pharma33.pt` 转为 `flavor/pharma33_weights.npz`（5.8 MB）
- 校验:numpy 前向与 torch + PyG 参考实现逐分子一致（最大 logit 偏差 ~1e-7）
- 校验:data1 验证集 macro-AUC 0.8833，与 checkpoint 记录的 val_auc 0.8796 吻合
- 新增:运行时入口 `py/predict.py`（stdin JSON → `@@FLAVOR_RESULT@@` JSON），支持 SMILES / SDF / 场景原子坐标三种输入
- 新增:`build_exe.py` 用 PyInstaller 打包单文件 flavor.exe（58 MB，内置 numpy + rdkit + 权重，无需 pytorch）
- 新增:插件面板支持分子来源切换、阈值/条数调节、结果概率条展示与 CSV 导出
- 变更:pack.py 打包项加入 flavor.exe
- 新增:侧边栏图标 icon.png（苯环字形，透明背景 256×256）
- 修复:场景分子优先读取原分子文件的键级；无显式氢且读不到文件时给出警告并降级处理（原先会输出乱电荷的无效分子）
- 校验:exe 三种输入（SDF / SMILES / 场景原子坐标）结果一致；磷酸钙判为 odorless 0.95、布洛芬判为 fatty/oily/pungent
- 修复:宿主 API 改名导致的报错 `wmapi_files.get_moleInfo is not a function`——程序实际是
  `get_mole_info(name)`（异步），模板 d.ts 已过期；`wmapi.d.ts` 按程序声明重新同步
- 新增:分子结构读取三层兼容（get_mole_info → get_moleInfo → wmapi_scene.get_mole_geom）
- 新增:面板「接口自检」按钮，输出宿主实际暴露的 wmapi_* 接口清单
- 核对:改用 wmview 技能的本地 HTTP 服务（`/get_apidoc`、`/run_script`）核对运行中程序的真实 API——
  `wmapi_files` 实际方法为 get_file_list / get_moleFold / get_mole_info / get_option / get_plugFold /
  get_rootFold / get_select / on_select / show_file / wait_root，确认没有 get_moleInfo
- 修复:`get_plugFold()` 返回的是 **plugs 根目录**（不是插件目录），flavor.exe 路径应为
  `<plugs>/wmview.flavor/flavor.exe`；同时补上「程序路径」输入框便于开发联调时手动指定
- 修复:路径不存在时 `run_exe` 会抛异常（而不是返回 false），改为逐个候选路径 try/catch 尝试
- 校验:在运行中的程序里用真实宿主数据跑通整条链路——`dingerxi.gjf`（丁二烯）→ spicy 0.607 / citrus 0.583 / herbal 0.560
- 新增:面板底部加入「插件介绍」与「使用教程」两个面板（教程用 el-collapse 分快速上手 / 分子来源 / 看懂结果 / 常见问题四节，默认展开第一节）
- 新增:推理核 wasm 化第一步——`flavor-core/`（纯 Rust）：GATv2 + MLP 前向、扁平权重加载、原生 CLI（含 `--parity` 对拍）、`wasm-bindgen` 导出
- 新增:`py/export_weights_flat.py` 把 npz 转成扁平 f32（`pharma33_weights.bin` + .json 索引，6.30 MB），供 wasm 一次 fetch 直接使用
- 新增:`py/export_parity.py` 用 RDKit + numpy 导出前向对拍用例（特征 + 参考 logits/概率）
- 校验:flavor-core 原生 CLI 与 numpy 对拍 12 个用例，最大 logit 偏差 4.77e-7、最大概率偏差 8.94e-8、top-1 标签 12/12 一致（阈值 1e-4）
- 校验:wasm 产物 123 KB，权重加载 8.5 ms，单分子前向 22 ms；node 里跑同一批用例最大概率偏差 7.53e-8、top-1 全一致
- 侦察:实测预编译 `@rdkit/rdkit`（MinimalLib，7.3 MB wasm，与 venv 的 rdkit 2026.03.6 同版本）能力边界——提供 get_mol/get_json（含 aromaticAtoms/aromaticBonds/atomRings）/递归 SMARTS 匹配/add_hs（保留 3D 坐标）/get_coords；但无逐原子 Crippen/TPSA、无 Gasteiger、无 3D 构象生成（ETKDG/MMFF）、无 DetermineBonds
- 侦察:量化模型敏感度——构象噪声（ETKDG 换种子 vs 平面 2D）maxΔp≈0.027、top10 重叠≈90%；逐特征分量清零影响：TPSA 0.080 > MR 0.036 > xyz/距离 0.035 >> Gasteiger 电荷 0.003 ≈ logP 0.002；小幅数值噪声（1e-2）几乎无影响
- 新增:化学层整体搬到前端（`src/chem/`）——rdkit.js 做解析/芳香性/成环/SMARTS/加氢，逐原子 TPSA 与 Crippen 按 RDKit 源码+数据表移植，Gasteiger 按 `GasteigerParams.cpp` 移植，药效团由 `BaseFeatures.fdef` 驱动，另含场景分子键级推断与端到端管线
- 新增:`py/export_tables.py` 从本机 RDKit 导出前端表（Crippen 110 条 SMARTS、fdef、元素量）；`src/chem/tables/`
- 新增:`py/export_chem_oracle.py` + `tools/validate-chem.ts` 化学层对拍（同一 molblock 双侧输入）
- 校验:化学层对拍 12 个 val 分子——逐原子 Crippen/TPSA、原子特征 x、edge_index、键特征均为 **0 偏差**；药效团特征清单 **12/12 完全一致**；最终概率最大偏差 1.6e-3、top-1 标签 12/12 一致（残余偏差来自 Gasteiger 杂化推断，≤0.15）
- 修复:rdkit.js 三处实测坑——`get_coords()` 返回 JS 数组非 JSON、`get_substruct_matches()` 无匹配返回 `{}`、解析不了 RDKit 原样合并的嵌套 `$()`（Donor/Acceptor）已改等价扁平写法回退
- 修复:`GetTotalDegree()` 需计入隐式氢；TPSA 的键计数需跳过与氢相连的键（否则 O–H 分支匹配不上）
- 变更:前端源码统一收进 `src/`（`index.ts` / `Sider.vue` / `comps`），同步更新 vite `exposes`、`index.html` 入口与 tsconfig include
- 变更:**推理整体改为纯前端 + wasm**——`Sider.vue` 不再 `run_exe`，改为 in-process 调用
  `src/chem/pipeline.ts`（rdkit.js 特征 + `flavor-core` wasm 前向）；新增 `src/chem/browser.ts`
  负责装配运行时（fetch 两个 wasm + 权重 + 标签 + fdef）
- 变更:构建流程接入 wasm——`npm run build` = type-check + `build:wasm`(wasm-pack) + vite build；
  wasm/权重作为 vite 资源发布到 `dist/assets`，`base: './'` 下用 `new URL(..., import.meta.url)` 取到
- 变更:`pack.py` 不再打包 `flavor.exe`（安装包从 ~62 MB 降到 ~14 MB，全是 wasm 与权重）
- 变更:`info.json`/`index.ts` 的设备限制去掉——`device: ['windows', 'android']`（不再依赖可执行文件）
- 修复:rdkit.js(MinimalLib) 在宿主 WebView 里把 wasm 路径算成宿主页面目录（取到 index.html，
  报 `expected magic word 00 61 73 6d, found 3c 21 64 6f`），且它不读 `Module.wasmBinary`；
  改为在 `initChem` 里用 `instantiateWasm` 接管实例化（`locateFile` 兜底）
- 修复:rdkit.js 加载失败时 `modulePromise` 会缓存被拒的 promise，导致后续重试直接失败；
  现在 `loadFlavorRuntime` 失败会清缓存，「推理核」面板也提供了重试按钮
- 校验:在运行中的 wmview WebView 里跑通整条链路（`tools/smoke.ts` → asset.localhost 取
  wasm 与权重）——RDKit 2026.03.6、138 类、27 条药效团定义、整链路加载 235 ms、
  `CCCOC(=O)C=Cc1ccco1` → sweet 0.587 / fruity 0.579 / caramellic 0.564
- 新增:`tools/compare-3d.ts` 量化 SMILES 路径的近似代价（前端 2D 坐标 vs exe 的 ETKDG+MMFF 3D）：
  12 个 val 分子最大 Δp 0.36、平均 0.21、top-1 9/12、top-10 平均重叠 7.1/10
- 变更:面板与文档如实标注 SMILES 路径的精度（药效团特征含绝对坐标项，2D 会偏粗），
  引导用户优先用场景分子；「程序路径」输入框随 exe 一起移除，换成「推理核」状态显示
- 校验:跨实现交叉验证——同一份场景分子负载（`lianxi-onsym.gjf` 的 syms/xyzs/bonds）下，
  前端与 `flavor.exe` 的结果逐标签一致（spicy 0.6345 vs 0.634452、sweet 0.5620 vs 0.562023、
  pungent 0.5611 vs 0.561079 …）
- 校验:在运行中的程序里点真实面板走通端到端——重新加载界面后「推理核」显示
  `已就绪 · RDKit 2026.03.6 · 138 个标签 · 载入 215 ms`，点「预测气味」得到
  `命中 6 项（≥ 0.50）／共 138 项 · C=C=C · 3 个重原子 · 3 个药效团`，概率条与 exe 一致
- 修复:场景分子键级推断由「按键序贪心」改为**精确求解**（搜索 + 剪枝 + 键长先验，
  要求每个原子的价恰好填满，无解才退回贪心并告警）。根因：贪心依赖连接表顺序，
  丁二烯只要中间骨架键先出现就会把双键放错位置、末端碳多补一个氢 → 结构变成 2-丁烯
- 修复:`buildMolblock` 的程序行把「3D」写到 MDL 规范要求的第 21–22 列（原先写在 27 列），
  RDKit 才能正确按 3D 判断立体化学
- 新增:无显式氢时改用「只认键长上明确的多重键」（严格阈值），芳香键刻意留单键交给氢补，
  避免把无氢苯这类欠定情形猜成累积双键
- 新增:`tools/test-bondorder.ts` 键级推断回归测试——拿 12 个 val 分子的真实 molblock 当基准，
  只喂「原子 + 坐标 + 连接表」，比对还原出的结构与实际概率
- 校验:键级推断效果——**带显式氢 12/12 还原正确**（改前 8/12），与参考实现最大概率偏差
  0.002（改前最大 0.5 量级）；无显式氢 10/12（改前 0/12），最大偏差 0.26（改前 0.41），
  剩下 2 个是刻意不猜的芳香环且会告警
- 校验:用真实文件复验——`dingerxi.gjf`（丁二烯）从 2-丁烯恢复为 `C=CC=C` 且不再告警，
  top 标签 spicy 0.6074 / citrus 0.5835 / herbal 0.5596 与 Python 版一致；面板显示
  `命中 6 项（≥ 0.50）／共 138 项 · C=CC=C · 4 个重原子 · 4 个药效团`
- 同步主程序 API: 按 wmview_app 当前声明更新 src/wmapi.d.ts（模板副本已漂移的那部分）
  - wmapi_cores: 新增 add_appdoc(path, content) / show_doc(path)
  - wmapi_files: show_file(name) → load_file(name, show)；新增 on_option(cb)（get_select / get_option / on_select 已在）
  - wmapi_scene: 新增 set_atom_select(mole_name, atms) / on_atom_select(cb)；
    add_systm 的 dirs 明确为 3 个坐标轴单位向量（主程序已加校验，少一项会抛错）
  - plugin.d.ts: device 由可选改为必填（与主程序 Plugin 声明一致，入口 index.ts 已提供）
- 发布: 版本 0.1.1 上架插件市场 —— 包上传到 OSS `plugins/flavor/0.1.1/wmview.flavor.zip`，
  图标同步 `plugins/flavor/icon.png`，市场库 `public.plugins` 同步为 version=0.1.1、platform=windows,android
