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
