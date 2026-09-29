# 插件开发 AI 指南（AGENTS.md）

> 本文件随 plugin-deps 模板包一起分发到插件项目，供在插件仓库中工作的 AI 编码助手
> 阅读。目标：让 AI 能按 wmview 插件规范修改代码、正确构建，并避免破坏插件与主程序的
> 单实例共享约定。

## 1. 插件是什么

一个 wmview 插件 = 一个 **Module Federation remote**（Vite 项目）+ 一份 `Plugin` 契约导出。
主程序（wmview_app）是 MF **Host**，以 singleton 方式向插件提供 `vue` / `element-plus` / `three`，
插件**不能**把这些库打进自己的产物，也不能再走 `window.vue`/`window.three` 之类的全局注入。

## 2. 项目文件约定

模板（本目录顶层文件）会被 `wmview-plugin init` 直接解压到插件项目根目录：

- `index.ts` —— 插件入口：**必须同时导出**
  - `export const plugin`：本地/生产安装加载用（主程序读 `module.plugin`）
  - `export default plugin`：开发模式加载用（回退读取 `module.default`）
- `Sider.vue` —— 插件主界面组件（侧边栏面板）
- `comps/Panel-Group.vue`、`comps/Panel-Item.vue` —— 通用面板组件，**每次随主程序
  `src/comps` 同步拷贝**，改动只应在主程序侧发生，本目录里不要手工改这两个文件
- 实际插件项目还会自己维护：`package.json`、`tsconfig*`、`src/`（业务代码）、
  `vite.config.ts` 等

### Plugin 契约字段

```ts
interface Plugin {
  name: string                                  // 唯一名，如 'myplugin'
  sider: { name: string; icon: string; comp: Component }
  setting?: { name: string; comp: Component }   // 可选：设置页 tab
  scene_color?: string                          // 可选：选中插件时场景背景色
  apidoc?: string                               // 可选：供脚本补全/悬停的 API 文档文本
  position: 'left' | 'right'
}
```

## 3. 技术约束（务必遵守）

1. **共享依赖单实例**：`vue`、`element-plus`、`three` 由主程序以 singleton 提供。
   插件侧 import 后**禁止二次打包**（不得在产物里出现第二份 vue/element-plus/three），
   否则会出现 Element Plus 全局注册失效、`provide/inject`（含 `inject('scene')`）、
   Pinia 跨实例等疑难 BUG。
2. **样式**：主程序已全局加载 `element-plus` 样式与通用组件所需 CSS，插件正常使用即可。
3. **不要直接依赖主程序内部实现**：不能 `import` 主程序的 `@/stores/*`、`@/scripts/*`、
   `src/plugin/*` 等私有模块；能力一律通过 **`window.wmapi_*` API** 调用。
4. **场景对象**：主程序通过 Vue `provide('scene', scene)` 提供场景对象，
   插件里 `const scene = inject('scene')` 取得（类型参考 `WmObj.d.ts`）。
   仅此一个 provide 有保证，不要假设其它 inject。
5. **图标**：`sider.icon` 给可访问的图片路径；主程序在侧边栏**原样显示**该图片（仅本地/开发插件外层加彩色环），
   建议提供透明背景、主体居中、接近方形的 PNG，以彩色图标效果最佳。

## 4. wmapi：插件与主程序交互的能力桥

> 本模板自带 `wmapi.d.ts`，内含 **wmapi_cores / wmapi_files / wmapi_scene** 的完整类型声明，
> 插件代码与 AI 都以它为准（编辑它即可获得补全与检查）。

所有能力挂在 `window` 上（其余如 editor/script/wfana 的 d.ts 不在模板内，运行期可用
`window.wmapi_script.get_apidoc()` 拉取完整 d.ts 文本喂给 LLM/补全）：

- `wmapi_cores` —— 核心：`run_exe`(运行外部程序)、`notify`、`add_reslog`/`add_logText`、
  `show_loading`/`hide_loading`、`read_text`/`save_text`、`save_file_dialog`、
  `set_bottom_message`、`get_show_siders` 等
- `wmapi_files` —— 文件/根目录：`wait_root`、`get_rootFold`、`get_moleFold`、
  `get_plugFold`、`load_file(name, show)`、`get_file_list` 等
- `wmapi_scene` / `wmapi_editor` / `wmapi_script` / `wmapi_wfana` —— 场景、建模、
  脚本、波函数分析等按需查阅对应 d.ts

使用要点：
- 目录类接口（`readDir`/`readTextFile`、`get_rootFold` 等）之前先 `await wmapi_files.wait_root()`，
  否则根目录还是空字符串，会拼出 `/plugs` 这类错误路径；
- `run_exe` 事件流：stdout/stderr 以事件 `{source}:RunExe:stdout|stderr` 推送，
  用 `@tauri-apps/api/event` 的 `listen` 接收（source 需与主程序约定的前缀一致）。
- `run_exe` 返回值：`Promise<[boolean, string]>`，`[是否成功退出, 完整输出(stdout+stderr)]`。

## 5. 构建与运行

```bash
npm install
npm run dev        # 插件 dev server，固定端口 3001；主程序会自动加载(蓝色环/名字带 @dev)
npm run type-check # 类型检查（改动后必须通过）
npm run build      # 产出 dist/（含 index.js 入口 + assets，供本地安装）
```

开发中插件的加载：主程序（开发模式与正式安装版都）默认连接 localhost:3001，没起 dev server 时
静默跳过；不需要联调时可在 插件 → 开发指南 里关掉开关（安卓端需先
`adb reverse tcp:3001 tcp:3001`）。开发服务器重启或插件代码改动后重新加载界面(Ctrl+R)。

生产/本地安装：把 `dist/` 内容（不是整个 dist 目录）拷到主程序 `plugs/<插件目录>/`，
入口文件为 `index.js`，主程序会自动识别并加载。
插件侧如用 Module Federation，请把 `federation({ filename: 'index.js', ... })` 保证产物入口叫 index.js。

## 6. AI 工作注意事项

- 开工前先读：本 AGENTS.md、`index.ts`、`Sider.vue`、`vite.config.ts`、相关 `*.d.ts`；
- 涉及 API 时不要凭记忆写，去对应 `wmapi*.d.ts` 里核对方法签名与参数；
- 每次改动后跑 `npm run type-check`（必须零错误），涉及构建/产物再跑 `npm run build`；
- UI 优先复用 `comps/Panel-Group.vue` / `Panel-Item.vue` + Element Plus 组件，
  不要为相似面板造新的布局轮子；
- 提交信息遵循简洁的 Git 风格（subject ≤50 字符、动词开头）。

## 7. 本插件补充约定（wmview.flavor）

> 本文件会被 `wmview-plugin update` 覆盖，项目自身的说明以 `README.md` 与 `notes/latest.md` 为准。

- **前端源码统一放在 `src/`**（`src/index.ts` 入口、`src/Sider.vue` 面板、`src/comps/` 通用组件），
  与第 2 节描述的模板根目录布局不同；`vite.config.ts` 的 `exposes`、`index.html` 入口脚本与
  `tsconfig.app.json` 的 include 均按 `src/` 配置。`wmview-plugin update` 往根目录重新落地的
  模板文件（index.ts / Sider.vue / comps）需要手工搬进 `src/` 并同步上述三处路径。
- **只用 `window.wmapi_*` API 与主程序交互**，不读取、不引用主程序仓库的源码/内部模块。
- **运行时是纯前端 + Rust→wasm，没有任何外部进程**：化学感知在 `src/chem/`（rdkit.js +
  按 RDKit 源码移植的逐原子描述符与药效团），前向核在 `flavor-core/`（纯 Rust，编成 wasm），
  两者由 `src/chem/browser.ts` 装配、`src/chem/pipeline.ts` 串起来。**不要**改回 `run_exe`：
  它在 Android 沙箱里不允许执行私有目录的二进制，改回就等于放弃移动端。
- 权重由 `flavor/pharma33_weights.bin` + `.json`（扁平 f32）提供，来自
  `py/export_weights_flat.py`；表（Crippen / fdef / 元素量）来自 `py/export_tables.py`。
  改了 Rust 必须 `npm run build:wasm` 重编 wasm（`npm run build` 已包含）。
- 修改特征或前向实现后，务必跑 `npx tsx tools/validate-chem.ts`（化学层 vs 本机 RDKit，
  逐项偏差应为 0 / ≤1.6e-3）与 `node flavor-core/verify-wasm.mjs`（wasm vs numpy，
  阈值：最大 logit 偏差 < 1e-4）。
- `py/` 是**开发用的参考实现**（导表、导权重、对拍基准、旧的 exe 版本），不参与运行时，
  也不随插件包分发；改动 `py/` 不需要重新打包插件。
- **`wmapi.d.ts` 是宿主声明的同步副本**：模板自带的版本已过期（写成 `get_moleInfo` / `get_files` /
  `get_selects`，程序里没有），更新模板后需按 `src/wmapi.d.ts`、`src/plugin/filelist/wmapi.d.ts`、
  `src/plugin/scene/wmapi.d.ts` 重新同步；调用宿主 API 前先确认方法名与是否异步。
