import type { Plugin } from './plugin'
import Sider from './Sider.vue'
import icon from './icon.png'

/**
 * 插件契约（见 AGENTS.md / plugin.d.ts）
 *
 * 必须同时导出：
 * - `plugin` 命名导出：本地安装加载（主程序读 module.plugin）
 * - `default` 导出：开发模式加载（主程序读 module.default）
 *
 * 其它可选字段：
 * - apidoc?: string      补全/悬停用的 API 文档（d.ts 文本）
 * - setting?: { name, comp }  设置页 tab
 * - scene_color?: string 选中该插件时场景背景色
 */
export const plugin: Plugin = {
  name: 'flavor',
  sider: { name: 'flavor', icon: icon, comp: Sider },
  position: 'right',
  // 推理走进程内 wasm（不依赖 run_exe 执行外部程序），所以 Windows 与 Android 都能用
  device: ['windows', 'android'],
}

export default plugin
