import { fileURLToPath, URL } from 'node:url'

import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import vueDevTools from 'vite-plugin-vue-devtools'
import { federation } from '@module-federation/vite'

export default defineConfig({
  plugins: [
    vue(),
    vueDevTools(),
    federation({
      // 名称规则：安装目录 wmview.<短名> 对应 remote 名 wmview_<短名>（'.' -> '_'）。
      // 本插件安装目录为 wmview.flavor，因此 remote 名是 wmview_flavor；
      // 与主程序 dev 联调时若需要自动加载，再临时改成 'wmview_xtb'（见 AGENTS.md 第 5 节）。
      name: 'wmview_flavor',
      filename: 'index.js', // 入口文件名固定，主程序只认 index.js
      exposes: {
        './index': './src/index.ts', // 与插件入口文件保持一致（源码统一放 src/）
      },
      // 暴露模块时把组件样式一起带上，避免消费端拿不到 CSS
      bundleAllCSS: true,
      // 插件侧不生成给消费端用的类型（类型由 wmapi.d.ts + 插件 d.ts 维护）
      dts: false,
      // vue/element-plus 由主程序(MF Host)以 singleton 提供，插件不再各自打包
      shared: {
        vue: { singleton: true, requiredVersion: false },
        'element-plus': { singleton: true, requiredVersion: false },
      },
    }),
  ],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  // 插件产物被宿主放到 plugs/wmview.flavor/ 子目录下加载（asset.localhost），
  // 必须使用相对 base，否则产物里的资源引用会从宿主根目录解析而 404。
  base: './',
  server: {
    // 固定端口：主程序 dev 启动时自动加载 http://localhost:3001/index.js
    port: 3001,
    strictPort: true,
    cors: true,
    hmr: true,
    origin: 'http://localhost:3001',
  },
  build: {
    // 保持产物可读，便于调试
    minify: false,
  },
})
