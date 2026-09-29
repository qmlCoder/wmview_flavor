/**
 * 把 `tools/smoke.ts` 打成独立小包，用于在**真实 WebView 里**验证运行时链路。
 *
 * 为什么要这么绕：插件本体只暴露 `plugin`，内部的化学层与推理核不对外导出；
 * 而这几行代码要验证的恰恰是「宿主环境里能不能 fetch 到 wasm/权重、rdkit.js 能不能
 * 初始化、前向能不能跑」。单独打一个小包丢进插件目录，再用宿主 API import 即可。
 *
 * 用法（见 README「验证」一节的第三条）：
 *   npx vite build --config tools/smoke.vite.config.ts
 *   把 build/smoke 拷到 <plugs>/wmview.flavor/smoke/，然后在宿主里 import 它的 smoke.js
 */
import { fileURLToPath, URL } from 'node:url'

import { defineConfig } from 'vite'

export default defineConfig({
  base: './',
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('../src', import.meta.url)),
    },
  },
  build: {
    outDir: fileURLToPath(new URL('../build/smoke', import.meta.url)),
    emptyOutDir: true,
    minify: false,
    rollupOptions: {
      input: fileURLToPath(new URL('./smoke.ts', import.meta.url)),
      // 默认 app 构建会把入口的导出当作「没人用」而删掉，冒烟测试要靠它拿到 smoke()
      preserveEntrySignatures: 'strict',
      output: {
        entryFileNames: 'smoke.js',
        chunkFileNames: '[name]-[hash].js',
        assetFileNames: 'assets/[name]-[hash][extname]',
      },
    },
  },
})
