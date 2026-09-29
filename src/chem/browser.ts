/**
 * 浏览器 / 插件侧的推理运行时：一次性加载 rdkit.js、flavor-core wasm、模型权重与标签，
 * 之后预测全程在本进程内完成（没有外部进程、没有 Python、不联网）。
 *
 * 资源随插件包一起发布，用 Vite 的资源导入（`?url`）拿到地址再 fetch：
 * 插件在宿主里由 `http://asset.localhost/plugs/wmview.flavor/` 提供，
 * 相对地址会跟着产物所在目录解析，所以 dev server 与安装版用的是同一套代码。
 */

import rdkitWasmUrl from '@rdkit/rdkit/RDKit_minimal.wasm?url'
import coreWasmUrl from '../../flavor-core/pkg/flavor_core_bg.wasm?url'
import weightsBinUrl from '../../flavor/pharma33_weights.bin?url'
import weightsIndexUrl from '../../flavor/pharma33_weights.json?url'
import labels from '../../flavor/labels.json'

// 权重索引与标签随源码一起打进产物（分别为 6.6 KB / 1.9 KB）；
// 两个 wasm 与 6.3 MB 的权重数据按资源文件发布，运行时 fetch。
import { initSync, load_weights, num_classes, predict as forwardWasm } from '../../flavor-core/pkg/flavor_core.js'
import { parseFdef, type FeatureDef } from './fdef'
import type { ForwardFn } from './pipeline'
import { chem, initChem } from './rdkit'
import { BASE_FEATURES_FDEF } from './tables/fdef'

export interface FlavorRuntime {
  /** 138 个气味标签（顺序与模型输出一致） */
  labels: string[]
  /** BaseFeatures.fdef 里的药效团定义 */
  featureDefs: FeatureDef[]
  /** wasm 前向（供 `predict()` 调用） */
  forward: ForwardFn
  /** RDKit 版本号，用于「接口自检」核对 */
  rdkitVersion: string
  /** 模型类别数（自检用，应为 138） */
  classes: number
  /** 首次加载耗时（毫秒） */
  loadMs: number
}

/** 各阶段的进度提示，面板直接展示 */
export type LoadProgress = (message: string) => void

let runtimePromise: Promise<FlavorRuntime> | null = null
let loaded: FlavorRuntime | null = null

/** 已加载完成的运行时（未加载完返回 null） */
export function flavorRuntime(): FlavorRuntime | null {
  return loaded
}

async function fetchBytes(url: string, what: string): Promise<ArrayBuffer> {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`${what} 加载失败（HTTP ${res.status}）：${url}`)
  return res.arrayBuffer()
}

async function fetchText(url: string, what: string): Promise<string> {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`${what} 加载失败（HTTP ${res.status}）：${url}`)
  return res.text()
}

async function buildRuntime(report: LoadProgress): Promise<FlavorRuntime> {
  const t0 = performance.now()

  report('加载 RDKit 与推理核…')
  const [rdkitBytes, coreBytes, weightsIndex] = await Promise.all([
    fetchBytes(rdkitWasmUrl, 'RDKit wasm'),
    fetchBytes(coreWasmUrl, '推理核 wasm'),
    fetchText(weightsIndexUrl, '权重索引'),
  ])

  // 两个 wasm 都手工实例化：地址由构建时决定，不依赖 emscripten 的自动定位。
  await initChem({ wasmBinary: rdkitBytes, wasmUrl: rdkitWasmUrl })
  initSync({ module: new Uint8Array(coreBytes) })

  report('加载模型权重（6.3 MB）…')
  const weightsBin = await fetchBytes(weightsBinUrl, '模型权重')
  const classes = load_weights(weightsIndex, new Uint8Array(weightsBin))

  const runtime: FlavorRuntime = {
    labels: labels as string[],
    featureDefs: parseFdef(BASE_FEATURES_FDEF),
    forward: (x, nodes, edgeIndex, edgeAttr, edges, pharma) =>
      forwardWasm(x, nodes, edgeIndex, edgeAttr, edges, pharma),
    rdkitVersion: chem().version(),
    classes,
    loadMs: performance.now() - t0,
  }
  loaded = runtime
  return runtime
}

/**
 * 加载运行时（幂等，多处并发调用只会真正加载一次）。
 * 失败时清空缓存，便于用户重试。
 */
export function loadFlavorRuntime(report: LoadProgress = () => {}): Promise<FlavorRuntime> {
  if (!runtimePromise) {
    runtimePromise = buildRuntime(report).catch((err: unknown) => {
      runtimePromise = null
      throw err
    })
  }
  return runtimePromise
}

/** 类别数自检（未加载时返回 0） */
export function loadedClasses(): number {
  return num_classes()
}
