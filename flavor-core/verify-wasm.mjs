// 验证 wasm 链路：加载 pkg/ 里的 wasm，喂与 numpy 对拍相同的 12 个用例，
// 比较概率偏差、top-1 标签一致性，并测量单分子推理耗时。
// 用法: node verify-wasm.mjs
import fs from 'node:fs'
import { initSync, load_weights, num_classes, predict } from './pkg/flavor_core.js'

const ROOT = new URL('../', import.meta.url)
const readText = (p) => fs.readFileSync(new URL(p, ROOT), 'utf8')
const readBin = (p) => fs.readFileSync(new URL(p, ROOT))

initSync({ module: readBin('flavor-core/pkg/flavor_core_bg.wasm') })

const t0 = performance.now()
const classes = load_weights(readText('flavor/pharma33_weights.json'), new Uint8Array(readBin('flavor/pharma33_weights.bin')))
const tLoad = performance.now() - t0
console.log(`wasm 权重加载完成：${classes} 个类别，耗时 ${tLoad.toFixed(1)} ms`)

const cases = JSON.parse(readText('build/parity/cases.json')).cases
console.log(`用例数：${cases.length}，类别数自检：${num_classes()}`)

const argmax = (v) => {
  let best = 0
  for (let i = 1; i < v.length; i++) if (v[i] > v[best]) best = i
  return best
}

let maxDiff = 0
let allMatch = true
let maxDiffName = ''
for (const c of cases) {
  const probs = predict(
    new Float32Array(c.x),
    c.nodes,
    new Int32Array(c.edge_index),
    new Float32Array(c.edge_attr),
    c.edges,
    new Float32Array(c.pharma),
  )
  const d = probs.reduce((m, v, i) => Math.max(m, Math.abs(v - c.probs[i])), 0)
  const ok = argmax(probs) === argmax(c.probs)
  allMatch &&= ok
  if (d > maxDiff) {
    maxDiff = d
    maxDiffName = c.name
  }
  console.log(`  ${ok ? 'OK  ' : 'DIFF'} ${c.name.slice(0, 40).padEnd(42)} 最大概率偏差 ${d.toExponential(3)}`)
}

// 计时：挑最大的用例重复跑
const big = cases.reduce((a, b) => (a.nodes * a.edges > b.nodes * b.edges ? a : b))
const x = new Float32Array(big.x)
const ei = new Int32Array(big.edge_index)
const ea = new Float32Array(big.edge_attr)
const ph = new Float32Array(big.pharma)
predict(x, big.nodes, ei, ea, big.edges, ph) // 预热
const N = 50
const t1 = performance.now()
for (let i = 0; i < N; i++) predict(x, big.nodes, ei, ea, big.edges, ph)
const per = (performance.now() - t1) / N

console.log('\n=== 结论 ===')
console.log(`最大概率偏差：${maxDiff.toExponential(3)}（${maxDiffName}）`)
console.log(`top-1 标签全部一致：${allMatch}`)
console.log(`单分子前向耗时：${per.toFixed(3)} ms（用例 ${big.name.slice(0, 30)}，${big.nodes} 原子 ${big.edges} 键）`)
console.log(maxDiff < 1e-6 && allMatch ? '判定：通过' : '判定：不通过')
