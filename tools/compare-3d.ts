/**
 * 量化「SMILES 路径」的近似代价：前端用 2D 平面坐标，Python 版用 ETKDG + MMFF94 的 3D 构象。
 *
 * 化学层本身与 RDKit 逐项一致（见 `tools/validate-chem.ts`，双侧同一份 molblock），
 * 所以两边剩下的差异只来自构象：RDKit 的药效团识别里有基于距离的聚类（Hydrophobe /
 * Aromatic 等），2D 与 3D 的原子间距不同 → 药效团个数与位置不同 → 概率不同。
 *
 * 这个脚本就是把这个差异量出来，供 README 里写清「SMILES 路径的精度」。
 *
 * 用法: npx tsx tools/compare-3d.ts [--exe <路径>] [--n 12]
 */

import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { initChem, parseMol } from '../src/chem/rdkit'
import { parseFdef, type FeatureDef } from '../src/chem/fdef'
import { parseCoords, parseGraph } from '../src/chem/graph'
import { graphFeatures, pharmacophoreVector } from '../src/chem/features'
import { BASE_FEATURES_FDEF } from '../src/chem/tables/fdef'
import { initSync, load_weights, predict as forward } from '../flavor-core/pkg/flavor_core.js'

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const readBin = (p: string) => fs.readFileSync(path.join(ROOT, p))
const readText = (p: string) => fs.readFileSync(path.join(ROOT, p), 'utf8')

const args = process.argv.slice(2)
const argValue = (name: string, fallback: string): string => {
  const i = args.indexOf(name)
  return i >= 0 && args[i + 1] ? args[i + 1] : fallback
}
const EXE = argValue('--exe', 'build/deploy-backup/flavor.exe')
const N = Number(argValue('--n', '12'))

interface OracleCase {
  name: string
}

const argmax = (v: ArrayLike<number>): number => {
  let best = 0
  for (let i = 1; i < v.length; i += 1) if (v[i] > v[best]) best = i
  return best
}

/** 前端路径：与 `chem/pipeline.ts` 的 buildMolecule 保持一致（SMILES → 2D 坐标 + 加氢） */
function frontendProbs(smiles: string, labels: string[], defs: FeatureDef[]): number[] {
  const mol = parseMol(smiles, { removeHs: true })
  try {
    mol.set_new_coords?.()
    if (!mol.add_hs_in_place()) throw new Error('加氢失败')
    const heavy = mol.copy()
    if (!heavy) throw new Error('copy 失败')
    heavy.remove_hs_in_place()
    const graph = graphFeatures(parseGraph(heavy.get_json()))
    const pharma = pharmacophoreVector(mol, parseGraph(mol.get_json()), parseCoords(mol.get_coords()), defs)
    const probs = forward(graph.x, graph.nodes, graph.edgeIndex, graph.edgeAttr, graph.edges, pharma.vector)
    return Array.from(probs)
  } finally {
    mol.deleteLater()
  }
}

/**
 * Python 路径：把 SMILES 交给 flavor.exe（ETKDG + MMFF94 生成 3D 构象）。
 * 注意 exe 的结果是**按概率降序**排列的，要按标签名映射回与模型输出一致的顺序。
 */
function exeProbs(smiles: string, labels: string[]): number[] {
  const out = execFileSync(path.join(ROOT, EXE), ['--smiles', smiles], {
    encoding: 'utf8',
    maxBuffer: 64 * 1024 * 1024,
  })
  const line = out.split(/\r?\n/).find((l) => l.startsWith('@@FLAVOR_RESULT@@'))
  if (!line) throw new Error('exe 没有输出结果行')
  const reply = JSON.parse(line.slice('@@FLAVOR_RESULT@@'.length)) as {
    results: { label: string; prob: number }[]
  }
  const probs = new Array<number>(labels.length).fill(Number.NaN)
  for (const r of reply.results) {
    const i = labels.indexOf(r.label)
    if (i >= 0) probs[i] = r.prob
  }
  const missing = probs.findIndex((v) => Number.isNaN(v))
  if (missing >= 0) throw new Error(`exe 结果里缺少标签 ${labels[missing]}`)
  return probs
}

async function main(): Promise<void> {
  await initChem({ wasmBinary: readBin('node_modules/@rdkit/rdkit/dist/RDKit_minimal.wasm') })
  initSync({ module: readBin('flavor-core/pkg/flavor_core_bg.wasm') })
  load_weights(readText('flavor/pharma33_weights.json'), new Uint8Array(readBin('flavor/pharma33_weights.bin')))
  const labels = JSON.parse(readText('flavor/labels.json')) as string[]
  const defs = parseFdef(BASE_FEATURES_FDEF)

  const cases = (JSON.parse(readText('build/chem/oracle.json')) as { cases: OracleCase[] }).cases.slice(0, N)
  console.log(`对比 ${cases.length} 个分子：前端(2D 坐标) vs flavor.exe(ETKDG+MMFF 3D)\n`)

  let worst = 0
  let worstName = ''
  let top1 = 0
  let overlapSum = 0
  const diffs: number[] = []

  for (const c of cases) {
    const a = exeProbs(c.name, labels)
    const b = frontendProbs(c.name, labels, defs)
    const d = a.reduce((m, v, i) => Math.max(m, Math.abs(v - b[i])), 0)
    diffs.push(d)
    if (d > worst) {
      worst = d
      worstName = c.name
    }
    const okTop1 = argmax(a) === argmax(b)
    if (okTop1) top1 += 1
    const top = (v: number[]) =>
      Array.from(v.keys())
        .sort((x, y) => v[y] - v[x])
        .slice(0, 10)
        .map((i) => labels[i])
    const ta = new Set(top(a))
    const overlap = top(b).filter((l) => ta.has(l)).length
    overlapSum += overlap
    console.log(
      `${okTop1 ? 'OK  ' : 'DIFF'} ${c.name.slice(0, 30).padEnd(32)} 最大Δp ${d.toFixed(4)} top10 重叠 ${overlap}/10`,
    )
  }

  const mean = diffs.reduce((s, v) => s + v, 0) / diffs.length
  console.log('\n=== 汇总（SMILES 路径：2D vs 3D 构象） ===')
  console.log(`最大 Δp       ${worst.toFixed(4)}（${worstName}）`)
  console.log(`平均 Δp       ${mean.toFixed(4)}`)
  console.log(`top-1 一致    ${top1}/${cases.length}`)
  console.log(`top-10 平均重叠 ${(overlapSum / cases.length).toFixed(1)}/10`)
}

main().catch((err) => {
  console.error(err)
  process.exit(1)
})
