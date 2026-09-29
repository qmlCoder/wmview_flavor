/**
 * 键级推断回归测试：拿真实 molblock（含 3D 坐标和真实键级）当基准，
 * 只把「原子 / 坐标 / 连接表」喂给场景分子路径（丢掉键级），看能不能还原出真结构。
 *
 * 这正是宿主给插件的数据形态：`get_mole_info()` 只给 syms/xyzs/bonds，没有键级。
 *
 * 用法: npx tsx tools/test-bondorder.ts [--n 12]
 */

import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { buildMolblock, inferBondOrders } from '../src/chem/bondorder'
import { parseFdef } from '../src/chem/fdef'
import { predict } from '../src/chem/pipeline'
import { initChem, parseMol, type RDKitMol } from '../src/chem/rdkit'
import { ELEMENT_SYMBOLS } from '../src/chem/tables/elements'
import { BASE_FEATURES_FDEF } from '../src/chem/tables/fdef'
import { initSync, load_weights, predict as forward } from '../flavor-core/pkg/flavor_core.js'

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const readBin = (p: string) => fs.readFileSync(path.join(ROOT, p))
const readText = (p: string) => fs.readFileSync(path.join(ROOT, p), 'utf8')

const args = process.argv.slice(2)
const N = Number(args[args.indexOf('--n') + 1] || '12')

interface RawMol {
  molecules: {
    atoms: { z?: number; chg?: number }[]
    bonds: { atoms: number[]; bo?: number }[]
  }[]
}

/** 去掉氢之后的规范 SMILES（两边同口径比较） */
function heavySmiles(mol: RDKitMol): string {
  const copy = mol.copy()
  if (!copy) return '(copy failed)'
  copy.remove_hs_in_place()
  const s = stripStereo(copy.get_smiles())
  copy.deleteLater()
  return s
}

/**
 * 只比「连接关系」：MinimalLib 这个构建会忽略 `isomericSmiles:false`，
 * 所以直接把立体标记（`@` `@@` 与 `/` `\`）从规范 SMILES 里删掉再比。
 * 顺带说明一处已知的口径差异：从 3D 坐标反推时，我们自己生成的 molblock
 * 会额外标出 C=C 的 E/Z，而原始 SDF 来的 molblock 不会——两者构型一致，
 * 对预测（特征里不含立体信息）没有影响。
 */
function stripStereo(smiles: string): string {
  return smiles.replace(/@@|@|\\|\//g, '')
}

/** 从 molblock 取出「原子 / 坐标 / 连接表」（丢掉键级），可选择性去掉氢 */
function sceneInput(molblock: string, dropH: boolean) {
  const mol = parseMol(molblock, { removeHs: false })
  const raw = JSON.parse(mol.get_json()) as RawMol
  const coords = mol.get_coords()
  const m = raw.molecules[0]
  const all = m.atoms.map((a, i) => ({
    sym: ELEMENT_SYMBOLS[(a.z ?? 6) - 1] ?? 'C',
    xyz: coords[i],
  }))
  const keep = all.map((a, i) => ({ a, i })).filter((x) => !(dropH && x.a.sym === 'H'))
  const idx = new Map(keep.map((x, k) => [x.i, k]))
  const bonds = m.bonds
    .filter((b) => idx.has(b.atoms[0]) && idx.has(b.atoms[1]))
    .map((b) => [idx.get(b.atoms[0])!, idx.get(b.atoms[1])!] as [number, number])
  return {
    truthHeavy: heavySmiles(mol),
    mol,
    syms: keep.map((x) => x.a.sym),
    xyzs: keep.map((x) => x.a.xyz),
    bonds,
  }
}

async function main(): Promise<void> {
  await initChem({ wasmBinary: readBin('node_modules/@rdkit/rdkit/dist/RDKit_minimal.wasm') })
  initSync({ module: readBin('flavor-core/pkg/flavor_core_bg.wasm') })
  load_weights(readText('flavor/pharma33_weights.json'), new Uint8Array(readBin('flavor/pharma33_weights.bin')))
  const labels = JSON.parse(readText('flavor/labels.json')) as string[]
  const featureDefs = parseFdef(BASE_FEATURES_FDEF)
  const oracle = JSON.parse(readText('build/chem/oracle.json')) as {
    cases: { name: string; molblock: string; probs: number[] }[]
  }
  const cases = oracle.cases.slice(0, N)

  let passH = 0
  let passNoH = 0
  const diffsH: number[] = []
  const diffsNoH: number[] = []
  console.log(`用例 ${cases.length} 个（真值 = molblock 自带的键级；Δp = 与 RDKit 参考前三方一致的概率最大偏差）\n`)
  console.log(`${'分子'.padEnd(30)} 带显式氢                              无显式氢`)

  for (const c of cases) {
    const truth = sceneInput(c.molblock, true).truthHeavy
    const cells: string[] = []
    for (const dropH of [false, true] as const) {
      const scene = sceneInput(c.molblock, dropH)
      const { orders, unsatisfied } = inferBondOrders(
        scene.syms,
        scene.bonds,
        scene.syms.map(() => 0),
        { coords: scene.xyzs },
      )
      // 走真实管线（含补氢、sanitize、药效团与 wasm 前向）
      const reply = predict(
        { syms: scene.syms, xyzs: scene.xyzs, bonds: scene.bonds, charge: 0 },
        { labels, featureDefs, forward, threshold: 0.5 },
      )
      let got: string
      if (reply.ok) {
        got = stripStereo(reply.smiles ?? '')
      } else {
        got = `(失败 ${reply.error})`
      }
      // 概率偏差（按标签名对齐后再比）
      let dP = Number.NaN
      if (reply.ok && reply.results.length === labels.length) {
        const byLabel = new Map(reply.results.map((r) => [r.label, r.prob]))
        dP = labels.reduce((m, l, i) => Math.max(m, Math.abs((byLabel.get(l) ?? Number.NaN) - c.probs[i])), 0)
      }
      const ok = got === truth
      if (dropH) {
        if (ok) passNoH += 1
        diffsNoH.push(dP)
      } else if (ok) {
        passH += 1
        diffsH.push(dP)
      } else {
        diffsH.push(dP)
      }
      cells.push(
        `${ok ? 'OK  ' : 'FAIL'} ${got.slice(0, 18).padEnd(20)} Δp ${dP.toFixed(3)} ${orders.length ? '' : ''}${unsatisfied.size ? `未满足${unsatisfied.size}` : '价键满足'}`,
      )
      scene.mol.deleteLater()
    }
    console.log(`${c.name.slice(0, 28).padEnd(30)} ${cells[0]} | ${cells[1]}`)
  }

  console.log('\n=== 汇总 ===')
  console.log(`带显式氢：${passH}/${cases.length} 还原正确`)
  console.log(`无显式氢：${passNoH}/${cases.length} 还原正确`)
  const fmt = (v: number[]) => (v.some((x) => !Number.isFinite(x)) ? '含 NaN' : Math.max(...v).toFixed(3))
  console.log(`最大 Δp：带显式氢 ${fmt(diffsH)}；无显式氢 ${fmt(diffsNoH)}`)
}

main().catch((err) => {
  console.error(err)
  process.exit(1)
})
