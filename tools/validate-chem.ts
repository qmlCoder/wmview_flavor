/**
 * 化学层对拍：把 `py/export_chem_oracle.py` 导出的基准与前端实现逐项比较。
 *
 * 比较项（同一份 molblock 作为双侧输入，排除构象差异）：
 *   1. 逐原子 Crippen(logP/MR) / TPSA 贡献 / Gasteiger 电荷
 *   2. 药效团特征清单（家族 + 原子索引）
 *   3. 图特征 x / edge_index / edge_attr
 *   4. 药效团 33 维向量
 *   5. 最终概率（wasm 前向）
 *
 * 用法: npx tsx tools/validate-chem.ts
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { graphFeatures, pharmacophoreVector } from '../src/chem/features';
import { parseFdef, pharmacophores } from '../src/chem/fdef';
import { parseCoords, parseGraph } from '../src/chem/graph';
import { crippenContribs, gasteigerCharges, tpsaContribs } from '../src/chem/peratom';
import { initChem, parseMol, release } from '../src/chem/rdkit';
import { CRIPPEN_TABLE } from '../src/chem/tables/crippen';
import { BASE_FEATURES_FDEF } from '../src/chem/tables/fdef';

import { initSync, load_weights, predict as forward } from '../flavor-core/pkg/flavor_core.js';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const readBin = (p: string) => fs.readFileSync(path.join(ROOT, p));
const readText = (p: string) => fs.readFileSync(path.join(ROOT, p), 'utf8');

interface OracleCase {
  name: string;
  molblock: string;
  x: number[];
  nodes: number;
  edge_index: number[];
  edges: number;
  edge_attr: number[];
  pharma: number[];
  probs: number[];
  crippen: [number, number][];
  tpsa: number[];
  charges: number[];
  features: { family: string; atoms: number[] }[];
}

const maxAbsDiff = (a: ArrayLike<number>, b: number[]): number => {
  let m = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i += 1) m = Math.max(m, Math.abs(Number(a[i]) - b[i]));
  return m;
};

const argmax = (v: ArrayLike<number>): number => {
  let best = 0;
  for (let i = 1; i < v.length; i += 1) if (v[i] > v[best]) best = i;
  return best;
};

async function main(): Promise<void> {
  await initChem(readBin('node_modules/@rdkit/rdkit/dist/RDKit_minimal.wasm'));
  initSync({ module: readBin('flavor-core/pkg/flavor_core_bg.wasm') });
  load_weights(readText('flavor/pharma33_weights.json'), new Uint8Array(readBin('flavor/pharma33_weights.bin')));

  const defs = parseFdef(BASE_FEATURES_FDEF);
  const oracle = JSON.parse(readText('build/chem/oracle.json')) as { cases: OracleCase[] };

  console.log(`用例 ${oracle.cases.length} 个，fdef 特征定义 ${defs.length} 条\n`);

  const worst = { crippen: 0, tpsa: 0, charges: 0, x: 0, edgeIndex: 0, edgeAttr: 0, pharma: 0, probs: 0 };
  let featureMismatch = 0;
  let allArgmax = true;

  for (const c of oracle.cases) {
    const mol = parseMol(c.molblock, { removeHs: false });
    try {
      const heavy = mol.copy();
      if (!heavy) throw new Error('copy 失败');
      heavy.remove_hs_in_place();
      const graphH = parseGraph(mol.get_json());
      const graphHeavy = parseGraph(heavy.get_json());
      const coords = parseCoords(mol.get_coords());

      const crippen = crippenContribs(mol, graphH, CRIPPEN_TABLE);
      const tpsa = tpsaContribs(graphH);
      const charges = gasteigerCharges(graphH);
      const crippenFlat: number[] = [];
      for (const r of crippen) crippenFlat.push(r.logp, r.mr);
      const crippenRef: number[] = [];
      for (const [a, b] of c.crippen) crippenRef.push(a, b);

      const dCrippen = maxAbsDiff(crippenFlat, crippenRef);
      const dTpsa = maxAbsDiff(tpsa, c.tpsa);
      const dCharges = maxAbsDiff(charges, c.charges);

      const found = pharmacophores(mol, defs);
      let sameSet = found.length === c.features.length;
      if (sameSet) {
        for (let i = 0; i < found.length; i += 1) {
          const a = found[i];
          const b = c.features[i];
          if (
            a.family !== b.family ||
            a.atoms.length !== b.atoms.length ||
            a.atoms.some((v, k) => v !== b.atoms[k])
          ) {
            sameSet = false;
            break;
          }
        }
      }
      if (!sameSet) featureMismatch += 1;

      const gf = graphFeatures(graphHeavy);
      const pharma = pharmacophoreVector(mol, graphH, coords, defs);
      // 调试用：把前端算出的特征落盘，便于用原生 CLI 复现 wasm 里的崩溃
      if (c === oracle.cases[0]) {
        fs.mkdirSync(path.join(ROOT, 'build', 'chem'), { recursive: true });
        fs.writeFileSync(
          path.join(ROOT, 'build', 'chem', 'ts-case0.json'),
          JSON.stringify({
            nodes: gf.nodes,
            edges: gf.edges,
            x: Array.from(gf.x),
            edge_index: Array.from(gf.edgeIndex),
            edge_attr: Array.from(gf.edgeAttr),
            pharma: Array.from(pharma.vector),
          }),
        );
      }
      const dX = maxAbsDiff(gf.x, c.x);
      const dEi = maxAbsDiff(gf.edgeIndex, c.edge_index);
      const dEa = maxAbsDiff(gf.edgeAttr, c.edge_attr);
      const dPharma = maxAbsDiff(pharma.vector, c.pharma);

      const probs = forward(gf.x, gf.nodes, gf.edgeIndex, gf.edgeAttr, gf.edges, pharma.vector);
      const dProbs = maxAbsDiff(probs, c.probs);
      const okArg = argmax(probs) === argmax(c.probs);
      allArgmax &&= okArg;

      worst.crippen = Math.max(worst.crippen, dCrippen);
      worst.tpsa = Math.max(worst.tpsa, dTpsa);
      worst.charges = Math.max(worst.charges, dCharges);
      worst.x = Math.max(worst.x, dX);
      worst.edgeIndex = Math.max(worst.edgeIndex, dEi);
      worst.edgeAttr = Math.max(worst.edgeAttr, dEa);
      worst.pharma = Math.max(worst.pharma, dPharma);
      worst.probs = Math.max(worst.probs, dProbs);

      console.log(
        `${sameSet ? 'OK  ' : 'DIFF'} ${c.name.slice(0, 30).padEnd(32)} ` +
          `crip ${dCrippen.toExponential(1)} tpsa ${dTpsa.toExponential(1)} q ${dCharges.toExponential(1)}` +
          ` | x ${dX.toExponential(1)} ei ${dEi.toExponential(1)} ea ${dEa.toExponential(1)}` +
          ` ph ${dPharma.toExponential(1)} | p ${dProbs.toExponential(1)}${okArg ? '' : ' TOP1✗'}`,
      );
      release(heavy);
    } finally {
      release(mol);
    }
  }

  console.log('\n=== 汇总（与 RDKit 的最大绝对偏差） ===');
  console.log(`逐原子 Crippen        ${worst.crippen.toExponential(3)}`);
  console.log(`逐原子 TPSA 贡献      ${worst.tpsa.toExponential(3)}`);
  console.log(`逐原子 Gasteiger 电荷 ${worst.charges.toExponential(3)}`);
  console.log(`原子特征 x            ${worst.x.toExponential(3)}`);
  console.log(`edge_index            ${worst.edgeIndex.toExponential(3)}`);
  console.log(`键特征 edge_attr      ${worst.edgeAttr.toExponential(3)}`);
  console.log(`药效团 33 维          ${worst.pharma.toExponential(3)}`);
  console.log(`最终概率              ${worst.probs.toExponential(3)}`);
  console.log(`药效团清单不一致分子数: ${featureMismatch}`);
  console.log(`top-1 全部一致: ${allArgmax}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
