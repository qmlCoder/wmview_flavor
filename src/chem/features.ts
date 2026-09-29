/**
 * 特征组装：图特征（11 维原子 / 4 维键）+ 药效团特征（33 维）。
 *
 * 口径与 `py/flavor_feats.py` 完全一致：
 * - 图特征取自**去氢**分子（等价于 Python 里 `Chem.RemoveHs` 后的重原子图）
 * - 药效团特征取自**带氢**的 3D 分子
 * - 药效团向量 = 逐特征 11 维做 max/mean/sum 三通道池化（所以特征顺序无关紧要）
 */

import { FAMILIES, pharmacophores, type FeatureDef } from './fdef';
import { ChemGraph } from './graph';
import { crippenContribs, gasteigerCharges, tpsaContribs } from './peratom';
import type { RDKitMol } from './rdkit';
import { CRIPPEN_TABLE } from './tables/crippen';
import { ATOMIC_WEIGHTS } from './tables/elements';

export const NODE_DIM = 11;
export const EDGE_DIM = 4;
export const PHARMA_DIM = 33;

/** 只能构成这几种键类型 one-hot 的键（与 `BOND_TYPE_CHOICES` 顺序一致）。 */
const BOND_ONE_HOT = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'] as const;

export interface GraphFeatures {
  /** nodes × 11 */
  x: Float32Array;
  nodes: number;
  /** 2 × edges */
  edgeIndex: Int32Array;
  /** edges × 4 */
  edgeAttr: Float32Array;
  edges: number;
}

/** 11 维原子特征（与 `atom_features` 一致）。 */
function atomFeatureRow(g: ChemGraph, i: number): number[] {
  const a = g.atoms[i];
  const sym = a.symbol;
  return [
    g.totalValence(i),
    a.aromatic ? 1 : 0,
    a.inRing ? 1 : 0,
    ATOMIC_WEIGHTS[a.z - 1] ?? 0,
    g.totalDegree(i),
    sym === 'N' ? 1 : 0,
    sym === 'O' ? 1 : 0,
    sym === 'S' ? 1 : 0,
    sym === 'F' ? 1 : 0,
    sym === 'Cl' ? 1 : 0,
    sym === 'Br' ? 1 : 0,
  ];
}

/** 4 维键特征：键类型 one-hot。 */
function bondFeatureRow(g: ChemGraph, bondIdx: number): number[] {
  const bond = g.bonds[bondIdx];
  const kind = bond.aromatic ? 'AROMATIC' : bond.order === 1 ? 'SINGLE' : bond.order === 2 ? 'DOUBLE' : bond.order === 3 ? 'TRIPLE' : '';
  return BOND_ONE_HOT.map((k) => (k === kind ? 1 : 0));
}

/** 由去氢分子图构造 GNN 输入（原子顺序与键顺序均取 RDKit 的索引顺序）。 */
export function graphFeatures(g: ChemGraph): GraphFeatures {
  const nodes = g.numAtoms;
  const x = new Float32Array(nodes * NODE_DIM);
  for (let i = 0; i < nodes; i += 1) {
    const row = atomFeatureRow(g, i);
    for (let k = 0; k < NODE_DIM; k += 1) x[i * NODE_DIM + k] = row[k];
  }

  // 与 Python 的 graph_arrays 一致：每条键展开成两条有向边（i→j、j→i），
  // 展平后是 [src 段..., dst 段...]，所以 edges 指**有向边数**、edge_index 长度是 2*edges。
  const edges = g.bonds.length * 2;
  const edgeIndex = new Int32Array(edges * 2);
  const edgeAttr = new Float32Array(edges * EDGE_DIM);
  for (let b = 0; b < g.bonds.length; b += 1) {
    const bond = g.bonds[b];
    const row = bondFeatureRow(g, b);
    const fwd = b * 2;
    const rev = fwd + 1;
    edgeIndex[fwd] = bond.a;
    edgeIndex[edges + fwd] = bond.b;
    edgeIndex[rev] = bond.b;
    edgeIndex[edges + rev] = bond.a;
    for (let k = 0; k < EDGE_DIM; k += 1) {
      edgeAttr[fwd * EDGE_DIM + k] = row[k];
      edgeAttr[rev * EDGE_DIM + k] = row[k];
    }
  }

  return { x, nodes, edgeIndex, edgeAttr, edges };
}

export interface PharmaResult {
  /** 33 维池化后的向量 */
  vector: Float32Array;
  /** 识别到的药效团数量 */
  count: number;
}

/** 某个药效团的 11 维原始向量（与 `PharmacophoreExtractor.extract` 里一致）。 */
function featureRow(
  family: string,
  atomIds: number[],
  coords: number[][],
  charges: number[],
  crippen: { logp: number; mr: number }[],
  tpsa: number[],
): number[] {
  const typeIdx = (FAMILIES as readonly string[]).indexOf(family);

  // 平均坐标
  let cx = 0;
  let cy = 0;
  let cz = 0;
  for (const idx of atomIds) {
    cx += coords[idx][0];
    cy += coords[idx][1];
    cz += coords[idx][2];
  }
  cx /= atomIds.length;
  cy /= atomIds.length;
  cz /= atomIds.length;

  // 电荷均值 / MR 之和 / logP 均值 / 局部 TPSA
  let chargeSum = 0;
  let mrSum = 0;
  let logpSum = 0;
  let tpsaSum = 0;
  for (const idx of atomIds) {
    const q = charges[idx];
    chargeSum += Number.isFinite(q) ? q : 0;
    mrSum += crippen[idx].mr;
    logpSum += crippen[idx].logp;
    tpsaSum += tpsa[idx];
  }

  // 到自身几何中心的距离统计
  let maxD = 0;
  let minD = 0;
  let meanD = 0;
  if (atomIds.length > 1) {
    const dists = atomIds.map((idx) => {
      const dx = coords[idx][0] - cx;
      const dy = coords[idx][1] - cy;
      const dz = coords[idx][2] - cz;
      return Math.sqrt(dx * dx + dy * dy + dz * dz);
    });
    maxD = Math.max(...dists);
    minD = Math.min(...dists);
    meanD = dists.reduce((s, v) => s + v, 0) / dists.length;
  }

  return [
    typeIdx,
    cx,
    cy,
    cz,
    chargeSum / atomIds.length,
    mrSum,
    logpSum / atomIds.length,
    tpsaSum,
    maxD,
    minD,
    meanD,
  ];
}

/**
 * 33 维药效团特征向量。
 *
 * `mol` 为带显式氢的分子，`graph` 为它对应的图，`coords` 为全部原子坐标。
 */
export function pharmacophoreVector(
  mol: RDKitMol,
  graph: ChemGraph,
  coords: number[][],
  defs: FeatureDef[],
): PharmaResult {
  const found = pharmacophores(mol, defs);
  const charges = gasteigerCharges(graph);
  const tpsa = tpsaContribs(graph);
  const crippen = crippenContribs(mol, graph, CRIPPEN_TABLE);

  const rows = found.map((f) => featureRow(f.family, f.atoms, coords, charges, crippen, tpsa));
  if (!rows.length) {
    return { vector: new Float32Array(PHARMA_DIM), count: 0 };
  }

  const out = new Float32Array(PHARMA_DIM);
  for (let col = 0; col < 11; col += 1) {
    let max = -Infinity;
    let sum = 0;
    for (const row of rows) {
      const v = row[col];
      if (v > max) max = v;
      sum += v;
    }
    out[col] = max; // max 池化
    out[11 + col] = sum / rows.length; // mean 池化
    out[22 + col] = sum; // sum 池化
  }

  return { vector: out, count: rows.length };
}
