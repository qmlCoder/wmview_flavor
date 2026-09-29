/**
 * 端到端推理管线：分子输入 → rdkit.js 感知 → 特征 → wasm 前向 → 结果。
 *
 * 与 Python 版（`py/predict.py` + `flavor_feats.py`）的对应关系：
 *   build_mol        → buildMolecule()
 *   PharmacophoreExtractor.extract → pharmacophoreVector()
 *   graph_arrays + graph_mol       → graphFeatures()
 *   FlavorModel.predict_proba      → opts.forward（flavor-core 的 wasm 导出）
 *
 * 前端不再需要外部进程：整个流程在 WebView 里跑（这也是支持 Android 的前提）。
 */

import {
  buildMolblock,
  inferBondOrders,
  inferBondOrdersFromGeometry,
  NO_BOND_ORDER_WARNING,
} from './bondorder';
import { graphFeatures, pharmacophoreVector, type GraphFeatures, type PharmaResult } from './features';
import { type FeatureDef } from './fdef';
import { parseCoords, parseGraph, type ChemGraph } from './graph';
import { parseMol, release, type RDKitMol } from './rdkit';

export interface FlavorInput {
  smiles?: string;
  molblock?: string;
  syms?: string[];
  xyzs?: number[][];
  bonds?: [number, number][];
  charge?: number;
}

export interface FlavorResultItem {
  label: string;
  prob: number;
  hit: boolean;
}

export interface FlavorReply {
  ok: boolean;
  error?: string;
  warning?: string | null;
  smiles?: string;
  atoms?: number;
  pharmacophores?: number;
  results: FlavorResultItem[];
}

/** wasm 前向函数（`flavor-core` 的 `predict` 导出）。 */
export type ForwardFn = (
  x: Float32Array,
  nodes: number,
  edgeIndex: Int32Array,
  edgeAttr: Float32Array,
  edges: number,
  pharma: Float32Array,
) => Float32Array;

export interface PredictOptions {
  labels: string[];
  featureDefs: FeatureDef[];
  forward: ForwardFn;
  threshold?: number;
  topk?: number;
}

export interface Molecule { mol: RDKitMol; warning: string | null }

/** SDF 可能含多个分子块，取第一块（与 Python 侧一致）。 */
function firstMolblock(block: string): string {
  const idx = block.indexOf('$$$$');
  return idx >= 0 ? block.slice(0, idx) : block;
}

/** 带氢且带坐标的分子：SMILES 走 2D 坐标（不做 ETKDG）、molblock/scene 用自带坐标。 */
function buildMolecule(input: FlavorInput): Molecule {
  let mol: RDKitMol;
  let warning: string | null = null;

  if (input.smiles) {
    mol = parseMol(input.smiles, { removeHs: true });
    // 与训练数据的「带氢 3D 构象」口径对齐：这里用 2D 平面坐标（实测差异在既有噪声内，
    // 详见 py/sensitivity.py）——不做 ETKDG/MMFF，避免把 RDKit 的构象生成也搬过来。
    mol.set_new_coords?.();
    if (!mol.add_hs_in_place()) throw new Error('加氢失败');
  } else if (input.molblock) {
    mol = parseMol(firstMolblock(input.molblock), { removeHs: false });
    if (mol.get_num_atoms(true) === mol.get_num_atoms(false)) {
      // 没有显式氢：按已有坐标补氢（等价 Python 的 AddHs(addCoords=True)）
      mol.add_hs_in_place();
    }
  } else if (input.syms?.length && input.xyzs?.length) {
    const built = buildSceneMolecule(input);
    mol = built.mol;
    warning = built.warning;
  } else {
    throw new Error('缺少分子输入（smiles / molblock / syms+xyzs 之一）');
  }

  if (!mol.has_coords()) {
    mol.set_new_coords?.();
  }
  return { mol, warning };
}

/**
 * 场景分子：只有原子 / 坐标 / 连接表（没有键级），需要自己把键级补出来。
 *
 * - **有显式氢**：氢把每个原子的价固定住了，用 `inferBondOrders` 精确求解
 *   （找不到解则退回贪心并告警）。这是最常见也最可靠的情形。
 * - **没有显式氢**：氢数和多重键数是同一个未知量，理论上欠定，只能按键长挑出
 *   有把握的多重键（`inferBondOrdersFromGeometry`），剩下的价交给氢补，
 *   并始终给出「结果仅供参考」的告警。
 */
function buildSceneMolecule(input: FlavorInput): Molecule {
  const syms = input.syms ?? [];
  const xyzs = input.xyzs ?? [];
  const bonds = input.bonds ?? [];
  const charge = input.charge ?? 0;

  const workSyms = syms.slice();
  const workXyzs = xyzs.map((c) => c.slice());
  const workBonds = bonds.map((b) => [b[0], b[1]] as [number, number]);
  const hasExplicitH = syms.some((s) => s === 'H' || s === 'h');

  const charges = workSyms.map(() => 0);
  if (charge !== 0 && workSyms.length) {
    // 总电荷放在第一个原子上（RDKit DetermineBonds 的电荷处理更复杂，这里只做粗略分配）
    charges[0] = charge;
  }

  let warning: string | null = null;
  let orders: number[];
  if (hasExplicitH) {
    const solved = inferBondOrders(workSyms, workBonds, charges, { coords: workXyzs });
    orders = solved.orders;
    if (solved.unsatisfied.size) warning = NO_BOND_ORDER_WARNING;
  } else {
    warning = NO_BOND_ORDER_WARNING;
    orders = inferBondOrdersFromGeometry(workSyms, workBonds, workXyzs);
  }

  const molblock = buildMolblock(workSyms, workXyzs, workBonds, orders);
  try {
    const mol = parseMol(molblock, { removeHs: false });
    // 没有显式氢时按剩余价补氢（等价 Python 的 AddHs(addCoords=True)），药效团特征需要氢
    if (!hasExplicitH) mol.add_hs_in_place();
    return { mol, warning };
  } catch {
    // 键级推断没通过 sanitize：退回单键骨架（芳香性等信息会缺失，结果仅供参考）
    const fallback = parseMol(buildMolblock(workSyms, workXyzs, workBonds, workBonds.map(() => 1)), { sanitize: false });
    return { mol: fallback, warning: NO_BOND_ORDER_WARNING };
  }
}

/** 计算所有特征（供推理与对拍共用）。 */
export function computeFeatures(
  mol: RDKitMol,
  heavyMol: RDKitMol,
  graphHeavy: ChemGraph,
  graphH: ChemGraph,
  coords: number[][],
  featureDefs: FeatureDef[],
): { graph: GraphFeatures; pharma: PharmaResult } {
  return {
    graph: graphFeatures(graphHeavy),
    pharma: pharmacophoreVector(mol, graphH, coords, featureDefs),
  };
}

/** 主入口：单分子预测。 */
export function predict(input: FlavorInput, opts: PredictOptions): FlavorReply {
  const threshold = opts.threshold ?? 0.5;
  const topk = opts.topk ?? 0;
  let built: Molecule | null = null;
  let heavy: RDKitMol | null = null;
  try {
    built = buildMolecule(input);
    const mol = built.mol;

    heavy = mol.copy();
    if (!heavy) throw new Error('复制分子失败');
    heavy.remove_hs_in_place();

    const graphH = parseGraph(mol.get_json());
    const graphHeavy = parseGraph(heavy.get_json());
    const coords = parseCoords(mol.get_coords());

    const { graph, pharma } = computeFeatures(mol, heavy, graphHeavy, graphH, coords, opts.featureDefs);
    const probs = opts.forward(graph.x, graph.nodes, graph.edgeIndex, graph.edgeAttr, graph.edges, pharma.vector);

    const order = Array.from(probs.keys()).sort((a, b) => probs[b] - probs[a]);
    const picked = topk > 0 ? order.slice(0, topk) : order;
    const results: FlavorResultItem[] = picked.map((i) => ({
      label: opts.labels[i] ?? `#${i}`,
      prob: Number(probs[i].toFixed(6)),
      hit: probs[i] >= threshold,
    }));

    return {
      ok: true,
      warning: built.warning,
      smiles: heavy.get_smiles(),
      atoms: graph.nodes,
      pharmacophores: pharma.count,
      results,
    };
  } catch (err) {
    return { ok: false, error: (err as Error).message, results: [] };
  } finally {
    release(built?.mol, heavy);
  }
}
