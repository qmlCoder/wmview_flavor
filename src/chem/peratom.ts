/**
 * 逐原子数值：TPSA 贡献、Crippen logP/MR、Gasteiger 电荷。
 *
 * rdkit.js 不暴露这三样（实测 `get_prop('_tpsaAtomContribs-0')` 等返回空），
 * 所以按 RDKit 的实现逐条移植：
 * - TPSA：`Code/GraphMol/Descriptors/MolSurf.cpp::getTPSAAtomContribs`（查表式分支，非 SMARTS）
 * - Crippen：`Code/GraphMol/Descriptors/Crippen.cpp::getCrippenAtomContribs` + `Data/Crippen.txt`
 * - Gasteiger：`Code/GraphMol/PartialCharges/GasteigerCharges.cpp` + `GasteigerParams.cpp`
 *
 * 三者都要求**带显式氢**的分子（与 `py/flavor_feats.py` 的 `mol_3d` 一致）。
 *
 * 敏感度实测（见 `py/component_sensitivity.py`）：TPSA 最关键（清零后 maxΔp≈0.08），
 * MR 次之（0.036），Gasteiger 电荷与 logP 几乎无影响（≤0.003）。
 */

import type { ChemGraph } from './graph';
import { smartsMatches, type RDKitMol } from './rdkit';
import { DAMP, DAMP_SCALE, H_PARAMS, IONXH, gasteigerParams } from './tables/gasteiger';

/**
 * TPSA 逐原子贡献（Ertl 碎片表）。
 *
 * 与 Python 侧 `rdMolDescriptors._CalcTPSAContribs(mol)` 一致：默认 `includeSandP=False`，
 * 所以只有 N / O 有贡献，其它原子一律 0。
 */
export function tpsaContribs(g: ChemGraph): number[] {
  const out = new Array<number>(g.numAtoms).fill(0);

  for (let i = 0; i < g.numAtoms; i += 1) {
    const z = g.atoms[i].z;
    if (z !== 7 && z !== 8) continue; // 只处理 N / O

    const { sing, doub, trip, arom } = g.bondCounts(i);
    const nbrs = g.heavyNeighbors(i).length;
    const nh = g.hNeighbors(i) + g.totalNumHs(i);
    const chg = g.atoms[i].chg;
    const in3Ring = g.atoms[i].in3Ring;

    let tmp = -1;
    if (z === 7) {
      switch (nbrs) {
        case 1:
          if (nh === 0 && chg === 0 && trip === 1) tmp = 23.79;
          else if (nh === 1 && chg === 0 && doub === 1) tmp = 23.85;
          else if (nh === 2 && chg === 0 && sing === 1) tmp = 26.02;
          else if (nh === 2 && chg === 1 && doub === 1) tmp = 25.59;
          else if (nh === 3 && chg === 1 && sing === 1) tmp = 27.64;
          break;
        case 2:
          if (nh === 0 && chg === 0 && sing === 1 && doub === 1) tmp = 12.36;
          else if (nh === 0 && chg === 0 && trip === 1 && doub === 1) tmp = 13.6;
          else if (nh === 1 && chg === 0 && sing === 2 && in3Ring) tmp = 21.94;
          else if (nh === 1 && chg === 0 && sing === 2 && !in3Ring) tmp = 12.03;
          else if (nh === 0 && chg === 1 && trip === 1 && sing === 1) tmp = 4.36;
          else if (nh === 1 && chg === 1 && doub === 1 && sing === 1) tmp = 13.97;
          else if (nh === 2 && chg === 1 && sing === 2) tmp = 16.61;
          else if (nh === 0 && chg === 0 && arom === 2) tmp = 12.89;
          else if (nh === 1 && chg === 0 && arom === 2) tmp = 15.79;
          else if (nh === 1 && chg === 1 && arom === 2) tmp = 14.14;
          break;
        case 3:
          if (nh === 0 && chg === 0 && sing === 3 && in3Ring) tmp = 3.01;
          else if (nh === 0 && chg === 0 && sing === 3 && !in3Ring) tmp = 3.24;
          else if (nh === 0 && chg === 0 && sing === 1 && doub === 2) tmp = 11.68;
          else if (nh === 0 && chg === 1 && sing === 2 && doub === 1) tmp = 3.01;
          else if (nh === 1 && chg === 1 && sing === 3) tmp = 4.44;
          else if (nh === 0 && chg === 0 && arom === 3) tmp = 4.41;
          else if (nh === 0 && chg === 0 && sing === 1 && arom === 2) tmp = 4.93;
          else if (nh === 0 && chg === 0 && doub === 1 && arom === 2) tmp = 8.39;
          else if (nh === 0 && chg === 1 && arom === 3) tmp = 4.1;
          else if (nh === 0 && chg === 1 && sing === 1 && arom === 2) tmp = 3.88;
          break;
        case 4:
          if (nh === 0 && sing === 4 && chg === 1) tmp = 0.0;
          break;
        default:
          break;
      }
      if (tmp < 0.0) {
        tmp = 30.5 - nbrs * 8.2 + nh * 1.5;
        if (tmp < 0) tmp = 0.0;
      }
    } else {
      // z === 8
      switch (nbrs) {
        case 1:
          if (nh === 0 && chg === 0 && doub === 1) tmp = 17.07;
          else if (nh === 1 && chg === 0 && sing === 1) tmp = 20.23;
          else if (nh === 0 && chg === -1 && sing === 1) tmp = 23.06;
          break;
        case 2:
          if (nh === 0 && chg === 0 && sing === 2 && in3Ring) tmp = 12.53;
          else if (nh === 0 && chg === 0 && sing === 2 && !in3Ring) tmp = 9.23;
          else if (nh === 0 && chg === 0 && arom === 2) tmp = 13.14;
          break;
        default:
          break;
      }
      if (tmp < 0.0) {
        tmp = 28.5 - nbrs * 8.6 + nh * 1.5;
        if (tmp < 0) tmp = 0.0;
      }
    }

    out[i] = tmp;
  }

  return out;
}

export interface CrippenContrib {
  logp: number;
  mr: number;
}

/**
 * Crippen 逐原子贡献（Wildman-Crippen）。
 *
 * RDKit 的算法：按表顺序对每条 SMARTS 做子结构匹配，取匹配中**查询原子 0** 对应的
 * 分子原子，若它还没被赋类型就写入该行的 logP/MR；所有原子都赋完就提前结束。
 */
export function crippenContribs(
  mol: RDKitMol,
  g: ChemGraph,
  table: [string, string, number, number][],
): CrippenContrib[] {
  const out: CrippenContrib[] = g.atoms.map(() => ({ logp: 0, mr: 0 }));
  const assigned = new Array<boolean>(g.numAtoms).fill(false);
  let remaining = g.numAtoms;

  for (const [, smarts, logp, mr] of table) {
    for (const match of smartsMatches(mol, smarts)) {
      const idx = match.atoms[0];
      if (idx === undefined || idx < 0 || idx >= g.numAtoms) continue;
      if (assigned[idx]) continue;
      assigned[idx] = true;
      out[idx] = { logp, mr };
      remaining -= 1;
    }
    if (remaining <= 0) break;
  }

  return out;
}

/**
 * 杂化推断，用于选 Gasteiger 参数模式（sp3/sp2/sp）。
 *
 * RDKit 的杂化在 sanitize 时算出，rdkit.js 没暴露；这里用等价的经验规则：
 * 有叁键或两个双键 → sp；有一个双键或芳香 → sp2；否则 sp3。
 */
function hybridMode(g: ChemGraph, i: number): string {
  const atom = g.atoms[i];
  if (atom.z === 1) return '*';
  const { doub, trip, arom } = g.bondCounts(i);
  if (trip >= 1 || doub >= 2) return 'sp';
  if (doub === 1 || arom >= 1) return 'sp2';
  return 'sp3';
}

/**
 * 形式电荷在共轭同种原子间均分（`Gasteiger::splitChargeConjugated`）。
 *
 * RDKit 用 `bond->getIsConjugated()` 判断共轭，rdkit.js 拿不到；这里用等价近似：
 * 芳香键，或「两端各自还连着双键/芳香键」的单键。中性分子（无形式电荷）下这段不生效。
 */
function splitChargeConjugated(g: ChemGraph, charges: number[], formal: number[]): void {
  const eps = 1e-8;
  const conjugated = (bondIdx: number): boolean => {
    const bond = g.bonds[bondIdx];
    if (bond.aromatic) return true;
    if (bond.order !== 1) return false;
    const unsat = (idx: number) => {
      const { doub, arom } = g.bondCounts(idx);
      return doub + arom > 0;
    };
    return unsat(bond.a) && unsat(bond.b);
  };

  for (let i = 0; i < g.numAtoms; i += 1) {
    let total = formal[i];
    if (Math.abs(total) <= eps || Math.abs(charges[i]) >= eps) continue;
    const marker = [i];
    for (let b1 = 0; b1 < g.bonds.length; b1 += 1) {
      const bond1 = g.bonds[b1];
      if (bond1.a !== i && bond1.b !== i) continue;
      if (!conjugated(b1)) continue;
      const mid = bond1.a === i ? bond1.b : bond1.a;
      for (let b2 = 0; b2 < g.bonds.length; b2 += 1) {
        if (b2 === b1) continue;
        const bond2 = g.bonds[b2];
        if (bond2.a !== mid && bond2.b !== mid) continue;
        if (!conjugated(b2)) continue;
        const other = bond2.a === mid ? bond2.b : bond2.a;
        if (g.atoms[i].z === g.atoms[other].z) {
          total += formal[other];
          marker.push(other);
        }
      }
    }
    const share = total / marker.length;
    for (const idx of marker) charges[idx] = share;
  }
}

/**
 * Gasteiger-Marsili 迭代均衡电荷（12 轮，阻尼 0.5 逐步减半）。
 *
 * 参数按 (元素, 模式) 查表；隐式氢按 RDKit 的做法单独跟踪（`hChrg`）。
 */
export function gasteigerCharges(g: ChemGraph): number[] {
  const n = g.numAtoms;
  const charges = new Array<number>(n).fill(0);
  const formal = g.atoms.map((a) => a.chg);
  const hChrg = new Array<number>(n).fill(0);
  const ionX = new Array<number>(n).fill(0);
  const energ = new Array<number>(n).fill(0);
  const params: [number, number, number][] = [];

  splitChargeConjugated(g, charges, formal);

  for (let i = 0; i < n; i += 1) {
    const atom = g.atoms[i];
    const p = gasteigerParams(atom.symbol, hybridMode(g, i));
    params.push(p);
    ionX[i] = atom.z === 1 ? IONXH : p[0] + p[1] + p[2];
  }

  let damp = DAMP;
  for (let iter = 0; iter < 12; iter += 1) {
    for (let i = 0; i < n; i += 1) {
      const p = params[i];
      energ[i] = p[0] + charges[i] * (p[1] + p[2] * charges[i]);
    }

    for (let i = 0; i < n; i += 1) {
      let dq = 0.0;
      for (const j of g.neighbors(i)) {
        const dx = energ[j] - energ[i];
        const sgn = dx < 0.0 ? 0 : 1;
        dq += dx / (sgn * (ionX[i] - ionX[j]) + ionX[j]);
      }

      const niHs = g.totalNumHs(i);
      if (niHs > 0) {
        const qHs = hChrg[i] / niHs;
        const enr = H_PARAMS[0] + qHs * (H_PARAMS[1] + H_PARAMS[2] * qHs);
        const dx = enr - energ[i];
        const sgn = dx < 0.0 ? 0 : 1;
        const dqH = dx / (sgn * (ionX[i] - IONXH) + IONXH);
        dq += niHs * dqH;
        hChrg[i] -= niHs * dqH * damp;
      }

      charges[i] += damp * dq;
    }

    damp *= DAMP_SCALE;
  }

  return charges;
}
