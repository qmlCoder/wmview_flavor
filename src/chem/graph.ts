/**
 * 把 rdkit.js `mol.get_json()` 的输出解析成好用的图结构。
 *
 * JSON 采用「默认值 + 覆盖」的紧凑格式：
 *   defaults.atom = {z:6, impHs:0, chg:0, ...}，原子只写与默认不同的字段；
 *   defaults.bond = {bo:1}，键同理。
 * 芳香性与成环信息在 extensions[0]（name = "rdkitRepresentation"）里：
 *   aromaticAtoms / aromaticBonds / atomRings。
 *
 * 注意：键的 `bo` 是**凯库勒式**键级，芳香键要按 `aromaticBonds` 判定
 * （对应 RDKit 的 `BondType.AROMATIC`，这是 bond_features 的第 4 个 one-hot 位）。
 */

import { ELEMENT_SYMBOLS } from './tables/elements';

export interface ChemAtom {
  /** 原子序数 */
  z: number;
  symbol: string;
  /** 隐式氢数（去氢分子上是氢的总数） */
  impHs: number;
  /** 显式氢计数属性（`[CH4]` 这种写法带出来的），一般为 0 */
  expHs: number;
  /** 形式电荷 */
  chg: number;
  aromatic: boolean;
  inRing: boolean;
  in3Ring: boolean;
}

export interface ChemBond {
  a: number;
  b: number;
  /** 凯库勒键级：1/2/3（芳香键这里通常是 1 或 2，看具体凯库勒式） */
  order: number;
  aromatic: boolean;
}

interface RawAtom {
  z?: number;
  impHs?: number;
  expHs?: number;
  chg?: number;
}

interface RawBond {
  bo?: number;
  atoms: [number, number];
}

interface RawExtension {
  name?: string;
  aromaticAtoms?: number[];
  aromaticBonds?: number[];
  atomRings?: number[][];
}

interface RawMolecule {
  atoms: RawAtom[];
  bonds?: RawBond[];
  extensions?: RawExtension[];
}

interface RawJson {
  defaults?: { atom?: RawAtom; bond?: { bo?: number } };
  molecules: RawMolecule[];
}

/** 分子图 + 常用查询（TPSA/原子特征要的数法都收在这里）。 */
export class ChemGraph {
  readonly atoms: ChemAtom[];
  readonly bonds: ChemBond[];
  private readonly nbrs: number[][];

  constructor(atoms: ChemAtom[], bonds: ChemBond[]) {
    this.atoms = atoms;
    this.bonds = bonds;
    this.nbrs = atoms.map(() => []);
    for (const bond of bonds) {
      this.nbrs[bond.a].push(bond.b);
      this.nbrs[bond.b].push(bond.a);
    }
  }

  get numAtoms(): number {
    return this.atoms.length;
  }

  neighbors(i: number): number[] {
    return this.nbrs[i];
  }

  /** 重原子邻居（TPSA 的 nNbrs 就是它）。 */
  heavyNeighbors(i: number): number[] {
    return this.nbrs[i].filter((j) => this.atoms[j].z !== 1);
  }

  degree(i: number): number {
    return this.nbrs[i].length;
  }

  /** 连接的氢原子数（分子带显式氢时就是真实氢数）。 */
  hNeighbors(i: number): number {
    let n = 0;
    for (const j of this.nbrs[i]) {
      if (this.atoms[j].z === 1) n += 1;
    }
    return n;
  }

  /** `atom.GetTotalNumHs()`：显式氢计数 + 隐式氢数。 */
  totalNumHs(i: number): number {
    return this.atoms[i].impHs + this.atoms[i].expHs;
  }

  /**
   * 该原子参与的各类型键的数量（TPSA 的 nSing/nDoub/nTrip/nArom）。
   *
   * 与 RDKit `getTPSAAtomContribs` 一致：**跳过与氢相连的键**（那些计入 nHs，
   * 否则 O-H 会被当成一根单键，导致 nh=1&sing=1 这类分支匹配不上）。
   */
  bondCounts(i: number): { sing: number; doub: number; trip: number; arom: number } {
    let sing = 0;
    let doub = 0;
    let trip = 0;
    let arom = 0;
    for (const bond of this.bonds) {
      if (bond.a !== i && bond.b !== i) continue;
      const other = bond.a === i ? bond.b : bond.a;
      if (this.atoms[other].z === 1) continue; // 与氢的键不计入
      if (bond.aromatic) arom += 1;
      else if (bond.order === 1) sing += 1;
      else if (bond.order === 2) doub += 1;
      else if (bond.order === 3) trip += 1;
    }
    return { sing, doub, trip, arom };
  }

  /** 显式价：键级之和（芳香键按 1.5），与 RDKit `getExplicitValence()` 一致。 */
  explicitValence(i: number): number {
    let sum = 0;
    for (const bond of this.bonds) {
      if (bond.a !== i && bond.b !== i) continue;
      sum += bond.aromatic ? 1.5 : bond.order;
    }
    return sum;
  }

  /** `atom.GetTotalValence()` = 显式价 + 氢数（RDKit 里是 int）。 */
  totalValence(i: number): number {
    return Math.trunc(this.explicitValence(i) + this.totalNumHs(i));
  }

  /** `atom.GetTotalDegree()`：连接的原子数**含隐式氢**（去氢分子上尤其重要）。 */
  totalDegree(i: number): number {
    return this.degree(i) + this.totalNumHs(i);
  }
}

/** 由 `get_json()` 文本构造图。 */
export function parseGraph(json: string): ChemGraph {
  const data = JSON.parse(json) as RawJson;
  const mol = data.molecules?.[0];
  if (!mol) throw new Error('分子 JSON 里没有 molecules 数组');

  const rawAtoms = mol.atoms ?? [];
  const rawBonds = mol.bonds ?? [];
  const ext = (mol.extensions ?? []).find((e) => e.name === 'rdkitRepresentation') ?? {};

  const aromaticAtoms = new Set(ext.aromaticAtoms ?? []);
  const aromaticBonds = new Set(ext.aromaticBonds ?? []);
  const rings = ext.atomRings ?? [];

  const inRing = new Array<boolean>(rawAtoms.length).fill(false);
  const in3Ring = new Array<boolean>(rawAtoms.length).fill(false);
  for (const ring of rings) {
    for (const idx of ring) {
      if (idx >= 0 && idx < rawAtoms.length) {
        inRing[idx] = true;
        if (ring.length === 3) in3Ring[idx] = true;
      }
    }
  }

  const atoms: ChemAtom[] = rawAtoms.map((raw, i) => {
    const z = raw.z ?? 6; // JSON 默认原子序数是 6（碳）
    return {
      z,
      symbol: ELEMENT_SYMBOLS[z - 1] ?? 'C',
      impHs: raw.impHs ?? 0,
      expHs: raw.expHs ?? 0,
      chg: raw.chg ?? 0,
      aromatic: aromaticAtoms.has(i),
      inRing: inRing[i],
      in3Ring: in3Ring[i],
    };
  });

  const bonds: ChemBond[] = rawBonds.map((raw, i) => ({
    a: raw.atoms[0],
    b: raw.atoms[1],
    order: raw.bo ?? 1,
    aromatic: aromaticBonds.has(i),
  }));

  return new ChemGraph(atoms, bonds);
}

/** `get_coords()` 直接返回 JS 数组（`[[x,y,z], ...]`），这里只做数值化与类型收束。 */
export function parseCoords(raw: unknown): number[][] {
  const arr = typeof raw === 'string' ? (JSON.parse(raw) as unknown[]) : (raw as unknown[]);
  if (!Array.isArray(arr)) return [];
  // 兼容「平铺数组」形式（[x,y,z,x,y,z,...]）
  if (arr.length && typeof arr[0] === 'number') {
    const flat = arr as number[];
    const out: number[][] = [];
    for (let i = 0; i + 2 < flat.length; i += 3) out.push([flat[i], flat[i + 1], flat[i + 2]]);
    return out;
  }
  return (arr as number[][]).map((c) => [Number(c[0]), Number(c[1]), Number(c[2])]);
}
