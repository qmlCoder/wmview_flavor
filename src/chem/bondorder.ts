/**
 * 场景分子（只有原子 + 坐标 + 连接表，没有键级）的键级推断。
 *
 * 现状（Python 版）用 `rdDetermineBonds.DetermineBonds`（xyz2mol：把键级分配写成
 * 整数规划求解），rdkit.js 没有暴露这个模块，所以这里自己实现：
 * **在连接表上精确求解一组满足所有原子价键需求的多重键分配**（搜索 + 剪枝，键长先验
 * 用来在多个解之间挑最合理的一个）。解不出来时退回「按键序贪心」，并给出警告。
 *
 * 为什么要精确求解：贪心只看局部、且依赖连接表顺序。丁二烯（C=CC=C）只要连接表里
 * 中间那根骨架键先出现，贪心就会把双键放错位置，末端碳多补一个氢 → 整个结构变成
 * 2-丁烯。精确求解要求每个原子的价都被恰好填满，从根上避免这类错误。
 *
 * 流程：连接表 → 求解键级 → 写成 molblock 交给 RDKit sanitize（芳香性由 RDKit 判定）。
 *
 * 注意：**显式氢是关键信息**。没有氢时「氢数」和「双键」是同一个未知量，理论上欠定
 * （Python 版的 xyz2mol 同样会解错，例如无氢苯会得到 `c1c#cc#cc#1`），这种情况仍然
 * 走警告路径。
 */

const DEFAULT_VALENCE: Record<string, number> = {
  H: 1,
  B: 3,
  C: 4,
  N: 3,
  O: 2,
  F: 1,
  Si: 4,
  P: 3,
  S: 2,
  Cl: 1,
  Br: 1,
  I: 1,
};

/**
 * 共价半径（Cordero 等，单位 Å，与 RDKit 的 `PeriodicTable::getRcovalent` 口径一致）。
 * 只用于「按键长给键级先验」——多个解都满足价键时，用它挑物理上合理的那个。
 */
const COVALENT_RADIUS: Record<string, number> = {
  H: 0.31,
  He: 0.28,
  Li: 1.28,
  Be: 0.96,
  B: 0.84,
  C: 0.76,
  N: 0.71,
  O: 0.66,
  F: 0.57,
  Ne: 0.58,
  Na: 1.66,
  Mg: 1.41,
  Al: 1.21,
  Si: 1.11,
  P: 1.07,
  S: 1.05,
  Cl: 1.02,
  Ar: 1.06,
  K: 2.03,
  Ca: 1.76,
  Br: 1.2,
  I: 1.39,
};

const RADIUS_FALLBACK = 0.9;
/** d / (r_i + r_j) 低于这两个阈值就先验为双键 / 叁键（C-C 1.54→1.01、C=C 1.34→0.88、C≡C 1.20→0.79） */
const RATIO_DOUBLE = 0.93;
const RATIO_TRIPLE = 0.82;
/**
 * 「无显式氢」时用的更严格阈值：只认键长上明确的多重键。
 * 芳香键（苯环 1.39 Å → 0.914）刻意落在阈值之外，因为有没有氢会改变它的正确写法，
 * 与其猜错不如留成单键、把剩下的价交给氢。
 */
const STRICT_RATIO_DOUBLE = 0.9;
const STRICT_RATIO_TRIPLE = 0.83;
/** 搜索节点上限，超了就退回贪心（防止病态输入把页面卡住） */
const SEARCH_BUDGET = 200000;

export const NO_BOND_ORDER_WARNING =
  '分子没有显式氢（或键级存在歧义），键级按价键规则近似推断，预测结果仅供参考；' +
  '建议改用 SMILES 输入，或使用带键级/带氢的 SDF、MOL、XYZ 文件';

/** 目标价（考虑常见形式电荷），与 RDKit 的默认价规则接近。 */
function targetValence(symbol: string, charge: number): number {
  switch (symbol) {
    case 'N':
      return charge === 1 ? 4 : charge === -1 ? 2 : 3;
    case 'O':
      return charge === -1 ? 1 : charge === 1 ? 3 : 2;
    case 'C':
      return charge === 0 ? 4 : 3;
    case 'S':
      return charge === 1 ? 3 : 2;
    case 'P':
      return 3;
    default:
      return DEFAULT_VALENCE[symbol] ?? 0;
  }
}

export interface SceneInput {
  syms: string[];
  xyzs: number[][];
  bonds: [number, number][];
  charge: number;
}

export interface BondOrderResult {
  /** 与输入连接表同序的键级 */
  orders: number[];
  /** 无法满足的价键需求（原子索引 → 缺多少），非空说明结果是近似的 */
  unsatisfied: Map<number, number>;
  /** 求解方式：exact = 精确解（价键全部满足）；greedy = 退回贪心近似 */
  method: 'exact' | 'greedy';
}

export interface InferOptions {
  /** 原子坐标；给了就按键长算先验（只影响多个解之间的取舍） */
  coords?: number[][];
}

/** 键长的「先验键级」：1 / 2 / 3。没有坐标时一律当单键。 */
function lengthRatio(syms: string[], coords: number[][] | undefined, a: number, b: number): number {
  if (!coords) return 1;
  const ca = coords[a];
  const cb = coords[b];
  if (!ca || !cb) return 1;
  const dx = ca[0] - cb[0];
  const dy = ca[1] - cb[1];
  const dz = ca[2] - cb[2];
  const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
  if (!Number.isFinite(d) || d <= 0) return 1;
  const rSum = (COVALENT_RADIUS[syms[a]] ?? RADIUS_FALLBACK) + (COVALENT_RADIUS[syms[b]] ?? RADIUS_FALLBACK);
  if (rSum <= 0) return 1;
  return d / rSum;
}

/** 键长的「先验键级」：1 / 2 / 3。没有坐标时一律当单键。 */
function priorOrder(syms: string[], coords: number[][] | undefined, a: number, b: number): number {
  const ratio = lengthRatio(syms, coords, a, b);
  if (ratio <= RATIO_TRIPLE) return 3;
  if (ratio <= RATIO_DOUBLE) return 2;
  return 1;
}

/**
 * 没有显式氢时的键级来源：只把「键长上明确是多重键」的键标出来（其余留单键），
 * 剩下的价由调用方补氢。
 *
 * 为什么不能像有氢时那样精确求解：氢数和双键数是同一个未知量，理论上欠定。
 * 连 Python 版的 `rdDetermineBonds`（xyz2mol 整数规划）也解不对——无氢苯会得到
 * `c1c#cc#cc#1`。这里只做「有把握的那部分」，并保证不超过原子价。
 */
export function inferBondOrdersFromGeometry(
  syms: string[],
  bonds: [number, number][],
  coords: number[][],
): number[] {
  const orders = bonds.map(([a, b]) => {
    const ratio = lengthRatio(syms, coords, a, b);
    if (ratio <= STRICT_RATIO_TRIPLE) return 3;
    if (ratio <= STRICT_RATIO_DOUBLE) return 2;
    return 1;
  });

  // 不能让任何原子的键级和超过目标价（超了就把最长的键降回来）
  const target = syms.map((s, i) => targetValence(s, 0));
  const incident: number[][] = syms.map(() => []);
  bonds.forEach(([a, b], i) => {
    incident[a].push(i);
    incident[b].push(i);
  });
  for (let i = 0; i < syms.length; i += 1) {
    let sum = incident[i].reduce((acc, bi) => acc + orders[bi], 0);
    if (sum <= target[i]) continue;
    // 键级高的先降（也更长），降到满足目标价为止
    const sorted = incident[i].slice().sort((x, y) => orders[y] - orders[x]);
    for (const bi of sorted) {
      while (sum > target[i] && orders[bi] > 1) {
        orders[bi] -= 1;
        sum -= 1;
      }
      if (sum <= target[i]) break;
    }
  }
  return orders;
}

/**
 * 精确求解：给每条键找一个键级（1..3），使每个原子的
 * `Σ(键级) + 隐式氢 = 目标价` 恰好成立（显式氢的价也为 1，因此不能加倍）。
 *
 * 搜索按「先验键级高的键先定」的顺序展开，候选键级按「离先验近」排序，
 * 于是第一个找到的解通常就是物理解。搜索规模用 `SEARCH_BUDGET` 兜底。
 *
 * @returns 成功时返回键级数组，无解或超预算时返回 null
 */
function solveExact(
  syms: string[],
  bonds: [number, number][],
  charges: number[],
  coords: number[][] | undefined,
): number[] | null {
  const n = syms.length;
  const target = syms.map((s, i) => targetValence(s, charges[i] ?? 0));
  const degree = new Array<number>(n).fill(0);
  for (const [a, b] of bonds) {
    degree[a] += 1;
    degree[b] += 1;
  }
  // rem[i]：该原子还「差多少价」——初始是目标价减去单键已占的部分
  const rem = target.map((t, i) => t - degree[i]);
  if (rem.some((v) => v < 0)) return null; // 单键就超价（例如 4 配位中性氮），无从下手

  const prior = bonds.map(([a, b]) => priorOrder(syms, coords, a, b));
  const order = bonds.map((_, i) => i).sort((x, y) => prior[y] - prior[x] || x - y);

  const orders = bonds.map(() => 1);
  // cap[i]：还没确定的键最多还能给原子 i 加多少（每根最多加 2）
  const cap = degree.map((d) => d * 2);
  let nodes = 0;

  const dfs = (k: number): boolean => {
    if (++nodes > SEARCH_BUDGET) return false;
    if (k === bonds.length) return rem.every((v) => v === 0);
    const bi = order[k];
    const [a, b] = bonds[bi];
    const pref = prior[bi] - 1; // 先验对应的增量
    // 候选增量：0 / 1 / 2，按「离先验近」排序（同样近时优先大键级）
    const cands = [0, 1, 2].filter((d) => d <= rem[a] && d <= rem[b]);
    cands.sort((x, y) => Math.abs(x - pref) - Math.abs(y - pref) || y - x);

    cap[a] -= 2;
    cap[b] -= 2;
    for (const d of cands) {
      rem[a] -= d;
      rem[b] -= d;
      orders[bi] = 1 + d;
      // 剪枝：剩余价不能为负，也不能超过剩余键能提供的上限
      if (rem[a] >= 0 && rem[b] >= 0 && rem[a] <= cap[a] && rem[b] <= cap[b] && dfs(k + 1)) {
        return true;
      }
      rem[a] += d;
      rem[b] += d;
    }
    cap[a] += 2;
    cap[b] += 2;
    orders[bi] = 1;
    return false;
  };

  return dfs(0) ? orders : null;
}

/** 兜底：按键序贪心分配键级（先补双键，再补叁键），解不出来时用它并告警。 */
function inferGreedy(syms: string[], bonds: [number, number][], charges: number[]): BondOrderResult {
  const orders = bonds.map(() => 1);
  const need = syms.map(() => 0);

  // 初始需求 = 目标价 - 单键已占用
  const degree = syms.map(() => 0);
  for (const [a, b] of bonds) {
    degree[a] += 1;
    degree[b] += 1;
  }
  for (let i = 0; i < syms.length; i += 1) {
    need[i] = targetValence(syms[i], charges[i] ?? 0) - degree[i];
  }

  // 双键
  for (let b = 0; b < bonds.length; b += 1) {
    const [i, j] = bonds[b];
    if (orders[b] !== 1) continue;
    if (need[i] > 0 && need[j] > 0) {
      orders[b] = 2;
      need[i] -= 1;
      need[j] -= 1;
    }
  }
  // 叁键
  for (let b = 0; b < bonds.length; b += 1) {
    const [i, j] = bonds[b];
    if (orders[b] !== 1) continue;
    if (need[i] >= 2 && need[j] >= 2) {
      orders[b] = 3;
      need[i] -= 2;
      need[j] -= 2;
    }
  }

  const unsatisfied = new Map<number, number>();
  for (let i = 0; i < syms.length; i += 1) {
    if (need[i] !== 0) unsatisfied.set(i, need[i]);
  }
  return { orders, unsatisfied, method: 'greedy' };
}

/**
 * 推断键级：先尝试精确求解（价键全部满足），失败再退回贪心。
 *
 * `charges[i]` 为该原子的形式电荷（场景路径目前只把总电荷挂在第一个原子上）。
 */
export function inferBondOrders(
  syms: string[],
  bonds: [number, number][],
  charges: number[],
  opts: InferOptions = {},
): BondOrderResult {
  const exact = solveExact(syms, bonds, charges, opts.coords);
  if (exact) return { orders: exact, unsatisfied: new Map(), method: 'exact' };
  return inferGreedy(syms, bonds, charges);
}

/** 写一个 V2000 molblock（RDKit 用它做 sanitize 与芳香性判定）。 */
export function buildMolblock(syms: string[], xyzs: number[][], bonds: [number, number][], orders: number[]): string {
  const lines: string[] = [];
  lines.push('wmview');
  // 程序行：MDL 规范要求维度代码「3D」落在第 21–22 列，RDKit 靠它判断构象是不是 3D。
  // 位置写错的话 RDKit 会当成 2D，立体化学（含 C=C 的 E/Z）就不会从坐标里推出来。
  lines.push('  wmview.flavor'.padEnd(20) + '3D');
  lines.push('');
  lines.push(`${String(syms.length).padStart(3)}${String(bonds.length).padStart(3)}  0  0  0  0  0  0  0  0999 V2000`);
  for (let i = 0; i < syms.length; i += 1) {
    const [x, y, z] = xyzs[i];
    lines.push(
      `${x.toFixed(4).padStart(10)}${y.toFixed(4).padStart(10)}${z.toFixed(4).padStart(10)} ${syms[i].padEnd(3)} 0  0  0  0  0  0  0  0  0  0  0  0`,
    );
  }
  for (let b = 0; b < bonds.length; b += 1) {
    const [i, j] = bonds[b];
    lines.push(`${String(i + 1).padStart(3)}${String(j + 1).padStart(3)}${String(orders[b]).padStart(3)}  0  0  0  0`);
  }
  lines.push('M  END');
  return lines.join('\n');
}
