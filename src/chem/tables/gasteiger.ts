/**
 * Gasteiger-Marsili 电荷参数表，抄录自 RDKit
 * `Code/GraphMol/PartialCharges/GasteigerParams.cpp`（defaultParamData + additionalParamData）。
 *
 * 每条：[元素符号, 模式, a(电负性), b(硬度), c]
 * 模式由原子的杂化推断（H 用 "*"，超价 S 用 "so"/"so2"）。
 */

export const IONXH = 20.02;
export const DAMP = 0.5;
export const DAMP_SCALE = 0.5;

export type GasteigerParam = [string, string, number, number, number];

export const GASTEIGER_PARAMS: GasteigerParam[] = [
  ['H', '*', 7.17, 6.24, -0.56],
  ['C', 'sp3', 7.98, 9.18, 1.88],
  ['C', 'sp2', 8.79, 9.32, 1.51],
  ['C', 'sp', 10.39, 9.45, 0.73],
  ['N', 'sp3', 11.54, 10.82, 1.36],
  ['N', 'sp2', 12.87, 11.15, 0.85],
  ['N', 'sp', 15.68, 11.7, -0.27],
  ['O', 'sp3', 14.18, 12.92, 1.39],
  ['O', 'sp2', 17.07, 13.79, 0.47],
  ['F', 'sp3', 14.66, 13.85, 2.31],
  ['Cl', 'sp3', 11.0, 9.69, 1.35],
  ['Br', 'sp3', 10.08, 8.47, 1.16],
  ['I', 'sp3', 9.9, 7.96, 0.96],
  ['S', 'sp3', 10.14, 9.13, 1.38],
  ['S', 'so', 10.14, 9.13, 1.38],
  ['S', 'so2', 12.0, 10.81, 1.2],
  ['S', 'sp2', 10.88, 9.49, 1.33],
  ['P', 'sp3', 8.9, 8.24, 0.96],
  ['X', '*', 0.0, 0.0, 0.0],
  ['P', 'sp2', 9.665, 8.53, 0.735],
  ['Si', 'sp3', 7.3, 6.567, 0.657],
  ['Si', 'sp2', 7.905, 6.748, 0.443],
  ['Si', 'sp', 9.065, 7.027, -0.002],
  ['B', 'sp3', 5.98, 6.82, 1.605],
  ['B', 'sp2', 6.42, 6.807, 1.322],
  ['Be', 'sp3', 3.845, 6.755, 3.165],
  ['Be', 'sp2', 4.005, 6.725, 3.035],
  ['Mg', 'sp2', 3.565, 5.572, 2.197],
  ['Mg', 'sp3', 3.3, 5.587, 2.447],
  ['Mg', 'sp', 4.04, 5.472, 1.823],
  ['Al', 'sp3', 5.375, 4.953, 0.867],
  ['Al', 'sp2', 5.795, 5.02, 0.695],
];

/** 元素 → 模式 → 参数（后出现的覆盖先出现的，与 RDKit 的 map 赋值一致）。 */
export const GASTEIGER_MAP: Map<string, [number, number, number]> = new Map(
  GASTEIGER_PARAMS.map(([elem, mode, a, b, c]) => [`${elem}|${mode}`, [a, b, c] as [number, number, number]]),
);

const FALLBACK: [number, number, number] = [0.0, 0.0, 0.0];

/** 与 RDKit `GasteigerParams::getParams(elem, mode)` 等价（含逐级回退）。 */
export function gasteigerParams(elem: string, mode: string): [number, number, number] {
  const direct = GASTEIGER_MAP.get(`${elem}|${mode}`);
  if (direct) return direct;
  const anyMode = GASTEIGER_MAP.get(`${elem}|*`);
  if (anyMode) return anyMode;
  return GASTEIGER_MAP.get('X|*') ?? FALLBACK;
}

/** 隐式氢的参数（`getParams("H", "*")`）。 */
export const H_PARAMS: [number, number, number] = gasteigerParams('H', '*');
