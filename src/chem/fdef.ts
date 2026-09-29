/**
 * `BaseFeatures.fdef` 解析 + 药效团特征识别。
 *
 * 严格复刻 RDKit 的行为（见 `Code/GraphMol/MolChemicalFeatures/FeatureParser.cpp`
 * 与 `MolChemicalFeatureFactory.cpp`）：
 *
 * 1. 续行：行尾 `\` 表示与下一行拼接；`#` 开头是注释。
 * 2. `AtomType 名字 SMARTS`：
 *    - 名字前加 `!` 表示「取反」；
 *    - 内部存储为 `$(SMARTS)`，同名重复定义会被合并进同一个 `$(...)` 组
 *      （非取反：把末尾 `]` 换成 `,$(新SMARTS)]`；取反：把首个 `[` 换成 `[!$(新);`）；
 *    - 定义时就把 `{其它类型}` 做**纯文本替换**展开。
 * 3. `DefineFeature 名字 SMARTS` + `Family X`（+ 可选 `Weights`）+ `EndFeature`：
 *    - SMARTS 同样做 `{类型}` 文本替换展开。
 * 4. 识别时按定义顺序匹配，得到若干匹配；**若同族中已有特征是该匹配的超集，
 *    则该匹配被丢弃**（RDKit 的去重规则）。
 * 5. 特征的原子列表按 SMARTS 查询原子顺序排列（不是排序后的集合）。
 */

import { smartsMatches, smartsQuery, type RDKitMol } from './rdkit';

export interface FeatureDef {
  subType: string;
  family: string;
  /** 按 RDKit 原样合并（嵌套 `$()`）的 SMARTS */
  smarts: string;
  /**
   * 与 `smarts` 语义等价的**扁平式**写法：把同名 AtomType 的多个定义拆成并列的
   * `$(...)` 原子测试。
   *
   * 之所以需要：rdkit.js（MinimalLib）的 SMARTS 解析器对「嵌套 `$()` 且内部含 `,` 或
   * `$(...)` 同级并列」的写法会直接返回 null（RDKit 本体能解析），而扁平式能通过，
   * 且匹配结果与 RDKit 一致（见 tools/validate-chem.ts）。仅在原式无法编译时使用。
   */
  smartsFlat: string;
}

/** 只保留插件用到的 8 个家族（与 `py/flavor_feats.py` 的 FAMILIES 一致）。 */
export const FAMILIES = [
  'Donor',
  'Acceptor',
  'NegIonizable',
  'PosIonizable',
  'Aromatic',
  'Hydrophobe',
  'LumpedHydrophobe',
  'ZnBinder',
] as const;

/** 去掉行尾续行符并把逻辑行拼起来。 */
function logicalLines(text: string): string[] {
  const out: string[] = [];
  let acc = '';
  for (const raw of text.split(/\r?\n/)) {
    const line = raw;
    if (line.endsWith('\\')) {
      acc += line.slice(0, -1);
      continue;
    }
    out.push(acc + line);
    acc = '';
  }
  if (acc) out.push(acc);
  return out;
}

/** 文本替换式展开：把所有 `{Name}` 换成已存储的定义（按名字排序，对齐 std::map 顺序）。 */
function expand(smarts: string, defs: Map<string, string>): string {
  let out = smarts;
  for (const key of [...defs.keys()].sort()) {
    const value = defs.get(key) as string;
    if (out.includes(key)) out = out.split(key).join(value);
  }
  return out;
}

/** 解析 fdef 文本 → 特征定义列表（保持文件顺序）。 */
export function parseFdef(text: string): FeatureDef[] {
  const atomTypes = new Map<string, string>();
  /** 同名 AtomType 的各个原始定义（按文件顺序），用于生成扁平式写法 */
  const atomTypeParts = new Map<string, string[]>();
  const defs: FeatureDef[] = [];
  const lines = logicalLines(text);

  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i].trim();
    if (!line || line.startsWith('#')) continue;
    const tokens = line.split(/[\s\t]+/).filter((t) => t.length > 0);
    const keyword = (tokens[0] ?? '').toUpperCase();

    if (keyword === 'ATOMTYPE') {
      if (tokens.length < 3) throw new Error(`AtomType 行缺字段: ${line}`);
      let name = tokens[1];
      const negater = name.startsWith('!');
      if (negater) name = name.slice(1);
      const key = `{${name}}`;
      let sma = `$(${tokens[2]})`;
      const base = atomTypes.get(key);
      if (base !== undefined) {
        if (negater) {
          sma = base.replace('[', `[!${sma};`);
        } else {
          const last = base.lastIndexOf(']');
          sma = `${base.slice(0, last)},${sma}]`;
        }
      }
      sma = expand(sma, atomTypes);
      atomTypes.set(key, sma);
      const parts = atomTypeParts.get(key) ?? [];
      parts.push(tokens[2]);
      atomTypeParts.set(key, parts);
    } else if (keyword === 'DEFINEFEATURE') {
      if (tokens.length < 3) throw new Error(`DefineFeature 行缺字段: ${line}`);
      const subType = tokens[1];
      const smarts = expand(tokens[2], atomTypes);
      const flatDefs = new Map<string, string>();
      for (const [key, parts] of atomTypeParts) {
        flatDefs.set(key, parts.map((p) => `$(${p})`).join(','));
      }
      const smartsFlat = expand(tokens[2], flatDefs);
      let family = '';
      let foundEnd = false;
      for (i += 1; i < lines.length; i += 1) {
        const inner = lines[i].trim();
        if (!inner || inner.startsWith('#')) continue;
        const it = inner.split(/[\s\t]+/).filter((t) => t.length > 0);
        const head = (it[0] ?? '').toUpperCase();
        if (head === 'ENDFEATURE') {
          foundEnd = true;
          break;
        }
        if (head === 'FAMILY') family = it[1] ?? '';
        // WEIGHTS 只影响特征权重，插件不使用，忽略
      }
      if (!foundEnd) throw new Error(`DefineFeature ${subType} 缺少 EndFeature`);
      defs.push({ subType, family, smarts, smartsFlat });
    } else {
      throw new Error(`fdef 中无法识别的关键字: ${kwFromTokens(tokens)}`);
    }
  }
  return defs;
}

function kwFromTokens(tokens: string[]): string {
  return tokens[0] ?? '';
}

export interface Pharmacophore {
  family: string;
  /** 参与该特征的原子索引（按 SMARTS 查询原子顺序） */
  atoms: number[];
}

/**
 * 识别分子里的药效团特征。
 *
 * `mol` 必须是**带显式氢**的分子（与 `py/flavor_feats.py` 里 `mol_3d` 一致），
 * 因为 Donor/Hydrophobe 一类定义依赖氢原子。
 */
export function pharmacophores(mol: RDKitMol, defs: FeatureDef[]): Pharmacophore[] {
  const kept: Pharmacophore[] = [];
  for (const def of defs) {
    if (!(FAMILIES as readonly string[]).includes(def.family)) continue;
    // MinimalLib 解析不了 RDKit 原样合并的嵌套 SMARTS 时，回退到等价的扁平写法
    let matches = smartsMatches(mol, def.smarts);
    if (!smartsQuery(def.smarts) && def.smartsFlat !== def.smarts) {
      matches = smartsMatches(mol, def.smartsFlat);
    }
    for (const match of matches) {
      const atoms = match.atoms;
      if (!atoms.length) continue;
      const set = new Set(atoms);
      // RDKit 去重：同族中已有的特征若是本次匹配的超集，就丢弃本次匹配
      const covered = kept.some((f) => f.family === def.family && isSuperset(f.atoms, set));
      if (covered) continue;
      kept.push({ family: def.family, atoms });
    }
  }
  return kept;
}

/** `a` 是否包含 `b` 的全部元素（std::includes 语义）。 */
function isSuperset(a: number[], b: Set<number>): boolean {
  const as = new Set(a);
  for (const v of b) {
    if (!as.has(v)) return false;
  }
  return true;
}
