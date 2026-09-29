/**
 * rdkit.js（`@rdkit/rdkit`，MinimalLib 的 WASM 构建）的加载与薄封装。
 *
 * 这里**不**做任何 Vite 特有的导入（如 `?url`），wasm 字节由调用方注入：
 * - 浏览器/插件：`src/chem/browser.ts` 用 `?url` 拿到资源再 fetch
 * - node 开发脚本：直接读 node_modules 里的 wasm 文件
 *
 * 实测（RDKit_minimal 2026.03.6）该 wasm 提供：get_mol/get_qmol、get_json（含
 * aromaticAtoms/aromaticBonds/atomRings）、递归 SMARTS 匹配、add_hs/remove_hs、
 * get_coords、get_descriptors。**不**提供逐原子 Crippen/TPSA、Gasteiger、3D 构象生成。
 */

import initRDKit from '@rdkit/rdkit';

/** MinimalLib 暴露的分子对象（只列出本插件用到的部分）。 */
export interface RDKitMol {
  is_valid(): boolean;
  has_coords(): number;
  get_json(): string;
  get_molblock(): string;
  /** `details` 可传 `{"isomericSmiles":false}` 之类，用来关掉立体化学标记 */
  get_smiles(details?: string): string;
  /** 逐原子坐标，JS 数组形式（不是 JSON 字符串） */
  get_coords(): number[][];
  get_substruct_matches(query: RDKitMol): string;
  get_num_atoms(heavyOnly?: boolean): number;
  get_num_bonds(): number;
  set_new_coords(useCoordGen?: boolean): boolean;
  add_hs(): string;
  add_hs_in_place(): boolean;
  remove_hs(): string;
  remove_hs_in_place(): boolean;
  copy(): RDKitMol | null;
  delete(): void;
  deleteLater(): void;
}

export interface RDKitModule {
  version(): string;
  get_mol(input: string, details?: string): RDKitMol | null;
  get_qmol(smarts: string): RDKitMol | null;
}

export interface MolParseOptions {
  sanitize?: boolean;
  removeHs?: boolean;
  strictParsing?: boolean;
}

let modulePromise: Promise<RDKitModule> | null = null;
let moduleInstance: RDKitModule | null = null;

export interface ChemInitOptions {
  /** wasm 字节；给定时用 `instantiateWasm` 直接实例化，不让 glue 自己去找文件 */
  wasmBinary?: ArrayBuffer | Uint8Array;
  /** wasm 地址，兜底用（万一 glue 仍要自己加载） */
  wasmUrl?: string;
}

/**
 * 初始化（幂等）。
 *
 * 这个 MinimalLib 构建**不读** `Module.wasmBinary`：它按自己算出来的 `scriptDirectory`
 * 去找 `RDKit_minimal.wasm`。在宿主 WebView 里 `WorkerGlobalScope` 存在，glue 会走 worker
 * 分支、把 `scriptDirectory` 算成**宿主页面目录**，于是取到 index.html，报
 * "expected magic word 00 61 73 6d, found 3c 21 64 6f"。所以这里直接接管
 * `instantiateWasm`（glue 里优先级最高的钩子），用调用方取好的字节自己实例化，
 * `locateFile` 再兜一道底。
 */
export function initChem(options: ChemInitOptions | ArrayBuffer | Uint8Array): Promise<RDKitModule> {
  if (!modulePromise) {
    const opts: ChemInitOptions =
      options instanceof ArrayBuffer || ArrayBuffer.isView(options)
        ? { wasmBinary: options as ArrayBuffer | Uint8Array }
        : options;
    const bytes = opts.wasmBinary ? new Uint8Array(opts.wasmBinary) : null;
    modulePromise = initRDKit({
      ...(bytes
        ? {
            instantiateWasm(
              imports: WebAssembly.Imports,
              callback: (instance: WebAssembly.Instance) => void,
            ): WebAssembly.Exports {
              void WebAssembly.instantiate(bytes, imports).then((result) => {
                callback(result.instance);
              });
              return {};
            },
          }
        : {}),
      ...(opts.wasmUrl ? { locateFile: () => opts.wasmUrl } : {}),
    }).then((m: RDKitModule) => {
      moduleInstance = m;
      return m;
    });
  }
  return modulePromise;
}

export function chemReady(): boolean {
  return moduleInstance !== null;
}

export function chem(): RDKitModule {
  if (!moduleInstance) throw new Error('rdkit.js 尚未初始化，请先 await initChem(wasm)');
  return moduleInstance;
}

/**
 * 解析分子输入（SMILES / molblock / SDF 首块）。
 * `removeHs: false` 是关键：MinimalLib 默认会去氢，而药效团特征需要氢原子。
 */
export function parseMol(input: string, opts: MolParseOptions = {}): RDKitMol {
  const details = JSON.stringify({
    sanitize: opts.sanitize ?? true,
    removeHs: opts.removeHs ?? false,
    strictParsing: opts.strictParsing ?? true,
  });
  const mol = chem().get_mol(input, details);
  if (!mol || !mol.is_valid()) throw new Error('无法解析分子输入');
  return mol;
}

/** SMARTS 查询缓存：同一 SMARTS 只编译一次（Crippen 表有上百条）。 */
const queryCache = new Map<string, RDKitMol | null>();

export function smartsQuery(smarts: string): RDKitMol | null {
  if (!queryCache.has(smarts)) {
    queryCache.set(smarts, chem().get_qmol(smarts));
  }
  return queryCache.get(smarts) ?? null;
}

/** 一个子结构匹配：`atoms[i]` 是查询第 i 个原子命中的分子原子索引。 */
export type Match = { atoms: number[]; bonds: number[] };

/** 全部匹配（原子索引按查询顺序，与 RDKit `feat.GetAtomIds()` 一致）。 */
export function smartsMatches(mol: RDKitMol, smarts: string): Match[] {
  const query = smartsQuery(smarts);
  if (!query) return [];
  const raw = mol.get_substruct_matches(query);
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw) as unknown;
    // 注意：MinimalLib 在**无匹配**时返回的是 "{}"（不是 "[]"），必须挡住
    if (!Array.isArray(parsed)) return [];
    return parsed as Match[];
  } catch {
    return [];
  }
}

/** 只取第一个匹配就够用时（例如 Crippen 的类型判定）用这个。 */
export function smartsMatchFirst(mol: RDKitMol, smarts: string): Match | null {
  const all = smartsMatches(mol, smarts);
  return all.length ? all[0] : null;
}

/** 释放 MinimalLib 对象（wasm 内存需显式回收）。 */
export function release(...mols: (RDKitMol | null | undefined)[]): void {
  for (const m of mols) {
    if (m) m.deleteLater();
  }
}
