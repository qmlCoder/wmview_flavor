/**
 * 运行时冒烟测试：在宿主 WebView 里跑一次完整的「分子 → 特征 → wasm 前向 → 概率」。
 *
 * 用法（在宿主里执行，见 README「验证」）：
 *   const m = await import('http://asset.localhost/plugs/wmview.flavor/smoke/smoke.js')
 *   return JSON.stringify(await m.smoke('CCCOC(=O)C=Cc1ccco1'))
 */

import { loadFlavorRuntime } from '../src/chem/browser'
import { predict, type FlavorInput } from '../src/chem/pipeline'

export interface SmokeResult {
  runtime: { rdkit: string; classes: number; labels: number; features: number; loadMs: number }
  reply: {
    ok: boolean
    error?: string
    warning?: string | null
    smiles?: string
    atoms?: number
    pharmacophores?: number
    top: { label: string; prob: number }[]
    count: number
  }
}

/**
 * 跑一次冒烟测试。`input` 可以是 SMILES 字符串，也可以是场景分子
 * （`{ syms, xyzs, bonds, charge }`）；省略时用苯环酯示例分子。
 */
export async function smoke(input: string | FlavorInput = 'CCCOC(=O)C=Cc1ccco1'): Promise<SmokeResult> {
  const runtime = await loadFlavorRuntime()
  const reply = predict(
    typeof input === 'string' ? { smiles: input } : input,
    {
      labels: runtime.labels,
      featureDefs: runtime.featureDefs,
      forward: runtime.forward,
      threshold: 0.5,
    },
  )
  return {
    runtime: {
      rdkit: runtime.rdkitVersion,
      classes: runtime.classes,
      labels: runtime.labels.length,
      features: runtime.featureDefs.length,
      loadMs: Math.round(runtime.loadMs),
    },
    reply: {
      ok: reply.ok,
      error: reply.error,
      warning: reply.warning,
      smiles: reply.smiles,
      atoms: reply.atoms,
      pharmacophores: reply.pharmacophores,
      top: reply.results.slice(0, 8).map((r) => ({ label: r.label, prob: Number(r.prob.toFixed(4)) })),
      count: reply.results.length,
    },
  }
}

// 也挂到 window 上，方便在宿主控制台直接调 smoke_()
;(window as unknown as { smoke_: typeof smoke }).smoke_ = smoke
