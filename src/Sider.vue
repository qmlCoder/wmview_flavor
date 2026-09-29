<!-- eslint-disable vue/multi-word-component-names -->
<style module>
div.tip {
  font-size: smaller;
  color: gray;
  margin-top: 8px;
}

div.row {
  display: flex;
  align-items: center;
  gap: 6px;
  width: 100%;
}

div.toolbar {
  display: flex;
  align-items: center;
  gap: 6px;
  margin: 5px;
}

div.list {
  width: 100%;
  max-height: 20rem;
  overflow-y: auto;
}

span.label {
  width: 6.5rem;
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
  font-size: small;
}

span.prob {
  width: 3.2rem;
  text-align: right;
  font-size: small;
  font-variant-numeric: tabular-nums;
}

div.bar {
  flex: 1;
}

span.empty {
  color: gray;
  font-size: small;
}

div.doc {
  width: 100%;
  box-sizing: border-box;
  padding: 0 2px 2px;
  font-size: small;
  line-height: 1.55;
  color: #333;
}

div.doc p {
  margin: 4px 0;
}

div.doc ul,
div.doc ol {
  margin: 4px 0;
  padding-left: 1.2rem;
}

div.doc li {
  margin: 3px 0;
}

div.doc code {
  padding: 0 3px;
  border-radius: 3px;
  background-color: #f2f3f5;
  font-size: smaller;
  word-break: break-all;
}

div.doc strong {
  color: #1f6feb;
}
</style>

<template>
  <PanelGroup title="分子气味预测">
    <PanelItem title="分子来源">
      <el-radio-group v-model="state.source" size="small">
        <el-radio-button value="scene">场景分子</el-radio-button>
        <el-radio-button value="smiles">SMILES</el-radio-button>
      </el-radio-group>
    </PanelItem>

    <PanelItem v-if="state.source === 'scene'" title="当前分子">
      <div :class="$style.row">
        <span :class="$style.empty">{{ state.moleName || '场景中没有分子' }}</span>
        <el-button size="small" text type="primary" @click="refresh_mole">刷新</el-button>
      </div>
    </PanelItem>

    <PanelItem
      v-else
      title="SMILES"
      tip="SMILES 没有 3D 坐标，这里用 RDKit 的 2D 平面坐标，精度不如场景分子"
    >
      <el-input
        v-model="state.smiles"
        size="small"
        clearable
        placeholder="例如 CCCOC(=O)C=Cc1ccco1"
      ></el-input>
    </PanelItem>

    <PanelItem title="总电荷" tip="场景分子推断键级时使用，离子请填对应电荷">
      <el-input-number
        v-model="state.charge"
        size="small"
        :min="-4"
        :max="4"
        :step="1"
      ></el-input-number>
    </PanelItem>

    <PanelItem title="推理核" tip="化学感知与模型前向都在本进程内运行（rdkit.js + Rust→wasm）">
      <div :class="$style.row">
        <span :class="$style.empty">{{ state.engineMsg }}</span>
        <el-button
          v-if="state.engine === 'error'"
          size="small"
          text
          type="primary"
          @click="reload_engine"
        >
          重试
        </el-button>
      </div>
    </PanelItem>
  </PanelGroup>

  <div :class="$style.toolbar">
    <el-button type="primary" size="small" :loading="state.running" @click="predict">预测气味</el-button>
    <el-button size="small" :disabled="!state.results.length" @click="export_csv">导出 CSV</el-button>
    <el-button size="small" text @click="self_check">接口自检</el-button>
  </div>

  <PanelGroup title="预测结果">
    <PanelItem title="命中阈值">
      <el-slider
        v-model="state.threshold"
        :min="0"
        :max="1"
        :step="0.01"
        size="small"
        :show-tooltip="false"
      ></el-slider>
    </PanelItem>

    <PanelItem title="显示条数">
      <div :class="$style.row">
        <el-input-number
          v-model="state.topN"
          size="small"
          :min="5"
          :max="138"
          :step="5"
        ></el-input-number>
        <el-checkbox v-model="state.onlyHit" size="small">只看命中</el-checkbox>
      </div>
    </PanelItem>

    <PanelItem title="气味标签" :row="false">
      <div :class="$style.list">
        <div v-for="item in shown" :key="item.label" :class="$style.row">
          <span :class="$style.label" :title="item.label">{{ item.label }}</span>
          <el-progress
            :class="$style.bar"
            :percentage="Math.round(item.prob * 100)"
            :stroke-width="10"
            :show-text="false"
          ></el-progress>
          <span :class="$style.prob">{{ (item.prob * 100).toFixed(1) }}%</span>
        </div>
        <span v-if="!shown.length" :class="$style.empty">还没有预测结果</span>
      </div>
    </PanelItem>

    <PanelItem title="统计">
      <span :class="$style.empty">
        命中 {{ hitCount }} 项（≥ {{ state.threshold.toFixed(2) }}）／共 {{ state.results.length }} 项
        <template v-if="state.summary">· {{ state.summary }}</template>
      </span>
    </PanelItem>
  </PanelGroup>

  <PanelGroup title="插件介绍">
    <div :class="$style.doc">
      <p>
        <strong>flavor</strong> 用开源模型 <strong>PharmaGNN</strong>（药效团特征 + GATv2
        图注意力网络）给分子做<strong>多标签气味预测</strong>：输入一个分子，输出 138
        个气味标签（fruity / floral / pungent …）的概率。
      </p>
      <ul>
        <li>输入：场景里打开的分子，或直接粘贴 SMILES</li>
        <li>输出：138 个标签的概率条，可导出 CSV，命中项同时写入结果日志</li>
        <li>推理在<strong>本进程内</strong>完成：化学感知用 rdkit.js，前向是 Rust 编译的 wasm，不联网、无外部程序</li>
        <li>因为不依赖可执行文件，Windows 与 Android 都能跑</li>
      </ul>
      <p :class="$style.tip">
        场景分子按坐标推断键级；SMILES 没有 3D 坐标，只能用 RDKit 的 2D 平面坐标，
        而药效团特征里含绝对坐标项，所以精度低于场景分子（实测 top-10 标签平均重叠 7/10）。
        想要最准的结果，请打开分子的 3D 结构文件再预测。
      </p>
    </div>
  </PanelGroup>

  <PanelGroup title="使用教程">
    <div :class="$style.doc">
      <el-collapse v-model="state.docOpen">
        <el-collapse-item name="quick" title="① 快速上手">
          <ol>
            <li>打开分子文件（SDF / MOL / PDB / xyz / gjf …），或把「分子来源」切到 SMILES 粘贴结构式。</li>
            <li>按需填写「总电荷」——离子分子要填对，否则键级推断会出错。</li>
            <li>设置「命中阈值」（越接近 1 越严格）与「显示条数 / 只看命中」。</li>
            <li>等「推理核」显示<strong>已就绪</strong>（首次约 1–3 秒），点「预测气味」即可在下方看到概率条。</li>
            <li>点「导出 CSV」选择保存路径，把完整结果存成表格。</li>
          </ol>
        </el-collapse-item>

        <el-collapse-item name="source" title="② 选哪种分子来源">
          <ul>
            <li>
              <strong>SMILES</strong>：最省事，但只有 2D 平面坐标（RDKit 不提供构象生成），
              药效团特征里的绝对坐标项会因此偏粗——<strong>想准就用场景分子</strong>。
            </li>
            <li><strong>场景分子（原文件可读）</strong>：直接用文件自带的键级，SDF / MOL / PDB 等最准确。</li>
            <li><strong>场景分子（仅坐标 + 连接表）</strong>：由 3D 坐标推断键级，分子带显式氢时准确。</li>
            <li><strong>场景分子（无显式氢且读不到文件）</strong>：只能按单键退化处理，结果仅供参考，建议改用 SMILES。</li>
          </ul>
        </el-collapse-item>

        <el-collapse-item name="result" title="③ 看懂结果">
          <ul>
            <li>概率条 = 该标签的置信度（0–100%）；「命中」= 概率 ≥ 阈值。</li>
            <li>「统计」显示命中项数与标签总数；用 SMILES 时还会给出重原子数与药效团数。</li>
            <li>这是多标签分类，同一分子同时命中多个气味（如 fatty / oily / pungent）属正常。</li>
          </ul>
        </el-collapse-item>

        <el-collapse-item name="faq" title="④ 常见问题">
          <ul>
            <li>
              「推理核」显示<strong>加载失败</strong>：多为插件包不完整（缺 <code>assets/</code> 里的
              wasm 或权重）。重新解压安装包再试；也可以点「重试」重新加载。
            </li>
            <li>首次预测前要等 1–3 秒：要加载 7.3 MB 的 RDKit wasm 与 6.3 MB 权重，之后每个分子只要几十毫秒。</li>
            <li>出现「键级推断」警告：分子缺少显式氢，结果仅供参考。</li>
            <li>想确认宿主接口：点「接口自检」，宿主实际暴露的 <code>wmapi_*</code> 方法会写进日志。</li>
          </ul>
        </el-collapse-item>
      </el-collapse>
    </div>
  </PanelGroup>
</template>

<script setup lang="ts">
import { computed, nextTick, onMounted, reactive } from 'vue'
import PanelGroup from './comps/Panel-Group.vue'
import PanelItem from './comps/Panel-Item.vue'
import { loadFlavorRuntime, type FlavorRuntime } from './chem/browser'
import { predict as run_predict, type FlavorInput, type FlavorResultItem } from './chem/pipeline'

/** 分子结构（syms / xyzs / bonds），兼容不同版本的程序接口 */
interface MoleGeom {
  syms: string[]
  xyzs: number[][]
  bonds: [number, number][]
}

const state = reactive({
  source: 'scene' as 'scene' | 'smiles',
  moleName: '' as string,
  smiles: '',
  charge: 0,
  threshold: 0.5,
  topN: 15,
  onlyHit: false,
  running: false,
  results: [] as FlavorResultItem[],
  summary: '',
  engine: 'idle' as 'idle' | 'loading' | 'ready' | 'error',
  engineMsg: '尚未加载',
  docOpen: ['quick'] as string[],
})

const hitCount = computed(() => state.results.filter((r) => r.prob >= state.threshold).length)
const shown = computed(() => {
  const list = state.onlyHit ? state.results.filter((r) => r.prob >= state.threshold) : state.results
  return list.slice(0, state.topN)
})

/** 读取场景里当前显示的分子名 */
const refresh_mole = () => {
  state.moleName = window.wmapi_scene?.get_mole_name() ?? ''
}

/** 推理核就绪后的状态文案 */
const engine_info = (rt: FlavorRuntime) =>
  `已就绪 · RDKit ${rt.rdkitVersion} · ${rt.classes} 个标签 · 载入 ${rt.loadMs.toFixed(0)} ms`

/** 确保推理核已加载（幂等；并发调用只会加载一次） */
const ensure_engine = async (): Promise<FlavorRuntime> => {
  if (state.engine === 'ready') {
    const rt = await loadFlavorRuntime()
    return rt
  }
  state.engine = 'loading'
  try {
    const rt = await loadFlavorRuntime((msg) => {
      state.engineMsg = msg
    })
    state.engine = 'ready'
    state.engineMsg = engine_info(rt)
    return rt
  } catch (err) {
    state.engine = 'error'
    state.engineMsg = `加载失败：${(err as Error).message}`
    throw err
  }
}

/** 手动重试加载（失败后按钮触发） */
const reload_engine = () => {
  state.engine = 'idle'
  void ensure_engine().catch(() => {})
}

/**
 * 读取分子结构。程序接口在不同版本里改过名：
 * - 当前版本：`wmapi_files.get_mole_info(name)`（异步）
 * - 旧模板：`wmapi_files.get_moleInfo(name)`（同步）
 * - 兜底：`wmapi_scene.get_mole_geom(name)`（只给坐标，没有连接表）
 */
const read_mole = async (name: string): Promise<MoleGeom> => {
  const files = window.wmapi_files as unknown as Record<string, unknown>
  const current = files['get_mole_info']
  if (typeof current === 'function') {
    const info = (await (current as (n: string) => Promise<MoleGeom>).call(
      window.wmapi_files,
      name,
    )) as MoleGeom
    if (info?.syms?.length) return info
  }
  const legacy = files['get_moleInfo']
  if (typeof legacy === 'function') {
    const info = (legacy as (n: string) => MoleGeom).call(window.wmapi_files, name)
    if (info?.syms?.length) return info
  }
  const geom = window.wmapi_scene.get_mole_geom(name)
  if (geom?.length) {
    return {
      syms: geom.map((a) => a[0]),
      xyzs: geom.map((a) => [a[1], a[2], a[3]]),
      bonds: [],
    }
  }
  throw new Error(`读不到分子结构：${name}`)
}

/** 组装预测请求：场景分子（优先用分子文件的键级，其次用坐标 + 连接表）或 SMILES */
const build_input = async (): Promise<FlavorInput> => {
  if (state.source === 'smiles') {
    const smiles = state.smiles.trim()
    if (!smiles) throw new Error('请先填入 SMILES')
    return { smiles }
  }
  const name = state.moleName || window.wmapi_scene?.get_mole_name() || ''
  if (!name) throw new Error('场景里还没有分子，先打开一个分子文件')

  // sdf/mol/mdl 文件里带真实键级，优先用它；其它格式（xyz/gjf/log）直接走坐标 + 连接表
  if (/\.(sdf|mol|mdl)$/i.test(name)) {
    try {
      const fold = (window.wmapi_files?.get_moleFold() ?? '').replace(/[\\/]+$/, '')
      if (fold) {
        const text = await window.wmapi_cores.read_text(`${fold}/${name}`)
        if (text && /M\s+END/.test(text) && /V2000|V3000/.test(text)) {
          return { molblock: text.split('$$$$')[0] }
        }
      }
    } catch {
      // 读文件失败就退回场景数据，由前端按坐标推断键级
    }
  }

  const info = await read_mole(name)
  return {
    syms: info.syms,
    xyzs: info.xyzs,
    bonds: info.bonds ?? [],
    charge: state.charge,
  }
}

const predict = async () => {
  if (state.running) return
  let input: FlavorInput
  try {
    input = await build_input()
  } catch (err) {
    window.wmapi_cores.notify((err as Error).message, 'warning')
    return
  }

  state.running = true
  window.wmapi_cores.show_loading('气味预测中…')
  try {
    const runtime = await ensure_engine()
    // 让加载面板先渲染出来，再做同步的特征提取 + 前向
    await nextTick()
    await new Promise((resolve) => setTimeout(resolve, 16))

    const reply = run_predict(input, {
      labels: runtime.labels,
      featureDefs: runtime.featureDefs,
      forward: runtime.forward,
      threshold: state.threshold,
    })
    if (!reply.ok) {
      window.wmapi_cores.notify(reply.error || '预测失败', 'error')
      window.wmapi_cores.add_logText('flavor', reply.error || '')
      return
    }
    state.results = reply.results ?? []
    state.summary = reply.smiles
      ? `${reply.smiles} · ${reply.atoms ?? 0} 个重原子 · ${reply.pharmacophores ?? 0} 个药效团`
      : ''
    if (reply.warning) {
      window.wmapi_cores.notify(reply.warning, 'warning')
    } else {
      window.wmapi_cores.notify(`预测完成，命中 ${hitCount.value} 项气味`, 'success')
    }
    window.wmapi_cores.add_reslog(
      'flavor',
      state.results
        .filter((r) => r.prob >= state.threshold)
        .slice(0, 20)
        .map((r) => `${r.label}: ${(r.prob * 100).toFixed(1)}%`)
        .join('\n'),
    )
  } catch (err) {
    window.wmapi_cores.notify(`预测失败：${(err as Error).message}`, 'error')
    window.wmapi_cores.add_logText('flavor', String((err as Error).stack ?? err))
  } finally {
    window.wmapi_cores.hide_loading()
    state.running = false
  }
}

const export_csv = async () => {
  const path = await window.wmapi_cores.save_file_dialog()
  if (!path) return
  const rows = ['label,probability,hit']
  for (const r of state.results) {
    rows.push(`${r.label},${r.prob.toFixed(6)},${r.prob >= state.threshold ? 1 : 0}`)
  }
  window.wmapi_cores.save_text(path, rows.join('\n'))
  window.wmapi_cores.notify('预测结果已导出', 'success')
}

/** 把宿主实际暴露的接口清单打到日志里，便于确认程序版本与 API 是否一致 */
const self_check = () => {
  const dump = (name: string, obj: unknown) => {
    if (!obj) return `${name}: (未定义)`
    return `${name}: ${Object.keys(obj as object).sort().join(', ')}`
  }
  const name = window.wmapi_scene?.get_mole_name?.() ?? ''
  window.wmapi_cores.add_logText(
    'flavor',
    [
      `平台: ${navigator.userAgent}`,
      `推理核: ${state.engineMsg}`,
      dump('wmapi_files', window.wmapi_files),
      dump('wmapi_cores', window.wmapi_cores),
      dump('wmapi_scene', window.wmapi_scene),
      `当前分子: ${name || '(无)'}`,
    ].join('\n'),
  )
  window.wmapi_cores.notify('接口清单已写入日志', 'info')
}

onMounted(() => {
  refresh_mole()
  // 进面板就先把推理核加载起来，用户点预测时就不用等
  void ensure_engine().catch(() => {})
})
</script>
