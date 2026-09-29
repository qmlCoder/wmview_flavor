//! PharmaGNN 前向计算，逐算子对应 `py/flavor_np.py`。
//!
//! 与 Python 版的差异只在浮点累加顺序：这里用 f64 累加再落回 f32（更接近真值），
//! numpy 侧是 f32 累加/BLAS 分块。实测最大偏差见 `--parity` 报告。

use crate::weights::Weights;

/// 与训练时一致的超参（改动需同时改 `py/flavor_np.py`）。
const NEGATIVE_SLOPE: f32 = 0.2;
const LAYER_NORM_EPS: f64 = 1e-5;
pub const GAT_HEADS: [usize; 5] = [8, 8, 4, 4, 2];
pub const NODE_DIM: usize = 11;
pub const EDGE_DIM: usize = 4;
pub const PHARMA_DIM: usize = 33;
/// edge_encoder 之后的边特征维度，也是 GAT 里 `lin_edge` 的输入维度。
const ENCODED_DIM: usize = 128;

/// MLP 里的算子种类。
#[derive(Clone, Copy, PartialEq)]
enum Step {
    Lin,
    Ln,
    Relu,
}

use Step::{Lin, Ln, Relu};

const NODE_ENCODER: &[(usize, Step)] = &[(0, Lin), (1, Ln), (2, Relu)];
const EDGE_ENCODER: &[(usize, Step)] = &[(0, Lin), (1, Ln), (2, Relu)];
const PHARMA_ENCODER: &[(usize, Step)] = &[
    (0, Lin),
    (1, Ln),
    (2, Relu),
    (4, Lin),
    (5, Ln),
    (6, Relu),
];
const CLASSIFIER: &[(usize, Step)] = &[
    (0, Lin),
    (1, Ln),
    (2, Relu),
    (4, Lin),
    (5, Ln),
    (6, Relu),
    (8, Lin),
    (9, Ln),
    (10, Relu),
    (12, Lin),
];

/// `y = x @ W.T (+ b)`；`x` 为 `n×k` 行主序，`W` 为 `o×k`。
fn linear(
    w: &Weights,
    x: &[f32],
    n: usize,
    k: usize,
    wname: &str,
    bname: Option<&str>,
) -> Vec<f32> {
    let ww = w.get(wname);
    let o = ww.len() / k;
    let mut y = vec![0f32; n * o];
    for i in 0..n {
        let xi = &x[i * k..(i + 1) * k];
        for j in 0..o {
            let wj = &ww[j * k..(j + 1) * k];
            let mut acc = 0f64;
            for t in 0..k {
                acc += xi[t] as f64 * wj[t] as f64;
            }
            y[i * o + j] = acc as f32;
        }
    }
    if let Some(bn) = bname {
        let bb = w.get(bn);
        for i in 0..n {
            for j in 0..o {
                y[i * o + j] += bb[j];
            }
        }
    }
    y
}

/// LayerNorm（最后一维归一化，`eps = 1e-5`）。
fn layer_norm(w: &Weights, x: &[f32], n: usize, d: usize, wname: &str, bname: &str) -> Vec<f32> {
    let ww = w.get(wname);
    let bb = w.get(bname);
    let mut out = vec![0f32; n * d];
    for i in 0..n {
        let row = &x[i * d..(i + 1) * d];
        let mean = row.iter().map(|v| *v as f64).sum::<f64>() / d as f64;
        let var = row
            .iter()
            .map(|v| {
                let dv = *v as f64 - mean;
                dv * dv
            })
            .sum::<f64>()
            / d as f64;
        let inv = 1.0 / (var + LAYER_NORM_EPS).sqrt();
        for j in 0..d {
            let norm = ((row[j] as f64 - mean) * inv) as f32;
            out[i * d + j] = norm * ww[j] + bb[j];
        }
    }
    out
}

fn leaky_relu(v: f32) -> f32 {
    if v > 0.0 {
        v
    } else {
        v * NEGATIVE_SLOPE
    }
}

pub fn sigmoid(v: f32) -> f32 {
    1.0 / (1.0 + (-v as f64).exp()) as f32
}

/// PharmaGNN 推理器。
pub struct Model {
    pub w: Weights,
    pub num_classes: usize,
}

impl Model {
    pub fn new(w: Weights) -> Self {
        let num_classes = w.shape("classifier.12.weight")[0];
        Self { w, num_classes }
    }

    /// 由扁平权重构造。
    pub fn from_parts(index_json: &str, bin: &[u8]) -> Result<Self, String> {
        Ok(Self::new(Weights::from_parts(index_json, bin)?))
    }

    /// 按 `steps` 依次施加 linear / layernorm / relu（dropout 推理时为恒等映射，跳过）。
    fn mlp(&self, mut cur: Vec<f32>, n: usize, mut d: usize, prefix: &str, steps: &[(usize, Step)]) -> Vec<f32> {
        for (idx, step) in steps {
            let base = format!("{prefix}.{idx}");
            match step {
                Lin => {
                    let wname = format!("{base}.weight");
                    let out_dim = self.w.shape(&wname)[0];
                    cur = linear(&self.w, &cur, n, d, &wname, Some(&format!("{base}.bias")));
                    d = out_dim;
                }
                Ln => {
                    cur = layer_norm(
                        &self.w,
                        &cur,
                        n,
                        d,
                        &format!("{base}.weight"),
                        &format!("{base}.bias"),
                    );
                }
                Relu => {
                    cur.iter_mut().for_each(|v| {
                        if *v < 0.0 {
                            *v = 0.0;
                        }
                    });
                }
            }
        }
        cur
    }

    /// 单层 GATv2Conv（`concat=False, add_self_loops=True`，自环边特征取入边均值）。
    ///
    /// `x`: `n×128`；`edge_index`: `2E`（前半 src、后半 dst）；`attr`: `E×128`（已编码）。
    /// 返回 `n×128`。
    fn gatv2(
        &self,
        x: &[f32],
        n: usize,
        edge_index: &[i64],
        attr: &[f32],
        e: usize,
        heads: usize,
        layer: usize,
    ) -> Vec<f32> {
        let w = &self.w;
        let c = x.len() / n; // 通道数（128）
        let hc = heads * c;
        let p = |s: &str| format!("gnn.{layer}.{s}");
        let attr_dim = ENCODED_DIM;

        let x_l = linear(w, x, n, c, &p("lin_l.weight"), Some(&p("lin_l.bias")));
        let x_r = linear(w, x, n, c, &p("lin_r.weight"), Some(&p("lin_r.bias")));

        // 1) 先按 PyG 语义去掉已有自环（图里本来就没有），再补自环
        let mut src: Vec<usize> = Vec::with_capacity(e + n);
        let mut dst: Vec<usize> = Vec::with_capacity(e + n);
        let mut eattr: Vec<f32> = Vec::with_capacity((e + n) * attr_dim);
        for t in 0..e {
            let a = edge_index[t] as usize;
            let b = edge_index[e + t] as usize;
            if a == b {
                continue;
            }
            src.push(a);
            dst.push(b);
            eattr.extend_from_slice(&attr[t * attr_dim..(t + 1) * attr_dim]);
        }
        let e_keep = dst.len();

        // 2) 自环边特征 = 该节点入边特征均值（无入边则为 0）
        let mut cnt = vec![0usize; n];
        let mut acc = vec![0f64; n * attr_dim];
        for t in 0..e_keep {
            let d = dst[t];
            cnt[d] += 1;
            for k in 0..attr_dim {
                acc[d * attr_dim + k] += eattr[t * attr_dim + k] as f64;
            }
        }
        for v in 0..n {
            for k in 0..attr_dim {
                let val = if cnt[v] == 0 {
                    0.0
                } else {
                    (acc[v * attr_dim + k] / cnt[v] as f64) as f32
                };
                eattr.push(val);
            }
            src.push(v);
            dst.push(v);
        }
        let en = e_keep + n;

        // 3) 边特征投影（该层无 bias）
        let edge_proj = if w.has(&p("lin_edge.weight")) {
            linear(w, &eattr, en, attr_dim, &p("lin_edge.weight"), None)
        } else {
            vec![0f32; en * hc]
        };

        // 4) 注意力打分：leaky_relu(x_l[src] + x_r[dst] + edge) · att
        let att = w.get(&p("att")); // (1, heads, c)
        let mut alpha = vec![0f32; en * heads];
        let has_proj = w.has(&p("lin_edge.weight"));
        for t in 0..en {
            let s = src[t];
            let d = dst[t];
            for j in 0..heads {
                let mut score = 0f64;
                for k in 0..c {
                    let mut a = x_l[s * hc + j * c + k] + x_r[d * hc + j * c + k];
                    if has_proj {
                        a += edge_proj[t * hc + j * c + k];
                    }
                    score += leaky_relu(a) as f64 * att[j * c + k] as f64;
                }
                alpha[t * heads + j] = score as f32;
            }
        }

        // 5) 按 dst 分组、每个 head 独立做 softmax
        let mut maxv = vec![f32::NEG_INFINITY; n * heads];
        for t in 0..en {
            for j in 0..heads {
                let v = alpha[t * heads + j];
                let m = &mut maxv[dst[t] * heads + j];
                if v > *m {
                    *m = v;
                }
            }
        }
        let mut denom = vec![0f64; n * heads];
        for t in 0..en {
            for j in 0..heads {
                let v = ((alpha[t * heads + j] - maxv[dst[t] * heads + j]) as f64).exp();
                alpha[t * heads + j] = v as f32;
                denom[dst[t] * heads + j] += v;
            }
        }
        for t in 0..en {
            for j in 0..heads {
                let d = dst[t];
                alpha[t * heads + j] = (alpha[t * heads + j] as f64 / denom[d * heads + j]) as f32;
            }
        }

        // 6) 加权聚合后对 head 取均值
        let mut agg = vec![0f64; n * hc];
        for t in 0..en {
            let s = src[t];
            let d = dst[t];
            for j in 0..heads {
                let a = alpha[t * heads + j] as f64;
                for k in 0..c {
                    agg[d * hc + j * c + k] += a * x_l[s * hc + j * c + k] as f64;
                }
            }
        }
        let bias = if w.has(&p("bias")) {
            Some(w.get(&p("bias")))
        } else {
            None
        };
        let mut out = vec![0f32; n * c];
        for v in 0..n {
            for k in 0..c {
                let mut acc = 0f64;
                for j in 0..heads {
                    acc += agg[v * hc + j * c + k];
                }
                let mut val = (acc / heads as f64) as f32;
                if let Some(b) = bias {
                    val += b[k];
                }
                out[v * c + k] = val;
            }
        }
        out
    }

    /// 输出 logits（sigmoid 之前，138 维）。
    ///
    /// `x`: `nodes×11`；`edge_index`: `2*edges`；`edge_attr`: `edges×4`；`pharma`: 33 维。
    pub fn logits(
        &self,
        x: &[f32],
        nodes: usize,
        edge_index: &[i64],
        edge_attr: &[f32],
        edges: usize,
        pharma: &[f32],
    ) -> Vec<f32> {
        let mut h = self.mlp(x.to_vec(), nodes, NODE_DIM, "node_encoder", NODE_ENCODER);
        let mut ea = self.mlp(
            edge_attr.to_vec(),
            edges,
            EDGE_DIM,
            "edge_encoder",
            EDGE_ENCODER,
        );

        for (i, &heads) in GAT_HEADS.iter().enumerate() {
            let res = self.gatv2(&h, nodes, edge_index, &ea, edges, heads, i);
            for t in 0..h.len() {
                h[t] += res[t];
            }
            h = layer_norm(
                &self.w,
                &h,
                nodes,
                ENCODED_DIM,
                &format!("norms.{i}.weight"),
                &format!("norms.{i}.bias"),
            );
            h.iter_mut().for_each(|v| {
                if *v < 0.0 {
                    *v = 0.0;
                }
            });
        }

        // 全局平均池化（单分子）
        let mut graph = vec![0f64; ENCODED_DIM];
        for v in 0..nodes {
            for k in 0..ENCODED_DIM {
                graph[k] += h[v * ENCODED_DIM + k] as f64;
            }
        }
        let graph: Vec<f32> = graph
            .iter()
            .map(|v| (v / nodes.max(1) as f64) as f32)
            .collect();

        let pharma_repr = self.mlp(
            pharma.to_vec(),
            1,
            PHARMA_DIM,
            "pharma_encoder",
            PHARMA_ENCODER,
        );

        let mut fused = graph;
        fused.extend_from_slice(&pharma_repr);
        let d = fused.len();

        self.mlp(fused, 1, d, "classifier", CLASSIFIER)
    }

    /// 138 个气味标签的概率。
    pub fn predict_proba(
        &self,
        x: &[f32],
        nodes: usize,
        edge_index: &[i64],
        edge_attr: &[f32],
        edges: usize,
        pharma: &[f32],
    ) -> Vec<f32> {
        self.logits(x, nodes, edge_index, edge_attr, edges, pharma)
            .into_iter()
            .map(sigmoid)
            .collect()
    }
}
