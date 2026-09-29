//! wmview.flavor 的推理核。
//!
//! 与 `py/flavor_np.py` 逐算子等价：node/edge 编码器 → 5 层 GATv2（8/8/4/4/2 头，
//! 残差 + LayerNorm + ReLU）→ 全局平均池化 → 药效团 MLP → 拼接 → 分类头 → sigmoid。
//!
//! 模块划分：
//! - [`weights`] 扁平权重（`*.json` 索引 + `*.bin` f32 数据），原生与 wasm 共用
//! - [`model`] 前向计算
//!
//! 数值一致性由 `py/export_parity.py` 导出的用例 + 本 crate 的 `--parity` 模式验证。

pub mod model;
pub mod weights;

#[cfg(target_arch = "wasm32")]
pub mod wasm;

pub use model::Model;
pub use weights::Weights;
