//! wasm 绑定：插件在 WebView 里直接调用，不再需要外部进程。
//!
//! 用法（前端）：
//! ```js
//! import init, { load_weights, predict } from './flavor_core.js'
//! await init()
//! load_weights(indexJson, new Uint8Array(binBuffer))
//! const probs = predict(x, nodes, edgeIndex, edgeAttr, edges, pharma)  // Float32Array
//! ```

use std::cell::RefCell;

use wasm_bindgen::prelude::*;

use crate::model::Model;

thread_local! {
    static MODEL: RefCell<Option<Model>> = const { RefCell::new(None) };
}

/// 加载权重。`index_json` 是张量索引，`bin` 是 f32 小端的扁平数据。
#[wasm_bindgen]
pub fn load_weights(index_json: &str, bin: &[u8]) -> Result<usize, JsValue> {
    let model = Model::from_parts(index_json, bin).map_err(|e| JsValue::from_str(&e))?;
    let n = model.num_classes;
    MODEL.with(|m| *m.borrow_mut() = Some(model));
    Ok(n)
}

#[wasm_bindgen]
pub fn is_loaded() -> bool {
    MODEL.with(|m| m.borrow().is_some())
}

#[wasm_bindgen]
pub fn num_classes() -> usize {
    MODEL.with(|m| m.borrow().as_ref().map(|x| x.num_classes).unwrap_or(0))
}

fn with_model<T>(f: impl FnOnce(&Model) -> T) -> Result<T, JsValue> {
    MODEL.with(|m| {
        let borrow = m.borrow();
        match borrow.as_ref() {
            Some(model) => Ok(f(model)),
            None => Err(JsValue::from_str("权重尚未加载")),
        }
    })
}

/// 单分子前向，返回 138 个概率。
///
/// - `x`: `nodes × 11`（行主序）
/// - `edge_index`: `2 * edges`（前半 src、后半 dst）
/// - `edge_attr`: `edges × 4`
/// - `pharma`: 33 维
#[wasm_bindgen]
pub fn predict(
    x: &[f32],
    nodes: usize,
    edge_index: &[i32],
    edge_attr: &[f32],
    edges: usize,
    pharma: &[f32],
) -> Result<Vec<f32>, JsValue> {
    let idx: Vec<i64> = edge_index.iter().map(|v| *v as i64).collect();
    with_model(|m| m.predict_proba(x, nodes, &idx, edge_attr, edges, pharma))
}

/// 同上，但返回 sigmoid 之前的 logits（便于与参考实现对着调）。
#[wasm_bindgen]
pub fn logits(
    x: &[f32],
    nodes: usize,
    edge_index: &[i32],
    edge_attr: &[f32],
    edges: usize,
    pharma: &[f32],
) -> Result<Vec<f32>, JsValue> {
    let idx: Vec<i64> = edge_index.iter().map(|v| *v as i64).collect();
    with_model(|m| m.logits(x, nodes, &idx, edge_attr, edges, pharma))
}
