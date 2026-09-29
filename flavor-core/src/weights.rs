//! 扁平权重加载：`pharma33_weights.json`（索引）+ `pharma33_weights.bin`（f32 小端）。
//!
//! 之所以不用原生的 `.npz`：npz 是 zip 容器，wasm 里解压不划算；扁平格式一次
//! `fetch` 成 `ArrayBuffer` 就能直接当成 f32 切片用，且省掉 zlib 依赖。

use std::collections::HashMap;

/// 单个张量在二进制里的位置与形状（形状按行主序）。
#[derive(Debug, Clone, serde::Deserialize)]
pub struct TensorInfo {
    pub shape: Vec<usize>,
    pub offset: usize,
}

/// 权重索引文件。
#[derive(Debug, Clone, serde::Deserialize)]
pub struct WeightIndex {
    pub tensors: HashMap<String, TensorInfo>,
}

/// 只读权重表。
pub struct Weights {
    data: Vec<f32>,
    index: WeightIndex,
}

impl Weights {
    /// 由索引 JSON 文本 + f32 小端字节流构造。
    pub fn from_parts(index_json: &str, bin: &[u8]) -> Result<Self, String> {
        let index: WeightIndex =
            serde_json::from_str(index_json).map_err(|e| format!("权重索引解析失败：{e}"))?;
        if bin.len() % 4 != 0 {
            return Err(format!("权重二进制长度 {} 不是 4 的倍数", bin.len()));
        }
        let data: Vec<f32> = bin
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        for (name, t) in &index.tensors {
            let n: usize = t.shape.iter().product();
            if t.offset + n > data.len() {
                return Err(format!(
                    "张量 {name} 越界：offset {} + {n} > {}",
                    t.offset,
                    data.len()
                ));
            }
        }
        Ok(Self { data, index })
    }

    /// 数据元素总数（用于自检）。
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// 张量名列表（用于自检/缺失诊断）。
    pub fn names(&self) -> Vec<String> {
        let mut v: Vec<String> = self.index.tensors.keys().cloned().collect();
        v.sort();
        v
    }

    pub fn has(&self, name: &str) -> bool {
        self.index.tensors.contains_key(name)
    }

    pub fn shape(&self, name: &str) -> &[usize] {
        match self.index.tensors.get(name) {
            Some(t) => &t.shape,
            None => panic!("缺少权重张量 {name}"),
        }
    }

    /// 取张量的连续 f32 切片（行主序）。
    pub fn get(&self, name: &str) -> &[f32] {
        match self.index.tensors.get(name) {
            Some(t) => {
                let n: usize = t.shape.iter().product();
                &self.data[t.offset..t.offset + n]
            }
            None => panic!("缺少权重张量 {name}"),
        }
    }
}
