//! 原生 CLI：既能按 stdin JSON 做单次预测，也能跑与 numpy 的对拍（`--parity`）。
//!
//! ```text
//! flavor-core --weights flavor --parity build/parity/cases.json
//! flavor-core --weights flavor < payload.json
//! ```
//!
//! `payload.json` 字段：`nodes` / `x` / `edges` / `edge_index` / `edge_attr` / `pharma`。

use std::io::Read;
use std::path::{Path, PathBuf};

use flavor_core::{Model, Weights};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Payload {
    nodes: usize,
    edges: usize,
    x: Vec<f32>,
    edge_index: Vec<i64>,
    edge_attr: Vec<f32>,
    pharma: Vec<f32>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    nodes: usize,
    edges: usize,
    x: Vec<f32>,
    edge_index: Vec<i64>,
    edge_attr: Vec<f32>,
    pharma: Vec<f32>,
    logits: Vec<f32>,
    probs: Vec<f32>,
}

#[derive(Deserialize)]
struct Cases {
    cases: Vec<Case>,
}

#[derive(Serialize)]
struct CaseReport {
    name: String,
    max_logit_diff: f64,
    max_prob_diff: f64,
    argmax_match: bool,
}

#[derive(Serialize)]
struct Report {
    cases: usize,
    max_logit_diff: f64,
    max_logit_diff_case: String,
    max_prob_diff: f64,
    all_argmax_match: bool,
    per_case: Vec<CaseReport>,
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0usize;
    for (i, x) in v.iter().enumerate() {
        if *x > v[best] {
            best = i;
        }
    }
    best
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (*x as f64 - *y as f64).abs())
        .fold(0.0f64, f64::max)
}

fn load_model(dir: &Path) -> Result<Model, String> {
    let index = dir.join("pharma33_weights.json");
    let bin = dir.join("pharma33_weights.bin");
    let index_json = std::fs::read_to_string(&index)
        .map_err(|e| format!("读不到 {}：{e}", index.display()))?;
    let bytes = std::fs::read(&bin).map_err(|e| format!("读不到 {}：{e}", bin.display()))?;
    let w = Weights::from_parts(&index_json, &bytes)?;
    println!(
        "权重已加载：{} 个张量 / {} 个 f32（{:.2} MB）",
        w.names().len(),
        w.numel(),
        w.numel() as f64 * 4.0 / 1024.0 / 1024.0
    );
    Ok(Model::new(w))
}

fn run_parity(model: &Model, path: &Path) -> Result<(), String> {
    let text =
        std::fs::read_to_string(path).map_err(|e| format!("读不到 {}：{e}", path.display()))?;
    let cases: Cases = serde_json::from_str(&text).map_err(|e| format!("对拍用例解析失败：{e}"))?;

    let mut report = Report {
        cases: cases.cases.len(),
        max_logit_diff: 0.0,
        max_logit_diff_case: String::new(),
        max_prob_diff: 0.0,
        all_argmax_match: true,
        per_case: Vec::new(),
    };

    for c in &cases.cases {
        let logits = model.logits(&c.x, c.nodes, &c.edge_index, &c.edge_attr, c.edges, &c.pharma);
        let probs: Vec<f32> = logits.iter().copied().map(flavor_core::model::sigmoid).collect();
        let ld = max_abs_diff(&logits, &c.logits);
        let pd = max_abs_diff(&probs, &c.probs);
        let matched = argmax(&logits) == argmax(&c.probs);
        if ld > report.max_logit_diff {
            report.max_logit_diff = ld;
            report.max_logit_diff_case = c.name.clone();
        }
        report.max_prob_diff = report.max_prob_diff.max(pd);
        report.all_argmax_match &= matched;
        report.per_case.push(CaseReport {
            name: c.name.clone(),
            max_logit_diff: ld,
            max_prob_diff: pd,
            argmax_match: matched,
        });
    }

    println!("\n=== 与 numpy 实现的对拍 ===");
    println!("用例数：{}", report.cases);
    println!(
        "最大 logit 偏差：{:.3e}（{}）",
        report.max_logit_diff, report.max_logit_diff_case
    );
    println!("最大概率偏差：{:.3e}", report.max_prob_diff);
    println!("top-1 标签全部一致：{}", report.all_argmax_match);
    println!(
        "\n{}",
        serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?
    );

    // 仓库既有口径：与 torch 参考实现的对拍阈值是 1e-4
    if report.max_logit_diff < 1e-4 {
        println!("\n判定：通过（最大 logit 偏差 < 1e-4）");
        Ok(())
    } else {
        Err(format!(
            "判定：不通过，最大 logit 偏差 {:.3e} ≥ 1e-4",
            report.max_logit_diff
        ))
    }
}

fn run_single(model: &Model) -> Result<(), String> {
    let mut raw = String::new();
    std::io::stdin()
        .read_to_string(&mut raw)
        .map_err(|e| e.to_string())?;
    let p: Payload = serde_json::from_str(&raw).map_err(|e| format!("输入 JSON 解析失败：{e}"))?;
    let probs = model.predict_proba(
        &p.x,
        p.nodes,
        &p.edge_index,
        &p.edge_attr,
        p.edges,
        &p.pharma,
    );
    let out = serde_json::json!({ "num_classes": probs.len(), "probs": probs });
    println!("{}", serde_json::to_string(&out).map_err(|e| e.to_string())?);
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut weights_dir: PathBuf = PathBuf::from("flavor");
    let mut parity: Option<PathBuf> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--weights" => {
                i += 1;
                weights_dir = PathBuf::from(args.get(i).expect("--weights 需要目录参数"));
            }
            "--parity" => {
                i += 1;
                parity = Some(PathBuf::from(args.get(i).expect("--parity 需要文件参数")));
            }
            other => {
                eprintln!("未知参数：{other}");
                eprintln!("用法：flavor-core --weights <目录> [--parity <cases.json>]");
                std::process::exit(2);
            }
        }
        i += 1;
    }

    let model = match load_model(&weights_dir) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };
    println!("输出类别数：{}", model.num_classes);

    let result = match &parity {
        Some(p) => run_parity(&model, p),
        None => run_single(&model),
    };
    if let Err(e) = result {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
