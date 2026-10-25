//! Where is the time actually going? Breaks a `decouple` / `alg1` build into its
//! phases — **sampling** (`sample_continued`), order detection, construction, eval
//! — on a large grid, to confirm whether sampling was the bottleneck the
//! parallelization targets. Also times sampling alone serial-vs-parallel.
//!
//! Run: `cargo run --release --example sampling_bench [nip] [n]`.

use rmlf::tensor::sample_continued;
use rmlf::{Points, decouple::decouple};
use std::hint::black_box;
use std::time::Instant;

/// A deliberately *non-trivial* target so the per-eval cost is realistic (a cheap
/// closure would make sampling look artificially negligible). Mix of transcendental
/// ops, ~what a real model evaluation costs.
fn target(x: &[f64]) -> f64 {
    let mut acc = 0.0;
    for (i, &xi) in x.iter().enumerate() {
        acc += (xi * (i as f64 + 1.3)).sin() * (xi * 0.7).cos() + (1.0 + xi * xi).ln();
    }
    acc / (1.0 + x.iter().map(|v| v * v).sum::<f64>())
}

fn main() {
    let mut args = std::env::args().skip(1);
    let nip: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(40);
    let n: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(3);

    let bounds = vec![(-1.0, 1.0); n];
    let nips = vec![nip; n];
    let pts = Points::alternating(&bounds, &nips).unwrap();
    let grid: usize = (0..n)
        .map(|l| pts.column[l].len() + pts.row[l].len())
        .product();
    println!("n={n} nip={nip}  full grid = {grid} samples\n");

    // --- Phase: sampling alone (this is what we parallelized) ----------------
    let t = Instant::now();
    let (tab, _replaced) = sample_continued(&target, &pts);
    let t_sample = t.elapsed();
    black_box(&tab);

    // --- Whole decouple build (sampling + order detect + construct) ----------
    let t = Instant::now();
    let (model, info) = decouple(&target, &bounds, &nips).unwrap();
    let t_build = t.elapsed();

    // --- Eval cost (off the hot path, for scale) -----------------------------
    let q: Vec<f64> = (0..n).map(|l| 0.13 + 0.01 * l as f64).collect();
    let t = Instant::now();
    for _ in 0..10_000 {
        black_box(model.eval(black_box(&q)));
    }
    let t_eval = t.elapsed();

    let ms = |d: std::time::Duration| d.as_secs_f64() * 1e3;
    println!(
        "sample_continued (parallel) : {:>8.1} ms   <- the sampling pass",
        ms(t_sample)
    );
    println!(
        "full decouple build         : {:>8.1} ms   (sampling + detect + construct)",
        ms(t_build)
    );
    println!(
        "  -> non-sampling phases      : {:>8.1} ms   ({:.0}% of build)",
        ms(t_build) - ms(t_sample),
        100.0 * (ms(t_build) - ms(t_sample)) / ms(t_build)
    );
    println!(
        "  -> sampling share of build  : {:>8.0}%",
        100.0 * ms(t_sample) / ms(t_build)
    );
    println!(
        "10k evals                   : {:>8.1} ms   ({:.1} us/eval)",
        ms(t_eval),
        ms(t_eval) / 10.0
    );
    println!("detected degree             : {:?}", info.degree);
}
