//! Timing the alg2 greedy grid scan. Two levers:
//!   1. **per-point eval** — dense vs TT-accelerated (`compress`), on a separable
//!      rational where the model compresses to low TT rank;
//!   2. **candidate count** (#10) — the `Full` scan visits all `nip^n` points
//!      (exponential in `n`); `Coordinate` scans `O(Σ dims)` per iteration. The
//!      second example pushes `n` to where the full point count alone is the wall.

use rmlf::{Alg2Options, Candidates, Points, alg2};
use std::time::Instant;

fn main() {
    // Separable order-2 rational (k = 3 per axis): the model is TT-rank 1, so the
    // grid error scan can use the factorized batched eval.
    let f = |x: &[f64]| -> f64 { x.iter().map(|&v| 1.0 / ((v + 2.0) * (v + 3.0))).product() };
    let n = 6;
    let nip = 7;
    let pts = Points::alternating(&vec![(-1.0, 1.0); n], &vec![nip; n]).unwrap();
    println!("=== per-point eval lever (TT) ===");
    println!(
        "n = {n}, nip = {nip}  (full grid = {} points)",
        (nip).pow(n as u32)
    );

    let dense_opt = Alg2Options {
        compress: None,
        ..Alg2Options::default()
    };
    let tt_opt = Alg2Options {
        compress: Some(1e-12),
        ..Alg2Options::default()
    };

    let t = Instant::now();
    let (_m, i_dense) = alg2(&f, &pts, &dense_opt).unwrap();
    let dt_dense = t.elapsed();

    let t = Instant::now();
    let (_m, i_tt) = alg2(&f, &pts, &tt_opt).unwrap();
    let dt_tt = t.elapsed();

    println!(
        "dense scan: {dt_dense:?}  (order {:?}, best_err {:.2e})",
        i_dense.order, i_dense.best_error
    );
    println!(
        "TT scan:    {dt_tt:?}  (order {:?}, best_err {:.2e})",
        i_tt.order, i_tt.best_error
    );
    println!(
        "speedup: {:.1}x",
        dt_dense.as_secs_f64() / dt_tt.as_secs_f64()
    );

    // #10: candidate-count lever. At higher n the *point count* is the wall — the
    // Full scan visits nip^n points per iteration regardless of per-point cost.
    // Coordinate ascent scans O(Σ dims) per iteration instead.
    println!("\n=== candidate-count lever (#10: Full vs Coordinate) ===");
    let g = |x: &[f64]| -> f64 { x.iter().map(|&v| 1.0 / (v + 2.5)).product() };
    let n = 8;
    let nip = 5;
    let pts = Points::alternating(&vec![(-1.0, 1.0); n], &vec![nip; n]).unwrap();
    println!(
        "n = {n}, nip = {nip}  (full grid = {} points)",
        (nip as u64).pow(n as u32)
    );

    let full_opt = Alg2Options {
        candidates: Candidates::Full,
        ..Alg2Options::default()
    };
    let coord_opt = Alg2Options {
        candidates: Candidates::Coordinate,
        ..Alg2Options::default()
    };

    let t = Instant::now();
    let (_m, i_full) = alg2(&g, &pts, &full_opt).unwrap();
    let dt_full = t.elapsed();

    let t = Instant::now();
    let (_m, i_coord) = alg2(&g, &pts, &coord_opt).unwrap();
    let dt_coord = t.elapsed();

    println!(
        "Full scan:       {dt_full:?}  (order {:?}, best_err {:.2e})",
        i_full.order, i_full.best_error
    );
    println!(
        "Coordinate scan: {dt_coord:?}  (order {:?}, best_err {:.2e})",
        i_coord.order, i_coord.best_error
    );
    println!(
        "speedup: {:.1}x",
        dt_full.as_secs_f64() / dt_coord.as_secs_f64()
    );
}
