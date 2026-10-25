//! Show the low-rank (tensor-train) compression of a structured high-dimensional
//! model: build it dense via alg1, compress, and report storage + rank + the
//! eval speedup, confirming the TT eval matches the dense one.

use rmlf::{Alg1Options, CompressedBarycentric, Points, alg1};
use std::time::Instant;

fn main() {
    // Separable n-D rational, order 2 per axis (k = 3): the coefficient tensor is
    // TT-rank 1, so dense storage 3^n collapses to ~3n in TT.
    let f = |x: &[f64]| x.iter().map(|&v| 1.0 / ((v + 2.0) * (v + 3.0))).product();
    let n = 7;
    let nip = 8;
    let p = Points::alternating(&vec![(-1.0, 1.0); n], &vec![nip; n]).unwrap();

    let (model, info) = alg1(&f, &p, &Alg1Options::default()).unwrap();
    println!("n = {n}, detected order = {:?}", info.order);

    let tt = CompressedBarycentric::from_model(&model, 1e-10);
    let dense = model.c.len();
    let tt_store = tt.denom.storage() + tt.numer.storage();
    println!(
        "coefficient storage: dense = {dense}, TT = {tt_store}  ({:.1}x smaller), max TT rank = {}",
        dense as f64 / tt_store as f64,
        tt.max_rank()
    );

    // Query grid.
    let m = 50_000;
    let mut rng = 0x1234_5678u64;
    let mut next = || {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        (rng >> 11) as f64 / (1u64 << 53) as f64
    };
    let pts_x: Vec<Vec<f64>> = (0..m)
        .map(|_| (0..n).map(|_| -0.9 + 1.8 * next()).collect())
        .collect();
    let xs: Vec<&[f64]> = pts_x.iter().map(|v| v.as_slice()).collect();

    let t = Instant::now();
    let dense_vals = model.eval_grid(&xs);
    let dt_dense = t.elapsed();

    let t = Instant::now();
    let tt_vals = tt.eval_grid(&xs);
    let dt_tt = t.elapsed();

    let max_diff = dense_vals
        .iter()
        .zip(&tt_vals)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    println!("eval {m} points: dense {dt_dense:?}, TT {dt_tt:?}");
    println!("max |dense - TT| = {max_diff:.2e}");
}
