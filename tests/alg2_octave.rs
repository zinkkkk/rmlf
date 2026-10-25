//! alg2 (greedy/adaptive) parity against Octave `mlf.alg2`.
//!
//! Golden values produced by running the reference `+mlf` package under
//! `octave-cli` on the separable order-1 rational `1/((x1+3)(x2+2))`, sampled on
//! the `examples.m` point convention (`p_c = linspace`, `p_r = p_c + dx/2`) with
//! `nip = 6`. Octave converges in 2 greedy iterations to order `[1,1]` and
//! recovers the function to machine precision (`#2 rel max error 3.3e-16`).

use approx::assert_relative_eq;
use rmlf::{Alg2Options, Points, alg2};

/// Build points the way `mlf.examples` does: per axis,
/// `p_c = linspace(lo, hi, nip)` and `p_r = p_c + dx` with
/// `dx = |p_c[end] - p_c[end-1]| / 2`.
fn examples_points(lo: f64, hi: f64, nip: usize, n: usize) -> Points {
    let pc: Vec<f64> = (0..nip)
        .map(|i| lo + (hi - lo) * i as f64 / (nip - 1) as f64)
        .collect();
    let dx = (pc[nip - 1] - pc[nip - 2]).abs() / 2.0;
    let pr: Vec<f64> = pc.iter().map(|&v| v + dx).collect();
    Points::new(vec![pc; n], vec![pr; n]).unwrap()
}

#[test]
fn alg2_separable_rational_matches_octave() {
    let f = |x: &[f64]| 1.0 / ((x[0] + 3.0) * (x[1] + 2.0));
    let pts = examples_points(-1.0, 1.0, 6, 2);

    let (model, info) = alg2(&f, &pts, &Alg2Options::default()).unwrap();

    // Octave: converged at iteration 2, order [1, 1], machine-precision fit.
    assert_eq!(info.order, vec![1, 1], "order = {:?}", info.order);
    assert!(info.best_iter <= 2, "best_iter = {}", info.best_iter);
    assert!(info.best_error < 1e-12, "best_error = {}", info.best_error);

    // Octave r(0.13,-0.4) = 0.199680511182109, r(0.5,0.5) = 0.114285714285714.
    assert_relative_eq!(
        model.eval(&[0.13, -0.4]),
        0.199680511182109,
        epsilon = 1e-10
    );
    assert_relative_eq!(model.eval(&[0.5, 0.5]), 0.114285714285714, epsilon = 1e-10);

    // And it equals the true function at those points.
    assert_relative_eq!(model.eval(&[0.13, -0.4]), f(&[0.13, -0.4]), epsilon = 1e-10);
    assert_relative_eq!(model.eval(&[0.5, 0.5]), f(&[0.5, 0.5]), epsilon = 1e-10);
}
