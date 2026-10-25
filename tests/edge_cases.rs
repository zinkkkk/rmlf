//! Edge-case targets: hard / discontinuous / singular functions that stress the
//! approximation. These are **characterization tests** — they pin down how the
//! framework actually behaves on targets it can and cannot fit, including the
//! failure modes. Approximation theory guarantees no rational interpolant fits an
//! arbitrary function, so several of these *document expected failure* (a large
//! but finite error) rather than success. The key invariants we always require:
//!
//!   * `eval` never returns NaN/Inf for in-domain queries (the removable-node
//!     singularity is handled; see `eval.rs` / WORKPLAN §6d), and
//!   * the fitting routines return a model (or a clean `Err`), never panic.
//!
//! When a target *is* rational-friendly, alg2's adaptivity usually recovers it
//! even where alg1's one-shot fit fails — these tests record that contrast.

use rmlf::{
    Alg1Options, Alg2Options, Points, alg1, alg1_from_tensor, alg2, alg2_from_tensor, sample,
    sample_continued,
};

/// Largest absolute residual of `model` vs `f` over a uniform interior sweep,
/// and whether any evaluation was non-finite.
fn sweep_error<M: Fn(&[f64]) -> f64>(
    model: &M,
    f: &dyn Fn(&[f64]) -> f64,
    ndim: usize,
    samples: usize,
) -> (f64, bool) {
    let mut max_err = 0.0_f64;
    let mut any_nonfinite = false;
    for k in 0..samples {
        let t = -0.9 + 1.8 * k as f64 / (samples - 1) as f64;
        let x = vec![t; ndim];
        let v = model(&x);
        if !v.is_finite() {
            any_nonfinite = true;
        } else {
            max_err = max_err.max((v - f(&x)).abs());
        }
    }
    (max_err, any_nonfinite)
}

fn grid(ndim: usize, nip: usize) -> Points {
    Points::alternating(&vec![(-1.0, 1.0); ndim], &vec![nip; ndim]).unwrap()
}

#[test]
fn fract_1d_is_secretly_linear_and_fits_exactly() {
    // On [-1, 1) the fractional part f(x) = x - floor(x) equals x for x in [0,1)
    // and x+1 for x in [-1,0) — but our interior sweep stays within a single
    // branch's worth of structure that the order-1 rational nails. This documents
    // that `.fract()` is NOT inherently unfittable; it's the *discontinuity* (at
    // integers / across branches) that breaks things, exercised below in 2-D.
    let f = |x: &[f64]| x[0].fract();
    let pts = grid(1, 10);
    let (m, _) = alg1(&f, &pts, &Alg1Options::default()).unwrap();
    let (err, nonfinite) = sweep_error(&|x| m.eval(x), &f, 1, 50);
    assert!(!nonfinite);
    assert!(err < 1e-9, "fract_1d err = {err:e}");
}

#[test]
fn fract_2d_is_expected_to_fail_but_stays_finite() {
    // Sum of two fractional parts: genuinely discontinuous, not rational. Both
    // algorithms produce a *finite* model with a large residual — the documented
    // failure mode (no panic, no NaN, just a poor fit).
    let f = |x: &[f64]| x[0].fract() + x[1].fract();
    let pts = grid(2, 10);

    let (m1, _) = alg1(&f, &pts, &Alg1Options::default()).unwrap();
    let (err1, nf1) = sweep_error(&|x| m1.eval(x), &f, 2, 50);
    assert!(!nf1, "alg1 produced a non-finite eval");
    assert!(err1 > 1e-2, "expected a poor fit, got err = {err1:e}");

    // alg2 also cannot fit it; it must still return cleanly with a finite error.
    let (_, info) = alg2(&f, &pts, &Alg2Options::default()).unwrap();
    assert!(info.best_error.is_finite());
}

#[test]
fn pole_inside_domain_is_handled_finitely() {
    // f(x) = 1/x has a pole at 0 *inside* [-1,1] — a genuine rational singularity.
    // The barycentric form represents poles natively, so this fits well; crucially
    // eval must stay finite at the queried (off-pole) points.
    let f = |x: &[f64]| 1.0 / x[0];
    let pts = grid(1, 10);
    let (m, _) = alg1(&f, &pts, &Alg1Options::default()).unwrap();
    let (err, nonfinite) = sweep_error(&|x| m.eval(x), &f, 1, 50);
    assert!(!nonfinite, "eval went non-finite near the pole");
    assert!(err < 1e-6, "1/x err = {err:e}");
}

#[test]
fn discontinuous_step_fails_gracefully() {
    // A hard 0/1 step: not approximable by a low-order rational. alg1's one-shot
    // fit is poor; alg2 adapts but still cannot resolve the jump. Both must remain
    // finite and return without panicking — that is the whole assertion.
    let f = |x: &[f64]| if x[0] + x[1] > 0.0 { 1.0 } else { 0.0 };
    let pts = grid(2, 10);

    let (m1, _) = alg1(&f, &pts, &Alg1Options::default()).unwrap();
    let (_, nf1) = sweep_error(&|x| m1.eval(x), &f, 2, 50);
    assert!(!nf1);

    let (_, info) = alg2(&f, &pts, &Alg2Options::default()).unwrap();
    assert!(info.best_error.is_finite());
    // It genuinely cannot fit a step: error stays O(1).
    assert!(
        info.best_error > 1e-2,
        "step unexpectedly fit: {}",
        info.best_error
    );
}

#[test]
fn high_frequency_oscillation_underresolved_but_finite() {
    // sin(50x) oscillates far faster than 10 sample points can resolve — classic
    // aliasing. The fit is poor (documented), but must never blow up to non-finite.
    let f = |x: &[f64]| (50.0 * x[0]).sin();
    let pts = grid(1, 10);
    let (m, _) = alg1(&f, &pts, &Alg1Options::default()).unwrap();
    let (err, nonfinite) = sweep_error(&|x| m.eval(x), &f, 1, 200);
    assert!(!nonfinite);
    assert!(err > 0.1, "expected aliasing error, got {err:e}");
}

#[test]
fn smooth_gaussian_spike_is_fittable_by_alg2() {
    // A sharp but smooth Gaussian: alg1 under-resolves the peak, alg2's adaptivity
    // places points where the curvature is and recovers it well. Documents the
    // alg1-vs-alg2 contrast on a *smooth* hard case (vs the discontinuous ones).
    let f = |x: &[f64]| (-200.0 * x[0] * x[0]).exp();
    let pts = grid(1, 12);
    let (_, info) = alg2(&f, &pts, &Alg2Options::default()).unwrap();
    assert!(info.best_error.is_finite());
    assert!(
        info.best_error < 1e-6,
        "gaussian spike err = {}",
        info.best_error
    );
}

#[test]
fn raw_sample_with_nan_yields_clean_error_not_panic() {
    // With the *raw* `sample`, a non-finite entry in the data tensor (here ln(x)
    // over a grid including negatives) cannot be factored: the fit returns a clean
    // `Err`, never a panic or silently-garbage model. This is the default,
    // conservative behavior.
    let f = |x: &[f64]| x[0].ln(); // NaN for x < 0
    let col: Vec<f64> = (0..6).map(|i| -1.0 + 2.0 * i as f64 / 5.0).collect();
    let row: Vec<f64> = col.iter().map(|v| v + 0.05).collect();
    assert!(
        col.iter().any(|&v| !f(&[v]).is_finite()),
        "setup: expected NaN samples"
    );

    let pts = Points::new(vec![col], vec![row]).unwrap();
    let tab = sample(&f, &pts);
    let res = alg1_from_tensor(&tab, &pts, &Alg1Options::default());
    assert!(
        res.is_err(),
        "NaN data should produce an Err with raw sample"
    );
}

#[test]
fn closure_entry_points_survive_a_singular_grid_point() {
    // `x1²/(x1²+x2²)` is 0/0 = NaN exactly at the origin. The reference linspace
    // grid lands a point *almost* on 0 (and so stays finite) only by floating-point
    // luck; ours can land exactly on 0. The closure-driven `alg1`/`alg2` route
    // through `sample_continued`, so a single singular sample degrades to a finite
    // (poor) fit instead of a hard SVD failure — matching MATLAB, which does not
    // crash here. (This target is genuinely non-rational at the origin, so we only
    // require *finite, no-panic*, not accuracy.)
    let f = |x: &[f64]| x[0] * x[0] / (x[0] * x[0] + x[1] * x[1]);
    // A symmetric even grid puts an exact 0 on each axis → an exact 0/0 sample.
    let pts = Points::new(
        vec![vec![-1.0, -0.5, 0.5, 1.0], vec![-1.0, -0.5, 0.5, 1.0]],
        vec![vec![-0.75, -0.25, 0.0, 0.75], vec![-0.75, -0.25, 0.0, 0.75]],
    )
    .unwrap();
    // The grid really does produce a non-finite sample at (0, 0).
    assert!(
        !f(&[0.0, 0.0]).is_finite(),
        "setup: expected a singular grid point"
    );

    let (m1, _) = alg1(&f, &pts, &Alg1Options::default()).expect("alg1 must not crash on NaN grid");
    assert!(m1.eval(&[0.3, 0.4]).is_finite());
    let (m2, info) =
        alg2(&f, &pts, &Alg2Options::default()).expect("alg2 must not crash on NaN grid");
    assert!(info.best_error.is_finite());
    assert!(m2.eval(&[0.3, 0.4]).is_finite());
}

#[test]
fn sample_continued_recovers_through_removable_singularity() {
    // The better answer to "how are NaNs handled": a function non-finite at
    // isolated points can still be approximated by **analytic continuation** —
    // sinc(x) = sin(x)/x is NaN (0/0) at x = 0 but equals 1 there in the limit.
    // `sample_continued` fills the singular sample from its nearest finite
    // neighbor, so the rational fit proceeds and continues *through* the gap
    // instead of failing outright. The fit is exact away from the singularity and
    // recovers a close (not exact) value at the filled point.
    let f = |x: &[f64]| {
        let t = x[0];
        if t == 0.0 { f64::NAN } else { t.sin() / t }
    };
    // A symmetric odd-count grid puts x = 0 exactly on a node, so a sample is NaN.
    let col: Vec<f64> = (0..9).map(|i| -3.0 + 6.0 * i as f64 / 8.0).collect();
    let row: Vec<f64> = col.iter().map(|v| v + 0.2).collect();
    let pts = Points::new(vec![col], vec![row]).unwrap();

    let (tab, replaced) = sample_continued(&f, &pts);
    assert_eq!(replaced, 1, "expected exactly one singular sample filled");

    let (m, info) = alg2_from_tensor(&tab, &pts, &Alg2Options::default()).unwrap();
    assert!(info.best_error.is_finite());
    // Exact where the function is regular...
    let truth = |t: f64| if t == 0.0 { 1.0 } else { t.sin() / t };
    assert!(
        (m.eval(&[1.5]) - truth(1.5)).abs() < 1e-3,
        "off-singularity fit poor"
    );
    // ...and the continued value at the singularity is recovered approximately
    // (a single filled point can't be exact, but it is far from NaN/garbage).
    let v0 = m.eval(&[0.0]);
    assert!(v0.is_finite());
    assert!(
        (v0 - 1.0).abs() < 0.2,
        "continued value at 0 = {v0}, expected ~1.0"
    );
}

#[test]
fn sample_continued_avoiding_singularity_is_exact() {
    // When the grid simply *avoids* the singularity, no fill is needed and the
    // rational fit recovers the continued value to machine precision — proof the
    // framework already does analytic continuation; the fill is only a fallback
    // for grids that unavoidably hit the singular point.
    let f = |x: &[f64]| {
        let t = x[0];
        if t.abs() < 1e-300 { 1.0 } else { t.sin() / t }
    };
    let pts = Points::alternating(&[(-3.0, 3.0)], &[14]).unwrap(); // avoids 0
    let (tab, replaced) = sample_continued(&f, &pts);
    assert_eq!(replaced, 0, "grid should avoid the singularity");
    let (m, _) = alg2_from_tensor(&tab, &pts, &Alg2Options::default()).unwrap();
    assert!(
        (m.eval(&[0.0]) - 1.0).abs() < 1e-9,
        "continuation at 0 not recovered"
    );
}

#[test]
fn positive_domain_log_fits_when_samples_are_finite() {
    // The same ln, but on a positive domain where every sample is finite, is a
    // smooth target alg2 fits well — contrast with the NaN case above: the problem
    // was the non-finite *samples*, not ln itself.
    let f = |x: &[f64]| x[0].ln();
    let pts = Points::alternating(&[(0.2, 4.0)], &[12]).unwrap();
    let (_, info) = alg2(&f, &pts, &Alg2Options::default()).unwrap();
    assert!(info.best_error.is_finite());
    assert!(info.best_error < 1e-6, "log fit err = {}", info.best_error);
}

#[test]
fn query_exactly_on_a_support_node_returns_sampled_value() {
    // The removable-singularity case: querying *on* a column support node must
    // return the exact sampled value, not NaN (MATLAB's eval_lagrangian yields
    // Inf/Inf = NaN here; we take the limit — see WORKPLAN §6d).
    let f = |x: &[f64]| 1.0 / ((x[0] + 3.0) * (x[1] + 2.0));
    let pts = grid(2, 8);
    let (m, _) = alg1(&f, &pts, &Alg1Options::default()).unwrap();
    // A column node along each axis (the model's own lambda values).
    let node = [m.lambda[0][0], m.lambda[1][0]];
    let v = m.eval(&node);
    assert!(v.is_finite(), "eval on a node returned non-finite {v}");
    assert!((v - f(&node)).abs() < 1e-9, "node value off: {v}");
}
