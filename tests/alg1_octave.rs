//! alg1 parity against Octave on the getting-started Polya-Szego example.

use approx::assert_relative_eq;
use rmlf::Points;
use rmlf::{Alg1Options, alg1};

/// Reproduce the exact getting-started setup: ip = linspace(-1,1,10) per axis,
/// p_c = ip(2:2:end), p_r = ip(1:2:end).
fn getting_started_points() -> Points {
    let ip: Vec<f64> = (0..10).map(|i| -1.0 + 2.0 * i as f64 / 9.0).collect();
    let col: Vec<f64> = (1..10).step_by(2).map(|i| ip[i]).collect(); // 2:2:end (1-based)
    let row: Vec<f64> = (0..10).step_by(2).map(|i| ip[i]).collect(); // 1:2:end
    Points::new(
        vec![col.clone(), col.clone(), col.clone()],
        vec![row.clone(), row.clone(), row.clone()],
    )
    .unwrap()
}

#[test]
fn polya_szego_recovers_and_matches_octave() {
    let f = |x: &[f64]| x[0] * x[1] + x[0] * x[2] + x[1] * x[2];
    let pts = getting_started_points();
    let mut opt = Alg1Options::default();
    opt.order.tol = 1e-8;
    // Use the default context subsampling (`Some(5)`) — the same heuristic MATLAB's
    // `compute_order` uses (`ord_N = 5`) and what both codes agree on: Octave also
    // returns [1,1,1] here with N=5. (With *all* contexts the per-context rank
    // genuinely varies on this coupled polynomial and both codes detect higher,
    // ambiguous order — Octave [1,3,3], us [4,4,4]; subsampling is the documented
    // way around it. See WORKPLAN §12.)

    let (model, info) = alg1(&f, &pts, &opt).unwrap();
    assert_eq!(info.order, vec![1, 1, 1]);

    // Octave g(0.1,-0.3,0.5) = -0.13 exactly.
    let x = [0.1, -0.3, 0.5];
    assert_relative_eq!(model.eval(&x), f(&x), epsilon = 1e-9);

    // Octave kept pc = [-0.777778, 1] along each axis.
    for lambda in &model.lambda {
        assert_relative_eq!(lambda[0], -0.7777777777777778, epsilon = 1e-9);
        assert_relative_eq!(lambda[1], 1.0, epsilon = 1e-9);
    }
}

/// Catalog cases where Octave's `alg1` reaches ~machine precision and our earlier
/// (stacked-unfolding) order detection over-detected order and *failed*. Octave
/// ground truth (validated directly via `octave-cli` on `+mlf`): CAS 11 order
/// [1,3], 36 [1,1,2], 56 [2,2], all off-grid error ~1e-13. After reverting to
/// MATLAB's max-per-context order detection (WORKPLAN §10 #3 / §12) our `alg1`
/// matches. This guards against re-introducing the over-detection regression.
#[test]
fn alg1_recovers_octave_passing_cases() {
    use rmlf::examples;
    let mut rng: u64 = 0x1234_5678;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng >> 11) as f64 / (1u64 << 53) as f64
    };
    for cas in [11u32, 36, 56] {
        let ex = examples::get(cas).unwrap();
        let f = ex.f;
        let pts = ex.points(None);
        let bnd = ex.axis_bounds();
        let (m, _) = alg1(&f, &pts, &Alg1Options::default()).unwrap();
        // Off-grid max error over random interior points (matches the Octave check).
        let mut maxe = 0.0_f64;
        for _ in 0..3000 {
            let x: Vec<f64> = bnd
                .iter()
                .map(|&(lo, hi)| lo + (hi - lo) * (0.05 + 0.9 * next()))
                .collect();
            let t = f(&x);
            if t.is_finite() {
                let e = (m.eval(&x) - t).abs();
                if e.is_finite() {
                    maxe = maxe.max(e);
                }
            }
        }
        assert!(
            maxe < 1e-6,
            "CAS {cas} alg1 off-grid max error {maxe:e} (Octave ~1e-13)"
        );
    }
}
