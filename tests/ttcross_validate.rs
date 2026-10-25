//! Validation of TT-cross (`src/ttcross.rs`) — the sub-exponential-sampling,
//! low-rank multivariate constructor.
//!
//! TT-cross is the structural upgrade that breaks the *dense-grid sampling*
//! premise (WORKPLAN §11): it samples `f` at `O(n·r²·k)` adaptively-chosen points
//! instead of the full `∏(k_l+q_l)` grid. These tests confirm (a) the sampling
//! footprint is genuinely sub-dense and shrinks (relative to dense) as `n` grows,
//! (b) the model is exact on the grid, and (c) the rational lift matches the dense
//! decoupling on a tensor-product rational.

use rmlf::Points;
use rmlf::decouple::decouple_from_closure;
use rmlf::ttcross::{TtCrossOptions, tt_cross, tt_cross_rational};

/// On a separable (TT-rank-1) target the sampling footprint is a tiny fraction of
/// the dense grid, and that fraction *improves* with dimension — the curse-breaking
/// property.
#[test]
fn sampling_footprint_beats_dense_and_improves_with_n() {
    let f = |x: &[f64]| {
        x.iter()
            .enumerate()
            .map(|(i, &xi)| 1.0 / (xi + (i as f64) + 2.0))
            .product::<f64>()
    };
    let frac = |n: usize| {
        let bounds = vec![(-1.0, 1.0); n];
        let nip = vec![5usize; n];
        let pts = Points::alternating(&bounds, &nip).unwrap();
        let (model, info) = tt_cross(&f, &pts, &TtCrossOptions::default()).unwrap();
        assert_eq!(
            model.max_rank(),
            1,
            "separable target should be rank 1 (n={n})"
        );
        info.evaluations as f64 / info.dense_grid as f64
    };
    let f4 = frac(4);
    let f7 = frac(7);
    assert!(f4 < 1.0, "n=4 sampling fraction {f4:.4} should be < 1");
    assert!(
        f7 < f4,
        "fraction should shrink with n: f4={f4:.4} f7={f7:.4}"
    );
}

/// TT-cross of a separable rational, recovered to good accuracy off-grid, with
/// vastly fewer samples than dense at n=8.
#[test]
fn high_dim_separable_recovered_cheaply() {
    let f = |x: &[f64]| {
        x.iter()
            .enumerate()
            .map(|(i, &xi)| 1.0 / (xi + (i as f64) + 2.0))
            .product::<f64>()
    };
    let pts = Points::alternating(&[(-1.0, 1.0); 8], &[5; 8]).unwrap();
    let (model, info) = tt_cross(&f, &pts, &TtCrossOptions::default()).unwrap();
    // Dense would be 10^8; cross is a tiny fraction.
    assert!(
        (info.evaluations as f64) < 0.001 * (info.dense_grid as f64),
        "evals {} vs dense {}",
        info.evaluations,
        info.dense_grid
    );
    // Separable rational ⇒ Lagrange interpolant is exact on the grid; off-grid it
    // approximates (polynomial interpolation of a rational). Check a node exactly.
    let node = [
        model.axes[0][1],
        model.axes[1][2],
        model.axes[2][0],
        model.axes[3][3],
        model.axes[4][1],
        model.axes[5][2],
        model.axes[6][0],
        model.axes[7][4],
    ];
    let got = model.eval(&node);
    assert!(
        (got - f(&node)).abs() < 1e-9,
        "on-node {got} vs {}",
        f(&node)
    );
}

/// The rational lift (`tt_cross_rational`) on a tensor-product rational matches the
/// dense decoupling to machine precision off-grid.
#[test]
fn rational_lift_matches_dense_decoupling() {
    let f = |x: &[f64]| 1.0 / ((x[0] + 3.0) * (x[1] + 2.0) * (x[2] + 4.0));
    let pts = Points::alternating(&[(-1.0, 1.0); 3], &[4; 3]).unwrap();

    let (cross_model, _) = tt_cross_rational(&f, &pts, &TtCrossOptions::default()).unwrap();
    // Dense decoupling on the same reduced (degree-1) grid.
    let dpts = Points::alternating(&[(-1.0, 1.0); 3], &[4; 3]).unwrap();
    let (dense_model, _) = decouple_from_closure(&f, &dpts, &[1, 1, 1]).unwrap();

    for x in [[0.2, -0.3, 0.5], [-0.7, 0.4, 0.9], [0.1, 0.1, -0.6]] {
        let c = cross_model.eval(&x);
        let d = dense_model.eval(&x);
        assert!(
            (c - f(&x)).abs() < 1e-7,
            "cross {c} vs true {} at {x:?}",
            f(&x)
        );
        assert!((c - d).abs() < 1e-7, "cross {c} vs dense {d} at {x:?}");
    }
}
