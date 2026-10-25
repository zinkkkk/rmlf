//! Low-rank (tensor-train) model parity: a `CompressedBarycentric` must agree
//! with the dense `Barycentric` it was built from, and structured high-`n`
//! targets must actually compress (small TT rank).

use approx::assert_relative_eq;
use rmlf::{Alg1Options, CompressedBarycentric, Points, alg1};

/// alternating-split points, the well-conditioned convention.
fn pts(n: usize, nip: usize) -> Points {
    Points::alternating(&vec![(-1.0, 1.0); n], &vec![nip; n]).unwrap()
}

#[test]
fn tt_eval_matches_dense_3d_rational() {
    let f = |x: &[f64]| 1.0 / ((x[0] + 3.0) * (x[1] + 2.0) * (x[2] + 4.0));
    let p = pts(3, 8);
    let (model, _) = alg1(&f, &p, &Alg1Options::default()).unwrap();

    let tt = CompressedBarycentric::from_model(&model, 1e-12);

    // A separable rational of order 1 has a rank-1 coefficient tensor.
    assert_eq!(tt.max_rank(), 1, "separable model should be TT-rank 1");

    for x in [[0.1, -0.3, 0.5], [-0.7, 0.2, 0.9], [0.33, 0.0, -0.21]] {
        assert_relative_eq!(tt.eval(&x), model.eval(&x), epsilon = 1e-9);
        assert_relative_eq!(tt.eval(&x), f(&x), epsilon = 1e-7);
    }
}

#[test]
fn tt_eval_matches_dense_on_node() {
    // Evaluate exactly on a support node: the removable-singularity limit must
    // agree between the dense and TT paths.
    let f = |x: &[f64]| 1.0 / ((x[0] + 3.0) * (x[1] + 2.0));
    let p = pts(2, 8);
    let (model, _) = alg1(&f, &p, &Alg1Options::default()).unwrap();
    let tt = CompressedBarycentric::from_model(&model, 1e-12);

    let node = [model.lambda[0][0], model.lambda[1][0]];
    assert_relative_eq!(tt.eval(&node), model.eval(&node), epsilon = 1e-9);
    // Pinned on one axis only.
    let half = [model.lambda[0][0], 0.137];
    assert_relative_eq!(tt.eval(&half), model.eval(&half), epsilon = 1e-9);
}

#[test]
fn tt_compresses_separable_high_dim() {
    // A separable 6-D order-2 rational: the coefficient tensor is TT-rank 1, so
    // the dense 3^6 storage collapses to ~O(n) in TT, and TT eval still matches.
    // (A *coupled* function like 1/(sum) is genuinely not low-rank and would not
    // compress — only separable / sum-of-few-terms structure does.)
    let f = |x: &[f64]| x.iter().map(|&v| 1.0 / ((v + 2.0) * (v + 3.0))).product();
    let p = pts(6, 8);
    let (model, _) = alg1(&f, &p, &Alg1Options::default()).unwrap();
    let tt = CompressedBarycentric::from_model(&model, 1e-10);

    assert_eq!(tt.max_rank(), 1, "separable model should be TT-rank 1");
    let dense_storage = model.c.len();
    let tt_storage = tt.denom.storage() + tt.numer.storage();
    assert!(
        tt_storage < dense_storage,
        "TT storage {tt_storage} should beat dense {dense_storage}"
    );

    for x in [
        [0.1, -0.3, 0.5, 0.2, -0.1, 0.4],
        [-0.5, 0.5, -0.5, 0.5, -0.5, 0.5],
    ] {
        assert_relative_eq!(tt.eval(&x), model.eval(&x), epsilon = 1e-7);
    }
}
