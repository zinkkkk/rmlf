//! N-D recursion parity against Octave golden values.
//!
//! These use exactly `order + 1` points per axis so each 1-D Loewner is
//! rank-deficient by exactly one and the null vector is unique (no ambiguity
//! from an over-sampled, multi-dimensional null space). Order reduction for
//! over-sampled data is the job of `compute_order` / `points_selection` (M3).

use approx::assert_relative_eq;
use rmlf::internals::loewner_null_rec_svd0;
use rmlf::{Barycentric, Points, column_values, sample};

#[test]
fn rec3_matches_octave() {
    // f(s1,s2,s3) = 1/((s1+3)(s2+2)(s3+4)): order 1 per axis, 2 points per axis.
    let f = |x: &[f64]| 1.0 / ((x[0] + 3.0) * (x[1] + 2.0) * (x[2] + 4.0));
    let pts = Points::new(
        vec![vec![0.0, 1.0], vec![0.0, 1.0], vec![0.0, 1.0]],
        vec![vec![0.5, 1.5], vec![0.5, 1.5], vec![0.5, 1.5]],
    )
    .unwrap();
    let tab = sample(&f, &pts);
    let res = loewner_null_rec_svd0(&pts, tab.view()).unwrap();

    let expect: [f64; 8] = [
        -0.4,
        0.5,
        0.6,
        -0.75,
        0.533333333333333,
        -0.666666666666667,
        -0.8,
        1.0,
    ];
    assert_eq!(res.c.len(), 8);
    for (got, want) in res.c.iter().zip(expect) {
        assert_relative_eq!(got, &want, max_relative = 1e-12);
    }

    // Full chain: eval recovers the true value (Octave: 0.0267222489444712).
    let w = column_values(&pts, tab.view());
    let model = Barycentric {
        lambda: pts.column.clone(),
        w,
        c: res.c,
    };
    let got = model.eval(&[0.3, 0.7, 0.2]);
    assert_relative_eq!(got, f(&[0.3, 0.7, 0.2]), epsilon = 1e-12);
}
