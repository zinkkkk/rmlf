use rmlf::internals::{Normalize, NullSpace, SvdNull, loewner_1d, loewner_null_rec_svd0};
use rmlf::prelude::*;
use rmlf::{Candidates, OrderOptions, Tensor, compute_order, sample};
use rstsr::prelude::rt::{DeviceFaer, asarray};
use rstsr::prelude::*;
type C = Complex64;

fn close(got: C, expected: C, tol: f64) {
    assert!(
        (got - expected).norm() <= tol * (1.0 + expected.norm()),
        "got {got}, expected {expected}"
    );
}

fn rational(x: &[C]) -> C {
    (C::new(1.0, 2.0) + C::new(0.5, -0.3) * x[0] + C::new(-0.2, 0.7) * x[1])
        / (C::new(4.0, 0.4)
            + C::new(0.8, 0.2) * x[0]
            + C::new(0.3, -0.5) * x[1]
            + C::new(0.2, 0.1) * x[0] * x[1])
}

fn complex_grid() -> Points<C> {
    Points::chebyshev_segments(
        &[
            (C::new(-0.7, 0.1), C::new(1.0, 0.7)),
            (C::new(-0.2, -0.4), C::new(0.9, 0.2)),
        ],
        &[10, 10],
    )
    .unwrap()
}

#[test]
fn alg1_complex_closure_and_tensor_all_coefficient_methods() {
    let points = complex_grid();
    let tab = sample(&rational, &points);
    let orders = compute_order(&points, tab.view(), &OrderOptions::default());
    assert_eq!(orders, vec![1, 1]);
    for method in [
        CoefficientMethod::Recursive,
        CoefficientMethod::Full,
        CoefficientMethod::Auto,
    ] {
        let opts = Alg1Options {
            method,
            ..Default::default()
        };
        let (m, info) = alg1(&rational, &points, &opts).unwrap();
        let (data, _) = rmlf::alg1_from_tensor(&tab, &points, &opts).unwrap();
        assert_eq!(info.order, vec![1, 1]);
        let queries: Vec<Vec<C>> = (0..31)
            .map(|i| {
                vec![
                    C::new(0.03 * i as f64, 0.17),
                    C::new(-0.23 + 0.02 * i as f64, -0.19),
                ]
            })
            .collect();
        let xs: Vec<&[C]> = queries.iter().map(Vec::as_slice).collect();
        for (i, y) in m.eval_grid(&xs).iter().enumerate() {
            close(*y, rational(&queries[i]), 1e-11);
            close(*y, data.eval(&queries[i]), 1e-11);
        }
    }
}

#[test]
fn complex_recursion_matches_octave_phase_and_ordering() {
    let points = Points::new(
        vec![
            vec![C::new(0.0, 0.3), C::new(1.0, 0.7)],
            vec![C::new(-0.2, -0.4), C::new(0.9, 0.2)],
        ],
        vec![
            vec![C::new(0.5, -0.1), C::new(1.5, 0.4)],
            vec![C::new(0.3, 0.5), C::new(1.3, -0.6)],
        ],
    )
    .unwrap();
    let tab = sample(&rational, &points);
    let rec = rmlf::internals::loewner_null_rec(
        &points,
        tab.view(),
        &SvdNull {
            normalize: Normalize::Last,
        },
    )
    .unwrap();
    // Octave's mlf.loewner_null_rec(...,'svd'), generated with the same nodes.
    let golden = [
        C::new(0.7232374843865319, -0.02194868291805717),
        C::new(-0.8170311551477767, 0.1037135769777506),
        C::new(-0.8823659906057393, -0.02883812855599041),
        C::new(1.0, 0.0),
    ];
    for (&got, want) in rec.c.iter().zip(golden) {
        close(got, want, 2e-14);
    }
    let l = rmlf::internals::loewner_full(&points, tab.view());
    for i in 0..l.nrows() {
        let residual: C = (0..l.ncols()).map(|j| l[(i, j)] * rec.c[j]).sum();
        assert!(residual.norm() < 1e-13);
    }
}

#[test]
fn alg2_complex_dense_scan_modes_and_tensor_entry_point() {
    let points = complex_grid();
    let tab = sample(&rational, &points);
    for candidates in [
        Candidates::Full,
        Candidates::Coordinate,
        Candidates::Subsample(45),
    ] {
        let opt = Alg2Options {
            candidates,
            tol: 1e-11,
            max_iter: Some(15),
            patience: None,
            ..Default::default()
        };
        let (m, info) = alg2(&rational, &points, &opt).unwrap();
        assert!(info.error_history.iter().all(|v| v.is_finite()));
        for i in 0..17 {
            let x = [C::new(-0.13 + 0.04 * i as f64, 0.23), C::new(0.37, -0.12)];
            close(m.eval(&x), rational(&x), 1e-9);
        }
        let (data, _) = rmlf::alg2_from_tensor(&tab, &points, &opt).unwrap();
        let x = [C::new(0.2, 0.3), C::new(0.4, -0.1)];
        close(data.eval(&x), m.eval(&x), 1e-9);
    }
}

#[test]
fn complex_support_nodes_partial_nodes_and_real_lift() {
    let points = complex_grid();
    let (m, _) = alg1(&rational, &points, &Default::default()).unwrap();
    for &a in &m.lambda[0] {
        for &b in &m.lambda[1] {
            close(m.eval(&[a, b]), rational(&[a, b]), 1e-14);
        }
    }
    for &a in &m.lambda[0] {
        let x = [a, C::new(0.31, 0.22)];
        close(m.eval(&x), rational(&x), 1e-12);
    }
    let real = |x: &[f64]| 1.0 / ((x[0] + 3.0) * (x[1] + 2.0));
    let points = Points::alternating(&[(-1.0, 1.0); 2], &[8; 2]).unwrap();
    let (m, _) = alg1(&real, &points, &Default::default()).unwrap();
    let lifted = m.to_complex();
    let x = [C::new(0.2, 0.7), C::new(-0.3, -0.4)];
    close(lifted.eval(&x), 1.0 / ((x[0] + 3.0) * (x[1] + 2.0)), 1e-12);
    // A real grid may also sample a complex-valued target by embedding its nodes.
    let (m, _) = alg1(&rational, &points.to_complex(), &Default::default()).unwrap();
    close(m.eval(&x), rational(&x), 1e-11);
}

#[test]
fn complex_conjugate_pairs_do_not_force_a_complex_target_to_be_real() {
    let points = Points::new(
        vec![vec![C::new(0.2, 0.6), C::new(0.2, -0.6)]],
        vec![vec![C::new(0.8, 0.3), C::new(0.8, -0.3)]],
    )
    .unwrap();
    let f = |x: &[C]| C::new(1.0, 2.0) / (x[0] + C::new(3.0, 0.4));
    let (m, _) = alg1(&f, &points, &Default::default()).unwrap();
    close(m.eval(&[C::new(0.1, 0.2)]), f(&[C::new(0.1, 0.2)]), 1e-13);
}

#[test]
fn complex_tensor_validation_and_continuation() {
    let points = Points::new(
        vec![vec![C::new(0.0, 0.0), C::new(1.0, 0.0)]],
        vec![vec![C::new(0.5, 0.0), C::new(1.5, 0.0)]],
    )
    .unwrap();
    let device = DeviceFaer::default();
    let bad: Tensor<C> = asarray((vec![C::new(1.0, 0.0); 3], [3].c(), &device)).into_dim();
    assert!(rmlf::alg1_from_tensor(&bad, &points, &Default::default()).is_err());
    let nan: Tensor<C> = asarray((
        vec![
            C::new(1.0, 0.0),
            C::new(0.0, f64::NAN),
            C::new(0.5, 0.0),
            C::new(0.25, 0.0),
        ],
        [4].c(),
        &device,
    ))
    .into_dim();
    assert!(matches!(
        rmlf::alg1_from_tensor(&nan, &points, &Default::default()),
        Err(rmlf::LoewnerError::NonFinite { count: 1, .. })
    ));
    let f = |x: &[C]| {
        if x[0] == C::new(0.0, 0.0) {
            C::new(f64::NAN, 0.0)
        } else {
            C::new(2.0, 3.0)
        }
    };
    let (filled, count) = rmlf::sample_continued(&f, &points);
    assert_eq!(count, 1);
    let (m, _) = rmlf::alg1_from_tensor(&filled, &points, &Default::default()).unwrap();
    close(m.eval(&[C::new(0.3, 0.2)]), C::new(2.0, 3.0), 1e-13);
}

#[test]
fn wide_complex_nullspace_and_zero_spectrum() {
    let l = faer::Mat::from_fn(1, 3, |_, j| C::new(j as f64 + 1.0, j as f64 - 0.5));
    let c = SvdNull::svd0().null(&l).unwrap();
    let residual: C = (0..3).map(|j| l[(0, j)] * c.vector[j]).sum();
    assert!(residual.norm() < 1e-13);
    // Both scalar types must return a TRUE null direction of a wide matrix,
    // not the last nonzero singular direction from a thin SVD.
    let real = faer::Mat::from_fn(1, 3, |_, j| j as f64 + 1.0);
    let rc = SvdNull::svd0().null(&real).unwrap();
    let residual: f64 = (0..3).map(|j| real[(0, j)] * rc.vector[j]).sum();
    assert!(residual.abs() < 1e-13);
    let zero = faer::Mat::<C>::zeros(2, 2);
    assert_eq!(SvdNull::svd0().null(&zero).unwrap().sigma, vec![0.0, 0.0]);
    let lambda = [C::new(0.0, 1.0), C::new(1.0, -0.5)];
    let mu = [C::new(0.3, 0.2), C::new(1.3, 0.7)];
    let f = |z: C| C::new(1.0, 1.0) / (z + 3.0);
    let l = loewner_1d(&lambda, &lambda.map(f), &mu, &mu.map(f));
    assert!(SvdNull::svd0().null(&l).unwrap().sigma[1] < 1e-12);
}

#[test]
fn complex_grid_builders_and_duplicate_detection() {
    let bounds = [(C::new(0.0, 1.0), C::new(0.0, 3.0))];
    let p = Points::reference_segments(&bounds, &[3]).unwrap();
    assert_eq!(
        p.column[0],
        vec![C::new(0.0, 1.0), C::new(0.0, 2.0), C::new(0.0, 3.0)]
    );
    assert_eq!(p.row[0][0], C::new(0.5, 1.0));
    let z = C::new(0.2, 0.3);
    assert!(matches!(
        Points::new(vec![vec![z]], vec![vec![z]]),
        Err(rmlf::LoewnerError::ComplexNonDisjoint { .. })
    ));
    assert!(Points::alternating_segments(&bounds, &[1]).is_err());
}

#[test]
fn complex_separable_three_dimensional_recursion() {
    let f = |x: &[C]| {
        1.0 / ((x[0] + C::new(3.0, 0.2)) * (x[1] + C::new(2.0, -0.3)) * (x[2] + C::new(4.0, 0.1)))
    };
    let points =
        Points::chebyshev_segments(&[(C::new(-0.5, 0.2), C::new(0.8, 0.5)); 3], &[8; 3]).unwrap();
    let (m, info) = alg1(&f, &points, &Default::default()).unwrap();
    assert_eq!(info.order, vec![1, 1, 1]);
    let x = [C::new(0.23, 0.18), C::new(-0.12, -0.3), C::new(0.4, 0.27)];
    close(m.eval(&x), f(&x), 1e-12);
    let tab = sample(&f, &points);
    let sel = rmlf::points_selection(&points, tab.view(), &info.order, true);
    let rec = loewner_null_rec_svd0(&sel.points, sel.tab.view()).unwrap();
    for (&c, &want) in rec.c.iter().zip(&m.c) {
        close(c, want, 1e-12);
    }
}

#[test]
fn complex_pivoted_selection_and_greedy_construction_policies() {
    let points = complex_grid();
    let x = [C::new(0.27, -0.08), C::new(0.38, 0.19)];
    for minimal in [false, true] {
        let opt = Alg1Options {
            select: rmlf::SelectStrategy::Pivoted,
            minimal,
            ..Default::default()
        };
        let (model, _) = alg1(&rational, &points, &opt).unwrap();
        close(model.eval(&x), rational(&x), 1e-11);
    }
    for method in [CoefficientMethod::Full, CoefficientMethod::Auto] {
        let opt = Alg2Options {
            method,
            tol: 1e-11,
            ..Default::default()
        };
        let (model, _) = alg2(&rational, &points, &opt).unwrap();
        close(model.eval(&x), rational(&x), 1e-10);
    }
    // Lawson placement uses real nonnegative weights on complex magnitudes.
    let opt = Alg2Options {
        lawson: Some(1),
        tol: 1e-11,
        ..Default::default()
    };
    let (model, _) = alg2(&rational, &points, &opt).unwrap();
    close(model.eval(&x), rational(&x), 1e-10);
}
