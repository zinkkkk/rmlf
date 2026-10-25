use rmlf::examples::complex;
use rmlf::prelude::*;
type C = Complex64;

#[test]
fn entire_commented_catalog_matches_original_octave_formulas() {
    let golden: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/complex_reference.json")).unwrap();
    let entries = golden.as_array().unwrap();
    assert_eq!(entries.len(), complex::legacy_ids().len());
    assert_eq!(complex::all().len(), entries.len() + 1);
    for entry in entries {
        let id = entry["id"].as_u64().unwrap() as u32;
        let case = complex::get(id).unwrap();
        assert!(complex::legacy_ids().contains(&(id as usize)));
        assert_eq!(case.ndim, id as usize);
        assert_eq!(case.degrees.unwrap().len(), case.ndim);
        let nodes = case.points(None).unwrap();
        assert_eq!(
            nodes.column_sizes(),
            case.degrees
                .unwrap()
                .iter()
                .map(|d| d + 1)
                .collect::<Vec<_>>()
        );
        for (sample, value) in entry["values"].as_array().unwrap().iter().enumerate() {
            let x: Vec<C> = (1..=case.ndim)
                .map(|j| {
                    C::new(
                        0.75 + 0.025 * j as f64 + 0.11 * sample as f64,
                        0.05 + 0.007 * j as f64 - 0.02 * sample as f64,
                    )
                })
                .collect();
            let want = C::new(value[0].as_f64().unwrap(), value[1].as_f64().unwrap());
            let got = (case.f)(&x);
            assert!(
                (got - want).norm() < 1e-14 * (1.0 + want.norm()),
                "case {id}: {got} vs {want}"
            );
        }
    }
}

#[test]
fn every_axis_of_every_legacy_complex_example_fits_off_grid() {
    // Every variable participates, without constructing an exponential n-D grid.
    for &id in complex::legacy_ids() {
        let case = complex::get(id as u32).unwrap();
        let fixed: Vec<C> = (0..case.ndim)
            .map(|j| C::new(1.05 + 0.009 * j as f64, 0.06 + 0.002 * j as f64))
            .collect();
        for (axis, &degree) in case.degrees.unwrap().iter().enumerate() {
            let f = |s: &[C]| {
                let mut x = fixed.clone();
                x[axis] = s[0];
                (case.f)(&x)
            };
            let pts = Points::chebyshev_segments(
                &[(C::new(0.4, 0.1), C::new(1.8, 0.3))],
                &[2 * (degree + 1)],
            )
            .unwrap();
            let opt = Alg1Options {
                order: rmlf::OrderOptions {
                    tol: -1.0,
                    ..Default::default()
                },
                ..Default::default()
            };
            let (model, _) = alg1(&f, &pts, &opt).unwrap();
            for i in 0..7 {
                let x = [C::new(0.47 + 0.2 * i as f64, 0.16 + 0.017 * i as f64)];
                let want = f(&x);
                let got = model.eval(&x);
                assert!(
                    (got - want).norm() < 2e-8 * (1.0 + want.norm()),
                    "case {id}, axis {axis}, degree {degree}: {got} vs {want}"
                );
            }
        }
    }
}

#[test]
fn every_legacy_example_fits_a_coupled_two_axis_section() {
    for &id in complex::legacy_ids() {
        let case = complex::get(id as u32).unwrap();
        // First two variables exercise the high-degree terms and their coupling.
        let fixed = vec![C::new(1.1, 0.05); case.ndim];
        let f = |s: &[C]| {
            let mut x = fixed.clone();
            x[..2].copy_from_slice(s);
            (case.f)(&x)
        };
        let d = case.degrees.unwrap();
        let pts = Points::chebyshev_segments(
            &[(C::new(0.4, 0.1), C::new(1.8, 0.3)); 2],
            &[2 * (d[0] + 1), 2 * (d[1] + 1)],
        )
        .unwrap();
        let opt = Alg1Options {
            order: rmlf::OrderOptions {
                tol: -1.0,
                ..Default::default()
            },
            ..Default::default()
        };
        let (model, _) = alg1(&f, &pts, &opt).unwrap();
        for i in 0..5 {
            let x = [
                C::new(0.63 + 0.21 * i as f64, 0.18),
                C::new(0.57 + 0.17 * i as f64, 0.22),
            ];
            let want = f(&x);
            let got = model.eval(&x);
            assert!(
                (got - want).norm() < 2e-7 * (1.0 + want.norm()),
                "case {id}: {got} vs {want}"
            );
        }
    }
}

#[test]
fn active_mixed_domain_case_59() {
    let case = complex::get(59).unwrap();
    let pts = case.points(None).unwrap();
    assert_eq!(pts.column_sizes(), vec![30, 30]);
    let (model, info) = alg1(&case.f, &pts, &Default::default()).unwrap();
    let mut worst = 0.0_f64;
    for i in 0..25 {
        let x = [
            C::new(0.0, 0.12 + 3.91 * i as f64),
            C::new(0.17 + 0.72 * i as f64, 0.0),
        ];
        let want = (case.f)(&x);
        let got = model.eval(&x);
        worst = worst.max((got - want).norm() / (1.0 + want.norm()));
    }
    assert!(
        worst < 1e-6,
        "case 59, order {:?}, max scaled error {worst:e}",
        info.order
    );
}
