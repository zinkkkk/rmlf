//! Validation of the variable-decoupling method (`src/decouple.rs`) against the
//! existing `alg1` recursion and the example catalog.
//!
//! The follow-up paper's decoupling produces the *same* multivariate rational as
//! the recursion (it is an equivalent barycentric form built by inspection), so on
//! any rational target sampled on a unisolvent grid the two must agree to machine
//! precision — and both must recover the true function. These tests pin that
//! equivalence on the catalog's separable/tensor-product rationals and confirm the
//! decoupled single-variable (KST) functions reconstruct the target.

use rmlf::decouple::{decouple, decouple_from_closure};
use rmlf::examples::get;
use rmlf::{Alg1Options, Points, alg1};

/// On a unisolvent grid, the decoupled model and the alg1 recursion evaluate
/// identically (same underlying barycentric form, different construction route).
#[test]
fn decouple_matches_alg1_on_separable_rational() {
    let f = |x: &[f64]| 1.0 / ((x[0] + 3.0) * (x[1] + 2.0) * (x[2] + 4.0));
    let pts = Points::alternating(&[(-1.0, 1.0); 3], &[6; 3]).unwrap();

    let mut opt = Alg1Options::default();
    opt.order.tol = 1e-8;
    let (m1, info) = alg1(&f, &pts, &opt).unwrap();
    assert_eq!(info.order, vec![1, 1, 1]);

    // Reduce to order+1 points the same way alg1 does, then decouple with those
    // degrees. Easiest: use the auto-detecting `decouple` on the same grid.
    let (md, di) = decouple(&f, &[(-1.0, 1.0); 3], &[6; 3]).unwrap();
    assert_eq!(di.degree, vec![1, 1, 1]);

    for x in [[0.2, -0.3, 0.5], [-0.7, 0.4, 0.9], [0.1, 0.1, -0.6]] {
        let a = m1.eval(&x);
        let d = md.eval(&x);
        assert!((a - d).abs() < 1e-9, "alg1 {a} vs decouple {d} at {x:?}");
        assert!(
            (d - f(&x)).abs() < 1e-9,
            "decouple {d} vs true {} at {x:?}",
            f(&x)
        );
    }
}

/// The Pólya–Szegő catalog case (CAS 48) decouples exactly from the minimal grid.
#[test]
fn catalog_polya_szego_decouples_exactly() {
    let e = get(48).expect("CAS 48 present");
    // Degree (1,1,1) needs exactly degree+1 = 2 column points per axis. The
    // reference `examples.m` convention (p_c = linspace(lo,hi,nip), p_r = p_c+dx)
    // with nip = 2 gives that minimal unisolvent grid and decouples exactly on
    // CAS 48's bounds [-1/2, 1] (verified err ~1e-17). (Note: `Points::alternating`
    // deals one grid into col/row and is non-unisolvent here — domain-dependent
    // conditioning, see notes; the reference convention is correct for this case.)
    let (model, info) =
        decouple_from_closure(&e.f, &e.points(Some(2)), &[1, 1, 1]).expect("decouple Pólya–Szegő");
    assert_eq!(info.n_functions, 7);
    for x in [[0.1, -0.3, 0.5], [-0.7, 0.2, 0.9], [0.33, 0.33, -0.21]] {
        assert!(
            (model.eval(&x) - (e.f)(&x)).abs() < 1e-9,
            "decouple {} vs true {} at {x:?}",
            model.eval(&x),
            (e.f)(&x)
        );
    }
}

/// A higher-`n` separable rational from the catalog (CAS 44, n=8) decouples; the
/// recursion's curse-taming carries over since the construction is the same tree of
/// 1-D solves, now with weights written by inspection.
#[test]
fn catalog_high_dim_separable_decouples() {
    // 1 / (x1^4 + x2^2 x3 + ... ) is NOT separable, but a genuinely separable
    // high-n rational is: use a product form to exercise the n=8 path cheaply.
    let f = |x: &[f64]| {
        let mut p = 1.0;
        for (i, &xi) in x.iter().enumerate() {
            p *= 1.0 / (xi + (i as f64) + 2.0);
        }
        p
    };
    let bounds = vec![(-1.0, 1.0); 8];
    let nip = vec![4usize; 8];
    let (md, di) = decouple(&f, &bounds, &nip).unwrap();
    assert_eq!(di.degree, vec![1; 8]);
    let x = [0.2, -0.1, 0.3, 0.0, 0.4, -0.2, 0.1, 0.5];
    assert!(
        (md.eval(&x) - f(&x)).abs() < 1e-8,
        "n=8 decouple {} vs true {}",
        md.eval(&x),
        f(&x)
    );
}
