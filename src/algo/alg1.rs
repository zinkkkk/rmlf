//! Algorithm 1: direct multivariate Loewner approximation. Replaces `mlf.alg1`.
//!
//! Pipeline: sample the target on the combined grid → detect the order along
//! each axis → select `order + 1` points and reduce the tensor → build the
//! barycentric coefficients with the recursive null space → assemble an
//! evaluable [`Barycentric`] model.

use crate::error::Result;
use crate::eval::{Barycentric, column_values};
use crate::nullspace::{NullSpace, SvdNull};
use crate::order::{OrderOptions, compute_order_report};
use crate::points::Points;
use crate::recurse::loewner_null_separable;
use crate::scalar::Scalar;
use crate::select::{SelectStrategy, points_selection_with};
use crate::source::TargetFn;
use crate::tensor::{Tensor, sample_continued};

use super::coefficient::{CoefficientMethod, coefficients};

/// Options for [`alg1()`].
#[derive(Debug, Clone)]
pub struct Alg1Options {
    /// Order-detection options. If `tol` is set to a negative value, order
    /// detection is skipped and every axis uses `column_len - 1` (the maximal
    /// order the data supports), matching `mlf.alg1`'s `ord_tol == -1` path.
    pub order: OrderOptions,
    /// Keep the row set minimal (square 1-D Loewner per axis). `true` mirrors
    /// the common `data_min` default.
    pub minimal: bool,
    /// How the `order + 1` support points are chosen per axis. Default
    /// [`SelectStrategy::Spread`] (the reference even split). [`SelectStrategy::
    /// Pivoted`] makes one-shot alg1 robust on coupled targets the even split
    /// cannot resolve (see WORKPLAN §6d/#7).
    pub select: SelectStrategy,
    /// How to construct the final barycentric coefficient tensor.  `Full` forms
    /// one multivariate Loewner matrix; this is unrelated to Algorithm 2's
    /// residual-grid [`crate::Candidates::Full`] setting.
    pub method: CoefficientMethod,
}

impl Default for Alg1Options {
    fn default() -> Self {
        Alg1Options {
            order: OrderOptions::default(),
            minimal: true,
            select: SelectStrategy::Spread,
            method: CoefficientMethod::Recursive,
        }
    }
}

/// Information returned alongside the model.
#[derive(Debug, Clone)]
pub struct Alg1Info {
    /// Detected (or forced) order per axis.
    pub order: Vec<usize>,
    /// Coefficient-construction flop count.
    pub flop: u128,
}

/// Build an approximation of a function `f` of `n` variables, sampling it on the
/// grid defined by `points`.
pub fn alg1<S: Scalar, F: TargetFn<S> + Sync>(
    f: &F,
    points: &Points<S>,
    opt: &Alg1Options,
) -> Result<(Barycentric<S>, Alg1Info)> {
    // Sample with analytic-continuation fill: if the grid lands on an isolated
    // singularity (e.g. `x1²/(x1²+x2²)` at the origin → 0/0 = NaN), a single
    // non-finite sample would otherwise make the SVD fail outright. `sample_
    // continued` fills it from finite neighbors so the fit degrades gracefully to
    // a (possibly poor) finite model instead of erroring — matching the reference,
    // which avoids the exact-zero only by floating-point luck. Identical to
    // `sample` when every sample is finite. The data-tensor entry points stay
    // strict (the caller owns their data).
    let (tab, _) = sample_continued(f, points);
    alg1_from_tensor(&tab, points, opt)
}

/// Build an approximation from an already-sampled data tensor `tab` (the
/// data-driven entry point, when no closure is available).
///
/// `tab` must have been sampled on the combined `column ++ row` grid of
/// `points`, row-major (use [`crate::tensor::sample`]).
pub fn alg1_from_tensor<S: Scalar>(
    tab: &Tensor<S>,
    points: &Points<S>,
    opt: &Alg1Options,
) -> Result<(Barycentric<S>, Alg1Info)> {
    alg1_with_null(tab, points, opt, &SvdNull::svd0())
}

/// As [`alg1_from_tensor`], with an explicit null-space strategy for the
/// top-level leading solve (the recursion uses last-entry normalization below
/// the top regardless; see [`crate::recurse`]).
pub fn alg1_with_null<S: Scalar, N: NullSpace<S>>(
    tab: &Tensor<S>,
    points: &Points<S>,
    opt: &Alg1Options,
    null: &N,
) -> Result<(Barycentric<S>, Alg1Info)> {
    points.check_disjoint()?;
    crate::tensor::validate(tab, points)?;

    // Order detection (or forced maximal order). When detection runs it also
    // reports the per-axis unfolding rank, which gates the separability shortcut.
    let (order, separable) = if opt.order.tol < 0.0 {
        (
            points.column_sizes().iter().map(|&k| k - 1).collect(),
            false,
        )
    } else {
        let report = compute_order_report(points, tab.view(), &opt.order);
        // Separable iff every axis's mode-l unfolding is rank 1 (and n > 1; the
        // 1-D case is already a single solve, so the shortcut buys nothing).
        // Keep the legacy real optimization, but do not extend its heuristic
        // to complex inputs: rank-one 1-D slices alone do not establish
        // separability (a coupled bilinear denominator is a counterexample).
        let sep = S::IS_REAL && report.rank.len() > 1 && report.rank.iter().all(|&r| r == 1);
        (report.order, sep)
    };

    // Point selection + tensor reduction.
    let sel = points_selection_with(points, tab.view(), &order, opt.minimal, opt.select);

    // Recursive barycentric coefficients on the reduced problem. A separable
    // target (#6) takes the n-SVD outer-product shortcut instead of the full
    // recursion's k_0× redundant child solves; both emit the same rank-1 `c`.
    let rec = if separable && opt.method == CoefficientMethod::Recursive {
        loewner_null_separable(&sel.points, sel.tab.view(), null)?
    } else {
        coefficients(opt.method, &sel.points, sel.tab.view(), null)?
    };

    // Values at the reduced column grid.
    let w = column_values(&sel.points, sel.tab.view());

    let model = Barycentric {
        lambda: sel.points.column.clone(),
        w,
        c: rec.c,
    };
    Ok((
        model,
        Alg1Info {
            order,
            flop: rec.flop,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn recovers_polya_szego_polynomial() {
        // Getting-started example: H = x1*x2 + x1*x3 + x2*x3 on [-1,1]^3.
        let f = |x: &[f64]| x[0] * x[1] + x[0] * x[2] + x[1] * x[2];
        let pts = Points::alternating(&[(-1.0, 1.0); 3], &[10; 3]).unwrap();

        let mut opt = Alg1Options::default();
        opt.order.tol = 1e-8;
        let (model, info) = alg1(&f, &pts, &opt).unwrap();
        assert_eq!(info.order, vec![1, 1, 1]);

        for (a, b, c) in [(0.1, -0.3, 0.5), (-0.7, 0.2, 0.9), (0.33, 0.33, -0.21)] {
            let x = [a, b, c];
            assert_relative_eq!(model.eval(&x), f(&x), epsilon = 1e-9);
        }
    }

    #[test]
    fn recovers_oversampled_rational() {
        // Order-1 separable rational, deliberately oversampled (7 pts/axis).
        let f = |x: &[f64]| 1.0 / ((x[0] + 3.0) * (x[1] + 2.0) * (x[2] + 4.0));
        let col: Vec<f64> = (0..7).map(|i| i as f64 * 0.3).collect();
        let row: Vec<f64> = (0..7).map(|i| i as f64 * 0.3 + 0.15).collect();
        let pts = Points::new(
            vec![col.clone(), col.clone(), col.clone()],
            vec![row.clone(), row.clone(), row.clone()],
        )
        .unwrap();
        let (model, info) = alg1(&f, &pts, &Alg1Options::default()).unwrap();
        assert_eq!(info.order, vec![1, 1, 1]);
        let x = [0.42, 1.1, 0.27];
        assert_relative_eq!(model.eval(&x), f(&x), epsilon = 1e-9);
    }

    #[test]
    fn separable_shortcut_yields_correct_model() {
        // A separable target (all unfolding ranks == 1) routes through the #6
        // fast path. Verify the resulting model interpolates correctly at
        // off-grid points; entry-for-entry parity with the full recursion is
        // covered by `recurse::tests::separable_matches_recursion`.
        let f = |x: &[f64]| 1.0 / ((x[0] + 3.0) * (x[1] + 2.0) * (x[2] + 4.0));
        let col: Vec<f64> = (0..7).map(|i| i as f64 * 0.3).collect();
        let row: Vec<f64> = (0..7).map(|i| i as f64 * 0.3 + 0.15).collect();
        let pts = Points::new(
            vec![col.clone(), col.clone(), col.clone()],
            vec![row.clone(), row.clone(), row.clone()],
        )
        .unwrap();

        let (model, info) = alg1(&f, &pts, &Alg1Options::default()).unwrap();
        assert_eq!(info.order, vec![1, 1, 1]);
        for x in [[0.42, 1.1, 0.27], [-0.3, 0.8, 1.4], [2.1, 0.05, 0.9]] {
            assert_relative_eq!(model.eval(&x), f(&x), epsilon = 1e-9);
        }
    }

    #[test]
    fn full_method_builds_the_final_tensor() {
        let f = |x: &[f64]| 1.0 / (3.0 + x[0] + 0.5 * x[1] + x[0] * x[1]);
        let pts = Points::alternating(&[(-0.8, 0.8); 2], &[8; 2]).unwrap();
        let opt = Alg1Options {
            method: CoefficientMethod::Full,
            ..Alg1Options::default()
        };
        let (model, _) = alg1(&f, &pts, &opt).unwrap();
        for x in [[0.13, -0.27], [-0.51, 0.42], [0.61, 0.09]] {
            assert_relative_eq!(model.eval(&x), f(&x), epsilon = 1e-9);
        }
    }
}
