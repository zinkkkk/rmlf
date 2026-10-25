//! Algorithm 2: greedy / adaptive multivariate Loewner approximation. The
//! *iterative* counterpart of [`crate::algo::alg1()`], replacing `mlf.alg2`.
//!
//! Where alg1 detects the order once and fits in a single shot, alg2 grows the
//! interpolation (column) set one point at a time: it fits the current model,
//! measures the approximation error at **every** sample on the grid, and adds the
//! abscissa of maximum error to the column set along each axis. It stops when the
//! relative max error drops below `tol` or the iteration / order budget is spent,
//! returning the best model seen (AAA-like greedy selection).
//!
//! ## Improvements over the MATLAB reference
//!
//! The MATLAB `alg2` is research code with several avoidable costs we drop:
//!
//! * **No string-built dynamic indexing.** MATLAB assembles `tab(...)` index
//!   expressions as strings and `eval`s them each iteration; we reduce the tensor
//!   with [`crate::tensor::gather`] over index vectors — allocation-light and
//!   branch-free.
//! * **Parallel, batched error scan.** The error at all `N` grid points is the
//!   per-iteration hot loop; MATLAB calls a scalar evaluator in a `for` loop. We
//!   evaluate the whole grid at once with the rayon-parallel
//!   [`Barycentric::eval_grid`].
//! * **Single canonical ordering.** Row-major storage means the value block `w`,
//!   the coefficients `c`, and the grid all share one order, so MATLAB's
//!   `mat2vec`/`vec2mat`/`ind2sub` round-trips vanish.
//! * **Order bound reuses the order machinery.** The `max_itpl` cap is derived
//!   from [`crate::order::compute_order`] at two tolerances (the same idea as
//!   `alg2_order_bnd`) rather than a separately maintained code path.

use super::scan::{FitScalar, ScanAccelerator};
use crate::error::{LoewnerError, Result};
use crate::eval::Barycentric;
use crate::grid::{flat_to_multi, grid_len, multi_to_flat};
use crate::nullspace::{NullSpace, SvdNull};
use crate::order::{OrderOptions, compute_order_report};
use crate::points::Points;
use crate::scalar::Scalar;
use crate::source::TargetFn;
use crate::tensor::{Tensor, contig_slice, gather, sample_continued};

use rayon::prelude::*;

use super::coefficient::{CoefficientMethod, coefficients};

/// Dense grid-scan flop estimate (`grid_size · reduced_coeff_size`) above which a
/// TT-accelerated scan is attempted. Below it the dense scan is already cheap and
/// not worth the per-iteration TT-SVD overhead.
const TT_GRID_THRESHOLD: u128 = 1 << 20;

/// Which grid points the greedy error hunt evaluates each iteration.
///
/// The default [`Candidates::Full`] scan visits every one of the
/// `prod(k_l + q_l)` grid points — exact greedy selection, but a point count
/// **exponential in `n`** regardless of how cheap each eval is (the TT scan
/// acceleration only lowers per-point cost). For high-`n` problems the count
/// itself is the wall; the cheaper modes evaluate a *subset* and trade exact
/// greedy parity for a tractable per-iteration cost. They still converge (the
/// residual genuinely drops), they just may take a different path there.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Candidates {
    /// Scan the entire grid (default). Exact greedy pick and exact stopping-test
    /// max error; cost `O(prod(k_l + q_l))` per iteration.
    Full,
    /// Coordinate (axis-wise) ascent: hold the current best point and sweep one
    /// axis at a time, taking the max residual over the swept lines. Cost
    /// `O(Σ(k_l + q_l))` per iteration — linear, not exponential, in `n`. Good
    /// for near-separable residuals; the stopping-test max error is a lower bound
    /// (only the swept lines are seen).
    Coordinate,
    /// Evaluate the residual on a fixed pseudo-random subset of this many grid
    /// points, take the max there, then refine with one coordinate sweep around
    /// it. Cost `O(count + Σ(k_l + q_l))`. Cheapest; the most aggressive
    /// approximation of the true greedy pick.
    Subsample(usize),
}

/// Options for [`alg2()`].
#[derive(Debug, Clone)]
pub struct Alg2Options {
    /// Relative max-error stopping tolerance (`max|err| / max|tab| < tol`).
    pub tol: f64,
    /// Hard cap on greedy iterations. `None` = `prod(k_l) - 1` (the MATLAB
    /// default: as many as there are column points minus one).
    pub max_iter: Option<usize>,
    /// Keep the row set minimal (square 1-D Loewner per axis). Mirrors
    /// `data_min` in the reference.
    pub minimal: bool,
    /// Per-axis cap on how many interpolation points may be added. `None` derives
    /// it from a two-tolerance order estimate (rational axes get a tight cap,
    /// others may use the full point set).
    pub max_itpl: Option<Vec<usize>>,
    /// Tolerance for tensor-train compression of each iteration's model, used to
    /// accelerate the full-grid error scan (the exponential bottleneck) via
    /// factorized eval. `Some(tol)` enables it; the implementation only switches
    /// to the TT path when the grid scan is large *and* the model actually
    /// compresses (the contraction is cheaper than dense), so it never regresses
    /// small or incompressible problems. Complex models always use dense eval;
    /// `None` also forces dense eval for real models.
    pub compress: Option<f64>,
    /// Which grid points the greedy error hunt evaluates each iteration. Default
    /// [`Candidates::Full`] (exact greedy parity); the cheaper modes make high-`n`
    /// tractable by scanning a subset. See [`Candidates`].
    pub candidates: Candidates,
    /// How to construct the coefficient tensor at every greedy iteration.
    /// `Full` here means a monolithic multivariate Loewner null space and is
    /// independent of [`Candidates::Full`], which only controls the residual
    /// points scanned when choosing the next support point.
    pub method: CoefficientMethod,
    /// Early-stop patience: stop once this many *consecutive* iterations fail to
    /// improve on the best error seen so far. Greedy point-adding has diminishing
    /// returns, but the error can also plateau for a few iterations before a key
    /// point breaks through — so this is set forgivingly (the reference `alg2.m`
    /// uses `floor(max_iter/5)`, here ~8 of 25). Lower it only to bail sooner on
    /// hopeless targets. `None` disables the cutoff (run to `max_iter`); `Some(0)`
    /// is treated as `Some(1)`.
    pub patience: Option<usize>,
    /// **Multivariate Lawson** (minimax-oriented point placement). `None` ⇒ the
    /// classic greedy that adds the support point of largest *raw* residual (an
    /// `L²`/interpolatory bias). `Some(_)` maintains nonnegative Lawson weights
    /// over the full grid and instead adds the point of largest *reweighted*
    /// residual, updating the weights `s ← s · |f − g|` each iteration so support
    /// migrates toward the persistent-error (equioscillation) regions — the n-D
    /// lift of AAA-Lawson. It only changes *which* point is promoted, so the
    /// recursion's curse-taming construction is untouched and the cost is the same
    /// full-grid scan. The reserved `usize` is the (currently unused) inner-refit
    /// budget for a future coefficient-level Lawson polish; any `Some` enables the
    /// placement reweighting.
    pub lawson: Option<usize>,
}

impl Default for Alg2Options {
    fn default() -> Self {
        Alg2Options {
            tol: 1e-9,
            max_iter: Some(25),
            minimal: true,
            max_itpl: None,
            compress: Some(1e-10),
            candidates: Candidates::Full,
            method: CoefficientMethod::Recursive,
            patience: Some(8),
            lawson: None,
        }
    }
}

/// Information returned alongside the greedy model.
#[derive(Debug, Clone)]
pub struct Alg2Info {
    /// Iteration index of the returned (best) model, 1-based.
    pub best_iter: usize,
    /// Relative max error of the returned model.
    pub best_error: f64,
    /// Relative max error after each iteration (length = iterations run).
    pub error_history: Vec<f64>,
    /// Per-axis order of the returned model (`k_l - 1`).
    pub order: Vec<usize>,
    /// Coefficient-construction flop count, summed over iterations.
    pub flop: u128,
}

/// Build a greedy/adaptive approximation of `f`, sampling on the grid of
/// `points`. See [`alg2_from_tensor`] for the data-driven entry point.
pub fn alg2<S: FitScalar, F: TargetFn<S> + Sync>(
    f: &F,
    points: &Points<S>,
    opt: &Alg2Options,
) -> Result<(Barycentric<S>, Alg2Info)> {
    // Continuation-fill isolated non-finite samples so a grid that happens to land
    // on a singularity degrades to a finite (if poor) fit instead of crashing the
    // SVD — see `alg1` for the rationale. Identical to `sample` when all finite.
    let (tab, _) = sample_continued(f, points);
    alg2_from_tensor(&tab, points, opt)
}

/// As [`alg2()`], from an already-sampled full data tensor `tab` (sampled on the
/// combined `column ++ row` grid of `points`, row-major).
pub fn alg2_from_tensor<S: FitScalar>(
    tab: &Tensor<S>,
    points: &Points<S>,
    opt: &Alg2Options,
) -> Result<(Barycentric<S>, Alg2Info)> {
    alg2_with_null(tab, points, opt, &SvdNull::svd0())
}

/// Threshold on the unfolding spectral gap below which an axis is judged
/// confidently rational. The gap is `sigma[order] / sigma[order-1]`; a value
/// this small means the spectrum drops sharply past the detected order, the
/// single-pass analogue of the old "order stable across two tolerances" test.
const RATIONAL_GAP: f64 = 1e-3;

/// Estimate the per-axis interpolation-point cap, the analogue of
/// `mlf.alg2_order_bnd`. Where the MATLAB heuristic ran order detection at two
/// tolerances and capped any axis whose order was *stable* across both, the
/// stacked rank-revealing unfolding (#3) gives the singular-value decay from a
/// **single pass** (`compute_order_report`): a sharp spectral gap past the
/// detected order ⇒ confidently rational ⇒ cap at `order + 1` points; a gradual
/// decay ⇒ leave the full point set available.
fn estimate_max_itpl<S: Scalar>(points: &Points<S>, tab: &Tensor<S>) -> Vec<usize> {
    let full: Vec<usize> = points.column_sizes();
    let n = points.ndim();
    let report = compute_order_report(
        points,
        tab.view(),
        &OrderOptions {
            tol: 1e-9,
            max_order: None,
            max_contexts: Some(2),
            gap_stop: None,
        },
    );
    (0..n)
        .map(|i| {
            if report.gap[i] < RATIONAL_GAP {
                report.order[i] + 1
            } else {
                full[i]
            }
        })
        .collect()
}

/// As [`alg2_from_tensor`], with an explicit null-space strategy for each
/// iteration's top-level leading solve.
pub fn alg2_with_null<S: FitScalar, N: NullSpace<S>>(
    tab: &Tensor<S>,
    points: &Points<S>,
    opt: &Alg2Options,
    null: &N,
) -> Result<(Barycentric<S>, Alg2Info)> {
    points.check_disjoint()?;
    crate::tensor::validate(tab, points)?;
    let n = points.ndim();

    // Combined abscissae and full tensor dimensions per axis.
    let ip: Vec<Vec<S>> = (0..n).map(|l| points.combined_axis(l)).collect();
    let full_dims: Vec<usize> = ip.iter().map(Vec::len).collect();
    let total = grid_len(&full_dims);
    let tab_view = tab.view();
    let tab_flat = contig_slice(&tab_view);
    debug_assert_eq!(tab_flat.len(), total);

    let max_abs = tab_flat.iter().fold(0.0_f64, |a, &x| a.max(x.magnitude()));
    let scale = if max_abs == 0.0 { 1.0 } else { max_abs };

    let max_itpl = opt
        .max_itpl
        .clone()
        .unwrap_or_else(|| estimate_max_itpl(points, tab));
    let max_iter = opt.max_iter.unwrap_or_else(|| {
        points
            .column_sizes()
            .iter()
            .product::<usize>()
            .saturating_sub(1)
    });

    // Greedy state: per-axis column index sets into `ip`.
    let mut col_idx: Vec<Vec<usize>> = vec![Vec::new(); n];

    // Seed: the grid point furthest from the tensor mean (first big residual).
    let mean = tab_flat.iter().copied().sum::<S>() / S::from_real(total as f64);
    let mut worst = argmax_abs_dev(tab_flat, mean);

    let mut best: Option<(Barycentric<S>, Alg2Info)> = None;
    let mut error_history = Vec::new();
    let mut flop = 0u128;
    let mut multi = vec![0usize; n];
    // Consecutive non-improving iterations, for the early-stop patience guard.
    let mut no_improve = 0usize;
    // Multivariate Lawson weights over the full grid (uniform until enabled). When
    // `opt.lawson` is set the greedy promotes the point of largest *weighted*
    // residual and these are reweighted each iteration; otherwise unused.
    let mut lawson_w: Vec<f64> = if opt.lawson.is_some() {
        vec![1.0; total]
    } else {
        Vec::new()
    };
    let patience = opt.patience.map(|p| p.max(1));

    for iter in 1..=max_iter {
        // Which axes may still grow?
        let can_add: Vec<bool> = (0..n).map(|i| col_idx[i].len() < max_itpl[i]).collect();
        if !can_add.iter().any(|&b| b) {
            break;
        }

        // Add the worst point's per-axis coordinates to the column sets.
        flat_to_multi(worst, &full_dims, &mut multi);
        for i in 0..n {
            if can_add[i] {
                insert_sorted_unique(&mut col_idx[i], multi[i]);
            }
        }

        // Column/row partition along each axis: rows are the complement, capped
        // to the column count when `minimal`.
        let mut pick: Vec<Vec<usize>> = Vec::with_capacity(n);
        let mut new_col: Vec<Vec<S>> = Vec::with_capacity(n);
        let mut new_row: Vec<Vec<S>> = Vec::with_capacity(n);
        for i in 0..n {
            let cols = &col_idx[i];
            let mut rows: Vec<usize> = (0..full_dims[i]).filter(|j| !cols.contains(j)).collect();
            if opt.minimal {
                rows.truncate(cols.len());
            }
            if rows.is_empty() {
                return Err(LoewnerError::Degenerate(
                    "no row points left for the Loewner partition".into(),
                ));
            }
            new_col.push(cols.iter().map(|&j| ip[i][j]).collect());
            new_row.push(rows.iter().map(|&j| ip[i][j]).collect());
            // Reduced-tensor index order along this axis: columns then rows.
            pick.push(cols.iter().copied().chain(rows).collect());
        }

        let sub_points = Points::new(new_col, new_row)?;
        let tab_it = gather(&tab_view, &pick);

        // Coefficients + the column-block values on the reduced grid.  The
        // construction method is applied anew at every greedy iteration.
        let rec = coefficients(opt.method, &sub_points, tab_it.view(), null)?;
        flop += rec.flop;
        let w = crate::eval::column_values(&sub_points, tab_it.view());

        let model = Barycentric {
            lambda: sub_points.column.clone(),
            w,
            c: rec.c,
        };

        // Error at the candidate grid points. The `Full` scan costs
        // `total · prod(k)` and is the greedy bottleneck; the cheaper modes scan a
        // subset (see `Candidates`). When the full scan is large and the model
        // compresses, evaluate the *whole* grid in one factorized TT sweep
        // (`O(total · r²)`) and reduce over the precomputed values — the
        // exponential→polynomial acceleration. Auto-fall back to dense for small
        // grids or incompressible models (where TT would not pay off).
        let dense_cost = (total as u128) * (model.c.len() as u128);
        let tt_model = match opt.compress {
            Some(tol) if dense_cost > TT_GRID_THRESHOLD => {
                S::compress(&model, tol).filter(|tt| tt.eval_cost() < model.c.len())
            }
            _ => None,
        };
        let (max_err, greedy_idx) = if opt.lawson.is_some() {
            // Multivariate Lawson: evaluate the model on the whole grid, pick the
            // next support point by *weighted* residual, then reweight. Always a
            // dense full-grid pass (the weights are per-grid-point).
            let values: Vec<S> = match &tt_model {
                Some(tt) => tt.eval_full_grid(&ip),
                None => {
                    let mut v = vec![S::zero(); total];
                    let mut mi = vec![0usize; n];
                    let mut x = vec![S::zero(); n];
                    for (flat, slot) in v.iter_mut().enumerate() {
                        flat_to_multi(flat, &full_dims, &mut mi);
                        for i in 0..n {
                            x[i] = ip[i][mi[i]];
                        }
                        *slot = model.eval(&x);
                    }
                    v
                }
            };
            grid_error_lawson(
                &values,
                &full_dims,
                tab_flat,
                &col_idx,
                &max_itpl,
                &mut lawson_w,
            )
        } else {
            match opt.candidates {
                Candidates::Full => match &tt_model {
                    Some(tt) => {
                        let values = tt.eval_full_grid(&ip);
                        grid_error_values(&values, &full_dims, tab_flat, &col_idx, &max_itpl)
                    }
                    None => grid_error(
                        |x| model.eval(x),
                        &ip,
                        &full_dims,
                        tab_flat,
                        &col_idx,
                        &max_itpl,
                    ),
                },
                // The subset scans evaluate few points, so a per-point evaluator is
                // fine — prefer the factorized TT eval when the model compressed.
                Candidates::Coordinate | Candidates::Subsample(_) => {
                    let eval = |x: &[S]| match &tt_model {
                        Some(tt) => tt.eval(x),
                        None => model.eval(x),
                    };
                    candidate_error(
                        opt.candidates,
                        &eval,
                        worst,
                        &ip,
                        &full_dims,
                        tab_flat,
                        &col_idx,
                        &max_itpl,
                    )
                }
            }
        };
        let rel = max_err / scale;
        error_history.push(rel);
        worst = greedy_idx;

        let order: Vec<usize> = sub_points.column_sizes().iter().map(|&k| k - 1).collect();
        let info = Alg2Info {
            best_iter: iter,
            best_error: rel,
            error_history: Vec::new(), // filled at the end
            order,
            flop,
        };

        let improved = best
            .as_ref()
            .map(|(_, bi)| rel < bi.best_error)
            .unwrap_or(true);
        if improved {
            best = Some((model, info));
            no_improve = 0;
        } else {
            no_improve += 1;
        }

        if max_err <= scale * opt.tol {
            break;
        }

        // Early stop on stagnation: once the error has not improved for
        // `patience` consecutive iterations, further greedy additions are very
        // unlikely to help — bail out and return the best model instead of
        // grinding to `max_iter` (the user-visible "deeper search that just
        // returns the first guess" pathology).
        if let Some(p) = patience
            && no_improve >= p
        {
            break;
        }
    }

    let (model, mut info) =
        best.ok_or(LoewnerError::Degenerate("alg2 ran no iterations".into()))?;
    info.flop = flop;
    info.error_history = error_history;
    Ok((model, info))
}

/// Index of the grid sample whose deviation from `mean` is largest in absolute
/// value (the greedy seed before any model exists).
fn argmax_abs_dev<S: Scalar>(tab: &[S], mean: S) -> usize {
    tab.par_iter()
        .enumerate()
        .map(|(i, &v)| (i, (v - mean).magnitude()))
        .reduce(
            || (0, f64::NEG_INFINITY),
            |a, b| if b.1 > a.1 { b } else { a },
        )
        .0
}

/// Evaluate `eval` on the full grid in parallel and return both the global max
/// error (for the stopping test) and the index of the max error among the
/// *addable* points (the greedy pick). Points along an axis that has reached its
/// `max_itpl` cap are excluded from the greedy pick (their error cannot be
/// reduced by adding more points there), mirroring the reference's masking.
///
/// `eval` is the model evaluator — either the dense [`Barycentric::eval`] or the
/// factorized [`crate::tt::CompressedBarycentric::eval`]; the grid scan is the
/// exponential hot loop, so swapping in the cheaper evaluator is the whole point
/// of the TT integration.
fn grid_error<S: Scalar, E: Fn(&[S]) -> S + Sync>(
    eval: E,
    ip: &[Vec<S>],
    full_dims: &[usize],
    tab_flat: &[S],
    col_idx: &[Vec<usize>],
    max_itpl: &[usize],
) -> (f64, usize) {
    let n = ip.len();
    let total = tab_flat.len();

    // Which axes are saturated: a saturated axis masks *all* its coordinates from
    // the greedy pick; an unsaturated axis masks only its already-chosen columns.
    let saturated: Vec<bool> = (0..n).map(|i| col_idx[i].len() >= max_itpl[i]).collect();

    let (max_err, _, greedy_idx) = (0..total)
        .into_par_iter()
        // (err, greedy_err, flat) per point; per-worker scratch avoids
        // re-allocating the multi-index/coordinate buffers on every point.
        .map_init(
            || (vec![0usize; n], vec![S::zero(); n]),
            |(multi, x), flat| {
                flat_to_multi(flat, full_dims, multi);
                for i in 0..n {
                    x[i] = ip[i][multi[i]];
                }
                let err = (tab_flat[flat] - eval(x)).magnitude();
                let err = if err.is_finite() { err } else { 0.0 };

                // Eligible for the greedy pick?
                let eligible = (0..n).all(|i| {
                    if saturated[i] {
                        false
                    } else {
                        !col_idx[i].contains(&multi[i])
                    }
                });
                let greedy_err = if eligible { err } else { f64::NEG_INFINITY };
                (err, greedy_err, flat)
            },
        )
        .reduce(
            || (0.0_f64, f64::NEG_INFINITY, 0usize),
            |a, b| {
                let max_err = a.0.max(b.0);
                let (g, idx) = if b.1 > a.1 { (b.1, b.2) } else { (a.1, a.2) };
                (max_err, g, idx)
            },
        );
    (max_err, greedy_idx)
}

/// As [`grid_error`], but reading model values from a precomputed grid `values`
/// (one per grid point, row-major) instead of evaluating point-by-point — used
/// with the batched TT [`CompressedBarycentric::eval_full_grid`]. No per-point
/// eval, just the error reduction and the greedy-eligibility masking.
fn grid_error_values<S: Scalar>(
    values: &[S],
    full_dims: &[usize],
    tab_flat: &[S],
    col_idx: &[Vec<usize>],
    max_itpl: &[usize],
) -> (f64, usize) {
    let n = full_dims.len();
    let total = tab_flat.len();
    debug_assert_eq!(values.len(), total);
    let saturated: Vec<bool> = (0..n).map(|i| col_idx[i].len() >= max_itpl[i]).collect();
    // Per-axis: which coordinates are *ineligible* (saturated axis, or an
    // already-chosen column). Precomputed so the per-point check is O(n) lookups
    // rather than O(n · cols) membership scans.
    let blocked: Vec<Vec<bool>> = (0..n)
        .map(|i| {
            let mut b = vec![saturated[i]; full_dims[i]];
            for &c in &col_idx[i] {
                b[c] = true;
            }
            b
        })
        .collect();

    let (max_err, _, greedy_idx) = (0..total)
        .into_par_iter()
        .map_init(
            || vec![0usize; n],
            |multi, flat| {
                let err = (tab_flat[flat] - values[flat]).magnitude();
                let err = if err.is_finite() { err } else { 0.0 };
                flat_to_multi(flat, full_dims, multi);
                let eligible = (0..n).all(|i| !blocked[i][multi[i]]);
                let greedy_err = if eligible { err } else { f64::NEG_INFINITY };
                (err, greedy_err, flat)
            },
        )
        .reduce(
            || (0.0_f64, f64::NEG_INFINITY, 0usize),
            |a, b| {
                let max_err = a.0.max(b.0);
                let (g, idx) = if b.1 > a.1 { (b.1, b.2) } else { (a.1, a.2) };
                (max_err, g, idx)
            },
        );
    (max_err, greedy_idx)
}

/// Multivariate-Lawson greedy scan. Like [`grid_error_values`], but the greedy
/// pick maximizes the **Lawson-weighted** residual `lawson_w[flat] · |f − g|`
/// (steering the next support point toward persistent-error regions), while
/// `max_err` remains the true *unweighted* max for the stopping test. Afterwards
/// the weights are updated in place `lawson_w[flat] ← lawson_w[flat] · |f − g|`
/// and renormalized to mean 1 — the n-D IRLS reweighting that drives the support
/// toward an equioscillating (minimax) configuration.
///
/// Construction is unchanged (the recursion still builds the model); only *which*
/// point is promoted differs, so the curse-taming and per-iteration cost hold.
fn grid_error_lawson<S: Scalar>(
    values: &[S],
    full_dims: &[usize],
    tab_flat: &[S],
    col_idx: &[Vec<usize>],
    max_itpl: &[usize],
    lawson_w: &mut [f64],
) -> (f64, usize) {
    let n = full_dims.len();
    let total = tab_flat.len();
    debug_assert_eq!(values.len(), total);
    debug_assert_eq!(lawson_w.len(), total);
    let saturated: Vec<bool> = (0..n).map(|i| col_idx[i].len() >= max_itpl[i]).collect();
    let blocked: Vec<Vec<bool>> = (0..n)
        .map(|i| {
            let mut b = vec![saturated[i]; full_dims[i]];
            for &c in &col_idx[i] {
                b[c] = true;
            }
            b
        })
        .collect();

    // Per-point residual, the true max, and the weighted-residual argmax among
    // eligible points. Sequential (it also reweights in place); the grid scan is
    // memory-bound and this stays O(total).
    let mut multi = vec![0usize; n];
    let mut max_err = 0.0_f64;
    let mut best_w = f64::NEG_INFINITY;
    let mut best_idx = 0usize;
    for flat in 0..total {
        let err = (tab_flat[flat] - values[flat]).magnitude();
        let err = if err.is_finite() { err } else { 0.0 };
        if err > max_err {
            max_err = err;
        }
        flat_to_multi(flat, full_dims, &mut multi);
        let eligible = (0..n).all(|i| !blocked[i][multi[i]]);
        if eligible {
            let w = lawson_w[flat] * err;
            if w > best_w {
                best_w = w;
                best_idx = flat;
            }
        }
        // Damped IRLS reweight toward the current residual. Using √err (rather than
        // err) is the relaxed Lawson update — it converges more slowly but avoids
        // over-concentrating on a single singularity and starving other regions
        // (raw err lets one corner dominate and the off-grid error elsewhere blows
        // up). A small relative floor keeps every region in play.
        lawson_w[flat] *= err.sqrt();
    }

    // Floor weights at a small fraction of the mean so no region is abandoned,
    // then renormalize to mean 1 (neither vanishing nor blowing up).
    let sum: f64 = lawson_w.iter().sum();
    if sum > 0.0 {
        let mean = sum / total as f64;
        let floor = 1e-3 * mean;
        let mut new_sum = 0.0;
        for w in lawson_w.iter_mut() {
            if *w < floor {
                *w = floor;
            }
            new_sum += *w;
        }
        let scale = total as f64 / new_sum;
        for w in lawson_w.iter_mut() {
            *w *= scale;
        }
    } else {
        for w in lawson_w.iter_mut() {
            *w = 1.0;
        }
    }
    (max_err, best_idx)
}

/// Cheaper greedy error hunt over a *subset* of the grid (the [`Candidates`]
/// modes other than `Full`). Returns `(max_err, greedy_idx)` with the same
/// contract as [`grid_error`], but `max_err` is the max over the scanned subset
/// — a lower bound on the true global max (only matters for the stopping test,
/// which then stops slightly later, never falsely early in a way that breaks
/// convergence).
///
/// `seed` is the previous iteration's worst flat index, used as the center of the
/// coordinate sweep (coordinate ascent) and as the refinement center after a
/// subsample draw.
#[allow(clippy::too_many_arguments)]
fn candidate_error<S: Scalar, E: Fn(&[S]) -> S>(
    mode: Candidates,
    eval: &E,
    seed: usize,
    ip: &[Vec<S>],
    full_dims: &[usize],
    tab_flat: &[S],
    col_idx: &[Vec<usize>],
    max_itpl: &[usize],
) -> (f64, usize) {
    let n = full_dims.len();
    let total = tab_flat.len();
    let saturated: Vec<bool> = (0..n).map(|i| col_idx[i].len() >= max_itpl[i]).collect();

    // Residual at a flat grid index, plus its greedy eligibility. Reuses scratch
    // buffers supplied by the caller to stay allocation-light in the hot path.
    let probe = |flat: usize, multi: &mut [usize], x: &mut [S]| -> (f64, f64) {
        flat_to_multi(flat, full_dims, multi);
        for i in 0..n {
            x[i] = ip[i][multi[i]];
        }
        let err = (tab_flat[flat] - eval(x)).magnitude();
        let err = if err.is_finite() { err } else { 0.0 };
        let eligible = (0..n).all(|i| !saturated[i] && !col_idx[i].contains(&multi[i]));
        (err, if eligible { err } else { f64::NEG_INFINITY })
    };

    let mut multi = vec![0usize; n];
    let mut x = vec![S::zero(); n];
    let mut max_err = 0.0_f64;
    let mut best_g = f64::NEG_INFINITY;
    let mut best_idx = seed;

    // For `Subsample`, first probe a fixed pseudo-random subset and re-center the
    // coordinate sweep on its worst eligible point. The draw is deterministic
    // (a fixed odd stride mod `total`) so runs are reproducible.
    let mut center = seed;
    if let Candidates::Subsample(count) = mode {
        let count = count.min(total).max(1);
        // Odd multiplier coprime-ish to most sizes spreads the probes; the offset
        // breaks alignment with the row-major axis strides.
        let stride = 0x9E3779B1usize | 1;
        let mut flat = seed % total;
        for _ in 0..count {
            let (err, g) = probe(flat, &mut multi, &mut x);
            if err > max_err {
                max_err = err;
            }
            if g > best_g {
                best_g = g;
                best_idx = flat;
            }
            flat = (flat.wrapping_add(stride)) % total;
        }
        if best_g > f64::NEG_INFINITY {
            center = best_idx;
        }
    }

    // Coordinate **ascent**: from `center`, repeatedly sweep one axis at a time
    // and *commit* that axis to its residual-maximizing value before moving on, so
    // each axis migrates toward a high-residual (hence un-chosen) coordinate. A
    // few rounds let the coordinates co-adapt; the residual is ~0 at already-chosen
    // columns (the model interpolates there), so ascent walks off them naturally,
    // which is what makes a fully-eligible point reachable. `O(rounds · Σ dims)`.
    let mut base = vec![0usize; n];
    flat_to_multi(center, full_dims, &mut base);
    const ROUNDS: usize = 3;
    for _ in 0..ROUNDS {
        let mut moved = false;
        for axis in 0..n {
            // Best (highest-residual) coordinate for this axis with the others
            // held at the running `base`.
            let mut best_v = base[axis];
            let mut best_v_err = f64::NEG_INFINITY;
            for v in 0..full_dims[axis] {
                base[axis] = v;
                let flat = multi_to_flat(&base, full_dims);
                let (err, g) = probe(flat, &mut multi, &mut x);
                if err > max_err {
                    max_err = err;
                }
                if g > best_g {
                    best_g = g;
                    best_idx = flat;
                }
                // Ascend on the *raw* residual (not eligibility) so the sweep can
                // climb across already-chosen columns toward a free maximum.
                if err > best_v_err {
                    best_v_err = err;
                    best_v = v;
                }
            }
            if best_v != base[axis] {
                moved = true;
            }
            base[axis] = best_v;
        }
        if !moved {
            break;
        }
    }

    (max_err, best_idx)
}

/// Insert `v` into the sorted unique vector `xs` if absent.
fn insert_sorted_unique(xs: &mut Vec<usize>, v: usize) {
    match xs.binary_search(&v) {
        Ok(_) => {}
        Err(pos) => xs.insert(pos, v),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn alg2_recovers_separable_rational() {
        // Order-1 separable rational; greedy should converge to the exact model.
        let f = |x: &[f64]| 1.0 / ((x[0] + 3.0) * (x[1] + 2.0));
        let pts = Points::alternating(&[(-1.0, 1.0); 2], &[8; 2]).unwrap();
        let (model, info) = alg2(&f, &pts, &Alg2Options::default()).unwrap();
        assert!(info.best_error < 1e-8, "best_error = {}", info.best_error);
        for x in [[0.13, -0.4], [0.7, 0.2], [-0.55, 0.9]] {
            assert_relative_eq!(model.eval(&x), f(&x), epsilon = 1e-7);
        }
    }

    #[test]
    fn full_coefficient_method_runs_at_greedy_iterations() {
        // A coupled denominator requires the multivariate construction once the
        // support grid grows beyond one point per axis.  Keep the run short so
        // this specifically exercises repeated full coefficient builds.
        let f = |x: &[f64]| 1.0 / (3.0 + x[0] + 0.5 * x[1] + x[0] * x[1]);
        let pts = Points::alternating(&[(-0.8, 0.8); 2], &[8; 2]).unwrap();
        let opt = Alg2Options {
            method: CoefficientMethod::Full,
            max_iter: Some(6),
            patience: None,
            ..Alg2Options::default()
        };
        let (model, info) = alg2(&f, &pts, &opt).unwrap();
        assert!(info.error_history.len() >= 2);
        assert!(info.best_error < 1e-8, "best_error = {}", info.best_error);
        for x in [[0.13, -0.27], [-0.51, 0.42], [0.61, 0.09]] {
            assert_relative_eq!(model.eval(&x), f(&x), epsilon = 1e-7);
        }
    }

    #[test]
    fn alg2_recovers_3d_separable_rational() {
        // Order-1 separable rational in 3 variables — the recursion's natural
        // regime; greedy should drive the error to ~machine precision.
        let f = |x: &[f64]| 1.0 / ((x[0] + 3.0) * (x[1] + 2.0) * (x[2] + 4.0));
        let pts = Points::alternating(&[(-1.0, 1.0); 3], &[8; 3]).unwrap();
        let (model, info) = alg2(&f, &pts, &Alg2Options::default()).unwrap();
        assert!(info.best_error < 1e-7, "best_error = {}", info.best_error);
        let x = [0.1, -0.3, 0.5];
        assert_relative_eq!(model.eval(&x), f(&x), epsilon = 1e-7);
    }

    #[test]
    fn patience_stops_early_on_stagnation() {
        // A discontinuous sign product stagnates initially. Patience must cut
        // off early and return the best model from the visited iterations.
        // An unguarded run can improve later (in particular after reaching a
        // wide Loewner with a true right null space); patience cannot predict
        // that future improvement. A wide `max_itpl` ensures patience, not
        // saturation, binds.
        let f = |x: &[f64]| x[0].signum() * x[1].signum();
        let pts = Points::alternating(&[(-1.0, 1.0); 2], &[20; 2]).unwrap();

        let patient = Alg2Options {
            patience: Some(3),
            max_iter: Some(30),
            max_itpl: Some(vec![15, 15]),
            ..Alg2Options::default()
        };
        let no_guard = Alg2Options {
            patience: None,
            ..patient.clone()
        };

        let (m_p, i_patient) = alg2(&f, &pts, &patient).unwrap();
        let (_, i_full) = alg2(&f, &pts, &no_guard).unwrap();

        // The guard stopped much earlier: best at iter 1, then `patience` stalls.
        assert_eq!(i_patient.best_iter, 1);
        assert_eq!(i_patient.error_history.len(), i_patient.best_iter + 3);
        assert!(
            i_patient.error_history.len() < i_full.error_history.len(),
            "patient {} not fewer than full {}",
            i_patient.error_history.len(),
            i_full.error_history.len()
        );
        // Compare identical search horizons: both must return the best seen.
        let truncated = Alg2Options {
            max_iter: Some(i_patient.error_history.len()),
            ..no_guard
        };
        let (m_t, i_truncated) = alg2(&f, &pts, &truncated).unwrap();
        assert!(i_full.best_error <= i_patient.best_error);
        assert_relative_eq!(
            i_patient.best_error,
            i_truncated.best_error,
            epsilon = 1e-12
        );
        assert_eq!(m_p.c.len(), m_t.c.len());
        for (a, b) in m_p.c.iter().zip(&m_t.c) {
            assert_relative_eq!(a, b, epsilon = 1e-12);
        }
    }

    #[test]
    fn coordinate_search_converges_separable() {
        // The coordinate (axis-wise) candidate search scans only O(Σ dims) points
        // per iteration, not the full grid. On a separable target the residual is
        // axis-aligned, so coordinate ascent should still drive the error down.
        // The contract is *convergence*, not bit-parity with the Full scan.
        let f = |x: &[f64]| 1.0 / ((x[0] + 3.0) * (x[1] + 2.0) * (x[2] + 4.0));
        let pts = Points::alternating(&[(-1.0, 1.0); 3], &[8; 3]).unwrap();
        let opt = Alg2Options {
            candidates: Candidates::Coordinate,
            ..Alg2Options::default()
        };
        let (model, info) = alg2(&f, &pts, &opt).unwrap();
        assert!(info.best_error < 1e-7, "best_error = {}", info.best_error);
        for x in [[0.1, -0.3, 0.5], [0.7, 0.2, -0.4]] {
            assert_relative_eq!(model.eval(&x), f(&x), epsilon = 1e-6);
        }
    }

    #[test]
    fn subsample_search_converges_separable() {
        // The subsample search probes a fixed pseudo-random subset then refines
        // with a coordinate sweep. It must still converge on a separable target.
        let f = |x: &[f64]| 1.0 / ((x[0] + 3.0) * (x[1] + 2.0));
        let pts = Points::alternating(&[(-1.0, 1.0); 2], &[8; 2]).unwrap();
        let opt = Alg2Options {
            candidates: Candidates::Subsample(16),
            ..Alg2Options::default()
        };
        let (model, info) = alg2(&f, &pts, &opt).unwrap();
        assert!(info.best_error < 1e-7, "best_error = {}", info.best_error);
        for x in [[0.13, -0.4], [0.7, 0.2]] {
            assert_relative_eq!(model.eval(&x), f(&x), epsilon = 1e-6);
        }
    }

    #[test]
    fn coordinate_search_far_cheaper_than_full_high_dim() {
        // Sanity that the cheaper mode is actually usable where Full's point count
        // is the wall: a 6-D grid (6^6 ≈ 4.7e4 points here, but the *scan* is what
        // grows). Coordinate scans O(Σ dims) per iteration. We only assert it runs
        // and converges; timing parity is exercised in examples/alg2_scale.rs.
        let f = |x: &[f64]| -> f64 { x.iter().map(|&v| 1.0 / (v + 2.5)).product() };
        let pts = Points::alternating(&[(-1.0, 1.0); 6], &[5; 6]).unwrap();
        let opt = Alg2Options {
            candidates: Candidates::Coordinate,
            ..Alg2Options::default()
        };
        let (model, info) = alg2(&f, &pts, &opt).unwrap();
        assert!(info.best_error < 1e-6, "best_error = {}", info.best_error);
        let x = [0.2, -0.3, 0.1, 0.4, -0.2, 0.3];
        assert_relative_eq!(model.eval(&x), f(&x), epsilon = 1e-5);
    }

    #[test]
    fn tt_accelerated_scan_matches_dense_scan() {
        // A 5-D separable rational on a grid large enough to trip the TT path
        // (grid_size · reduced_coeff_size > TT_GRID_THRESHOLD). The greedy result
        // must be identical whether the grid error scan uses the dense or the
        // factorized TT evaluator — the TT path only changes *how* the error is
        // computed, never *what* model is selected.
        let f = |x: &[f64]| -> f64 { x.iter().map(|&v| 1.0 / ((v + 2.0) * (v + 3.0))).product() };
        let pts = Points::alternating(&[(-1.0, 1.0); 5], &[6; 5]).unwrap();

        let dense_opt = Alg2Options {
            compress: None,
            ..Alg2Options::default()
        };
        let tt_opt = Alg2Options {
            compress: Some(1e-12),
            ..Alg2Options::default()
        };

        let (m_dense, i_dense) = alg2(&f, &pts, &dense_opt).unwrap();
        let (m_tt, i_tt) = alg2(&f, &pts, &tt_opt).unwrap();

        assert_eq!(i_dense.order, i_tt.order);
        assert_eq!(i_dense.best_iter, i_tt.best_iter);
        assert_relative_eq!(i_dense.best_error, i_tt.best_error, epsilon = 1e-12);
        // Same coefficients => identical model.
        assert_eq!(m_dense.c.len(), m_tt.c.len());
        for (a, b) in m_dense.c.iter().zip(&m_tt.c) {
            assert_relative_eq!(a, b, epsilon = 1e-10);
        }
    }

    /// True max error of a 2-D model over a dense interior sweep (off the grid).
    #[cfg(test)]
    fn sweep_max_2d(m: &Barycentric, f: &dyn Fn(&[f64]) -> f64) -> f64 {
        let mut mx = 0.0_f64;
        let g = 200;
        for i in 1..g {
            for j in 1..g {
                let x = [
                    -1.0 + 2.0 * i as f64 / g as f64,
                    -1.0 + 2.0 * j as f64 / g as f64,
                ];
                mx = mx.max((m.eval(&x) - f(&x)).abs());
            }
        }
        mx
    }

    #[test]
    fn multivariate_lawson_improves_max_norm_on_peaked_target() {
        // A sharp 2-D Gaussian: plain max-residual greedy leaves a large off-grid
        // error region; the multivariate Lawson reweighting (minimax-oriented point
        // placement) targets it and substantially lowers the true max error. This
        // is the n-D lift of AAA-Lawson — the payoff is in the max-norm, not the
        // on-grid scan error.
        let f = |x: &[f64]| (-8.0 * (x[0] * x[0] + x[1] * x[1])).exp();
        let pts = Points::alternating(&[(-1.0, 1.0); 2], &[16, 16]).unwrap();

        let plain = Alg2Options {
            lawson: None,
            ..Alg2Options::default()
        };
        let lawson = Alg2Options {
            lawson: Some(1),
            ..Alg2Options::default()
        };
        let (m_p, _) = alg2(&f, &pts, &plain).unwrap();
        let (m_l, _) = alg2(&f, &pts, &lawson).unwrap();

        let max_p = sweep_max_2d(&m_p, &f);
        let max_l = sweep_max_2d(&m_l, &f);
        assert!(max_l.is_finite() && max_p.is_finite());
        // Lawson should be clearly better on this peaked target (the off-grid peak
        // is exactly where plain greedy under-samples). On the dense sweep the win
        // is large; require at least a solid improvement.
        assert!(
            max_l < 0.7 * max_p,
            "Lawson max {max_l:e} not meaningfully < plain max {max_p:e}"
        );
    }

    #[test]
    fn multivariate_lawson_does_not_hurt_a_rational() {
        // On a genuinely rational target both reach machine precision: the Lawson
        // reweighting must not degrade the easy case (the IRLS weights collapse to
        // uniform once the residual is ~0).
        let f = |x: &[f64]| 1.0 / (1.0 + 25.0 * (x[0] * x[0] + x[1] * x[1]));
        let pts = Points::alternating(&[(-1.0, 1.0); 2], &[14, 14]).unwrap();
        let lawson = Alg2Options {
            lawson: Some(1),
            ..Alg2Options::default()
        };
        let (m, info) = alg2(&f, &pts, &lawson).unwrap();
        assert!(
            info.best_error < 1e-9,
            "rational not recovered: {}",
            info.best_error
        );
        assert!(sweep_max_2d(&m, &f) < 1e-7);
    }
}
