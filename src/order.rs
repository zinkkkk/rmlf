//! Per-variable order (degree) detection. Replaces `mlf.compute_order`.
//!
//! Along each axis, build a 1-D Loewner matrix for each sampled frozen context,
//! normalize its singular values by its own leading value, and take the maximum
//! rank across contexts. This follows MATLAB's per-context order semantics.
//!
//! Stacking all contexts instead would measure the union of their column spaces
//! and over-count coupled functions whose slices have different denominators.
//! A problem-global scale floor treats a fully constant axis as order zero.
//! Contexts can be subsampled with `max_contexts`; `max_order` and `gap_stop`
//! provide optional caps and stabilization for non-rational targets.
//!
//! Real and complex data share the same algorithm; spectra and thresholds are
//! always real. Strided axis lines are gathered directly from the data tensor.

use crate::points::Points;
use crate::tensor::{View, axis_line};

use crate::scalar::Scalar;
use faer::Mat;
use rayon::prelude::*;

/// Options controlling order detection.
#[derive(Debug, Clone)]
pub struct OrderOptions {
    /// Normalized singular-value threshold; values `> tol` count toward rank.
    pub tol: f64,
    /// Optional per-axis cap on the detected order (`min` applied).
    pub max_order: Option<Vec<usize>>,
    /// Cap on frozen-context slices tested per axis (`None` = all). Contexts
    /// are sampled evenly across the full set, like MATLAB's `ord_N`. Each
    /// selected context is factored separately and the maximum rank is kept.
    pub max_contexts: Option<usize>,
    /// **Non-rational stabilization (opt-in).** When `Some(g)`, an axis whose
    /// singular values decay *gradually* — no spectral gap below `g` anywhere up
    /// to the `σ>tol` count — is truncated at its **largest-gap elbow** instead of
    /// the slowly-decaying tail. A rational target has a sharp gap at its true
    /// order, so this leaves it untouched; a non-rational target (e.g. `|x₁−x₂|³`)
    /// otherwise runs the order up to where the rational interpolant goes
    /// Runge-unstable (RMSE blows up). Stopping at the elbow is the most stable
    /// truncation — the reference achieves the same effect with a loose `ord_tol`.
    /// `None` keeps the pure `σ>tol` count (exact-rational behavior).
    pub gap_stop: Option<f64>,
}

impl Default for OrderOptions {
    fn default() -> Self {
        OrderOptions {
            // Matches the MATLAB reference `alg1.m` default `ord_tol = 1e-10`.
            tol: 1e-10,
            max_order: None,
            max_contexts: Some(5),
            gap_stop: None,
        }
    }
}

/// Singular values of `m`, descending. Empty on SVD failure or empty matrix.
fn singular_values<S: Scalar>(m: &Mat<S>) -> Vec<f64> {
    match m.thin_svd() {
        Ok(svd) => {
            let s = svd.S().column_vector();
            (0..s.nrows()).map(|i| s[i].real()).collect()
        }
        Err(_) => Vec::new(),
    }
}

/// Per-axis order evidence: the normalized singular-value spectrum of the *single
/// frozen context* that attains the **maximum rank** along axis `k`, plus the
/// largest absolute leading singular value over all contexts (for the global
/// constant-axis floor).
///
/// This is the MATLAB `compute_order` semantics — build one 1-D Loewner per
/// context, normalize by *its own* σ₀, count σ > tol, and take the **max** rank
/// over contexts — not the stacked mode-`k` unfolding we used before. Stacking
/// measures the *union* of every context's column space (the true mode-`k`
/// multilinear rank), which **over-counts the per-axis order**: for a target that
/// is order-`d` rational along each axis but whose slices span different `d`-D
/// subspaces (e.g. `x1²/(x1²+x2²)`), stacking reports rank ≫ `d` while the correct
/// per-axis order is `d` (validated against Octave: CAS 56 stacked → 7, MATLAB and
/// this max-per-context → 2). The degenerate-slice robustness (§6c) is preserved
/// differently: a near-constant context's Loewner is ~0, its normalized spectrum
/// after its own σ₀ is meaningless, but it contributes a *small* rank, and the
/// **max** over contexts is dominated by a genuine (non-degenerate) context — and
/// the global-scale floor in the caller still zeroes a fully-constant axis.
fn max_context_spectrum<S: Scalar>(
    tab: &View<'_, S>,
    k: usize,
    lambda: &[S],
    mu: &[S],
    contexts: &[Vec<usize>],
    tol: f64,
) -> (Vec<f64>, f64) {
    let (kk, qk) = (lambda.len(), mu.len());
    if kk == 0 || qk == 0 || contexts.is_empty() {
        return (Vec::new(), 0.0);
    }
    // Each context: build its qk × kk Loewner, SVD, keep the spectrum and its top
    // singular value. Parallel over contexts.
    let per_context: Vec<(Vec<f64>, f64)> = contexts
        .par_iter()
        .map(|ctx| {
            let line = axis_line(tab, k, ctx);
            let l = Mat::from_fn(qk, kk, |i, j| {
                (line[kk + i] - line[j]) / (mu[i] - lambda[j])
            });
            let s = singular_values(&l);
            let top = s.first().copied().unwrap_or(0.0);
            (s, top)
        })
        .collect();

    // Largest leading singular value over contexts — the axis's absolute scale.
    let axis_top = per_context.iter().map(|(_, t)| *t).fold(0.0_f64, f64::max);

    // Pick the context of maximum rank (σ_i/σ_0 > tol), tie-broken by the larger
    // absolute scale so a genuine context beats a degenerate (noise) one.
    let best = per_context
        .iter()
        .max_by(|a, b| {
            let ra = rank_above(&a.0, tol);
            let rb = rank_above(&b.0, tol);
            ra.cmp(&rb)
                .then(a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        })
        .map(|(s, _)| s.clone())
        .unwrap_or_default();
    (best, axis_top)
}

/// Count of singular values exceeding `tol` after normalizing by the largest.
fn rank_above(sigma: &[f64], tol: f64) -> usize {
    match sigma.first().copied() {
        Some(s0) if s0 > 0.0 => sigma.iter().filter(|&&v| v / s0 > tol).count(),
        _ => 0,
    }
}

/// Stabilized order for a *non-rational* (gradually decaying) spectrum.
///
/// `raw_rank` is the `σ>tol` count. The decision hinges on the gap **at**
/// `raw_rank` (`σ[raw_rank]/σ[raw_rank-1]`):
/// * **Sharp** (`< sharp`) — a genuine rank cliff right at the count: the axis is
///   rational at order `raw_rank`. Measured: rational cases drop `~1e-13..1e-16`
///   there, far below any `sharp`. Return `raw_rank` unchanged (no regression).
/// * **Gradual** (`≥ sharp`, e.g. `~0.1` for `|x₁−x₂|³`) — no cliff: the count is
///   chasing a slowly-decaying tail that drives the rational interpolant
///   Runge-unstable. Return the **steepest-drop elbow** in `1..raw_rank`, the most
///   stable truncation, instead.
///
/// So a rational axis is never touched; only a confirmed non-rational tail is cut.
fn gap_elbow(sigma: &[f64], raw_rank: usize, sharp: f64) -> usize {
    let hi = raw_rank.min(sigma.len());
    if hi <= 1 {
        return raw_rank;
    }
    // Gap exactly at the σ>tol count. If there is no σ past the count (exact rank
    // drop) the tail is empty ⇒ rational ⇒ keep raw_rank.
    let tail_gap = if sigma.len() > raw_rank && sigma[raw_rank - 1] > 0.0 {
        sigma[raw_rank] / sigma[raw_rank - 1]
    } else {
        0.0
    };
    if tail_gap < sharp {
        // A real cliff at raw_rank ⇒ rational ⇒ trust the count.
        return raw_rank;
    }
    // Gradual decay: stop at the steepest relative drop (the elbow).
    let mut best_pos = 1usize;
    let mut best_ratio = f64::INFINITY;
    for i in 1..hi {
        let prev = sigma[i - 1];
        if prev <= 0.0 {
            break;
        }
        let ratio = sigma[i] / prev;
        if ratio < best_ratio {
            best_ratio = ratio;
            best_pos = i;
        }
    }
    best_pos.max(1)
}

/// Per-axis diagnostics reusing the context SVDs from order detection.
#[derive(Debug, Clone)]
pub struct OrderReport {
    /// Detected order per axis (the capped rank — what `compute_order` returns).
    pub order: Vec<usize>,
    /// Maximum per-context rank before order caps. Rank one along every axis
    /// is used by the legacy real separability heuristic, but does not by
    /// itself prove separability for a coupled function.
    pub rank: Vec<usize>,
    /// Spectral gap ratio per axis: `sigma[order] / sigma[order-1]`, i.e. how
    /// sharply the spectrum drops just past the detected order. Small ⇒ a clear
    /// gap ⇒ confidently rational; near 1 ⇒ gradual decay ⇒ leave the axis open.
    /// `0.0` when there is no singular value past the detected order (exact rank
    /// drop), `1.0` for a degenerate/empty axis.
    pub gap: Vec<f64>,
}

/// Detect the order along every axis. Returns `ord[k]` for `k in 0..n`.
///
/// `tab` is the full data tensor sampled on the combined `column ++ row` grid.
pub fn compute_order<S: Scalar>(
    points: &Points<S>,
    tab: View<'_, S>,
    opt: &OrderOptions,
) -> Vec<usize> {
    compute_order_report(points, tab, opt).order
}

/// As [`compute_order`], but also returns the per-axis [`OrderReport`]
/// diagnostics (unfolding rank and spectral gap) from the *same* SVD pass.
pub fn compute_order_report<S: Scalar>(
    points: &Points<S>,
    tab: View<'_, S>,
    opt: &OrderOptions,
) -> OrderReport {
    let n = points.ndim();
    let dims: Vec<usize> = (0..n)
        .map(|l| points.column[l].len() + points.row[l].len())
        .collect();

    // Per axis (independent → parallel): the singular-value spectrum of the
    // max-rank frozen context, plus that axis's absolute leading singular value.
    let per_axis: Vec<(Vec<f64>, f64)> = (0..n)
        .into_par_iter()
        .map(|k| {
            let (lambda, mu) = (&points.column[k], &points.row[k]);
            let contexts = axis_contexts(&dims, k, opt.max_contexts);
            max_context_spectrum(&tab, k, lambda, mu, &contexts, opt.tol)
        })
        .collect();
    let svals: Vec<&Vec<f64>> = per_axis.iter().map(|(s, _)| s).collect();

    // Problem-global Loewner scale: the largest leading singular value over all
    // axes. An axis whose own scale is negligible against this carries no
    // dependence (order 0) — without this floor, a constant axis's noise would
    // self-normalize to spurious rank.
    let global_scale = per_axis.iter().map(|(_, t)| *t).fold(0.0_f64, f64::max);

    let mut order = Vec::with_capacity(n);
    let mut rank = Vec::with_capacity(n);
    let mut gap = Vec::with_capacity(n);

    for k in 0..n {
        let axis_scale = per_axis[k].1;
        let raw_rank = if global_scale == 0.0 || axis_scale <= opt.tol * global_scale {
            // Degenerate / constant axis (negligible against the problem scale).
            0
        } else {
            // Significant singular values of the max-rank context, normalized by
            // *that context's own* leading singular value (MATLAB's `sig/sig(1)`).
            rank_above(svals[k], opt.tol)
        };

        // Non-rational stabilization: if the spectrum decays *gradually* (no sharp
        // gap up to `raw_rank`), the σ>tol count chases the slowly-decaying tail to
        // an order where the rational interpolant goes Runge-unstable. Truncate at
        // the largest-gap elbow instead — the most stable rank. A genuinely
        // rational axis has a sharp gap at its true order, so its elbow == raw_rank
        // and nothing changes.
        let stabilized = match opt.gap_stop {
            Some(g) if raw_rank > 1 => gap_elbow(svals[k], raw_rank, g),
            _ => raw_rank,
        };

        // Cannot exceed the number of available column points; if it reaches
        // it, drop by one (mirrors MATLAB's "not enough data" guard).
        let k_pts = points.column[k].len();
        let mut o = stabilized.min(k_pts);
        if let Some(caps) = &opt.max_order {
            o = o.min(caps[k]);
        }
        if o == k_pts && k_pts > 0 {
            o = k_pts - 1;
        }

        // Spectral gap just past the detected order: sigma[order] / sigma[order-1].
        // A sharp drop (small ratio) means the rank is unambiguous.
        let g = if o == 0 || svals[k].len() <= o {
            // No data past the detected order, or a degenerate axis.
            if svals[k].is_empty() { 1.0 } else { 0.0 }
        } else {
            svals[k][o] / svals[k][o - 1]
        };

        order.push(o);
        rank.push(raw_rank);
        gap.push(g);
    }

    OrderReport { order, rank, gap }
}

/// Enumerate the frozen-context multi-indices for axis `k`: every combination of
/// the *other* axes' indices over their full dimension, optionally subsampled to
/// at most `cap` evenly spaced contexts. The `k`-th entry is a placeholder (0).
fn axis_contexts(dims: &[usize], k: usize, cap: Option<usize>) -> Vec<Vec<usize>> {
    let other: Vec<usize> = (0..dims.len()).filter(|&l| l != k).collect();
    let total: usize = other.iter().map(|&l| dims[l]).product::<usize>().max(1);

    let pick: Vec<usize> = match cap {
        Some(c) if c < total => (0..c).map(|i| i * (total - 1) / (c - 1).max(1)).collect(),
        _ => (0..total).collect(),
    };

    pick.into_iter()
        .map(|mut flat| {
            let mut ctx = vec![0usize; dims.len()];
            for &l in other.iter().rev() {
                ctx[l] = flat % dims[l];
                flat /= dims[l];
            }
            ctx
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::sample;

    #[test]
    fn detects_order_of_separable_rational() {
        // f = 1/((s1+3)(s2+2)(s3+4)) is order 1 along each axis. Oversample with
        // 6 points/axis; detection should report order 1 everywhere.
        let f = |x: &[f64]| 1.0 / ((x[0] + 3.0) * (x[1] + 2.0) * (x[2] + 4.0));
        let col: Vec<f64> = (0..6).map(|i| i as f64).collect();
        let row: Vec<f64> = (0..6).map(|i| i as f64 + 0.5).collect();
        let pts = Points::new(
            vec![col.clone(), col.clone(), col.clone()],
            vec![row.clone(), row.clone(), row.clone()],
        )
        .unwrap();
        let tab = sample(&f, &pts);
        let opt = OrderOptions {
            tol: 1e-9,
            max_order: None,
            max_contexts: None,
            gap_stop: None,
        };
        let ord = compute_order(&pts, tab.view(), &opt);
        assert_eq!(ord, vec![1, 1, 1]);
    }

    #[test]
    fn detects_polynomial_orders() {
        // f = s1^2 * s2 : order 2 in s1, order 1 in s2.
        let f = |x: &[f64]| x[0] * x[0] * x[1];
        let col: Vec<f64> = (0..6).map(|i| i as f64).collect();
        let row: Vec<f64> = (0..6).map(|i| i as f64 + 0.5).collect();
        let pts = Points::new(
            vec![col.clone(), col.clone()],
            vec![row.clone(), row.clone()],
        )
        .unwrap();
        let tab = sample(&f, &pts);
        let opt = OrderOptions {
            tol: 1e-9,
            max_order: None,
            max_contexts: None,
            gap_stop: None,
        };
        let ord = compute_order(&pts, tab.view(), &opt);
        assert_eq!(ord, vec![2, 1]);
    }

    #[test]
    fn constant_axis_is_order_zero() {
        // f depends only on x1; x2 is constant -> order [1, 0]. The mode-2
        // unfolding is all (floating-point) zero; the problem-global scale floor
        // must keep its order at 0 rather than self-normalizing noise to rank.
        let f = |x: &[f64]| x[0];
        let col: Vec<f64> = (0..6).map(|i| i as f64).collect();
        let row: Vec<f64> = (0..6).map(|i| i as f64 + 0.5).collect();
        let pts = Points::new(
            vec![col.clone(), col.clone()],
            vec![row.clone(), row.clone()],
        )
        .unwrap();
        let tab = sample(&f, &pts);
        let opt = OrderOptions {
            tol: 1e-9,
            max_order: None,
            max_contexts: None,
            gap_stop: None,
        };
        let ord = compute_order(&pts, tab.view(), &opt);
        assert_eq!(ord, vec![1, 0]);
    }

    #[test]
    fn report_flags_separable_and_sharp_gap() {
        // A clean order-1 separable rational: every axis has unfolding rank 1
        // (separable, #6) and a sharp spectral gap (confidently rational, #5).
        let f = |x: &[f64]| 1.0 / ((x[0] + 3.0) * (x[1] + 2.0) * (x[2] + 4.0));
        let col: Vec<f64> = (0..6).map(|i| i as f64).collect();
        let row: Vec<f64> = (0..6).map(|i| i as f64 + 0.5).collect();
        let pts = Points::new(
            vec![col.clone(), col.clone(), col.clone()],
            vec![row.clone(), row.clone(), row.clone()],
        )
        .unwrap();
        let tab = sample(&f, &pts);
        let opt = OrderOptions {
            tol: 1e-9,
            max_order: None,
            max_contexts: None,
            gap_stop: None,
        };
        let rep = compute_order_report(&pts, tab.view(), &opt);
        assert_eq!(rep.order, vec![1, 1, 1]);
        assert_eq!(rep.rank, vec![1, 1, 1]);
        // Exact rational ⇒ singular values past the order are at noise level,
        // so the gap ratio is tiny (well under the alg2 RATIONAL_GAP = 1e-3).
        for g in &rep.gap {
            assert!(*g < 1e-3, "gap {g} not sharp");
        }
    }

    #[test]
    fn degenerate_context_does_not_inflate_order() {
        // f = x1 * x2 is order 1 in each. With a *row* point at x2 = 0, the
        // context that freezes x2 at 0 makes the x1-line identically zero — a
        // degenerate slice. The old per-matrix self-normalization (§6c) inflated
        // such a slice to full rank; the stacked mode-k unfolding contributes
        // near-zero rows, so the order stays correct at 1. Exhaustive contexts.
        let f = |x: &[f64]| x[0] * x[1];
        let pts = Points::new(
            vec![vec![1.0, 2.0, 3.0], vec![1.0, 2.0, 3.0]],
            vec![vec![0.5, 1.5, 2.5], vec![0.0, 1.5, 2.5]], // x2 row includes 0
        )
        .unwrap();
        let tab = sample(&f, &pts);
        let opt = OrderOptions {
            tol: 1e-9,
            max_order: None,
            max_contexts: None, // test *all* contexts, including the degenerate one
            gap_stop: None,
        };
        let ord = compute_order(&pts, tab.view(), &opt);
        assert_eq!(ord, vec![1, 1]);
    }
}
