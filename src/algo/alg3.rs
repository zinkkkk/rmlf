//! Algorithm 3: AAA-style **resampling refinement**. The grid-growing counterpart
//! of [`crate::algo::alg2()`].
//!
//! `alg1`/`alg2` both work on a **fixed**, pre-sampled grid: alg1 fits once,
//! alg2 greedily promotes points *within* that grid. Neither adds resolution the
//! original grid never had. `alg3` closes that gap for the closure-driven case:
//! each round it fits the current grid, keeps the support nodes the fit selected,
//! **refines** those node arrays (inserting interior abscissae where the model
//! must resolve more — [`crate::points::refine`]), **resamples** the target on the
//! finer grid, and repeats. The kept set carries from round to round, so the grid
//! follows the function instead of being guessed up front.
//!
//! ```text
//!   nodes_0 ─fit─▶ model_0 ─keep support─▶ refine ─resample─▶ nodes_1 ─fit─▶ …
//! ```
//!
//! This needs a callable `f` (it resamples), so unlike alg1/alg2 there is no
//! data-tensor entry point. It is most useful when the right resolution is
//! unknown a priori and oversampling the whole domain up front is wasteful.

use crate::error::Result;
use crate::eval::Barycentric;
use crate::points::{Points, refine};
use crate::source::TargetFn;

use super::alg2::{Alg2Options, alg2};

/// Options for [`alg3()`].
#[derive(Debug, Clone)]
pub struct Alg3Options {
    /// Per-axis domain `[lo, hi]`. Refinement keeps inserted points inside these
    /// bounds (it only ever subdivides existing gaps).
    pub bounds: Vec<(f64, f64)>,
    /// Starting number of abscissae per axis for round 0 (Chebyshev-placed).
    pub start_nip: usize,
    /// Interior points inserted per gap each refinement round (the `p` of
    /// [`refine`]). Larger ⇒ faster grid growth, more samples per round.
    pub refine_per_gap: usize,
    /// Max refinement rounds.
    pub max_rounds: usize,
    /// Relative max-error stopping tolerance, measured on a fresh validation set
    /// (`max|f - g| / max|f| < tol`).
    pub tol: f64,
    /// Stop after this many consecutive rounds without a new best validation
    /// error. Validation error can wobble for a round or two before a deeper
    /// refinement pays off, so this is deliberately forgiving; set it lower only
    /// when rounds are expensive and you want to bail early. `None` runs to
    /// `max_rounds`.
    pub patience: Option<usize>,
    /// Hard ceiling on the combined sampling-grid size `∏(k_l + q_l)`. Refinement
    /// roughly *doubles* the points per axis each round, so the grid grows like
    /// `(start·2^round)^n` — multiplicative in `n` and explosive past a few rounds.
    /// Once the *next* round's grid would exceed this, refinement stops (the best
    /// model so far is returned). Without it a 3-D case can balloon to billions of
    /// samples and exhaust RAM. Default ~2M points (a few tens of MB).
    pub max_grid: usize,
    /// Options for the per-round inner fit (each round calls [`alg2()`]).
    pub inner: Alg2Options,
}

impl Alg3Options {
    /// Defaults for the given per-axis `bounds` (everything else conventional).
    pub fn new(bounds: Vec<(f64, f64)>) -> Self {
        Alg3Options {
            bounds,
            start_nip: 6,
            refine_per_gap: 1,
            max_rounds: 10,
            tol: 1e-9,
            patience: Some(4),
            max_grid: 2_000_000,
            inner: Alg2Options::default(),
        }
    }
}

/// Information returned alongside the refined model.
#[derive(Debug, Clone)]
pub struct Alg3Info {
    /// Refinement rounds actually run.
    pub rounds: usize,
    /// Relative validation error after each round (length = `rounds`).
    pub error_history: Vec<f64>,
    /// Relative validation error of the returned (best) model.
    pub best_error: f64,
    /// Per-axis abscissa count of the final grid (column ++ row).
    pub final_nip: Vec<usize>,
}

/// Build an approximation of `f` by AAA-style resampling refinement.
///
/// Round 0 fits a Chebyshev grid of `start_nip` points per axis; each later round
/// refines the previous round's *kept support nodes* and resamples. Returns the
/// best model seen (lowest validation error).
pub fn alg3<F: TargetFn + Sync>(f: &F, opt: &Alg3Options) -> Result<(Barycentric, Alg3Info)> {
    let n = opt.bounds.len();
    let nips = vec![opt.start_nip.max(2); n];
    let mut points = Points::chebyshev(&opt.bounds, &nips)?;

    let mut best: Option<(Barycentric, f64)> = None;
    let mut error_history = Vec::new();
    let mut rounds = 0usize;
    let mut final_nip = points.column_sizes();
    let mut no_improve = 0usize;
    let patience = opt.patience.map(|p| p.max(1));

    for _ in 0..opt.max_rounds.max(1) {
        rounds += 1;
        let (model, _) = alg2(f, &points, &opt.inner)?;

        // Validation error on a fresh quasi-random interior set (independent of
        // the fit grid, so it measures genuine generalization, not interpolation).
        let rel = validation_error(f, &model, &opt.bounds);
        error_history.push(rel);

        let improved = best.as_ref().map(|(_, e)| rel < *e).unwrap_or(true);
        if improved {
            best = Some((model.clone(), rel));
            no_improve = 0;
        } else {
            no_improve += 1;
        }
        final_nip = combined_sizes(&points);

        if rel <= opt.tol {
            break;
        }
        // Stop refining once it stops paying off (over-resolution degrades the
        // fit); the best model so far is retained.
        if let Some(p) = patience
            && no_improve >= p
        {
            break;
        }

        // Refine and rebuild the per-axis grid. The kept support nodes the fit
        // selected (`model.lambda`) seed each axis; the previous grid's combined
        // abscissae are folded in so resolution never shrinks, then every gap is
        // subdivided. This is the "feed back the kept nodes, refine, resample"
        // step. `refine` only subdivides existing gaps, so points stay in bounds.
        let mut new_column: Vec<Vec<f64>> = Vec::with_capacity(n);
        let mut new_row: Vec<Vec<f64>> = Vec::with_capacity(n);
        for (axis, lambda) in model.lambda.iter().enumerate() {
            // Seed = kept column support ∪ the round's full sampling abscissae, so
            // the grid is monotone-refining even when the fit kept few columns.
            let mut dense = lambda.clone();
            dense.extend(points.combined_axis(axis));
            dense.sort_by(|a, b| a.partial_cmp(b).expect("alg3: NaN node"));
            dense.dedup_by(|a, b| (*a - *b).abs() <= 1e-12 * (1.0 + a.abs()));
            // Subdivide every gap, then merge the new interior points in.
            let extra = refine(&dense, opt.refine_per_gap.max(1));
            dense.extend(extra);
            dense.sort_by(|a, b| a.partial_cmp(b).expect("alg3: NaN node"));
            dense.dedup_by(|a, b| (*a - *b).abs() <= 1e-12 * (1.0 + a.abs()));
            // Alternating column/row split (disjoint, both clustered like `dense`).
            let m = dense.len();
            new_column.push((1..m).step_by(2).map(|i| dense[i]).collect());
            new_row.push((0..m).step_by(2).map(|i| dense[i]).collect());
        }
        // Refinement is multiplicative in n; stop before the next grid explodes.
        let next_grid: u128 = new_column
            .iter()
            .zip(&new_row)
            .map(|(c, r)| (c.len() + r.len()) as u128)
            .product();
        if next_grid > opt.max_grid as u128 {
            break;
        }
        points = Points::new(new_column, new_row)?;
    }

    let (model, best_error) = best.expect("alg3 runs at least one round");
    Ok((
        model,
        Alg3Info {
            rounds,
            error_history,
            best_error,
            final_nip,
        },
    ))
}

/// Combined `column + row` abscissa count per axis.
fn combined_sizes(points: &Points) -> Vec<usize> {
    points
        .column_sizes()
        .iter()
        .zip(points.row_sizes())
        .map(|(c, r)| c + r)
        .collect()
}

const N_VALID: usize = 2000;

/// Relative max validation error of `model` vs `f` over `N_VALID` quasi-random
/// interior points of the box `bounds`. Deterministic (fixed-seed LCG) so rounds
/// are comparable; samples the interior 80% to stay clear of node singularities.
fn validation_error<F: TargetFn + Sync>(f: &F, model: &Barycentric, bounds: &[(f64, f64)]) -> f64 {
    let mut state: u64 = 0x9E3779B97F4A7C15;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let n = bounds.len();
    let mut x = vec![0.0; n];
    let mut max_err = 0.0_f64;
    let mut max_val = 0.0_f64;
    for _ in 0..N_VALID {
        for (l, &(lo, hi)) in bounds.iter().enumerate() {
            let t = 0.1 + 0.8 * next();
            x[l] = lo + (hi - lo) * t;
        }
        let truth = f.eval(&x);
        max_val = max_val.max(truth.abs());
        let e = (truth - model.eval(&x)).abs();
        if e.is_finite() {
            max_err = max_err.max(e);
        }
    }
    let scale = if max_val == 0.0 { 1.0 } else { max_val };
    max_err / scale
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn alg3_refines_separable_rational() {
        // A separable rational: starting coarse, refinement + resampling should
        // drive the validation error down to near machine precision.
        let f = |x: &[f64]| 1.0 / ((x[0] + 3.0) * (x[1] + 2.0));
        let opt = Alg3Options::new(vec![(-1.0, 1.0); 2]);
        let (model, info) = alg3(&f, &opt).unwrap();
        assert!(
            info.best_error < 1e-7,
            "best_error = {} after {} rounds",
            info.best_error,
            info.rounds
        );
        for x in [[0.13, -0.4], [0.7, 0.2], [-0.55, 0.9]] {
            assert_relative_eq!(model.eval(&x), f(&x), epsilon = 1e-6);
        }
    }

    #[test]
    fn alg3_grid_grows_each_round() {
        // On a target it cannot fit exactly, alg3 keeps refining: the final grid
        // must be denser than the start, and it must run multiple rounds.
        let f = |x: &[f64]| (x[0] * 4.0).tanh() * (x[1] * 3.0).cos();
        let mut opt = Alg3Options::new(vec![(-1.0, 1.0); 2]);
        opt.start_nip = 5;
        opt.max_rounds = 4;
        opt.tol = 1e-14; // force it to keep refining
        opt.patience = None; // disable early-stop so growth is the only limiter
        let (_, info) = alg3(&f, &opt).unwrap();
        assert!(info.rounds >= 2, "only {} rounds", info.rounds);
        assert!(
            info.final_nip.iter().all(|&m| m > 5),
            "grid did not grow: {:?}",
            info.final_nip
        );
    }
}
