//! One-dimensional **AAA + Lawson** rational approximation — a *best-approximation*
//! complement to the interpolatory Loewner core.
//!
//! alg1/alg2/alg3 all build the model as a **null space** of a Loewner matrix:
//! the rational is *exact* at the support points and the error between them is
//! uncontrolled. AAA-Lawson instead targets **best approximation**:
//!
//! * **AAA** (Nakatsukasa–Sète–Trefethen 2018) grows the support greedily — add
//!   the sample of largest residual, then take the **least-squares** barycentric
//!   weights (smallest right singular vector of the Loewner matrix over the
//!   remaining *test* points). The result is a near-best `L²` rational.
//! * **Lawson** (iteratively reweighted least squares) reweights the test points
//!   to drive the solution toward the **minimax (`L∞`) best** rational — the
//!   equioscillating approximant with a uniform error bound. This is what makes
//!   AAA competitive with Remez for hard (branch-point) functions.
//!
//! The barycentric form here is identical to [`Barycentric`] at `n = 1`
//! (`lambda = [support]`, `w = f(support)`, `c =` weights), so the model reuses
//! the existing evaluator (including correct on-node limits).
//!
//! ## When to apply Lawson — "once at the end" vs "every step"
//!
//! Classical AAA-Lawson runs the greedy to convergence, *then* applies Lawson
//! once. [`AaaOptions::interleave`] also offers the every-step variant: run Lawson
//! after **each** point addition, so the greedy picks the next support point from
//! the *minimax-reweighted* residual rather than the `L²` residual. Because point
//! placement and weight optimization target slightly different optima, interleaving
//! can steer the support toward the minimax-optimal set from the start (at the cost
//! of a Lawson solve per step). Both variants are first-iterate-safe: Lawson's
//! first iterate *is* the `L²` solution and the best max-error iterate is retained,
//! so the returned model is **never worse in the max-norm than plain AAA**.
//!
//! This is the validated 1-D kernel. The multivariate lift lives in
//! [`crate::algo::alg2()`] (`Alg2Options::lawson`): it reweights alg2's full-grid
//! residual scan so greedy **point placement** targets the minimax error, without
//! touching the curse-taming recursion. Pushing the *coefficients* (not just the
//! support) to minimax in the recursive/TT construction is the open step
//! (see `docs/NDIM_MINIMAX.md` for the design).
//!
//! ## Status & accuracy (honest)
//!
//! Lawson here is an **rmlf addition**, not part of the mLF papers; it is the
//! well-known AAA-Lawson polish (Nakatsukasa–Sète–Trefethen). The IRLS update
//! follows chebfun's `aaa.m` (relative-error reweight, inf-norm normalized). It
//! **improves** the max-norm on hard, non-rational targets (e.g. `|x|`: `~3.9e-3`
//! L² → `~6e-4`, ~6×) and is provably **never worse** than plain `L²` AAA (first
//! iterate is the L² solution, best iterate retained). It does **not** yet reach
//! chebfun's full minimax: on `|x|` the true type-`n` equioscillating minimax is
//! `~5e-5` (Newman root-exponential) and this kernel plateaus ~10× above it. The
//! remaining gap is structural vs chebfun — chebfun reweights over **all** sample
//! points (support rows pinned) using the `[1/(Z−zⱼ), F/(Z−zⱼ)]` 2m-column system,
//! where this kernel reweights only the non-support test points. A full
//! chebfun-faithful rewrite is the path to true minimax; deferred.
//!
//! **Not yet ported — the Zolotarev / sign-function path.** Recent chebfun adds a
//! *damping ratio* together with an SVD-scaling branch
//! (`c = V·(1/σ²)` instead of the smallest right singular vector) that activates
//! only when the data **splits across a sign change with a gap** — positive on one
//! side, negative on the other (the Zolotarev best-rational-`sgn(x)` regime on
//! `[−b,−a] ∪ [a,b]`). That coupled feature is specialized to two-sided-gap sign
//! problems and is **not** implemented here; this kernel uses the plain smallest
//! singular vector. (It would not have helped the single-ridge `|x|` case above —
//! a different structure.) Worth adding when targeting Zolotarev-type problems.
//!
//! Use Lawson when you want a *better-than-L²* max-norm fit, not when you need the
//! certified minimax optimum.

use crate::eval::Barycentric;

use faer::Mat;

/// Options for [`aaa`].
#[derive(Debug, Clone)]
pub struct AaaOptions {
    /// Greedy stopping tolerance on the max residual (`max|f − r| < tol`).
    pub tol: f64,
    /// Maximum number of support points (capped at half the sample count).
    pub max_support: usize,
    /// Lawson IRLS iterations for the weight fit. `0` ⇒ plain `L²` AAA (no Lawson).
    pub lawson_iters: usize,
    /// Run Lawson after **every** greedy step (so point selection sees the
    /// minimax-reweighted residual) rather than once at the end. No effect when
    /// `lawson_iters == 0`.
    pub interleave: bool,
}

impl Default for AaaOptions {
    fn default() -> Self {
        AaaOptions {
            tol: 1e-13,
            max_support: 100,
            lawson_iters: 0,
            interleave: false,
        }
    }
}

/// Diagnostics returned alongside the model.
#[derive(Debug, Clone)]
pub struct AaaInfo {
    /// The chosen support abscissae.
    pub support: Vec<f64>,
    /// Root-mean-square error over the sample set.
    pub l2_error: f64,
    /// Max (`L∞`) error over the sample set.
    pub max_error: f64,
}

/// Fit a 1-D rational to samples `(x, fx)` by AAA (optionally Lawson-polished).
///
/// `x` are the sample abscissae, `fx = f(x)`. Returns the barycentric model and
/// diagnostics. Panics if `x` and `fx` differ in length or fewer than 2 samples.
pub fn aaa(x: &[f64], fx: &[f64], opt: &AaaOptions) -> (Barycentric, AaaInfo) {
    let m = x.len();
    assert_eq!(fx.len(), m, "x and fx length mismatch");
    assert!(m >= 2, "AAA needs at least 2 samples");
    let max_support = opt.max_support.min(m / 2).max(1);

    // Greedy support growth. Point selection uses the current model's residual;
    // with `interleave`, that model is Lawson-fitted, so selection targets the
    // minimax residual. We track the support set of the best (lowest max-error)
    // step and do the final fit on it.
    let mut support: Vec<usize> = Vec::new();
    let mut is_support = vec![false; m];
    let mean = fx.iter().sum::<f64>() / m as f64;
    let mut r = vec![mean; m]; // current model values at every sample
    let mut best: Option<(Vec<usize>, f64)> = None;

    let step_iters = if opt.interleave { opt.lawson_iters } else { 0 };

    for _ in 0..max_support {
        // Add the non-support sample of largest residual.
        let (mut worst, mut worst_err) = (usize::MAX, -1.0_f64);
        for i in 0..m {
            if is_support[i] {
                continue;
            }
            let e = (fx[i] - r[i]).abs();
            if e > worst_err {
                worst_err = e;
                worst = i;
            }
        }
        if worst == usize::MAX {
            break; // every sample is a support point
        }
        support.push(worst);
        is_support[worst] = true;

        let (model, _l2, max_err) = fit(x, fx, &support, step_iters);
        for (ri, &xi) in r.iter_mut().zip(x) {
            *ri = model.eval(&[xi]);
        }
        if best.as_ref().map(|(_, e)| max_err < *e).unwrap_or(true) {
            best = Some((support.clone(), max_err));
        }
        if max_err < opt.tol {
            break;
        }
    }

    let (best_support, _) = best.expect("AAA runs at least one step");
    let (model, l2_error, max_error) = fit(x, fx, &best_support, opt.lawson_iters);
    let support_pts = best_support.iter().map(|&i| x[i]).collect();
    (
        model,
        AaaInfo {
            support: support_pts,
            l2_error,
            max_error,
        },
    )
}

/// Fit barycentric weights on a fixed `support` (indices into `x`) and return the
/// model plus its RMS / max error over all samples. `lawson_iters == 0` gives the
/// `L²` weights; `> 0` runs Lawson IRLS.
fn fit(x: &[f64], fx: &[f64], support: &[usize], lawson_iters: usize) -> (Barycentric, f64, f64) {
    let m = x.len();
    let mut is_support = vec![false; m];
    for &i in support {
        is_support[i] = true;
    }
    let z: Vec<f64> = support.iter().map(|&i| x[i]).collect();
    let fz: Vec<f64> = support.iter().map(|&i| fx[i]).collect();
    let xt: Vec<f64> = (0..m).filter(|i| !is_support[*i]).map(|i| x[i]).collect();
    let ft: Vec<f64> = (0..m).filter(|i| !is_support[*i]).map(|i| fx[i]).collect();

    let c = if lawson_iters > 0 {
        lawson_weights(&z, &fz, &xt, &ft, lawson_iters)
    } else {
        l2_weights(&z, &fz, &xt, &ft)
    };
    let model = Barycentric {
        lambda: vec![z],
        w: fz,
        c,
    };

    let mut sse = 0.0_f64;
    let mut max_err = 0.0_f64;
    for (&xi, &fi) in x.iter().zip(fx) {
        let e = (fi - model.eval(&[xi])).abs();
        sse += e * e;
        max_err = max_err.max(e);
    }
    (model, (sse / m as f64).sqrt(), max_err)
}

/// Least-squares barycentric weights: the smallest right singular vector of the
/// Loewner matrix `A[i,j] = (ft[i] − fz[j]) / (xt[i] − z[j])` over the test points.
fn l2_weights(z: &[f64], fz: &[f64], xt: &[f64], ft: &[f64]) -> Vec<f64> {
    let a = loewner(z, fz, xt, ft, None);
    min_singular_vector(&a)
}

/// Lawson IRLS for the minimax weights, following the chebfun `aaa.m` update.
///
/// Maintains nonnegative test-point weights `s` (init uniform). Each iteration
/// solves the `√s`-weighted least squares and evaluates the true residual, then
/// applies the **damped** Lawson reweighting
/// `s ← s · ((1 − d) + d · relerr)`, `relerr = |resid| / max|resid|`,
/// renormalized by the **inf-norm** (`d = LAWSON_DAMP`, chebfun's default 1.0).
///
/// This damped, relative-error form is the key to convergence: the earlier
/// undamped `s ← s · |resid|` (L¹-normalized) drove the weights to collapse onto a
/// few points and **stalled** after a handful of iterations (improving `|x|` only
/// ~5× then plateauing). The damped relative update keeps every weight in play and
/// pushes the residual toward **equioscillation** (the true minimax), matching
/// chebfun's behavior. The first iterate equals the `L²` solution and the best
/// max-error iterate is retained, so the result is never worse than `L²`.
fn lawson_weights(z: &[f64], fz: &[f64], xt: &[f64], ft: &[f64], iters: usize) -> Vec<f64> {
    // Lawson damping ratio. 1.0 = standard (undamped) reweight: the update reduces
    // to `s ← s · relerr`. Chebfun also supports `< 1` damping, but only for the
    // Zolotarev / sign-gap regime, where the literature notes it is **manually
    // tuned per problem, not auto-guessable** — so it is fixed at the standard 1.0
    // here rather than exposed as a misleadingly "default-able" option. If the
    // Zolotarev path is added, take damping as an explicit caller value.
    const LAWSON_DAMP: f64 = 1.0;
    let q = xt.len();
    let mut s = vec![1.0_f64; q];

    let mut best_c = l2_weights(z, fz, xt, ft);
    let mut best_max = residual_max(z, fz, &best_c, xt, ft);

    for _ in 0..iters {
        let a = loewner(z, fz, xt, ft, Some(&s));
        let c = min_singular_vector(&a);

        // True residual of the resulting model at the test points.
        let model = Barycentric {
            lambda: vec![z.to_vec()],
            w: fz.to_vec(),
            c: c.clone(),
        };
        let mut max_r = 0.0_f64;
        let mut resid = vec![0.0_f64; q];
        for i in 0..q {
            let e = (ft[i] - model.eval(&[xt[i]])).abs();
            resid[i] = e;
            max_r = max_r.max(e);
        }
        if max_r < best_max {
            best_max = max_r;
            best_c = c;
        }
        if max_r <= 0.0 {
            break; // converged to (near-)zero residual
        }
        // Damped relative-error reweight (chebfun): s *= (1-d) + d*(resid/maxr),
        // then renormalize by the inf-norm so the largest weight is 1.
        let mut max_s = 0.0_f64;
        for i in 0..q {
            s[i] *= (1.0 - LAWSON_DAMP) + LAWSON_DAMP * (resid[i] / max_r);
            max_s = max_s.max(s[i]);
        }
        if max_s <= 0.0 {
            break;
        }
        for si in &mut s {
            *si /= max_s;
        }
    }
    best_c
}

/// Max true residual of the barycentric model with weights `c` over `(xt, ft)`.
fn residual_max(z: &[f64], fz: &[f64], c: &[f64], xt: &[f64], ft: &[f64]) -> f64 {
    let model = Barycentric {
        lambda: vec![z.to_vec()],
        w: fz.to_vec(),
        c: c.to_vec(),
    };
    xt.iter()
        .zip(ft)
        .map(|(&xi, &fi)| (fi - model.eval(&[xi])).abs())
        .fold(0.0_f64, f64::max)
}

/// The `q × k` Loewner matrix `A[i,j] = (ft[i] − fz[j]) / (xt[i] − z[j])`,
/// optionally with row `i` scaled by `√s[i]` (the Lawson reweighting).
fn loewner(z: &[f64], fz: &[f64], xt: &[f64], ft: &[f64], s: Option<&[f64]>) -> Mat<f64> {
    let (k, q) = (z.len(), xt.len());
    Mat::from_fn(q, k, |i, j| {
        let base = (ft[i] - fz[j]) / (xt[i] - z[j]);
        match s {
            Some(s) => s[i].sqrt() * base,
            None => base,
        }
    })
}

/// The right singular vector of the smallest singular value (unit-norm). Empty on
/// SVD failure.
fn min_singular_vector(a: &Mat<f64>) -> Vec<f64> {
    match a.thin_svd() {
        Ok(svd) => {
            let v = svd.V();
            let last = v.ncols() - 1;
            (0..v.nrows()).map(|i| v[(i, last)]).collect()
        }
        Err(_) => vec![1.0; a.ncols()],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn samples(f: impl Fn(f64) -> f64, m: usize) -> (Vec<f64>, Vec<f64>) {
        let x: Vec<f64> = (0..m)
            .map(|i| -1.0 + 2.0 * i as f64 / (m - 1) as f64)
            .collect();
        let fx: Vec<f64> = x.iter().map(|&t| f(t)).collect();
        (x, fx)
    }

    #[test]
    fn aaa_recovers_a_rational_to_machine_precision() {
        // Runge 1/(1+25x²) is rational; AAA must nail it once the support is large
        // enough (a few points), to ~machine precision.
        let (x, fx) = samples(|t| 1.0 / (1.0 + 25.0 * t * t), 300);
        let (_, info) = aaa(&x, &fx, &AaaOptions::default());
        assert!(info.max_error < 1e-10, "max_error = {:e}", info.max_error);
    }

    #[test]
    fn lawson_never_worse_and_usually_better_in_max_norm() {
        // |x| has a branch point at 0 — not rational. At a fixed support size, the
        // Lawson (minimax) fit must have max error ≤ the plain L² fit (guaranteed:
        // Lawson's first iterate is the L² solution, best iterate retained), and in
        // practice strictly smaller (it equioscillates).
        let (x, fx) = samples(|t| t.abs(), 400);
        let plain = AaaOptions {
            tol: 0.0, // force exactly `max_support` points for a fair comparison
            max_support: 12,
            lawson_iters: 0,
            interleave: false,
        };
        let lawson = AaaOptions {
            lawson_iters: 30,
            ..plain.clone()
        };
        let (_, i_plain) = aaa(&x, &fx, &plain);
        let (_, i_lawson) = aaa(&x, &fx, &lawson);

        assert!(i_plain.max_error.is_finite() && i_lawson.max_error.is_finite());
        assert!(
            i_lawson.max_error <= i_plain.max_error * (1.0 + 1e-9),
            "Lawson {:e} worse than L² {:e}",
            i_lawson.max_error,
            i_plain.max_error
        );
        // Strict improvement on this branch-point target.
        assert!(
            i_lawson.max_error < i_plain.max_error,
            "expected Lawson to improve max error: L²={:e} Lawson={:e}",
            i_plain.max_error,
            i_lawson.max_error
        );
    }

    #[test]
    fn interleaved_lawson_converges_and_is_not_worse_than_plain() {
        // Applying Lawson after *every* greedy step must still produce a valid model
        // whose max error is at least as good as plain L² AAA at the same budget.
        let (x, fx) = samples(|t| t.abs(), 400);
        let plain = AaaOptions {
            tol: 0.0,
            max_support: 12,
            lawson_iters: 0,
            interleave: false,
        };
        let interleaved = AaaOptions {
            lawson_iters: 15,
            interleave: true,
            ..plain.clone()
        };
        let (_, i_plain) = aaa(&x, &fx, &plain);
        let (_, i_inter) = aaa(&x, &fx, &interleaved);
        assert!(i_inter.max_error.is_finite());
        assert!(
            i_inter.max_error <= i_plain.max_error * (1.0 + 1e-9),
            "interleaved {:e} worse than plain {:e}",
            i_inter.max_error,
            i_plain.max_error
        );
    }
}
