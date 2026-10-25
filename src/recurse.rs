//! The recursive null-space construction — the heart of the multivariate Loewner
//! framework, and the crate's reason to exist.
//!
//! Instead of forming the full `Q x K` n-D Loewner matrix and taking one huge
//! SVD (`O(K^3)`, exponential in `n`), the barycentric coefficient vector is
//! assembled from a *sequence of tiny 1-D Loewner null spaces*. This is the
//! paper's headline result (Theorem 5.3) and what tames the curse of
//! dimensionality. Replaces `mlf.loewner_null_rec{,2,N}`.
//!
//! ## The recursion (0-based, row-major)
//!
//! Along each axis `l`, the combined abscissae are `column ++ row`: the first
//! `k_l` indices are column support points, the next `q_l` are row points. The
//! data tensor `tab` is sampled on this combined grid (see [`crate::tensor`]).
//!
//! * **Leading null vector** `c_lead` (length `k_0`): take the slab of `tab`
//!   with every *trailing* axis frozen at its last column index, split its
//!   leading axis into `W = slab[0..k_0]` / `V = slab[k_0..]`, form the 1-D
//!   Loewner matrix along axis 0, and take its null space.
//! * **Combine**: for each leading column index `p` in `0..k_0`, recurse on the
//!   contiguous sub-tensor `tab[p, .., .., ..]` over axes `1..n`, obtaining
//!   `c_child`, and append `c_child * c_lead[p]`.
//! * **Base case** `n == 1`: split the 1-D slab and return its Loewner null
//!   space directly.
//!
//! Row-major storage makes `tab[p, ..]` a zero-copy contiguous slice and makes
//! the emitted coefficients land in exactly the canonical [`crate::grid`] order,
//! so no `mat2vec`/`vec2mat`/`permute` bookkeeping is needed.

use crate::error::Result;
use crate::loewner::loewner_1d;
use crate::nullspace::{Normalize, NullSpace, SvdNull};
use crate::points::Points;
use crate::tensor::{View, contig_slice, index_leading, scalar_at};

use crate::scalar::Scalar;
use rayon::prelude::*;

/// Output of the recursive construction.
#[derive(Debug, Clone)]
pub struct RecResult<S = f64> {
    /// Barycentric coefficients `c`, length `prod(k_l)`, in canonical order.
    pub c: Vec<S>,
    /// Rough flop count of the recursion (sum of per-Loewner `q * k^2`),
    /// for comparison against the `K^3` monolithic cost.
    pub flop: u128,
}

/// Compute the barycentric coefficients via the recursive 1-D Loewner null
/// spaces. `points` supplies the per-axis column/row split; `tab` is the data
/// tensor sampled on the combined grid (shape `[k_l + q_l]` per axis).
///
/// `null` is the null-space strategy used for the **top-level** leading axis.
///
/// Matching the MATLAB reference, only this very first leading solve uses the
/// caller's method (e.g. `'svd0'` = normalize by max-abs); every *nested*
/// solve, leading or per-slice, uses SVD normalized by the last entry
/// (`'svd'`). The final `c` is a null space, hence defined only up to scaling —
/// this convention is what makes the emitted vector match `mlf` entry-for-entry
/// and keeps the per-axis scalars multiplying consistently.
pub fn loewner_null_rec<S: Scalar, N: NullSpace<S>>(
    points: &Points<S>,
    tab: View<'_, S>,
    null: &N,
) -> Result<RecResult<S>> {
    let k: Vec<usize> = points.column_sizes();
    let nested = SvdNull {
        normalize: Normalize::Last,
    };

    // --- Top-level leading null vector ---------------------------------------
    let n = points.ndim();
    let k0 = k[0];
    let lead_slab = freeze_trailing_last_column(points, &tab, 0);
    let (lead, mut flop) = {
        let (lambda, mu) = (&points.column[0], &points.row[0]);
        let w = &lead_slab[..k0];
        let v = &lead_slab[k0..];
        let l = loewner_1d(lambda, w, mu, v);
        let f = (l.nrows() as u128) * (l.ncols() as u128).pow(2);
        (null.null(&l)?.vector, f)
    };

    if n == 1 {
        return Ok(RecResult { c: lead, flop });
    }

    // --- Parallel combine over the independent leading branches --------------
    // Each leading column index spawns an independent recursion over the
    // trailing axes; these branches share no state, so rayon fans them out.
    // Per-branch buffers are concatenated in canonical order afterward.
    let branches: Vec<Result<(Vec<S>, u128)>> = (0..k0)
        .into_par_iter()
        .map(|p| {
            let scale = lead[p];
            let sub = index_leading(&tab, p);
            let mut buf = Vec::new();
            let mut f = 0u128;
            recurse(points, &sub, &nested, 1, &mut buf, &mut f)?;
            for x in &mut buf {
                *x *= scale;
            }
            Ok((buf, f))
        })
        .collect();

    let mut c = Vec::with_capacity(k.iter().product());
    for branch in branches {
        let (buf, f) = branch?;
        c.extend_from_slice(&buf);
        flop += f;
    }
    Ok(RecResult { c, flop })
}

/// Recurse over axes `axis..n` (always `axis >= 1`), appending the coefficients
/// for this sub-tensor in canonical order.
///
/// Every solve at this depth and below uses `nested` (last-entry normalization);
/// the top-level leading solve is handled by [`loewner_null_rec`]. This stays
/// sequential: deep branches are tiny SVDs whose work is dwarfed by rayon's
/// spawn overhead, so the parallelism lives only at the top level's wide fan-out.
fn recurse<S: Scalar, M: NullSpace<S>>(
    points: &Points<S>,
    tab: &View<'_, S>,
    nested: &M,
    axis: usize,
    out: &mut Vec<S>,
    flop: &mut u128,
) -> Result<()> {
    let n = points.ndim();
    let k0 = points.column[axis].len();
    let lead_dim = tab.shape()[0];

    // --- Leading-axis null vector --------------------------------------------
    // Slab of `tab` with all trailing axes frozen at their last column index.
    let lead_slab = freeze_trailing_last_column(points, tab, axis);
    debug_assert_eq!(lead_slab.len(), lead_dim);

    let lead = {
        let (lambda, mu) = (&points.column[axis], &points.row[axis]);
        let w = &lead_slab[..k0];
        let v = &lead_slab[k0..];
        let l = loewner_1d(lambda, w, mu, v);
        *flop += (l.nrows() as u128) * (l.ncols() as u128).pow(2);
        nested.null(&l)?.vector
    };

    if axis + 1 == n {
        // Base case: this is the last axis; emit the leading null vector.
        out.extend_from_slice(&lead);
        return Ok(());
    }

    // --- Combine with the recursion over the remaining axes -------------------
    // For each leading column index, recurse on tab[p, .., ..] and scale.
    for (p, &scale) in lead.iter().enumerate().take(k0) {
        let sub = index_leading(tab, p);
        let start = out.len();
        recurse(points, &sub, nested, axis + 1, out, flop)?;
        for x in &mut out[start..] {
            *x *= scale;
        }
    }
    Ok(())
}

/// Gather the leading-axis vector of `tab` with every trailing axis frozen at
/// its last *column* index `k_l - 1`. Returns a vector of length `tab.dims()[0]`.
///
/// This is `tab(:, k_1-1, k_2-1, ..., k_{n-1}-1)` — the representative slice the
/// MATLAB code uses to build the leading 1-D Loewner matrix. For a 1-D view it
/// is just the whole contiguous line.
fn freeze_trailing_last_column<S: Scalar>(
    points: &Points<S>,
    tab: &View<'_, S>,
    axis: usize,
) -> Vec<S> {
    if tab.ndim() == 1 {
        return contig_slice(tab).to_vec();
    }
    let lead_dim = tab.shape()[0];
    let n = tab.ndim();
    // Multi-index: leading axis varies, every trailing axis pinned to its last
    // column index k_l - 1.
    let mut idx = vec![0usize; n];
    for (j, l) in ((axis + 1)..points.ndim()).enumerate() {
        idx[j + 1] = points.column[l].len() - 1;
    }
    (0..lead_dim)
        .map(|i| {
            idx[0] = i;
            scalar_at(tab, &idx)
        })
        .collect()
}

/// The default recursive construction using SVD-with-max-abs (`'svd0'`).
pub fn loewner_null_rec_svd0<S: Scalar>(
    points: &Points<S>,
    tab: View<'_, S>,
) -> Result<RecResult<S>> {
    loewner_null_rec(points, tab, &SvdNull::svd0())
}

/// Gather the line of `tab` that varies axis `axis` and freezes **every other**
/// axis at its last column index `k_l - 1`. Length `tab.shape()[axis]`.
///
/// Unlike [`freeze_trailing_last_column`] (which is tied to the recursion's
/// sub-view, always varying view-position 0), this addresses the real axis
/// `axis` of the full tensor — what the separability shortcut needs per axis.
fn axis_line_last_column<S: Scalar>(points: &Points<S>, tab: &View<'_, S>, axis: usize) -> Vec<S> {
    let n = tab.ndim();
    if n == 1 {
        return contig_slice(tab).to_vec();
    }
    let dim = tab.shape()[axis];
    let mut idx: Vec<usize> = (0..n).map(|l| points.column[l].len() - 1).collect();
    (0..dim)
        .map(|i| {
            idx[axis] = i;
            scalar_at(tab, &idx)
        })
        .collect()
}

/// Fast path for a known separable target: its coefficient tensor has the form
/// `C = c_0 ⊗ c_1 ⊗ … ⊗ c_{n-1}`, and each factor `c_l` is the
/// null vector of a single 1-D Loewner along axis `l`. We then need only `n` tiny
/// SVDs and one outer product — no recursion, none of the `k_0 ×` redundant child
/// solves the general recursion performs in this case.
///
/// The factors are built from exactly the slices the recursion uses
/// (`freeze_trailing_last_column`) and normalized with the same convention —
/// axis 0 by the caller's `null`, every later axis by last-entry — so the emitted
/// `c` matches [`loewner_null_rec`] entry-for-entry (a null space is defined only
/// up to scale; this convention fixes it consistently).
///
/// The caller must establish separability. This function returns a coefficient
/// tensor, not an optional result, and does not test the precondition. The
/// per-context ranks in [`crate::order::OrderReport`] alone are not a proof.
pub fn loewner_null_separable<S: Scalar, N: NullSpace<S>>(
    points: &Points<S>,
    tab: View<'_, S>,
    null: &N,
) -> Result<RecResult<S>> {
    let n = points.ndim();
    let nested = SvdNull {
        normalize: Normalize::Last,
    };

    // Per-axis factor c_l = null vector of the axis-l Loewner on the slice that
    // varies axis l and freezes every *other* axis at its last column index. For
    // a separable target this slice is `(other constants) · base_l`; the constant
    // scales the Loewner but not its null vector, so the factor matches what the
    // recursion extracts along axis l. Axis 0 uses the caller's normalization;
    // later axes use last-entry, matching the recursion's nested convention.
    let mut factors: Vec<Vec<S>> = Vec::with_capacity(n);
    let mut flop = 0u128;
    for axis in 0..n {
        let slab = axis_line_last_column(points, &tab, axis);
        let k0 = points.column[axis].len();
        let (lambda, mu) = (&points.column[axis], &points.row[axis]);
        let l = loewner_1d(lambda, &slab[..k0], mu, &slab[k0..]);
        flop += (l.nrows() as u128) * (l.ncols() as u128).pow(2);
        let vec = if axis == 0 {
            null.null(&l)?.vector
        } else {
            nested.null(&l)?.vector
        };
        factors.push(vec);
    }

    // Outer product in canonical row-major order: c[i_0,…,i_{n-1}] = ∏_l c_l[i_l].
    let total: usize = factors.iter().map(Vec::len).product();
    let mut c = vec![S::one(); total];
    let mut block = total;
    for factor in &factors {
        let kl = factor.len();
        block /= kl;
        for (flat, x) in c.iter_mut().enumerate() {
            *x *= factor[(flat / block) % kl];
        }
    }
    Ok(RecResult { c, flop })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::sample;
    use approx::assert_relative_eq;

    #[test]
    fn separable_matches_recursion() {
        // For a separable target the n-SVD outer-product shortcut (#6) must emit
        // the *same* coefficient vector as the full recursion, entry-for-entry
        // (the recursion already produces a rank-1 C here, just with k_0×
        // redundant child solves). Cross-checked on a 3-D separable rational.
        // Minimal (square, order+1 = 2 points/axis) problem, as produced by
        // point selection — the regime the shortcut is invoked in. Each axis
        // Loewner is then 2×2 rank-1 with a unique 1-D null space.
        let f = |x: &[f64]| 1.0 / ((x[0] + 3.0) * (x[1] + 2.0) * (x[2] + 4.0));
        let col: Vec<f64> = (0..2).map(|i| i as f64 * 0.3).collect();
        let row: Vec<f64> = (0..2).map(|i| i as f64 * 0.3 + 0.15).collect();
        let pts = Points::new(
            vec![col.clone(), col.clone(), col.clone()],
            vec![row.clone(), row.clone(), row.clone()],
        )
        .unwrap();
        let tab = sample(&f, &pts);
        let rec = loewner_null_rec_svd0(&pts, tab.view()).unwrap();
        let sep = loewner_null_separable(&pts, tab.view(), &SvdNull::svd0()).unwrap();
        assert_eq!(sep.c.len(), rec.c.len());
        for (got, want) in sep.c.iter().zip(&rec.c) {
            assert_relative_eq!(got, want, epsilon = 1e-12);
        }
    }

    #[test]
    fn rec2_matches_octave() {
        // Golden values from Octave mlf.loewner_null_rec for
        // f(s1,s2) = 1/((s1+3)(s2+2)), p_c={[0,1],[0,1]}, p_r={[0.5,1.5],[0.5,1.5]}.
        let f = |x: &[f64]| 1.0 / ((x[0] + 3.0) * (x[1] + 2.0));
        let pts = Points::new(
            vec![vec![0.0, 1.0], vec![0.0, 1.0]],
            vec![vec![0.5, 1.5], vec![0.5, 1.5]],
        )
        .unwrap();
        let tab = sample(&f, &pts);
        let res = loewner_null_rec_svd0(&pts, tab.view()).unwrap();
        let expect = [0.5, -0.75, -0.6666666666666667, 1.0];
        assert_eq!(res.c.len(), 4);
        for (got, want) in res.c.iter().zip(expect) {
            assert_relative_eq!(got, &want, epsilon = 1e-12);
        }
    }
}
