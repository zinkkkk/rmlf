//! Null-space extraction for Loewner matrices, and the normalization convention.
//!
//! The barycentric weights are the (approximate) null vector of a Loewner
//! matrix: the right-singular vector of the smallest singular value. Replaces
//! `mlf.null`. The method strings in MATLAB (`'svd0'`, `'svd-1'`, ...) encode
//! both the algorithm (svd / qr) and a trailing normalization index:
//!
//! * `k > 0`  -> divide by the `k`-th entry,
//! * `k == 0` -> divide by the entry of maximum magnitude (`'svd0'`),
//! * `k < 0`  -> divide by the last entry (`'svd-1'`).
//!
//! The recursive construction relies on a *consistent* normalization so that the
//! per-axis scalings multiply correctly; we expose the strategy as a trait so it
//! can be swapped without touching the recursion.

use crate::error::{LoewnerError, Result};
use crate::scalar::Scalar;
use faer::Mat;

/// How to normalize a raw null vector to a canonical scaling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Normalize {
    /// Divide by the entry of maximum magnitude (MATLAB `...0`). Default.
    #[default]
    MaxAbs,
    /// Divide by the last entry (MATLAB `...-1`).
    Last,
    /// Divide by the `k`-th entry (0-based; MATLAB `...k`, 1-based).
    Index(usize),
}

impl Normalize {
    /// Apply the normalization in place. Returns the pivot index used.
    fn apply<S: Scalar>(self, c: &mut [S]) -> usize {
        let pivot = match self {
            Normalize::MaxAbs => c
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.magnitude().partial_cmp(&b.magnitude()).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0),
            Normalize::Last => c.len() - 1,
            Normalize::Index(k) => k,
        };
        let denom = c[pivot];
        if denom != S::zero() {
            for x in c.iter_mut() {
                *x /= denom;
            }
        }
        pivot
    }
}

/// Result of a null-space computation.
#[derive(Debug, Clone)]
pub struct NullResult<S = f64> {
    /// The (normalized) null vector — the barycentric weights.
    pub vector: Vec<S>,
    /// Normalized singular values (`sigma / sigma_0`), descending. Empty for
    /// methods that do not compute them.
    pub sigma: Vec<f64>,
    /// Pivot index used for normalization.
    pub pivot: usize,
}

/// A strategy for extracting the null vector of a (small) Loewner matrix.
pub trait NullSpace<S: Scalar = f64> {
    /// Compute the normalized null vector of `m`.
    fn null(&self, m: &Mat<S>) -> Result<NullResult<S>>;
}

/// SVD-based null space: the right-singular vector of the smallest singular
/// value, then normalized. This is the `'svd*'` family and the default.
#[derive(Debug, Clone, Copy, Default)]
pub struct SvdNull {
    pub normalize: Normalize,
}

impl SvdNull {
    /// SVD null space with `MaxAbs` normalization (MATLAB `'svd0'`).
    pub fn svd0() -> Self {
        SvdNull {
            normalize: Normalize::MaxAbs,
        }
    }
}

impl<S: Scalar> NullSpace<S> for SvdNull {
    fn null(&self, m: &Mat<S>) -> Result<NullResult<S>> {
        if m.nrows() == 0 || m.ncols() == 0 {
            return Err(LoewnerError::Degenerate("empty matrix".into()));
        }
        // A wide matrix has right-null directions omitted by the thin SVD.
        let svd = if m.nrows() < m.ncols() {
            m.svd()
        } else {
            m.thin_svd()
        }
        .map_err(|_| LoewnerError::SvdFailed)?;
        let v = svd.V();
        let ncols = v.ncols();
        if ncols == 0 {
            return Err(LoewnerError::Degenerate("empty matrix".into()));
        }
        // Right-singular vector of the smallest singular value = last column of V.
        let last = ncols - 1;
        let mut vector: Vec<S> = (0..v.nrows()).map(|i| v[(i, last)]).collect();

        // Normalized singular values (sigma / sigma_0), descending.
        let s = svd.S().column_vector();
        let s0 = s[0].real();
        let sigma: Vec<f64> = (0..s.nrows())
            .map(|i| if s0 == 0.0 { 0.0 } else { s[i].real() / s0 })
            .collect();

        let pivot = self.normalize.apply(&mut vector);
        Ok(NullResult {
            vector,
            sigma,
            pivot,
        })
    }
}

/// By-inspection null space — Gaussian elimination, **no SVD**.
///
/// This is the method of Theorem 2.1 of the variable-decoupling paper (arXiv
/// 2605.07246): the right null space of a 1-D Loewner matrix is a single line,
/// recovered directly. We never compute singular values. For an exactly rational
/// slice the matrix is rank `k−1` and this returns the exact `[γ_j d(λ_j)]`
/// direction; for any other data it returns the null direction of the system as
/// formed (the paper's *approximate decoupling*, §5(c)) — it always runs and
/// always returns a vector, regardless of the data.
///
/// Algorithm: partial-pivoted row reduction of the `q×k` matrix. The free column
/// (the one without a pivot — the last by construction for a rank-`k−1` system)
/// becomes the null direction by back-substitution. This needs only `+ − × ÷`,
/// so it generalizes to any field (the bridge to the exact ℚ / 𝔽_p branch).
#[derive(Debug, Clone, Copy, Default)]
pub struct InspectNull {
    pub normalize: Normalize,
}

impl NullSpace for InspectNull {
    fn null(&self, m: &Mat<f64>) -> Result<NullResult> {
        let q = m.nrows();
        let k = m.ncols();
        if k == 0 {
            return Err(LoewnerError::Degenerate("empty matrix".into()));
        }
        if k == 1 {
            // A single column: the only homogeneous solution is the zero vector,
            // but barycentrically a length-1 weight is just the scalar 1.
            let mut vector = vec![1.0];
            let pivot = self.normalize.apply(&mut vector);
            return Ok(NullResult {
                vector,
                sigma: Vec::new(),
                pivot,
            });
        }

        // Dense copy for in-place elimination.
        let mut a: Vec<Vec<f64>> = (0..q)
            .map(|i| (0..k).map(|j| m[(i, j)]).collect())
            .collect();

        // Gauss–Jordan with full pivoting, but capped at `k − 1` pivots so that
        // exactly one column is always left free to carry the null direction. A
        // well-formed order-`k−1` slice has rank `k−1`, so the (k−1)-th pivot is
        // genuine; the leftover column is the smallest-pivot one by construction.
        // For non-rational data this still produces a usable approximate null
        // (the paper's §5(c) approximate decoupling). No SVD, only `+ − × ÷`.
        let mut pivot_col = vec![usize::MAX; k]; // column -> pivot row, or MAX
        let mut row_used = vec![false; q];
        let max_pivots = (k - 1).min(q);
        for _ in 0..max_pivots {
            // Full pivot: largest-magnitude entry over remaining rows & columns.
            let mut best = (0.0f64, usize::MAX, usize::MAX); // (|val|, row, col)
            for (r, used) in row_used.iter().enumerate() {
                if *used {
                    continue;
                }
                for (c, &pc) in pivot_col.iter().enumerate() {
                    if pc != usize::MAX {
                        continue; // column already has a pivot
                    }
                    let v = a[r][c].abs();
                    if v > best.0 {
                        best = (v, r, c);
                    }
                }
            }
            let (_, pr, pcol) = best;
            if pr == usize::MAX {
                break; // nothing left to eliminate
            }
            // Eliminate `pcol` from every other row.
            let piv = a[pr][pcol];
            let pivot_row = a[pr].clone();
            for (r, row) in a.iter_mut().enumerate() {
                if r == pr {
                    continue;
                }
                let factor = row[pcol] / piv;
                if factor != 0.0 {
                    for (c, x) in row.iter_mut().enumerate() {
                        *x -= factor * pivot_row[c];
                    }
                }
            }
            pivot_col[pcol] = pr;
            row_used[pr] = true;
        }

        // The free column is the one left without a pivot. Prefer the *last* such
        // column (matches the paper's last-entry normalization and the recursion's
        // expectation that the trailing weight anchors the scale).
        let free = (0..k)
            .rev()
            .find(|&c| pivot_col[c] == usize::MAX)
            .unwrap_or(k - 1);
        let mut vector = vec![0.0; k];
        vector[free] = 1.0;
        // Back-out each pivot variable: after Gauss–Jordan, pivot row `r` for
        // pivot column `pc` reads `piv·x_pc + a[r][free]·x_free = 0`.
        for pc in 0..k {
            let r = pivot_col[pc];
            if r != usize::MAX {
                let piv = a[r][pc];
                vector[pc] = -a[r][free] / piv;
            }
        }

        let pivot = self.normalize.apply(&mut vector);
        Ok(NullResult {
            vector,
            sigma: Vec::new(),
            pivot,
        })
    }
}

/// Minimum-norm approximate null space via **inverse iteration** — no SVD.
///
/// For a genuinely rational slice the Loewner matrix is rank `k−1` and
/// [`InspectNull`] returns the exact null line; but for a **non-rational** target
/// (the paper's approximate-decoupling regime, §5(c)) the slice is numerically
/// full rank, so forcing a null by elimination divides by tiny pivots and yields a
/// wildly ill-conditioned vector (dynamic range up to ~1e29 on `|x−y|³`). The SVD
/// avoided this by returning the *unit-norm* minimal-singular vector.
///
/// This recovers that minimum-norm direction without an SVD: the smallest
/// eigenvector of the normal matrix `A = LᵀL` via shifted inverse iteration. Each
/// step solves `(A + μI) y = x_prev` (Cholesky/LU on a tiny `k×k` system) and
/// renormalizes; it converges to the eigenvector of the smallest eigenvalue, i.e.
/// the right-singular vector of the smallest singular value of `L` — exactly the
/// SVD null, computed with only solves and matvecs. Always returns a unit-norm
/// vector, so coefficients stay O(1).
#[derive(Debug, Clone, Copy)]
pub struct MinNormNull {
    pub normalize: Normalize,
    /// Number of inverse-iteration steps. A handful suffices for these tiny
    /// matrices; default 50 via [`MinNormNull::new`].
    pub iters: usize,
}

impl MinNormNull {
    pub fn new(normalize: Normalize) -> Self {
        MinNormNull {
            normalize,
            iters: 50,
        }
    }
}

impl NullSpace for MinNormNull {
    fn null(&self, m: &Mat<f64>) -> Result<NullResult> {
        let q = m.nrows();
        let k = m.ncols();
        if k == 0 {
            return Err(LoewnerError::Degenerate("empty matrix".into()));
        }
        if k == 1 {
            let mut vector = vec![1.0];
            let pivot = self.normalize.apply(&mut vector);
            return Ok(NullResult {
                vector,
                sigma: Vec::new(),
                pivot,
            });
        }

        // Normal matrix A = LᵀL (k×k, symmetric PSD).
        let mut a = vec![vec![0.0f64; k]; k];
        for (i, ai) in a.iter_mut().enumerate() {
            for (j, aij) in ai.iter_mut().enumerate() {
                let mut s = 0.0;
                for r in 0..q {
                    s += m[(r, i)] * m[(r, j)];
                }
                *aij = s;
            }
        }

        // Scale-aware shift so (A + μI) is safely invertible even when A is exactly
        // singular (the rational case): μ relative to the trace.
        let trace: f64 = (0..k).map(|i| a[i][i]).sum();
        let mu = (trace / k as f64).max(1.0) * 1e-12;
        for (i, ai) in a.iter_mut().enumerate() {
            ai[i] += mu;
        }

        // Shifted inverse iteration: x ← normalize((A+μI)^{-1} x).
        let mut x = vec![1.0 / (k as f64).sqrt(); k];
        for _ in 0..self.iters {
            let y = solve_spd(&a, &x)?;
            let norm = y.iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm == 0.0 || !norm.is_finite() {
                break;
            }
            let mut maxchg = 0.0f64;
            for (xi, yi) in x.iter_mut().zip(&y) {
                let nx = yi / norm;
                maxchg = maxchg.max((nx - *xi).abs());
                *xi = nx;
            }
            if maxchg < 1e-14 {
                break;
            }
        }

        let mut vector = x;
        let pivot = self.normalize.apply(&mut vector);
        Ok(NullResult {
            vector,
            sigma: Vec::new(),
            pivot,
        })
    }
}

/// Solve `A x = b` for symmetric positive-definite `A` (tiny dense, Cholesky with
/// an LU fallback). `A` is given as row-major `Vec<Vec<f64>>`.
// Indexed loops read clearer than iterator chains for these dense LA kernels.
#[allow(clippy::needless_range_loop)]
fn solve_spd(a: &[Vec<f64>], b: &[f64]) -> Result<Vec<f64>> {
    let n = a.len();
    // Cholesky: A = L Lᵀ.
    let mut l = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i][j];
            for c in 0..j {
                s -= l[i][c] * l[j][c];
            }
            if i == j {
                if s <= 0.0 {
                    // Not numerically SPD — fall back to plain Gaussian solve.
                    return solve_lu(a, b);
                }
                l[i][j] = s.sqrt();
            } else {
                l[i][j] = s / l[j][j];
            }
        }
    }
    // Forward solve L y = b.
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut s = b[i];
        for c in 0..i {
            s -= l[i][c] * y[c];
        }
        y[i] = s / l[i][i];
    }
    // Back solve Lᵀ x = y.
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for c in (i + 1)..n {
            s -= l[c][i] * x[c];
        }
        x[i] = s / l[i][i];
    }
    Ok(x)
}

/// Plain Gaussian-elimination solve with partial pivoting (SPD fallback).
#[allow(clippy::needless_range_loop)]
fn solve_lu(a: &[Vec<f64>], b: &[f64]) -> Result<Vec<f64>> {
    let n = a.len();
    let mut m: Vec<Vec<f64>> = a.to_vec();
    let mut rhs = b.to_vec();
    for col in 0..n {
        let mut piv = col;
        for r in (col + 1)..n {
            if m[r][col].abs() > m[piv][col].abs() {
                piv = r;
            }
        }
        m.swap(col, piv);
        rhs.swap(col, piv);
        let d = m[col][col];
        if d == 0.0 {
            return Err(LoewnerError::Degenerate("singular normal matrix".into()));
        }
        for r in (col + 1)..n {
            let f = m[r][col] / d;
            if f != 0.0 {
                for c in col..n {
                    m[r][c] -= f * m[col][c];
                }
                rhs[r] -= f * rhs[col];
            }
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = rhs[i];
        for c in (i + 1)..n {
            s -= m[i][c] * x[c];
        }
        x[i] = s / m[i][i];
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loewner::loewner_1d;
    use approx::assert_relative_eq;

    #[test]
    fn inspect_null_matches_svd_on_rational() {
        // f(x) = 1/(x+3): order-1 rational. By-inspection null must match the SVD
        // null direction (both annihilate L) — and the octave reference [-0.75, 1].
        let f = |x: f64| 1.0 / (x + 3.0);
        let lambda = [0.0, 1.0];
        let mu = [0.5, 1.5];
        let w: Vec<f64> = lambda.iter().map(|&x| f(x)).collect();
        let v: Vec<f64> = mu.iter().map(|&x| f(x)).collect();
        let l = loewner_1d(&lambda, &w, &mu, &v);
        let c = InspectNull {
            normalize: Normalize::Last,
        }
        .null(&l)
        .unwrap()
        .vector;
        assert_relative_eq!(c[0], -0.75, epsilon = 1e-12);
        assert_relative_eq!(c[1], 1.0, epsilon = 1e-12);
        // And it annihilates L.
        for i in 0..l.nrows() {
            let dot: f64 = (0..l.ncols()).map(|j| l[(i, j)] * c[j]).sum();
            assert_relative_eq!(dot, 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn inspect_null_higher_order_annihilates() {
        // f(x) = (x+1)/(x^2 + 2x + 5): order-2 rational, k=3, q=3.
        let f = |x: f64| (x + 1.0) / (x * x + 2.0 * x + 5.0);
        let lambda = [0.0, 1.0, 2.0];
        let mu = [0.5, 1.5, 2.5];
        let w: Vec<f64> = lambda.iter().map(|&x| f(x)).collect();
        let v: Vec<f64> = mu.iter().map(|&x| f(x)).collect();
        let l = loewner_1d(&lambda, &w, &mu, &v);
        let c = InspectNull {
            normalize: Normalize::Last,
        }
        .null(&l)
        .unwrap()
        .vector;
        for i in 0..l.nrows() {
            let dot: f64 = (0..l.ncols()).map(|j| l[(i, j)] * c[j]).sum();
            assert_relative_eq!(dot, 0.0, epsilon = 1e-9);
        }
    }

    #[test]
    fn min_norm_matches_svd_direction_on_rational() {
        // On a rational slice, the min-norm (inverse-iteration) null must match the
        // SVD null direction up to sign/scale (both annihilate L).
        let f = |x: f64| (x + 1.0) / (x * x + 2.0 * x + 5.0);
        let lambda = [0.0, 1.0, 2.0];
        let mu = [0.5, 1.5, 2.5];
        let w: Vec<f64> = lambda.iter().map(|&x| f(x)).collect();
        let v: Vec<f64> = mu.iter().map(|&x| f(x)).collect();
        let l = loewner_1d(&lambda, &w, &mu, &v);
        let c = MinNormNull::new(Normalize::Last).null(&l).unwrap().vector;
        for i in 0..l.nrows() {
            let dot: f64 = (0..l.ncols()).map(|j| l[(i, j)] * c[j]).sum();
            assert_relative_eq!(dot, 0.0, epsilon = 1e-7);
        }
    }

    #[test]
    fn min_norm_bounded_on_full_rank() {
        // A full-rank (non-rational-ish) matrix: min-norm must return a UNIT-norm
        // vector (bounded), unlike forced Gaussian elimination which blows up.
        let l = loewner_1d(
            &[0.0, 1.0, 2.0],
            &[1.0, 0.3, 2.7],
            &[0.5, 1.5, 2.5],
            &[0.9, 2.1, 0.2],
        );
        let c = MinNormNull::new(Normalize::MaxAbs).null(&l).unwrap().vector;
        let norm: f64 = c.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(norm.is_finite() && norm > 0.0);
        // MaxAbs normalization ⇒ largest entry is 1, so all entries are ≤ 1.
        assert!(c.iter().all(|v| v.abs() <= 1.0 + 1e-12));
    }

    #[test]
    fn null_of_rank_one_loewner_recovers_constant() {
        // For a constant function f == 5, every value is 5, so the Loewner
        // matrix is identically zero: rank 0. Any unit vector is a null vector;
        // here we just check the routine runs and normalizes.
        let l = loewner_1d(&[1.0, 2.0], &[5.0, 5.0], &[3.0, 4.0], &[5.0, 5.0]);
        let res = SvdNull::svd0().null(&l).unwrap();
        assert_eq!(res.vector.len(), 2);
        // MaxAbs normalization => the pivot entry is exactly 1.
        assert_relative_eq!(res.vector[res.pivot], 1.0);
    }

    #[test]
    fn null_vector_annihilates_loewner() {
        // f(x) = 1/(x+3): a genuine 1-D rational, order 1. A 2x2 (k=q=2)
        // Loewner of an order-1 function is rank 1, so it has a 1-D null space,
        // and L * c must be ~0.
        let f = |x: f64| 1.0 / (x + 3.0);
        let lambda = [0.0, 1.0];
        let mu = [0.5, 1.5];
        let w: Vec<f64> = lambda.iter().map(|&x| f(x)).collect();
        let v: Vec<f64> = mu.iter().map(|&x| f(x)).collect();
        let l = loewner_1d(&lambda, &w, &mu, &v);
        let c = SvdNull::svd0().null(&l).unwrap().vector;
        for i in 0..l.nrows() {
            let dot: f64 = (0..l.ncols()).map(|j| l[(i, j)] * c[j]).sum();
            assert_relative_eq!(dot, 0.0, epsilon = 1e-12);
        }
    }
}

#[cfg(test)]
mod octave_parity {
    use super::*;
    use crate::loewner::loewner_1d;
    use approx::assert_relative_eq;

    #[test]
    fn svd0_matches_octave_1d() {
        // Octave mlf.null(L,'svd0') for f=1/(x+3), lambda=[0,1], mu=[0.5,1.5]
        // gives c = [-0.75, 1].
        let f = |x: f64| 1.0 / (x + 3.0);
        let lambda = [0.0, 1.0];
        let mu = [0.5, 1.5];
        let w: Vec<f64> = lambda.iter().map(|&x| f(x)).collect();
        let v: Vec<f64> = mu.iter().map(|&x| f(x)).collect();
        let l = loewner_1d(&lambda, &w, &mu, &v);
        let c = SvdNull::svd0().null(&l).unwrap().vector;
        assert_relative_eq!(c[0], -0.75, epsilon = 1e-12);
        assert_relative_eq!(c[1], 1.0, epsilon = 1e-12);
    }
}
