//! Barycentric (Lagrangian) evaluation of the multivariate model.
//!
//! Given column support points `lambda_l`, the sampled values `w(J)` at those
//! points, and the barycentric coefficients `c(J)`, the model is
//!
//! ```text
//!          sum_J c(J) w(J) / prod_l (x_l - lambda_l(j_l))
//! g(x) =  -------------------------------------------------
//!          sum_J c(J)       / prod_l (x_l - lambda_l(j_l))
//! ```
//!
//! summed over the canonical [`crate::grid`] order of multi-indices `J`.
//! Replaces `mlf.eval_lagrangian`. No solving, no tensor algebra — just two
//! reductions over the coefficient grid.

use crate::grid::flat_to_multi;
use crate::points::Points;
use crate::tensor::{View, scalar_at};

use crate::scalar::Scalar;
use rayon::prelude::*;

/// Extract the column-block values `w` from the full data tensor: the samples at
/// the column support points along every axis, flattened in canonical order.
///
/// This is `tab(0..k_0, 0..k_1, ..., 0..k_{n-1})` read row-major. Because the
/// column points occupy the *leading* indices along each axis, this is just the
/// recursive leading sub-block; we gather it directly without any `mat2vec`.
pub fn column_values<S: Scalar>(points: &Points<S>, tab: View<'_, S>) -> Vec<S> {
    let k = points.column_sizes();
    let total: usize = k.iter().product();
    let mut w = Vec::with_capacity(total);
    let mut idx = vec![0usize; k.len()];
    for flat in 0..total {
        // Multi-index over the *column* sub-grid (sizes `k`), canonical order.
        // The full tensor has larger trailing dims, so we index it explicitly
        // rather than walking its flat buffer.
        flat_to_multi(flat, &k, &mut idx);
        w.push(scalar_at(&tab, &idx));
    }
    w
}

/// A built model: the column support points, the values `w`, and the
/// barycentric coefficients `c`. Cheap to evaluate and to clone.
#[derive(Debug, Clone)]
pub struct Barycentric<S = f64> {
    /// Column support points per axis.
    pub lambda: Vec<Vec<S>>,
    /// Sampled values at the column grid (canonical order).
    pub w: Vec<S>,
    /// Barycentric coefficients (canonical order).
    pub c: Vec<S>,
}

impl<S: Scalar> Barycentric<S> {
    /// Number of variables.
    #[inline]
    pub fn ndim(&self) -> usize {
        self.lambda.len()
    }

    /// Per-axis number of support nodes.
    #[inline]
    fn sizes(&self) -> Vec<usize> {
        self.lambda.iter().map(Vec::len).collect()
    }

    /// Evaluate the model at `x` (length `n`).
    ///
    /// `g(x) = N(x) / D(x)` where both `N` and `D` are contractions of a
    /// coefficient tensor against the rank-1 weight structure `⊗_l w_l` (see
    /// `axis_weights`): `D` uses the coefficients `c`, `N` uses `c ⊙ w`.
    /// The contraction is performed axis-by-axis (last/fastest first), reusing the
    /// row-major contiguity — `O(prod k_l)` multiplies and only `Σ k_l` divisions
    /// (the reciprocals are formed once), versus the naive term-by-term sum's
    /// `O(n · prod k_l)` work with `prod k_l` divisions.
    pub fn eval(&self, x: &[S]) -> S {
        debug_assert_eq!(x.len(), self.ndim());
        let dims = self.sizes();
        let (mut sd, mut sn) = (Vec::new(), Vec::new());
        self.eval_core(x, &dims, &mut sd, &mut sn)
    }

    /// Evaluate at many points at once, returning one value per row of `xs`.
    ///
    /// Each query point is independent, so the batch is evaluated in parallel;
    /// each worker reuses a pair of scratch buffers across its points (no
    /// per-point allocation). `xs[i]` must have length `n`. Prefer this over
    /// looping [`Self::eval`] for bulk queries (error sweeps, plotting, the greedy
    /// [`crate::algo::alg2()`] candidate scan).
    pub fn eval_grid(&self, xs: &[&[S]]) -> Vec<S> {
        let dims = self.sizes();
        xs.par_iter()
            .map_init(
                || {
                    (
                        Vec::with_capacity(self.c.len()),
                        Vec::with_capacity(self.c.len()),
                    )
                },
                |(sd, sn), x| self.eval_core(x, &dims, sd, sn),
            )
            .collect()
    }

    /// Core evaluation: contract `c` (→ `D`) and `c ⊙ w` (→ `N`) against the
    /// rank-1 weight structure, reusing `scratch_d`/`scratch_n` as the working
    /// buffers. Returns `N / D`.
    fn eval_core(
        &self,
        x: &[S],
        dims: &[usize],
        scratch_d: &mut Vec<S>,
        scratch_n: &mut Vec<S>,
    ) -> S {
        // Full-node fast path / NaN guard: if `x` lands on a support node along
        // *every* axis, the barycentric limit is exactly the sampled value `w(J)`
        // — return it directly. This also fixes the `0/0 = NaN` failure when the
        // coefficient `c(J)` at that node is zero (a degenerate fit, e.g. a
        // collapsed near-rank-1 model on a non-rational target): the ratio
        // `c(J)·w(J) / c(J)` is `0/0`, but the interpolation value there is
        // unambiguously `w(J)`. See WORKPLAN §6d.
        if let Some(j) = self.full_node_index(x) {
            return self.w[j];
        }

        let weights = axis_weights(&self.lambda, x);

        scratch_d.clear();
        scratch_d.extend_from_slice(&self.c);
        let den = contract_rank1(scratch_d, dims, &weights);

        scratch_n.clear();
        scratch_n.extend(self.c.iter().zip(&self.w).map(|(&c, &w)| c * w));
        let num = contract_rank1(scratch_n, dims, &weights);

        // Guard the remaining `0/0`: a partial-node coincidence where the surviving
        // slice's denominator vanishes (degenerate model) would yield NaN. Return a
        // finite `0` rather than NaN so a poor fit never poisons downstream code
        // (error scans, plotting); a genuine pole gives `±Inf`, which we keep.
        if den == S::zero() {
            return if num == S::zero() {
                S::zero()
            } else {
                num / den
            };
        }
        num / den
    }

    /// If `x` coincides with a support node along **every** axis, return the flat
    /// index `J` of that grid point (canonical order); else `None`. Used by
    /// [`Self::eval_core`] to short-circuit to the exact sampled value `w(J)`.
    fn full_node_index(&self, x: &[S]) -> Option<usize> {
        let mut flat = 0usize;
        for (lam, &xl) in self.lambda.iter().zip(x) {
            let j = lam.iter().position(|&v| v == xl)?;
            flat = flat * lam.len() + j;
        }
        Some(flat)
    }
}

impl<S: Scalar> crate::model::Model<S> for Barycentric<S> {
    fn ndim(&self) -> usize {
        Barycentric::ndim(self)
    }
    fn eval(&self, x: &[S]) -> S {
        Barycentric::eval(self, x)
    }
    fn eval_grid(&self, xs: &[&[S]]) -> Vec<S> {
        Barycentric::eval_grid(self, xs)
    }
}

/// Per-axis evaluation weights at `x`: the barycentric reciprocal
/// `1 / (x_l - lambda_l(j))` for each node `j`, **unless** `x_l` lands exactly on
/// a node, in which case the unit indicator selecting that node.
///
/// The indicator is the removable-singularity limit written cleanly: as
/// `x_l -> lambda_l(j)` the factor `1/(x_l - lambda_l(j))` diverges and dominates
/// both numerator and denominator, so it cancels in the ratio and only the
/// `j_l = j` slice survives — exactly what selecting that index does. With all
/// axes pinned this returns the sampled value `w(J)`. (The MATLAB
/// `eval_lagrangian` instead yields `Inf/Inf = NaN` on a node.)
///
/// Shared by the dense [`Barycentric`] eval and the low-rank
/// [`crate::tt::CompressedBarycentric`] eval, which contract the same weight
/// structure against differently-stored coefficient tensors.
pub(crate) fn axis_weights<S: Scalar>(lambda: &[Vec<S>], x: &[S]) -> Vec<Vec<S>> {
    lambda
        .iter()
        .zip(x)
        .map(|(lam, &xl)| axis_weight(lam, xl))
        .collect()
}

/// The weight vector along one axis at coordinate `xl`: see `axis_weights`.
pub(crate) fn axis_weight<S: Scalar>(lam: &[S], xl: S) -> Vec<S> {
    match lam.iter().position(|&v| v == xl) {
        Some(j) => {
            let mut e = vec![S::zero(); lam.len()];
            e[j] = S::one();
            e
        }
        None => lam.iter().map(|&v| S::one() / (xl - v)).collect(),
    }
}

/// Contract the dense row-major tensor held in `buf` (shape `dims`) against the
/// rank-1 structure `⊗_l weights[l]`, reducing it in place to the scalar
/// `Σ_J buf[J] · ∏_l weights[l][j_l]`.
///
/// Axes are contracted last (fastest/contiguous) first: each pass sums
/// contiguous length-`k_l` runs weighted by `weights[l]`, compacting the result
/// to the front of `buf`. Total work is `O(prod k_l)` multiply-adds with no
/// divisions (the caller supplies reciprocals). This is the dense special case
/// of the factorized tensor contraction the low-rank path generalizes.
fn contract_rank1<S: Scalar>(buf: &mut [S], dims: &[usize], weights: &[Vec<S>]) -> S {
    let mut block = buf.len();
    for (l, wl) in weights.iter().enumerate().rev() {
        let kl = dims[l];
        let outer = block / kl;
        for o in 0..outer {
            let base = o * kl;
            let mut s = S::zero();
            for (j, &w) in wl.iter().enumerate() {
                s += buf[base + j] * w;
            }
            buf[o] = s;
        }
        block = outer;
    }
    buf[0]
}

impl Barycentric<f64> {
    /// Lift the fitted real rational to complex coordinates for analytic
    /// continuation. This evaluates the same rational; it does not refit data.
    pub fn to_complex(&self) -> Barycentric<num_complex::Complex64> {
        let lift = |v: &[f64]| {
            v.iter()
                .map(|&x| num_complex::Complex64::new(x, 0.0))
                .collect()
        };
        Barycentric {
            lambda: self.lambda.iter().map(|a| lift(a)).collect(),
            w: lift(&self.w),
            c: lift(&self.c),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recurse::loewner_null_rec_svd0;
    use crate::tensor::sample;
    use approx::assert_relative_eq;

    #[test]
    fn eval_recovers_2d_rational_octave() {
        // Same model as the recursion golden test; Octave eval(0.3,0.7) gives
        // exactly the true value 0.112233445566779.
        let f = |x: &[f64]| 1.0 / ((x[0] + 3.0) * (x[1] + 2.0));
        let pts = Points::new(
            vec![vec![0.0, 1.0], vec![0.0, 1.0]],
            vec![vec![0.5, 1.5], vec![0.5, 1.5]],
        )
        .unwrap();
        let tab = sample(&f, &pts);
        let c = loewner_null_rec_svd0(&pts, tab.view()).unwrap().c;
        let w = column_values(&pts, tab.view());
        let model = Barycentric {
            lambda: pts.column.clone(),
            w,
            c,
        };
        let got = model.eval(&[0.3, 0.7]);
        assert_relative_eq!(got, f(&[0.3, 0.7]), epsilon = 1e-12);
    }

    #[test]
    fn eval_grid_matches_scalar_eval() {
        let f = |x: &[f64]| 1.0 / ((x[0] + 3.0) * (x[1] + 2.0));
        let pts = Points::new(
            vec![vec![0.0, 1.0], vec![0.0, 1.0]],
            vec![vec![0.5, 1.5], vec![0.5, 1.5]],
        )
        .unwrap();
        let tab = sample(&f, &pts);
        let c = loewner_null_rec_svd0(&pts, tab.view()).unwrap().c;
        let w = column_values(&pts, tab.view());
        let model = Barycentric {
            lambda: pts.column.clone(),
            w,
            c,
        };
        let q0 = [0.3, 0.7];
        let q1 = [-0.4, 1.2];
        let q2 = [0.9, 0.1];
        let xs: [&[f64]; 3] = [&q0, &q1, &q2];
        let batch = model.eval_grid(&xs);
        for (x, got) in xs.iter().zip(&batch) {
            assert_relative_eq!(*got, model.eval(x), epsilon = 1e-15);
        }
    }

    #[test]
    fn column_values_are_leading_block() {
        // f(x1,x2)=x1*10+x2, column pts pick the leading indices.
        let pts = Points::new(vec![vec![1.0], vec![2.0, 3.0]], vec![vec![9.0], vec![8.0]]).unwrap();
        let tab = sample(&|x: &[f64]| x[0] * 10.0 + x[1], &pts);
        // column block: axis0 idx {0}, axis1 idx {0,1} -> values at (1,2),(1,3)
        let w = column_values(&pts, tab.view());
        assert_eq!(w, vec![12.0, 13.0]);
    }

    #[test]
    fn eval_on_node_with_zero_coefficient_is_finite() {
        // Reproduces the user-reported NaN: a degenerate/collapsed fit whose
        // coefficient tensor is zero at most nodes. Querying *on* such a node gave
        // `0/0 = NaN`. With the full-node fast path, eval returns the sampled value
        // `w(J)` there; off a full node the `0/0` guard returns a finite value.
        // 2-axis model, c is zero everywhere except the last (3,3) corner.
        let lambda = vec![vec![1.0, 2.0], vec![10.0, 20.0]];
        let w = vec![5.0, 6.0, 7.0, 8.0]; // sampled values at the 4 nodes
        let mut c = vec![0.0; 4];
        c[3] = 1.0; // only the (1,1) coefficient is nonzero (unit selector)
        let model = Barycentric { lambda, w, c };

        // Query exactly on the (0,0) node, where c == 0: must return w[0] = 5.0,
        // not NaN.
        let got = model.eval(&[1.0, 10.0]);
        assert!(got.is_finite(), "eval on zero-coeff node returned {got}");
        assert_relative_eq!(got, 5.0, epsilon = 1e-12);

        // The nonzero-coefficient node evaluates to its own sample.
        assert_relative_eq!(model.eval(&[2.0, 20.0]), 8.0, epsilon = 1e-12);

        // An off-grid query is finite (no NaN even though most coeffs are zero).
        assert!(model.eval(&[1.5, 15.0]).is_finite());
    }
}
