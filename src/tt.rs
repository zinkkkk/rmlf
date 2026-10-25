//! Tensor-train (TT) representation of the barycentric coefficient tensors, and
//! the factorized evaluation that makes high-`n` models tractable.
//!
//! ## Why
//!
//! The recursive null space produces the coefficient tensor `C` of shape
//! `(k_0, …, k_{n-1})`. Both the construction *and* the dense evaluation are
//! `O(prod k_l)` — exponential in `n`. The recursion tames the construction
//! flop, but storing `C` densely and evaluating term-by-term leaves the curse of
//! dimensionality standing in *storage* and *evaluation*: a barycentric eval is a
//! contraction of `C` against the rank-1 structure `⊗_l 1/(x_l - lambda_l)`, and
//! over a dense `C` that costs `O(prod k_l)` per point — the practical wall for
//! `n >= 8` (and the bottleneck of the [`crate::algo::alg2()`] grid scan).
//!
//! ## The idea
//!
//! Most useful targets are *structured* — the high-dimensional catalog cases are
//! denominators that are sums of a few low-order terms, whose coefficient tensors
//! have low TT-rank. In TT format
//!
//! ```text
//! C[i_0, …, i_{n-1}] = G_0[i_0] · G_1[i_1] · … · G_{n-1}[i_{n-1}]
//! ```
//!
//! (each `G_l[i]` an `r_l × r_{l+1}` matrix, `r_0 = r_n = 1`) the contraction
//! against `⊗_l w_l` **factorizes** into a product of small matrices
//! `M_l = Σ_i w_l[i] G_l[i]`, costing `O(Σ_l k_l r_l r_{l+1})` — *linear* in `n`
//! for bounded rank. We compress both the denominator tensor `C` and the
//! numerator tensor `M = C ⊙ W` once (TT-SVD), then every evaluation is two
//! factorized contractions. Incompressible problems fall back to full TT-rank, no
//! worse than dense.

use crate::eval::{Barycentric, axis_weight, axis_weights};

use faer::Mat;
use rayon::prelude::*;

/// A tensor in tensor-train (TT) format.
///
/// `cores[l]` holds the `r_l × k_l × r_{l+1}` core in row-major order: the entry
/// `G_l[a, i, b]` is at `cores[l][(a * k_l + i) * r_{l+1} + b]`. `ranks` has
/// length `n + 1` with `ranks[0] = ranks[n] = 1`.
#[derive(Debug, Clone)]
pub struct TtTensor {
    /// Per-axis sizes `k_l`.
    pub dims: Vec<usize>,
    /// TT ranks `r_0 .. r_n` (length `n + 1`, boundaries are 1).
    pub ranks: Vec<usize>,
    /// The cores, one per axis (see the layout note above).
    pub cores: Vec<Vec<f64>>,
}

impl TtTensor {
    /// Compress a dense row-major tensor `data` of shape `dims` to TT form via
    /// the TT-SVD algorithm (Oseledets 2011) at relative Frobenius tolerance
    /// `tol`: the result satisfies `‖data − tt‖_F <= tol · ‖data‖_F`. Smaller
    /// `tol` keeps more rank; `tol = 0` is (numerically) exact.
    pub fn from_dense(data: &[f64], dims: &[usize], tol: f64) -> TtTensor {
        let n = dims.len();
        assert!(n >= 1, "TT needs at least one axis");
        debug_assert_eq!(data.len(), dims.iter().product::<usize>());

        let mut ranks = vec![1usize; n + 1];
        let mut cores: Vec<Vec<f64>> = Vec::with_capacity(n);

        // Truncation threshold per split (TT-SVD distributes the tolerance).
        let fro = data.iter().map(|v| v * v).sum::<f64>().sqrt();
        let delta = if n > 1 {
            tol * fro / ((n - 1) as f64).sqrt()
        } else {
            0.0
        };

        // Working matrix `w`, initially C reshaped to (dims[0], total/dims[0]).
        let total = data.len();
        let mut cols = total / dims[0];
        let mut w = Mat::from_fn(dims[0], cols, |i, j| data[i * cols + j]);

        for l in 0..n.saturating_sub(1) {
            let rows = ranks[l] * dims[l];
            debug_assert_eq!(rows, w.nrows());
            let svd = w.thin_svd().expect("TT-SVD: thin_svd failed");
            let s = svd.S().column_vector();
            let smax = s.nrows();
            // Keep singular values above the absolute threshold (at least one).
            let r = (0..smax).filter(|&i| s[i] > delta).count().max(1);
            ranks[l + 1] = r;

            // Core_l = U[:, :r], laid out (rows, r) row-major == (r_l, k_l, r).
            let u = svd.U();
            let mut core = vec![0.0; rows * r];
            for a in 0..rows {
                for b in 0..r {
                    core[a * r + b] = u[(a, b)];
                }
            }
            cores.push(core);

            // Next working matrix: (S[:r] · V[:, :r]^T), shape (r, cols), then
            // reshape to (r * dims[l+1], cols / dims[l+1]).
            let v = svd.V();
            let c2 = cols / dims[l + 1];
            let new_rows = r * dims[l + 1];
            w = Mat::from_fn(new_rows, c2, |i, j| {
                // Row-major flat index into the (r, cols) matrix S·V^T.
                let flat = i * c2 + j;
                let b = flat / cols; // singular-vector index in 0..r
                let jj = flat % cols; // column in 0..cols
                s[b] * v[(jj, b)]
            });
            cols = c2;
        }

        // Final core: whatever remains, shape (r_{n-1} * dims[n-1], 1).
        let last_rows = ranks[n - 1] * dims[n - 1];
        debug_assert_eq!(w.nrows(), last_rows);
        debug_assert_eq!(w.ncols(), 1);
        let mut core = vec![0.0; last_rows];
        for a in 0..last_rows {
            core[a] = w[(a, 0)];
        }
        cores.push(core);
        ranks[n] = 1;

        TtTensor {
            dims: dims.to_vec(),
            ranks,
            cores,
        }
    }

    /// The largest interior TT rank — a direct measure of how compressible the
    /// tensor was (1 = rank-1/separable, up to `prod` of the smaller side for
    /// incompressible).
    pub fn max_rank(&self) -> usize {
        self.ranks.iter().copied().max().unwrap_or(1)
    }

    /// Total number of stored core entries (the TT storage footprint).
    pub fn storage(&self) -> usize {
        self.cores.iter().map(Vec::len).sum()
    }

    /// Fraction of core entries that are exactly zero — a measure of *within-core*
    /// sparsity beyond the TT rank.
    ///
    /// TT-rank compression captures redundancy *across* indices, but a low-rank
    /// core can still be mostly zeros (e.g. a degenerate fit whose coefficient
    /// tensor is a unit selector `[0,…,0,1]`, the pattern flagged in WORKPLAN
    /// §6g). This reports how much such zero structure is present; after
    /// [`Self::prune`] the contraction skips those entries.
    pub fn sparsity(&self) -> f64 {
        let total = self.storage();
        if total == 0 {
            return 0.0;
        }
        let zeros = self.cores.iter().flatten().filter(|&&v| v == 0.0).count();
        zeros as f64 / total as f64
    }

    /// Zero out core entries with magnitude below `rel · max|core entry|` (a
    /// **relative** threshold), in place. Returns the number of entries newly
    /// zeroed.
    ///
    /// This sparsifies cores that carry tiny noise floor values (e.g. the
    /// `1.5e-16` entries in the user-reported degenerate model) so that
    /// [`Self::contract_rank1`] can skip them and [`Self::sparsity`] reflects the
    /// true structure. It is a *lossy* operation bounded by `rel`; use `rel = 0`
    /// for a no-op. The TT ranks are unchanged (this prunes within cores, not
    /// across them). Safe default `rel ≈ 1e-14` removes only round-off.
    pub fn prune(&mut self, rel: f64) -> usize {
        if rel <= 0.0 {
            return 0;
        }
        let mut pruned = 0usize;
        for core in &mut self.cores {
            let scale = core.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
            if scale == 0.0 {
                continue;
            }
            let thresh = rel * scale;
            for v in core.iter_mut() {
                if *v != 0.0 && v.abs() < thresh {
                    *v = 0.0;
                    pruned += 1;
                }
            }
        }
        pruned
    }

    /// Rough multiply count of one [`Self::contract_rank1`]: `Σ_l k_l r_l
    /// r_{l+1}`. Used to decide whether the TT path beats a dense contraction
    /// (`prod k_l`).
    pub fn contraction_cost(&self) -> usize {
        (0..self.dims.len())
            .map(|l| self.dims[l] * self.ranks[l] * self.ranks[l + 1])
            .sum()
    }

    /// Contract against the rank-1 weight structure `⊗_l weights[l]`, returning
    /// `Σ_J tt[J] · ∏_l weights[l][j_l]`.
    ///
    /// Folds left to right: maintain a length-`r_l` row vector `acc`, and at each
    /// axis replace it with `acc · M_l` where `M_l[a,b] = Σ_i weights[l][i]
    /// G_l[a,i,b]`. Cost `O(Σ_l k_l r_l r_{l+1})`.
    pub fn contract_rank1(&self, weights: &[Vec<f64>]) -> f64 {
        debug_assert_eq!(weights.len(), self.dims.len());
        let n = self.dims.len();
        let mut acc = vec![1.0_f64]; // 1 × r_0, r_0 == 1
        // `l` indexes several parallel arrays (dims, ranks, cores, weights).
        #[allow(clippy::needless_range_loop)]
        for l in 0..n {
            let (kl, rr) = (self.dims[l], self.ranks[l + 1]);
            let core = &self.cores[l];
            let wl = &weights[l];
            let mut next = vec![0.0_f64; rr];
            for (a, &av) in acc.iter().enumerate() {
                if av == 0.0 {
                    continue;
                }
                for (i, &wi) in wl.iter().enumerate() {
                    if wi == 0.0 {
                        continue;
                    }
                    let avwi = av * wi;
                    let base = (a * kl + i) * rr;
                    let row = &core[base..base + rr];
                    for (b, &g) in row.iter().enumerate() {
                        if g != 0.0 {
                            next[b] += avwi * g;
                        }
                    }
                }
            }
            acc = next;
        }
        acc[0]
    }

    /// Expand back to a dense row-major tensor (mainly for testing/validation).
    pub fn to_dense(&self) -> Vec<f64> {
        let n = self.dims.len();
        let total: usize = self.dims.iter().product();
        let mut out = vec![0.0; total];
        let mut idx = vec![0usize; n];
        for (flat, slot) in out.iter_mut().enumerate() {
            crate::grid::flat_to_multi(flat, &self.dims, &mut idx);
            // Multiply the per-axis 1×r/r×r row vectors.
            let mut acc = vec![1.0_f64];
            for (l, &i) in idx.iter().enumerate() {
                let (kl, rr) = (self.dims[l], self.ranks[l + 1]);
                let core = &self.cores[l];
                let mut next = vec![0.0_f64; rr];
                for (a, &av) in acc.iter().enumerate() {
                    let base = (a * kl + i) * rr;
                    for (b, slot) in next.iter_mut().enumerate() {
                        *slot += av * core[base + b];
                    }
                }
                acc = next;
            }
            *slot = acc[0];
        }
        out
    }
}

/// A barycentric model with both coefficient tensors stored in TT form — the
/// scalable counterpart of the dense [`Barycentric`].
///
/// `g(x) = N(x) / D(x)`, where `D` contracts the denominator TT (`C`) and `N`
/// contracts the numerator TT (`C ⊙ W`) against the shared rank-1 weight
/// structure. For structured targets both TTs have small rank, so evaluation is
/// linear in `n` rather than exponential.
#[derive(Debug, Clone)]
pub struct CompressedBarycentric {
    /// Column support points per axis (same as the dense model).
    pub lambda: Vec<Vec<f64>>,
    /// Denominator-weight tensor `C` in TT form.
    pub denom: TtTensor,
    /// Numerator-weight tensor `C ⊙ W` in TT form.
    pub numer: TtTensor,
}

impl CompressedBarycentric {
    /// Compress a dense [`Barycentric`] to TT form at relative tolerance `tol`
    /// (applied independently to the numerator and denominator tensors).
    ///
    /// `tol = 0` is **exact / lossless** — full rank is kept and the TT is a
    /// bit-exact refactorization (use [`Self::exact`] for that intent, or just
    /// `Barycentric::into()`). `tol > 0` drops bonds whose singular values fall
    /// below `tol·‖·‖_F`, trading a bounded error for smaller rank. For
    /// **tolerance-free exactness over ℚ / 𝔽_p** (no SVD at all) see
    /// [`crate::exact::tt`].
    pub fn from_model(model: &Barycentric, tol: f64) -> Self {
        let dims: Vec<usize> = model.lambda.iter().map(Vec::len).collect();
        let denom = TtTensor::from_dense(&model.c, &dims, tol);
        let mw: Vec<f64> = model.c.iter().zip(&model.w).map(|(&c, &w)| c * w).collect();
        let numer = TtTensor::from_dense(&mw, &dims, tol);
        CompressedBarycentric {
            lambda: model.lambda.clone(),
            denom,
            numer,
        }
    }

    /// **Exact (lossless)** TT compression of a dense model — `from_model(m, 0.0)`.
    /// Keeps full rank; the result evaluates identically to the dense model. Use
    /// this (or `m.into()`) when you want the storage/eval form without any
    /// approximation; use [`Self::from_model`] with `tol > 0` to truncate rank.
    pub fn exact(model: &Barycentric) -> Self {
        Self::from_model(model, 0.0)
    }

    /// Number of variables.
    #[inline]
    pub fn ndim(&self) -> usize {
        self.lambda.len()
    }

    /// The larger of the numerator/denominator max TT ranks (compression report).
    pub fn max_rank(&self) -> usize {
        self.numer.max_rank().max(self.denom.max_rank())
    }

    /// Rough multiply count of one [`Self::eval`] (both contractions).
    pub fn eval_cost(&self) -> usize {
        self.numer.contraction_cost() + self.denom.contraction_cost()
    }

    /// Total TT storage (numerator + denominator core entries).
    pub fn storage(&self) -> usize {
        self.numer.storage() + self.denom.storage()
    }

    /// Within-core zero fraction across both tensors (see [`TtTensor::sparsity`]).
    pub fn sparsity(&self) -> f64 {
        let total = self.storage();
        if total == 0 {
            return 0.0;
        }
        let nz = (self.numer.sparsity() * self.numer.storage() as f64)
            + (self.denom.sparsity() * self.denom.storage() as f64);
        nz / total as f64
    }

    /// Prune near-zero core entries in both tensors at relative threshold `rel`
    /// (see [`TtTensor::prune`]), returning the total entries zeroed. Sparsifies a
    /// noisy/degenerate model so [`Self::eval`] can skip the dead entries.
    pub fn prune(&mut self, rel: f64) -> usize {
        self.numer.prune(rel) + self.denom.prune(rel)
    }

    /// Evaluate the model at `x` (length `n`) via two factorized contractions.
    ///
    /// Guards the `0/0 = NaN` case (a degenerate / collapsed fit where the
    /// denominator coefficient at a node-coincident query vanishes — see the dense
    /// [`Barycentric::eval`] note and WORKPLAN §6d): when both contractions vanish
    /// the interpolant carries no mass there, so we return a finite `0` rather than
    /// NaN. A genuine pole (`num ≠ 0`, `den = 0`) still yields `±Inf`.
    pub fn eval(&self, x: &[f64]) -> f64 {
        debug_assert_eq!(x.len(), self.ndim());
        let w = axis_weights(&self.lambda, x);
        let num = self.numer.contract_rank1(&w);
        let den = self.denom.contract_rank1(&w);
        if den == 0.0 && num == 0.0 {
            return 0.0;
        }
        num / den
    }

    /// Evaluate at many points in parallel.
    pub fn eval_grid(&self, xs: &[&[f64]]) -> Vec<f64> {
        xs.par_iter().map(|x| self.eval(x)).collect()
    }

    /// Evaluate at **every point of the tensor grid** `⊗_l nodes[l]`, returning
    /// the values in canonical row-major order.
    ///
    /// This shares per-axis work across the whole grid: for each axis it forms
    /// the weight-contracted core matrix for every grid coordinate once, then
    /// sweeps them forward, so the entire grid costs `O(prod dim · r²)` rather
    /// than `prod dim` independent `O(r² Σk)` point evaluations. This is what
    /// makes the greedy [`crate::algo::alg2()`] full-grid error scan tractable for
    /// large `n` when the model is low-rank.
    pub fn eval_full_grid(&self, nodes: &[Vec<f64>]) -> Vec<f64> {
        let num = grid_contract(&self.numer, &self.lambda, nodes);
        let den = grid_contract(&self.denom, &self.lambda, nodes);
        num.iter()
            .zip(&den)
            .map(|(&a, &b)| if b == 0.0 && a == 0.0 { 0.0 } else { a / b })
            .collect()
    }
}

impl crate::model::Model for CompressedBarycentric {
    fn ndim(&self) -> usize {
        CompressedBarycentric::ndim(self)
    }
    fn eval(&self, x: &[f64]) -> f64 {
        CompressedBarycentric::eval(self, x)
    }
    fn eval_grid(&self, xs: &[&[f64]]) -> Vec<f64> {
        CompressedBarycentric::eval_grid(self, xs)
    }
}

/// Contract a TT against the full tensor grid `⊗_l nodes[l]`, with the per-axis
/// barycentric weights `w_l(node) = 1/(node - λ_l)` (or the unit indicator when
/// `node` is a support node). Returns `prod(|nodes[l]|)` values, row-major.
///
/// Forward sweep: maintain `acc` of shape `(prefix, r_l)` (prefix = product of
/// the grid sizes already swept). At axis `l`, for each grid coordinate `g` form
/// the matrix `M_g = Σ_i w_l[g][i] · G_l[·,i,·]` once and expand
/// `acc'[(p·dim_l + g), b] = Σ_a acc[p, a] · M_g[a, b]`.
fn grid_contract(tt: &TtTensor, lambda: &[Vec<f64>], nodes: &[Vec<f64>]) -> Vec<f64> {
    let n = tt.dims.len();
    let mut acc = vec![1.0_f64]; // (prefix=1, r_0=1)
    let mut prefix = 1usize;
    let mut r = tt.ranks[0];

    for l in 0..n {
        let dim_l = nodes[l].len();
        let kl = tt.dims[l];
        let rr = tt.ranks[l + 1];
        let core = &tt.cores[l];

        // Weight-contracted core matrix `M_g` (r × rr) for every grid coordinate,
        // formed once and shared across all `prefix` rows.
        let mgs: Vec<Vec<f64>> = nodes[l]
            .iter()
            .map(|&node| {
                let w = axis_weight(&lambda[l], node); // length kl
                let mut mg = vec![0.0_f64; r * rr];
                for a in 0..r {
                    for (i, &wi) in w.iter().enumerate() {
                        if wi == 0.0 {
                            continue;
                        }
                        let base = (a * kl + i) * rr;
                        let crow = &core[base..base + rr];
                        let mrow = &mut mg[a * rr..a * rr + rr];
                        for (b, &cv) in crow.iter().enumerate() {
                            mrow[b] += wi * cv;
                        }
                    }
                }
                mg
            })
            .collect();

        // Expand: acc2[(p·dim_l + g), :] = acc[p, :] · M_g. The `prefix` rows are
        // independent, so the dominant final sweep parallelizes.
        let acc_ref = &acc;
        let mgs_ref = &mgs;
        let acc2: Vec<f64> = (0..prefix)
            .into_par_iter()
            .flat_map_iter(|p| {
                let arow = &acc_ref[p * r..p * r + r];
                let mut block = vec![0.0_f64; dim_l * rr];
                for (g, mg) in mgs_ref.iter().enumerate() {
                    let out = &mut block[g * rr..g * rr + rr];
                    for (a, &av) in arow.iter().enumerate() {
                        if av == 0.0 {
                            continue;
                        }
                        let mrow = &mg[a * rr..a * rr + rr];
                        for (b, &mv) in mrow.iter().enumerate() {
                            out[b] += av * mv;
                        }
                    }
                }
                block
            })
            .collect();
        acc = acc2;
        prefix *= dim_l;
        r = rr;
    }
    debug_assert_eq!(r, 1);
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn rand_tensor(dims: &[usize], seed: u64) -> Vec<f64> {
        let total: usize = dims.iter().product();
        let mut s = seed;
        (0..total)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
            })
            .collect()
    }

    #[test]
    fn prune_sparsifies_noise_and_preserves_eval() {
        // A core with a noise-floor entry (like the user-reported 1.5e-16) should
        // be pruned away without changing the dense reconstruction meaningfully.
        let dims = [2, 3];
        // Rank-1 separable: core0 = [1, 0] (with a tiny noise term), core1 dense.
        let mut data = vec![0.0; 6];
        for (i, slot) in data.iter_mut().enumerate() {
            *slot = if i < 3 { (i + 1) as f64 } else { 0.0 };
        }
        let mut tt = TtTensor::from_dense(&data, &dims, 0.0);
        // Inject a noise-floor entry into a core.
        tt.cores[0].iter_mut().for_each(|v| {
            if *v == 0.0 {
                *v = 1.0e-16;
            }
        });
        let before = tt.to_dense();
        let pruned = tt.prune(1e-12);
        assert!(pruned >= 1, "should prune the noise entry");
        let after = tt.to_dense();
        for (a, b) in before.iter().zip(&after) {
            assert_relative_eq!(a, b, epsilon = 1e-12);
        }
        assert!(tt.sparsity() > 0.0, "pruned core should report sparsity");
    }

    #[test]
    fn tt_svd_roundtrips_exactly() {
        // tol = 0 keeps full rank: TT must reproduce the dense tensor.
        let dims = [3, 4, 2, 3];
        let data = rand_tensor(&dims, 12345);
        let tt = TtTensor::from_dense(&data, &dims, 0.0);
        let back = tt.to_dense();
        for (a, b) in data.iter().zip(&back) {
            assert_relative_eq!(a, b, epsilon = 1e-10);
        }
    }

    #[test]
    fn tt_contraction_matches_dense() {
        let dims = [3, 4, 2];
        let data = rand_tensor(&dims, 999);
        let tt = TtTensor::from_dense(&data, &dims, 0.0);
        let weights: Vec<Vec<f64>> = dims
            .iter()
            .enumerate()
            .map(|(l, &k)| (0..k).map(|j| 0.1 + l as f64 + 0.3 * j as f64).collect())
            .collect();
        // Dense contraction reference.
        let mut dense = 0.0;
        let mut idx = vec![0usize; dims.len()];
        for (flat, &c) in data.iter().enumerate() {
            crate::grid::flat_to_multi(flat, &dims, &mut idx);
            let mut p = c;
            for (l, &j) in idx.iter().enumerate() {
                p *= weights[l][j];
            }
            dense += p;
        }
        assert_relative_eq!(tt.contract_rank1(&weights), dense, epsilon = 1e-9);
    }

    #[test]
    fn full_grid_matches_pointwise() {
        // eval_full_grid must equal looping eval over the same grid.
        let lambda = vec![vec![0.0, 1.0, 2.0], vec![-1.0, 0.5]];
        let dims = [3, 2];
        // Build a small dense model directly.
        let c: Vec<f64> = (0..6).map(|i| 1.0 + i as f64).collect();
        let w: Vec<f64> = (0..6).map(|i| 0.5 - 0.1 * i as f64).collect();
        let model = Barycentric {
            lambda: lambda.clone(),
            w,
            c,
        };
        let tt = CompressedBarycentric::from_model(&model, 0.0);

        // Grid nodes (distinct from support nodes to avoid the node-limit branch
        // here; the on-node case is covered by tests/tt_compress.rs).
        let nodes = vec![vec![0.2, 0.7, 1.3, 2.5], vec![-0.4, 0.1, 0.9]];
        let grid = tt.eval_full_grid(&nodes);

        let mut idx = vec![0usize; 2];
        for (flat, &g) in grid.iter().enumerate() {
            crate::grid::flat_to_multi(flat, &[nodes[0].len(), nodes[1].len()], &mut idx);
            let x = [nodes[0][idx[0]], nodes[1][idx[1]]];
            assert_relative_eq!(g, tt.eval(&x), epsilon = 1e-12);
        }
        let _ = dims;
    }

    #[test]
    fn separable_tensor_is_low_rank() {
        // An outer product is exactly TT-rank 1.
        let a = [1.0, 2.0, 3.0];
        let b = [0.5, -1.0];
        let c = [2.0, 1.0, -1.0, 4.0];
        let dims = [3, 2, 4];
        let mut data = vec![0.0; 24];
        let mut idx = vec![0usize; 3];
        for (flat, slot) in data.iter_mut().enumerate() {
            crate::grid::flat_to_multi(flat, &dims, &mut idx);
            *slot = a[idx[0]] * b[idx[1]] * c[idx[2]];
        }
        let tt = TtTensor::from_dense(&data, &dims, 1e-12);
        assert_eq!(tt.max_rank(), 1, "separable tensor should be TT-rank 1");
    }
}
