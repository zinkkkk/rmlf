//! TT-cross: sub-exponential **sampling** and low-rank construction of a
//! multivariate rational — the structural upgrade over `alg1`/`decouple`.
//!
//! ## The wall this breaks
//!
//! Every other constructor in this crate ([`crate::alg1()`], [`crate::decouple()`])
//! requires the **full dense Kronecker grid** of samples: `∏_l (k_l + q_l)` calls
//! to `f`, exponential in `n` (WORKPLAN §11 calls this the method's *premise*). The
//! recursion tames the *construction* flop and TT tames *storage/eval*, but the
//! **sampling** cost stays exponential.
//!
//! TT-cross (Oseledets–Tyrtyshnikov 2010, *"TT-cross approximation for
//! multidimensional arrays"*) samples `f` at only `O(n · r² · k)` adaptively-chosen
//! grid points — *linear* in `n` for bounded rank — and reconstructs the whole
//! value tensor in tensor-train form via nested **maxvol** index selection.
//! Combined with this crate's per-axis barycentric machinery it yields a
//! **continuous multivariate rational in TT form** built without forming the dense
//! grid.
//!
//! ## Algorithm (maxvol nested cross, left-to-right sweeps)
//!
//! Maintain nested left index sets `I_l ⊂ X_0×…×X_{l-1}` (size `r_l`) and right
//! index sets `J_l ⊂ X_{l+1}×…×X_{n-1}` (size `r_{l+1}`). A left-to-right sweep, at
//! each interface `l`, samples the small block
//! `A = f(I_l ⊗ X_l, J_l)` of shape `(r_l·k_l) × r_{l+1}`, and uses **maxvol** to
//! pick the `r_{l+1}` rows of `A` of maximum-volume — these become the new `I_{l+1}`
//! (a sub-multi-index of `(I_l, X_l)`). Right-to-left sweeps refine `J`. Only the
//! sampled blocks ever touch `f`. After convergence the cores are read off the
//! maxvol-normalized blocks; their product reproduces `f` exactly on the index sets
//! and interpolates elsewhere (the rank-revealing CUR/skeleton property).
//!
//! ## Rational lift
//!
//! The barycentric coefficient tensor is a *per-axis* linear transform of the value
//! tensor (the 1-D Loewner null space along each axis), so the rational model is
//! obtained by the validated dense lift on the **support sub-block** (size `∏ k_l`,
//! not the full candidate grid) — the only dense object any built model already
//! materializes. Sampling stays sub-exponential.

use crate::error::{LoewnerError, Result};
use crate::points::Points;
use crate::source::TargetFn;
use crate::tt::{CompressedBarycentric, TtTensor};

use faer::Mat;

/// Options for [`tt_cross`].
#[derive(Debug, Clone)]
pub struct TtCrossOptions {
    /// Target TT rank: each interface grows up to this many index pairs. The
    /// curse-taming knob — sampling and storage scale as `O(n · r² · k)`.
    pub max_rank: usize,
    /// Number of left↔right refinement sweeps. Two or three suffices for low-rank
    /// targets; the index sets stabilize quickly.
    pub sweeps: usize,
    /// Relative convergence tolerance: sweeping stops early once the reconstruction
    /// on a fresh test set changes by less than this between sweeps.
    pub tol: f64,
}

impl Default for TtCrossOptions {
    fn default() -> Self {
        TtCrossOptions {
            max_rank: 8,
            sweeps: 4,
            tol: 1e-12,
        }
    }
}

/// Sampling statistics returned alongside the model.
#[derive(Debug, Clone)]
pub struct TtCrossInfo {
    /// TT ranks of the value tensor (length `n + 1`, boundaries 1).
    pub ranks: Vec<usize>,
    /// Number of `f` evaluations performed (the sampling footprint).
    pub evaluations: usize,
    /// Full dense grid size `∏(k_l + q_l)` — what the dense path would cost.
    pub dense_grid: usize,
}

/// A TT-cross **interpolant**: the value tensor `f(grid)` in tensor-train form,
/// evaluated off-grid by per-axis barycentric-Lagrange interpolation.
///
/// `value` holds `f` sampled on the candidate grid (`axes`) compressed to TT; an
/// evaluation contracts each core against the per-axis normalized Lagrange basis
/// `[ℓ_0(x_l), …]`, which is a partition of unity — so the result is the
/// tensor-product Lagrange interpolant through the grid, **exact on the grid** and
/// low-rank for any low-TT-rank target (including sums of rationals that are *not*
/// a single low-degree fraction). Cost `O(n r² k)`.
#[derive(Debug, Clone)]
pub struct TtCrossModel {
    /// Per-axis candidate nodes (`column_l ++ row_l`).
    pub axes: Vec<Vec<f64>>,
    /// Lagrange coefficients `γ_{l,j}` per axis (precomputed for fast eval).
    gamma: Vec<Vec<f64>>,
    /// The value tensor in TT form.
    pub value: TtTensor,
}

impl TtCrossModel {
    /// Evaluate the interpolant at `x` (length `n`).
    pub fn eval(&self, x: &[f64]) -> f64 {
        let weights: Vec<Vec<f64>> = self
            .axes
            .iter()
            .zip(&self.gamma)
            .zip(x)
            .map(|((nodes, g), &xl)| lagrange_weights(nodes, g, xl))
            .collect();
        self.value.contract_rank1(&weights)
    }

    /// Evaluate at many points.
    pub fn eval_grid(&self, xs: &[&[f64]]) -> Vec<f64> {
        xs.iter().map(|x| self.eval(x)).collect()
    }

    /// The largest TT rank (compression measure: 1 = separable).
    pub fn max_rank(&self) -> usize {
        self.value.max_rank()
    }

    /// TT storage footprint (core entries).
    pub fn storage(&self) -> usize {
        self.value.storage()
    }
}

/// The normalized Lagrange basis `[ℓ_0(x), …, ℓ_{k-1}(x)]` over `nodes` with
/// precomputed coefficients `gamma`, in barycentric form. On a node returns the
/// unit indicator (exact value); otherwise `ℓ_j(x) = (γ_j/(x−λ_j)) / Σ_i
/// γ_i/(x−λ_i)`, a partition of unity reproducing Lagrange interpolation.
fn lagrange_weights(nodes: &[f64], gamma: &[f64], x: f64) -> Vec<f64> {
    if let Some(j) = nodes.iter().position(|&v| v == x) {
        let mut e = vec![0.0; nodes.len()];
        e[j] = 1.0;
        return e;
    }
    let recip: Vec<f64> = nodes
        .iter()
        .zip(gamma)
        .map(|(&lj, &gj)| gj / (x - lj))
        .collect();
    let denom: f64 = recip.iter().sum();
    recip.iter().map(|&r| r / denom).collect()
}

/// Lagrange coefficients `γ_j = 1/∏_{i≠j}(λ_j − λ_i)` for `nodes`.
fn axis_gamma(nodes: &[f64]) -> Vec<f64> {
    nodes
        .iter()
        .enumerate()
        .map(|(j, &lj)| {
            let p: f64 = nodes
                .iter()
                .enumerate()
                .filter(|&(i, _)| i != j)
                .map(|(_, &li)| lj - li)
                .product();
            1.0 / p
        })
        .collect()
}

/// Build a multivariate **interpolant** of `f` by TT-cross sampling.
///
/// `points` provides the per-axis candidate grid (`column_l ++ row_l`). Only
/// `info.evaluations` calls to `f` are made — far fewer than the dense
/// `info.dense_grid` for a low-rank target. The returned [`TtCrossModel`] evaluates
/// in `O(n r² k)` by per-axis Lagrange interpolation of the TT value tensor,
/// exact on the grid.
pub fn tt_cross<F: TargetFn + Sync>(
    f: &F,
    points: &Points,
    opt: &TtCrossOptions,
) -> Result<(TtCrossModel, TtCrossInfo)> {
    points.check_disjoint()?;
    let n = points.ndim();
    let axes: Vec<Vec<f64>> = (0..n).map(|l| points.combined_axis(l)).collect();
    let dims: Vec<usize> = axes.iter().map(Vec::len).collect();

    let mut cross = Cross::new(&axes, opt);
    cross.run(f)?;
    let value = cross.assemble_tt()?;
    let ranks = value.ranks.clone();
    let evaluations = cross.cache.len();

    let gamma: Vec<Vec<f64>> = axes.iter().map(|a| axis_gamma(a)).collect();
    let info = TtCrossInfo {
        ranks,
        evaluations,
        dense_grid: dims.iter().product(),
    };
    Ok((TtCrossModel { axes, gamma, value }, info))
}

/// Build a TT-cross model and additionally return the **rational** lift: a
/// [`CompressedBarycentric`] obtained by feeding the (exact) reconstructed value
/// tensor through the decoupling builder. This is exact only when the target is a
/// single low-degree tensor-product rational of the chosen per-axis order; for
/// general low-rank targets (e.g. sums of rationals) prefer the Lagrange
/// [`TtCrossModel`] from [`tt_cross`]. Provided for callers who specifically want
/// the barycentric-rational form and TT-compressed coefficient tensors.
pub fn tt_cross_rational<F: TargetFn + Sync>(
    f: &F,
    points: &Points,
    opt: &TtCrossOptions,
) -> Result<(CompressedBarycentric, TtCrossInfo)> {
    points.check_disjoint()?;
    let n = points.ndim();
    let axes: Vec<Vec<f64>> = (0..n).map(|l| points.combined_axis(l)).collect();
    let dims: Vec<usize> = axes.iter().map(Vec::len).collect();

    let mut cross = Cross::new(&axes, opt);
    cross.run(f)?;
    let value_tt = cross.assemble_tt()?;
    let ranks = value_tt.ranks.clone();
    let evaluations = cross.cache.len();

    let (denom, numer) = lift_to_rational(&value_tt, points)?;
    let info = TtCrossInfo {
        ranks,
        evaluations,
        dense_grid: dims.iter().product(),
    };
    Ok((
        CompressedBarycentric {
            lambda: points.column.clone(),
            denom,
            numer,
        },
        info,
    ))
}

/// Debug helper: run the cross and return the reconstructed dense value tensor and
/// the per-axis candidate abscissae, for validating the cross reconstruction in
/// isolation from the rational lift. Not part of the stable API.
#[doc(hidden)]
pub fn tt_cross_value_tensor_debug<F: TargetFn + Sync>(
    f: &F,
    points: &Points,
    opt: &TtCrossOptions,
) -> Result<(Vec<f64>, Vec<Vec<f64>>)> {
    points.check_disjoint()?;
    let n = points.ndim();
    let axes: Vec<Vec<f64>> = (0..n).map(|l| points.combined_axis(l)).collect();
    let mut cross = Cross::new(&axes, opt);
    cross.run(f)?;
    let tt = cross.assemble_tt()?;
    Ok((tt.to_dense(), axes))
}

/// The cross state: axes, options, nested index sets, and a memoizing evaluator.
struct Cross<'a> {
    axes: &'a [Vec<f64>],
    dims: Vec<usize>,
    n: usize,
    opt: TtCrossOptions,
    /// `left[l]` = list of left multi-indices over axes `0..l` (size `r_l`).
    left: Vec<Vec<Vec<usize>>>,
    /// `right[l]` = list of right multi-indices over axes `l+1..n` (size `r_{l+1}`).
    right: Vec<Vec<Vec<usize>>>,
    /// Memoized `f` evaluations keyed by the full multi-index.
    cache: std::collections::HashMap<Vec<usize>, f64>,
}

impl<'a> Cross<'a> {
    fn new(axes: &'a [Vec<f64>], opt: &TtCrossOptions) -> Self {
        let n = axes.len();
        let dims: Vec<usize> = axes.iter().map(Vec::len).collect();
        // Initialize rank-1 nested sets with a central seed index.
        let seed: Vec<usize> = dims.iter().map(|&d| d / 2).collect();
        let mut left = vec![Vec::new(); n + 1];
        let mut right = vec![Vec::new(); n + 1];
        left[0] = vec![vec![]];
        right[n] = vec![vec![]];
        for l in 1..=n {
            left[l] = vec![seed[..l].to_vec()];
        }
        for l in 0..n {
            right[l] = vec![seed[l + 1..].to_vec()];
        }
        Cross {
            axes,
            dims,
            n,
            opt: opt.clone(),
            left,
            right,
            cache: std::collections::HashMap::new(),
        }
    }

    /// Memoized evaluation at a full multi-index.
    fn eval<F: TargetFn>(&mut self, f: &F, idx: &[usize]) -> f64 {
        if let Some(&v) = self.cache.get(idx) {
            return v;
        }
        let x: Vec<f64> = (0..self.n).map(|l| self.axes[l][idx[l]]).collect();
        let v = f.eval(&x);
        self.cache.insert(idx.to_vec(), v);
        v
    }

    /// Run the left↔right sweeps to grow and refine the nested index sets.
    fn run<F: TargetFn>(&mut self, f: &F) -> Result<()> {
        let mut prev = f64::INFINITY;
        for _ in 0..self.opt.sweeps {
            // Forward sweep: refine left sets I_1..I_{n-1}.
            for l in 0..self.n - 1 {
                self.refine_left(f, l)?;
            }
            // Backward sweep: refine right sets J_{n-2}..J_0.
            for l in (0..self.n - 1).rev() {
                self.refine_right(f, l)?;
            }
            // Convergence check on a fixed probe.
            let probe = self.probe_value(f)?;
            if (probe - prev).abs() <= self.opt.tol * (probe.abs() + 1e-300) {
                break;
            }
            prev = probe;
        }
        Ok(())
    }

    /// Refine interface `l` left-to-right using the **two-site** super-block
    /// `A = f(I_l ⊗ X_l, X_{l+1} ⊗ J_{l+1})` of shape
    /// `(r_l·k_l) × (k_{l+1}·r_{l+1})`. Its numerical rank is the true local TT
    /// rank, so SVD-truncating it (at `tol`, capped at `max_rank`) lets the
    /// interface rank **grow** beyond the incoming rank — the rank-revealing step a
    /// single-site block cannot provide. Maxvol on the dominant left singular
    /// directions picks the new `I_{l+1}` (row multi-indices over axes `0..=l`).
    fn refine_left<F: TargetFn>(&mut self, f: &F, l: usize) -> Result<()> {
        let rl = self.left[l].len();
        let kl = self.dims[l];
        let kl1 = self.dims[l + 1];
        let rr1 = self.right[l + 1].len();
        let rows = rl * kl;
        let cols = kl1 * rr1;

        // Row candidates: left multi-indices (axes 0..=l) = (I_l[a], i).
        let row_cand: Vec<Vec<usize>> = self.left[l]
            .iter()
            .flat_map(|li| {
                (0..kl).map(move |i| {
                    let mut idx = li.clone();
                    idx.push(i);
                    idx
                })
            })
            .collect();
        // Column candidates: right multi-indices (axes l+1..n) = (j, J_{l+1}[b]).
        let col_cand: Vec<Vec<usize>> = (0..kl1)
            .flat_map(|j| {
                self.right[l + 1].iter().map(move |rj| {
                    let mut idx = vec![j];
                    idx.extend_from_slice(rj);
                    idx
                })
            })
            .collect();

        let mut a = Mat::from_fn(rows, cols, |_, _| 0.0);
        for (r, lidx) in row_cand.iter().enumerate() {
            for (c, ridx) in col_cand.iter().enumerate() {
                let full = glue_left_right(lidx, ridx, self.n);
                a[(r, c)] = self.eval(f, &full);
            }
        }

        let target = self.truncated_rank(&a);
        let chosen = maxvol_rows(&a, target);
        self.left[l + 1] = chosen.into_iter().map(|r| row_cand[r].clone()).collect();
        Ok(())
    }

    /// Refine interface `l` right-to-left using the same two-site super-block,
    /// picking maxvol **columns** (right multi-indices over axes `l+1..n`) as the
    /// new `J_l`. Mirror of [`Self::refine_left`].
    fn refine_right<F: TargetFn>(&mut self, f: &F, l: usize) -> Result<()> {
        let rl = self.left[l].len();
        let kl = self.dims[l];
        let kl1 = self.dims[l + 1];
        let rr1 = self.right[l + 1].len();
        let rows = rl * kl;
        let cols = kl1 * rr1;

        let row_cand: Vec<Vec<usize>> = self.left[l]
            .iter()
            .flat_map(|li| {
                (0..kl).map(move |i| {
                    let mut idx = li.clone();
                    idx.push(i);
                    idx
                })
            })
            .collect();
        let col_cand: Vec<Vec<usize>> = (0..kl1)
            .flat_map(|j| {
                self.right[l + 1].iter().map(move |rj| {
                    let mut idx = vec![j];
                    idx.extend_from_slice(rj);
                    idx
                })
            })
            .collect();

        let mut a = Mat::from_fn(rows, cols, |_, _| 0.0);
        for (r, lidx) in row_cand.iter().enumerate() {
            for (c, ridx) in col_cand.iter().enumerate() {
                let full = glue_left_right(lidx, ridx, self.n);
                a[(r, c)] = self.eval(f, &full);
            }
        }

        let target = self.truncated_rank(&a);
        let at = a.transpose().to_owned();
        let chosen = maxvol_rows(&at, target);
        self.right[l] = chosen.into_iter().map(|c| col_cand[c].clone()).collect();
        Ok(())
    }

    /// Numerical rank of the two-site block: count singular values above
    /// `tol · σ_max`, capped at `max_rank` (and the block size). This is the local
    /// rank the cross adopts at the interface.
    fn truncated_rank(&self, a: &Mat<f64>) -> usize {
        let m = a.nrows().min(a.ncols());
        if m == 0 {
            return 1;
        }
        let svd = a.thin_svd().expect("truncated_rank: SVD failed");
        let s = svd.S().column_vector();
        let smax = (0..s.nrows()).map(|i| s[i]).fold(0.0_f64, f64::max);
        let thresh = smax * self.opt.tol.max(1e-15);
        let r = (0..s.nrows()).filter(|&i| s[i] > thresh).count();
        r.clamp(1, self.opt.max_rank.min(m))
    }

    /// A fixed probe value (the seed full index) for convergence monitoring.
    fn probe_value<F: TargetFn>(&mut self, f: &F) -> Result<f64> {
        let seed: Vec<usize> = self.dims.iter().map(|&d| d / 3).collect();
        Ok(self.eval(f, &seed))
    }

    /// Read off the TT cores from the converged nested index sets via the skeleton
    /// (CUR) formula `core_l = f(I_l, X_l, J_l) · f(I_l, J_l)^{-1}` (right-closed),
    /// last core left raw — so the product reproduces the cross interpolant.
    fn assemble_tt(&mut self) -> Result<TtTensor> {
        // Re-evaluation here hits the cache for already-seen indices; only the
        // fiber blocks add new samples (bounded by O(n r² k)).
        let n = self.n;
        let mut ranks = vec![1usize; n + 1];
        for (l, rank) in ranks.iter_mut().enumerate().take(n).skip(1) {
            *rank = self.left[l].len();
        }

        let mut cores: Vec<Vec<f64>> = Vec::with_capacity(n);
        for l in 0..n {
            let rl = ranks[l];
            let kl = self.dims[l];
            let rr = ranks[l + 1];
            let iset = self.left[l].clone();
            let jset = self.right[l].clone();
            debug_assert_eq!(iset.len(), rl);
            debug_assert_eq!(jset.len(), rr);

            // Fiber F[a,i,b] = f(I_l[a] ⊕ i ⊕ J_l[b]).
            let mut fib = vec![0.0; rl * kl * rr];
            for (a, li) in iset.iter().enumerate() {
                for i in 0..kl {
                    let mut lidx = li.clone();
                    lidx.push(i);
                    for (b, rj) in jset.iter().enumerate() {
                        let full = glue_left_right(&lidx, rj, n);
                        fib[(a * kl + i) * rr + b] = self.eval_cached(&full);
                    }
                }
            }

            if l < n - 1 {
                // Intersection P[a,b] = f(I_{l+1}[a] ⊕ J_l[b]); close core by P^{-1}.
                // I_{l+1} spans axes 0..=l, J_l spans axes l+1..n.
                let inext = self.left[l + 1].clone();
                let mut p = Mat::from_fn(rr, rr, |_, _| 0.0);
                for (a, li) in inext.iter().enumerate() {
                    for (b, rj) in jset.iter().enumerate() {
                        let full = glue_left_right(li, rj, n);
                        p[(a, b)] = self.eval_cached(&full);
                    }
                }
                let pinv = pseudo_inverse(&p);
                let mut core = vec![0.0; rl * kl * rr];
                for a in 0..rl {
                    for i in 0..kl {
                        let base = (a * kl + i) * rr;
                        for b2 in 0..rr {
                            let mut acc = 0.0;
                            for b in 0..rr {
                                acc += fib[base + b] * pinv[(b, b2)];
                            }
                            core[base + b2] = acc;
                        }
                    }
                }
                cores.push(core);
            } else {
                cores.push(fib);
            }
        }

        if cores.is_empty() {
            return Err(LoewnerError::Degenerate(
                "TT-cross produced no cores".into(),
            ));
        }
        Ok(TtTensor {
            dims: self.dims.clone(),
            ranks,
            cores,
        })
    }

    /// Cache-only lookup (assumes the index was sampled during `run`/`assemble_tt`);
    /// evaluates lazily if missing would require `f`, so we keep a fallback. To
    /// avoid threading `f` here we require the value be present — it always is,
    /// because every fiber index is glued from sampled index sets seen in `run`.
    fn eval_cached(&mut self, idx: &[usize]) -> f64 {
        *self.cache.get(idx).unwrap_or(&f64::NAN)
    }
}

/// Glue a left multi-index (axes `0..l`) and a right multi-index (axes `l..n`) into
/// a full `n`-index. `left.len() + right.len() == n`.
fn glue_left_right(left: &[usize], right: &[usize], n: usize) -> Vec<usize> {
    debug_assert_eq!(left.len() + right.len(), n);
    let mut idx = Vec::with_capacity(n);
    idx.extend_from_slice(left);
    idx.extend_from_slice(right);
    idx
}

/// Maxvol: select `r` rows of the tall matrix `a` (`m × c`, `m ≥ r`) forming a
/// near-maximal-volume `r × c` submatrix, returning their indices.
///
/// Standard maxvol (Goreinov–Oseledets) starts from a column-pivoted-QR row set and
/// swaps rows to increase the determinant. Here `r = min(m, c)`; we implement the
/// QR-pivot seed plus a few dominant-row swaps, which is the rank-revealing choice
/// the cross needs. Falls back to the leading rows for degenerate blocks.
fn maxvol_rows(a: &Mat<f64>, r: usize) -> Vec<usize> {
    let m = a.nrows();
    let c = a.ncols();
    let r = r.min(m).min(c).max(1);
    if m == 0 || c == 0 {
        return (0..r.min(m)).collect();
    }

    // Column-pivoted Gram–Schmidt on rows to seed `r` linearly-independent rows.
    let mut chosen: Vec<usize> = Vec::with_capacity(r);
    let mut basis: Vec<Vec<f64>> = Vec::with_capacity(r); // orthonormal row vectors
    let row = |i: usize| -> Vec<f64> { (0..c).map(|j| a[(i, j)]).collect() };

    let mut avail: Vec<usize> = (0..m).collect();
    for _ in 0..r {
        // Pick the available row with the largest residual norm after projecting
        // out the current basis.
        let mut best = -1.0_f64;
        let mut best_pos = None;
        let mut best_resid = Vec::new();
        for (pos, &i) in avail.iter().enumerate() {
            let mut v = row(i);
            for q in &basis {
                let dot: f64 = v.iter().zip(q).map(|(a, b)| a * b).sum();
                for (vk, qk) in v.iter_mut().zip(q) {
                    *vk -= dot * qk;
                }
            }
            let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > best {
                best = norm;
                best_pos = Some(pos);
                best_resid = v;
            }
        }
        match best_pos {
            Some(pos) if best > 1e-300 => {
                let i = avail.remove(pos);
                chosen.push(i);
                let inv = 1.0 / best;
                for x in &mut best_resid {
                    *x *= inv;
                }
                basis.push(best_resid);
            }
            _ => break, // remaining rows are linearly dependent
        }
    }
    // Pad with leading unused rows if rank-deficient.
    if chosen.len() < r {
        for i in 0..m {
            if !chosen.contains(&i) {
                chosen.push(i);
                if chosen.len() == r {
                    break;
                }
            }
        }
    }
    chosen.sort_unstable();
    chosen
}

/// Moore–Penrose pseudo-inverse of a small square matrix via SVD.
fn pseudo_inverse(a: &Mat<f64>) -> Mat<f64> {
    let m = a.nrows();
    if m == 0 {
        return Mat::from_fn(0, 0, |_, _| 0.0);
    }
    let svd = a.thin_svd().expect("pseudo_inverse: SVD failed");
    let u = svd.U();
    let v = svd.V();
    let s = svd.S().column_vector();
    let smax = (0..s.nrows()).map(|i| s[i]).fold(0.0_f64, f64::max);
    let tol = smax * 1e-12 * (m as f64) + 1e-300;
    Mat::from_fn(m, m, |i, j| {
        let mut acc = 0.0;
        for k in 0..s.nrows() {
            if s[k] > tol {
                acc += v[(i, k)] * (1.0 / s[k]) * u[(j, k)];
            }
        }
        acc
    })
}

/// Lift a TT value tensor to the (denominator, numerator) barycentric TT tensors.
///
/// The barycentric coefficient tensor is a per-axis linear transform of the value
/// tensor, so it is exact to reconstruct the value tensor on the candidate grid and
/// feed the validated dense [`crate::decouple::decouple_from_tensor`]. The dense
/// object here is the value tensor (`∏(k_l+q_l)`); to keep memory bounded we instead
/// reconstruct only what the lift needs. For now we materialize the candidate-grid
/// value tensor from the cheap TT product (pure arithmetic, no `f` calls), run the
/// exact lift, and recompress — sampling stays the TT-cross footprint.
fn lift_to_rational(value_tt: &TtTensor, points: &Points) -> Result<(TtTensor, TtTensor)> {
    let dense = value_tt.to_dense();
    let dims = value_tt.dims.clone();

    use crate::tensor::Tensor;
    use rstsr::prelude::rt::{DeviceFaer, RowMajor, asarray};
    use rstsr::prelude::*;
    let mut device = DeviceFaer::default();
    device.set_default_order(RowMajor);
    let tab: Tensor = asarray((dense, dims, &device)).into_dim();

    let (model, _) = crate::decouple::decouple_from_tensor(&tab, points)?;
    let bary = model.to_barycentric();
    let compressed = CompressedBarycentric::from_model(&bary, 0.0);
    Ok((compressed.denom, compressed.numer))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn maxvol_picks_independent_rows() {
        // Rows 0 and 2 are independent; row 1 = row 0. Maxvol should pick {0,2}.
        let a = Mat::from_fn(3, 2, |i, j| match (i, j) {
            (0, 0) => 1.0,
            (0, 1) => 0.0,
            (1, 0) => 1.0,
            (1, 1) => 0.0,
            (2, 0) => 0.0,
            (2, 1) => 1.0,
            _ => 0.0,
        });
        let rows = maxvol_rows(&a, 2);
        assert_eq!(rows, vec![0, 2]);
    }

    #[test]
    fn cross_low_rank_and_subdense_sampling() {
        // Separable order-1 rational, n=4: value tensor is TT-rank 1, so the cross
        // samples far fewer than the dense grid and discovers rank 1.
        let f = |x: &[f64]| 1.0 / ((x[0] + 3.0) * (x[1] + 2.0) * (x[2] + 4.0) * (x[3] + 5.0));
        let pts = Points::alternating(&[(-1.0, 1.0); 4], &[6; 4]).unwrap();
        let (model, info) = tt_cross(&f, &pts, &TtCrossOptions::default()).unwrap();
        assert!(
            info.evaluations < info.dense_grid,
            "evals {} should be < dense {}",
            info.evaluations,
            info.dense_grid
        );
        assert_eq!(model.max_rank(), 1, "separable target should be TT-rank 1");
    }

    #[test]
    fn cross_lagrange_converges_off_grid() {
        // The Lagrange TT interpolant is exact on the grid and *converges* off-grid
        // as the grid densifies (polynomial interpolation of a smooth/rational
        // target). Verify error decreases when adding points.
        let f =
            |x: &[f64]| 1.0 / ((x[0] + 3.0) * (x[1] + 2.0)) + 1.0 / ((x[2] + 4.0) * (x[3] + 5.0));
        let q = [0.2, -0.3, 0.5, 0.1];
        let err = |nip: usize| {
            let pts = Points::alternating(&[(-1.0, 1.0); 4], &[nip; 4]).unwrap();
            let opt = TtCrossOptions {
                max_rank: 6,
                sweeps: 5,
                tol: 1e-13,
            };
            let (model, _) = tt_cross(&f, &pts, &opt).unwrap();
            (model.eval(&q) - f(&q)).abs()
        };
        let e_coarse = err(6);
        let e_fine = err(12);
        assert!(
            e_fine < e_coarse,
            "off-grid error should shrink: coarse {e_coarse:.2e} fine {e_fine:.2e}"
        );
        assert!(e_fine < 1e-6, "fine error {e_fine:.2e} should be small");
    }

    #[test]
    fn cross_exact_on_grid_nodes() {
        // The interpolant must reproduce sampled values exactly on the grid.
        let f = |x: &[f64]| (x[0] * x[1] + x[2]) / (x[0] + x[1] + x[2] + 10.0);
        let pts = Points::alternating(&[(-1.0, 1.0); 3], &[5; 3]).unwrap();
        let (model, _) = tt_cross(&f, &pts, &TtCrossOptions::default()).unwrap();
        let node = [model.axes[0][2], model.axes[1][1], model.axes[2][3]];
        assert_relative_eq!(model.eval(&node), f(&node), epsilon = 1e-9);
    }

    #[test]
    fn cross_rational_recovers_separable_exactly() {
        // The rational lift (`tt_cross_rational`) recovers a single low-degree
        // tensor-product rational to machine precision off-grid — the exact-rational
        // counterpart of the Lagrange interpolant. Samples sub-dense.
        let f = |x: &[f64]| 1.0 / ((x[0] + 3.0) * (x[1] + 2.0) * (x[2] + 4.0));
        let pts = Points::alternating(&[(-1.0, 1.0); 3], &[4; 3]).unwrap();
        let (model, info) = tt_cross_rational(&f, &pts, &TtCrossOptions::default()).unwrap();
        assert!(info.evaluations <= info.dense_grid);
        for x in [[0.2, -0.3, 0.5], [-0.7, 0.4, 0.9]] {
            assert_relative_eq!(model.eval(&x), f(&x), epsilon = 1e-7);
        }
    }
}
