//! **Exact TT-cross over any [`Field`]** — sub-exponential *sampling* of a
//! black-box tensor with **bit-exact** reconstruction, the exact-arithmetic
//! counterpart of the `f64` [`crate::ttcross`] and follow-up #2 of
//! `docs/TRANSFORMS.md` §5.
//!
//! ## The wall this breaks
//!
//! Every exact constructor so far ([`super::decouple`], [`super::tt::ExactTt::from_dense`])
//! needs the **full dense grid**: `∏ k_l` black-box evaluations, exponential in
//! `n`. Over an exact field there is no maxvol (no "volume" — no ordering), but
//! there is something *better*: exact rank. A rank-`r` matrix over a field has an
//! `r × r` **invertible** cross, and the skeleton identity
//!
//! ```text
//! A = A[:, J] · A[I, J]⁻¹ · A[I, :]        (exact, whenever rank A[I,J] = rank A = r)
//! ```
//!
//! is an *identity*, not an approximation. Nesting it along the TT chain
//! (Oseledets–Tyrtyshnikov cross, exact case) reconstructs any tensor of exact
//! TT-ranks `≤ max_rank` **bit-for-bit** from `O(n · r² · k)` adaptive samples —
//! the float version's `tol`/SVD machinery is replaced by RREF pivoting, which is
//! a theorem of the data.
//!
//! ## Algorithm
//!
//! Maintain nested left index sets `I_l ⊂ X_0×…×X_{l-1}` and right sets
//! `J_l ⊂ X_{l+1}×…×X_{n-1}`. At interface `l`, sample the two-site super-block
//! `A = f(I_l ⊗ X_l, X_{l+1} ⊗ J_{l+1})` and run exact **full-pivot Gaussian
//! elimination**: the pivot rows become the new `I_{l+1}`, the pivot columns the
//! new `J_l`. Full pivoting guarantees the intersection `f(I_{l+1}, J_l)` is
//! invertible — exactly the matrix the skeleton assembly inverts — so the read-off
//! cores `G_l = f(I_l, X_l, J_l) · f(I_{l+1}, J_l)⁻¹` reproduce the tensor exactly
//! once the sweeps have captured the true ranks. Two-site blocks let interface
//! ranks *grow* (rank adaptation); sweeps stop when the rank profile stabilizes.
//!
//! ## Honest failure mode
//!
//! If the true TT-rank exceeds `max_rank` the result is a rank-capped skeleton
//! *interpolant*: exact on the sampled crosses, generally wrong elsewhere — and
//! over a finite field there is no "small residual": wrong is wrong. Callers that
//! cannot bound the rank a priori should verify by random probes (cheap:
//! [`super::tt::ExactTt::entry`] is `O(Σ r² )`); the tests pin this failure mode.

use super::core::Field;
use super::tt::{ExactTt, TtDecoupled, matmul};
use crate::error::{LoewnerError, Result};
use std::collections::HashMap;

/// Options for [`exact_tt_cross`].
#[derive(Debug, Clone)]
pub struct ExactCrossOptions {
    /// Interface rank cap — the sampling/storage knob, `O(n · r² · k)` samples.
    /// Recovery is exact iff every true bond rank is `≤ max_rank`.
    pub max_rank: usize,
    /// Maximum forward+backward sweeps; the loop exits early once the rank
    /// profile stops changing.
    pub sweeps: usize,
    /// Seed for the pseudo-random initial index sets (deterministic).
    pub seed: u64,
}

impl Default for ExactCrossOptions {
    fn default() -> Self {
        ExactCrossOptions {
            max_rank: 8,
            sweeps: 4,
            seed: 0x5EED,
        }
    }
}

/// Sampling statistics returned alongside the reconstructed TT.
#[derive(Debug, Clone)]
pub struct ExactCrossInfo {
    /// Recovered bond rank profile (`len = n + 1`, boundaries 1).
    pub ranks: Vec<usize>,
    /// Number of distinct black-box evaluations (the sampling footprint).
    pub evaluations: usize,
    /// Dense grid size `∏ k_l` the full-grid path would sample.
    pub dense_grid: u128,
}

/// Reconstruct a black-box tensor `f(multi-index)` of shape `dims` in exact TT
/// form from `O(n · r² · k)` adaptively chosen samples. Bit-exact whenever the
/// true TT-ranks are `≤ opt.max_rank` and the pivot sweeps capture them (generic
/// data; see module docs for the failure mode).
pub fn exact_tt_cross<F: Field>(
    dims: &[usize],
    mut f: impl FnMut(&[usize]) -> F,
    opt: &ExactCrossOptions,
) -> Result<(ExactTt<F>, ExactCrossInfo)> {
    if dims.is_empty() || dims.contains(&0) {
        return Err(LoewnerError::Degenerate(
            "exact_tt_cross: dims must be nonempty and positive".into(),
        ));
    }
    let dense_grid: u128 = dims.iter().map(|&k| k as u128).product();
    let mut state = CrossState::new(dims, opt);

    if dims.len() == 1 {
        // Trivial chain: one core = the whole axis.
        let core: Vec<F> = (0..dims[0]).map(|j| state.eval(&mut f, &[j])).collect();
        let evaluations = state.cache.len();
        let tt = ExactTt {
            dims: dims.to_vec(),
            ranks: vec![1, 1],
            cores: vec![core],
        };
        let info = ExactCrossInfo {
            ranks: tt.ranks.clone(),
            evaluations,
            dense_grid,
        };
        return Ok((tt, info));
    }

    state.run(&mut f)?;
    let tt = state.assemble(&mut f)?;
    let info = ExactCrossInfo {
        ranks: tt.ranks.clone(),
        evaluations: state.cache.len(),
        dense_grid,
    };
    Ok((tt, info))
}

/// Cross state: nested index sets plus a memoizing black-box evaluator.
struct CrossState<F: Field> {
    dims: Vec<usize>,
    n: usize,
    opt: ExactCrossOptions,
    /// `left[l]` = multi-indices over axes `0..l` (size `r_l`); `left[0] = [[]]`.
    left: Vec<Vec<Vec<usize>>>,
    /// `right[l]` = multi-indices over axes `l+1..n` (size `r_{l+1}`);
    /// `right[n-1] = [[]]`.
    right: Vec<Vec<Vec<usize>>>,
    cache: HashMap<Vec<usize>, F>,
}

impl<F: Field> CrossState<F> {
    fn new(dims: &[usize], opt: &ExactCrossOptions) -> Self {
        let n = dims.len();
        // Two deterministic pseudo-random seed points (nested prefixes/suffixes)
        // so an unlucky all-zero seed region is less likely than with a single
        // central seed; the pivot search self-corrects from there.
        let mut s = opt.seed | 1;
        let mut next = |k: usize| -> usize {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as usize) % k
        };
        let seeds: Vec<Vec<usize>> = (0..2)
            .map(|_| dims.iter().map(|&k| next(k)).collect())
            .collect();
        let mut left = vec![Vec::new(); n];
        let mut right = vec![Vec::new(); n];
        left[0] = vec![vec![]];
        right[n - 1] = vec![vec![]];
        for l in 1..n {
            let mut pre: Vec<Vec<usize>> = seeds.iter().map(|sd| sd[..l].to_vec()).collect();
            pre.dedup();
            left[l] = pre;
        }
        for l in 0..n - 1 {
            let mut suf: Vec<Vec<usize>> = seeds.iter().map(|sd| sd[l + 1..].to_vec()).collect();
            suf.dedup();
            right[l] = suf;
        }
        CrossState {
            dims: dims.to_vec(),
            n,
            opt: opt.clone(),
            left,
            right,
            cache: HashMap::new(),
        }
    }

    fn eval(&mut self, f: &mut impl FnMut(&[usize]) -> F, idx: &[usize]) -> F {
        if let Some(v) = self.cache.get(idx) {
            return v.clone();
        }
        let v = f(idx);
        self.cache.insert(idx.to_vec(), v.clone());
        v
    }

    /// Row candidates of the interface-`l` block: `(I_l[a], i)` over axes `0..=l`.
    fn row_candidates(&self, l: usize) -> Vec<Vec<usize>> {
        self.left[l]
            .iter()
            .flat_map(|li| {
                (0..self.dims[l]).map(move |i| {
                    let mut idx = li.clone();
                    idx.push(i);
                    idx
                })
            })
            .collect()
    }

    /// Column candidates: `(j, J_{l+1}[b])` over axes `l+1..n`.
    fn col_candidates(&self, l: usize) -> Vec<Vec<usize>> {
        (0..self.dims[l + 1])
            .flat_map(|j| {
                self.right[l + 1].iter().map(move |rj| {
                    let mut idx = vec![j];
                    idx.extend_from_slice(rj);
                    idx
                })
            })
            .collect()
    }

    /// One interface update: sample the two-site block, full-pivot it, install
    /// the pivot rows as `I_{l+1}` and the pivot columns as `J_l` — the joint
    /// selection that makes the skeleton intersection invertible by construction.
    fn update_interface(&mut self, f: &mut impl FnMut(&[usize]) -> F, l: usize) {
        let rows = self.row_candidates(l);
        let cols = self.col_candidates(l);
        let mut block = Vec::with_capacity(rows.len() * cols.len());
        for ri in &rows {
            for ci in &cols {
                let full = glue(ri, ci);
                block.push(match self.cache.get(&full) {
                    Some(v) => v.clone(),
                    None => {
                        let v = f(&full);
                        self.cache.insert(full, v.clone());
                        v
                    }
                });
            }
        }
        let cap = self.opt.max_rank.min(rows.len()).min(cols.len());
        let (prow, pcol) = full_pivot_cross(rows.len(), cols.len(), &block, cap);
        if prow.is_empty() {
            // All-zero block: keep rank-1 placeholder sets (zero-tensor chain).
            self.left[l + 1] = vec![rows[0].clone()];
            self.right[l] = vec![cols[0].clone()];
            return;
        }
        self.left[l + 1] = prow.into_iter().map(|r| rows[r].clone()).collect();
        self.right[l] = pcol.into_iter().map(|c| cols[c].clone()).collect();
    }

    fn rank_profile(&self) -> Vec<usize> {
        (0..self.n - 1).map(|l| self.left[l + 1].len()).collect()
    }

    /// Alternate forward/backward sweeps until the rank profile stabilizes.
    fn run(&mut self, f: &mut impl FnMut(&[usize]) -> F) -> Result<()> {
        let mut prev: Vec<usize> = Vec::new();
        for _ in 0..self.opt.sweeps {
            for l in 0..self.n - 1 {
                self.update_interface(f, l);
            }
            for l in (0..self.n - 1).rev() {
                self.update_interface(f, l);
            }
            let ranks = self.rank_profile();
            if ranks == prev {
                break;
            }
            prev = ranks;
        }
        Ok(())
    }

    /// Skeleton read-off: `G_l = f(I_l ⊗ X_l, J_l) · f(I_{l+1}, J_l)⁻¹`
    /// (last core raw). The inverse exists by the full-pivot construction; a
    /// singular intersection over a *zero* fiber yields a zero core instead.
    fn assemble(&mut self, f: &mut impl FnMut(&[usize]) -> F) -> Result<ExactTt<F>> {
        let n = self.n;
        let mut ranks = vec![1usize; n + 1];
        for (l, r) in ranks.iter_mut().enumerate().take(n).skip(1) {
            *r = self.left[l].len();
        }
        let mut cores = Vec::with_capacity(n);
        for l in 0..n {
            let (rl, kl, rr) = (ranks[l], self.dims[l], ranks[l + 1]);
            let iset = self.left[l].clone();
            let jset = self.right[l].clone();
            debug_assert_eq!(iset.len(), rl);
            debug_assert_eq!(jset.len(), rr);

            // Fiber F[a,i,b] = f(I_l[a] ⊕ i ⊕ J_l[b]).
            let mut fib = Vec::with_capacity(rl * kl * rr);
            for li in &iset {
                for i in 0..kl {
                    let mut lidx = li.clone();
                    lidx.push(i);
                    for rj in &jset {
                        let full = glue(&lidx, rj);
                        fib.push(self.eval(f, &full));
                    }
                }
            }
            if l == n - 1 {
                cores.push(fib);
                continue;
            }
            // Intersection P[a,b] = f(I_{l+1}[a] ⊕ J_l[b]) — invertible by the
            // joint full-pivot selection.
            let inext = self.left[l + 1].clone();
            let mut p = Vec::with_capacity(rr * rr);
            for li in &inext {
                for rj in &jset {
                    let full = glue(li, rj);
                    p.push(self.eval(f, &full));
                }
            }
            match invert(rr, &p) {
                Some(pinv) => cores.push(matmul(&fib, rl * kl, rr, &pinv, rr)),
                None if fib.iter().all(F::is_zero) => cores.push(fib),
                None => {
                    return Err(LoewnerError::Degenerate(
                        "exact_tt_cross: singular skeleton intersection — \
                         increase sweeps or max_rank"
                            .into(),
                    ));
                }
            }
        }
        Ok(ExactTt {
            dims: self.dims.clone(),
            ranks,
            cores,
        })
    }
}

/// Glue a left multi-index (axes `0..l`) and right multi-index (axes `l..n`).
fn glue(left: &[usize], right: &[usize]) -> Vec<usize> {
    let mut idx = Vec::with_capacity(left.len() + right.len());
    idx.extend_from_slice(left);
    idx.extend_from_slice(right);
    idx
}

/// Exact Gaussian elimination with **full pivoting** on a `rows × cols`
/// row-major matrix: returns up to `cap` pivot (row, column) index sets, in
/// selection order. The submatrix at the pivot rows × pivot columns is
/// invertible (each pivot is nonzero in the fully-eliminated complement) —
/// the property the skeleton identity needs. Over a field any nonzero entry is
/// a valid pivot, so the scan takes the first one (no magnitudes to order by).
fn full_pivot_cross<F: Field>(
    rows: usize,
    cols: usize,
    a: &[F],
    cap: usize,
) -> (Vec<usize>, Vec<usize>) {
    debug_assert_eq!(a.len(), rows * cols);
    let mut m = a.to_vec();
    let mut used_r = vec![false; rows];
    let mut used_c = vec![false; cols];
    let mut prow = Vec::new();
    let mut pcol = Vec::new();
    while prow.len() < cap {
        // First nonzero entry of the active submatrix.
        let mut found = None;
        'scan: for i in 0..rows {
            if used_r[i] {
                continue;
            }
            for j in 0..cols {
                if !used_c[j] && !m[i * cols + j].is_zero() {
                    found = Some((i, j));
                    break 'scan;
                }
            }
        }
        let Some((pi, pj)) = found else { break };
        used_r[pi] = true;
        used_c[pj] = true;
        prow.push(pi);
        pcol.push(pj);
        // Eliminate column pj from the remaining active rows.
        let pivot_inv = m[pi * cols + pj].inv().expect("pivot is nonzero");
        for i in 0..rows {
            if used_r[i] {
                continue;
            }
            let factor = m[i * cols + pj].clone() * pivot_inv.clone();
            if factor.is_zero() {
                continue;
            }
            for j in 0..cols {
                if used_c[j] {
                    continue;
                }
                let sub = factor.clone() * m[pi * cols + j].clone();
                let v = m[i * cols + j].clone() - sub;
                m[i * cols + j] = v;
            }
            m[i * cols + pj] = F::zero();
        }
    }
    (prow, pcol)
}

/// Exact inverse of an `n × n` row-major matrix by Gauss–Jordan;
/// `None` if singular.
fn invert<F: Field>(n: usize, a: &[F]) -> Option<Vec<F>> {
    debug_assert_eq!(a.len(), n * n);
    let mut m = a.to_vec();
    let mut inv: Vec<F> = (0..n * n)
        .map(|t| {
            if t % (n + 1) == 0 {
                F::one()
            } else {
                F::zero()
            }
        })
        .collect();
    for col in 0..n {
        let pr = (col..n).find(|&r| !m[r * n + col].is_zero())?;
        for j in 0..n {
            m.swap(col * n + j, pr * n + j);
            inv.swap(col * n + j, pr * n + j);
        }
        let pinv = m[col * n + col].inv().expect("pivot is nonzero");
        for j in 0..n {
            m[col * n + j] = m[col * n + j].clone() * pinv.clone();
            inv[col * n + j] = inv[col * n + j].clone() * pinv.clone();
        }
        for r in 0..n {
            if r == col {
                continue;
            }
            let factor = m[r * n + col].clone();
            if factor.is_zero() {
                continue;
            }
            for j in 0..n {
                let s = factor.clone() * m[col * n + j].clone();
                m[r * n + j] = m[r * n + j].clone() - s;
                let s2 = factor.clone() * inv[col * n + j].clone();
                inv[r * n + j] = inv[r * n + j].clone() - s2;
            }
        }
    }
    Some(inv)
}

/// Sample a black-box **function on a node grid** (`x ↦ f(x)` at
/// `λ_0[j_0], …, λ_{n-1}[j_{n-1}]`) into an exact TT of grid values — the
/// curse-of-*sampling* breaker for the exact pipeline.
pub fn cross_values_on_grid<F: Field>(
    lambda: &[Vec<F>],
    mut f: impl FnMut(&[F]) -> F,
    opt: &ExactCrossOptions,
) -> Result<(ExactTt<F>, ExactCrossInfo)> {
    let dims: Vec<usize> = lambda.iter().map(Vec::len).collect();
    let mut x = vec![F::zero(); lambda.len()];
    exact_tt_cross(
        &dims,
        |idx| {
            for (l, &j) in idx.iter().enumerate() {
                x[l] = lambda[l][j].clone();
            }
            f(&x)
        },
        opt,
    )
}

/// **Black-box polynomial → compressed exact model, sub-exponentially sampled**:
/// TT-cross the grid values, then build the barycentric model *by inspection*
/// (Thm 2.1: rank-1 `⊗γ` denominator, numerator = values ⊙ γ — no SVD, no null
/// space, no dense grid). Exact for any polynomial of per-axis degree
/// `≤ k_l − 1` with value-tensor TT-ranks `≤ max_rank`. This composes the two
/// curse-breakers: KST-by-inspection kills the *construction* cost, cross kills
/// the *sampling* cost — `O(n · r² · k)` evaluations end to end.
pub fn cross_poly_model<F: Field>(
    lambda: Vec<Vec<F>>,
    f: impl FnMut(&[F]) -> F,
    opt: &ExactCrossOptions,
) -> Result<(TtDecoupled<F>, ExactCrossInfo)> {
    let (values, info) = cross_values_on_grid(&lambda, f, opt)?;
    Ok((TtDecoupled::poly_from_values(lambda, &values), info))
}

/// **Variable-disjoint factorization certificate for an ARBITRARY bipartition,
/// grid-free** — the §2E bond-rank certificate freed from both its
/// restrictions (contiguous splits only; dense grid required).
///
/// For a split `S ⊔ S^c` of the axes, `f` factors as
/// `g(x_S)·h(x_{S^c})` on the grid **iff** the unfolding matrix
/// `M[(x_S), (x_{S^c})]` has rank 1. This routine reorders the axes as
/// `S ++ S^c` (a transparent relabelling of the *oracle*, costing nothing —
/// no tensor is ever permuted) and runs the exact cross on the permuted black
/// box; the bond rank at position `|S|` of the recompressed result is the
/// unfolding rank, capped at `opt.max_rank`.
///
/// Returns `(rank_at_split, permuted_tt, info)`. Interpretation: `1` ⇒
/// factorization certified on the sampled crosses (verify by cheap random
/// probes of `permuted_tt` if the rank budget was tight); `≥ 2` ⇒ certified
/// non-factorizable *if* the cross captured the true rank (always true when
/// the tensor's TT-ranks in the permuted order are `≤ max_rank`).
pub fn split_rank<F: Field>(
    dims: &[usize],
    mut f: impl FnMut(&[usize]) -> F,
    split: &[usize],
    opt: &ExactCrossOptions,
) -> Result<(usize, ExactTt<F>, ExactCrossInfo)> {
    let n = dims.len();
    let mut in_split = vec![false; n];
    for &s in split {
        if s >= n {
            return Err(LoewnerError::Degenerate(format!(
                "split_rank: axis {s} out of range (n = {n})"
            )));
        }
        if std::mem::replace(&mut in_split[s], true) {
            return Err(LoewnerError::Degenerate(format!(
                "split_rank: axis {s} repeated"
            )));
        }
    }
    if split.is_empty() || split.len() == n {
        return Err(LoewnerError::Degenerate(
            "split_rank: split must be a proper nonempty subset of the axes".into(),
        ));
    }
    // Permutation: new order = split ++ complement; perm[new_pos] = old_axis.
    let perm: Vec<usize> = split
        .iter()
        .copied()
        .chain((0..n).filter(|l| !in_split[*l]))
        .collect();
    let pdims: Vec<usize> = perm.iter().map(|&l| dims[l]).collect();
    let mut orig_idx = vec![0usize; n];
    let (tt, info) = exact_tt_cross(
        &pdims,
        |idx| {
            for (new_pos, &old_axis) in perm.iter().enumerate() {
                orig_idx[old_axis] = idx[new_pos];
            }
            f(&orig_idx)
        },
        opt,
    )?;
    let tt = tt.recompress();
    let rank = tt.ranks[split.len()];
    Ok((rank, tt, info))
}

#[cfg(test)]
mod tests {
    use super::super::field::Fp;
    use super::super::rational::Rational;
    use super::*;

    const P: u64 = 998_244_353;

    fn fp(v: i128) -> Fp {
        Fp::new(v, P)
    }

    /// A random TT of given bond rank built as a sum of rank-1 terms.
    fn random_low_rank_fp(dims: &[usize], terms: usize, seed: u64) -> ExactTt<Fp> {
        let mut s = seed | 1;
        let mut next = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            fp((s >> 13) as i128)
        };
        let mut acc: Option<ExactTt<Fp>> = None;
        for _ in 0..terms {
            let factors: Vec<Vec<Fp>> = dims
                .iter()
                .map(|&k| (0..k).map(|_| next()).collect())
                .collect();
            let t = ExactTt::rank_one(&factors);
            acc = Some(match acc {
                None => t,
                Some(a) => a.add(&t),
            });
        }
        acc.unwrap().recompress()
    }

    #[test]
    fn recovers_low_rank_fp_tensor_bit_exactly_subdense() {
        let dims = [4usize, 5, 4, 5, 4];
        let target = random_low_rank_fp(&dims, 3, 7);
        let dense = target.to_dense();
        let opt = ExactCrossOptions {
            max_rank: 4,
            ..Default::default()
        };
        let (tt, info) = exact_tt_cross(&dims, |idx| target.entry(idx), &opt).unwrap();
        assert_eq!(tt.to_dense(), dense, "bit-exact recovery");
        assert!(
            (info.evaluations as u128) < info.dense_grid,
            "evals {} < dense {}",
            info.evaluations,
            info.dense_grid
        );
        assert_eq!(tt.bond_ranks(), target.bond_ranks(), "true ranks found");
    }

    #[test]
    fn recovers_over_q_as_well() {
        // f(i,j,k) = (i+1)(j+2) + k — bond ranks [2, 2] over ℚ.
        let dims = [3usize, 3, 3];
        let f = |idx: &[usize]| {
            Rational::integer(((idx[0] + 1) * (idx[1] + 2)) as i64)
                + Rational::integer(idx[2] as i64)
        };
        let mut dense = Vec::new();
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    dense.push(f(&[i, j, k]));
                }
            }
        }
        let (tt, _) = exact_tt_cross(&dims, |idx| f(idx), &ExactCrossOptions::default()).unwrap();
        assert_eq!(tt.to_dense(), dense);
    }

    #[test]
    fn zero_tensor_and_single_axis_edge_cases() {
        let (tt, _) = exact_tt_cross(&[3, 4, 2], |_| fp(0), &ExactCrossOptions::default()).unwrap();
        assert!(tt.to_dense().iter().all(Fp::is_zero));

        let (tt1, info1) = exact_tt_cross(
            &[5],
            |idx| fp(idx[0] as i128 + 1),
            &ExactCrossOptions::default(),
        )
        .unwrap();
        assert_eq!(tt1.to_dense(), (1..=5).map(fp).collect::<Vec<_>>());
        assert_eq!(info1.evaluations, 5);
    }

    #[test]
    fn rank_cap_below_true_rank_fails_honestly() {
        // True rank 4 target, cap 2: the result is exact on the sampled crosses
        // but must differ somewhere — over a field there is no "close".
        let dims = [4usize, 6, 4];
        let target = random_low_rank_fp(&dims, 4, 99);
        assert!(
            target.bond_ranks().iter().any(|&r| r > 2),
            "target rank > 2"
        );
        let opt = ExactCrossOptions {
            max_rank: 2,
            ..Default::default()
        };
        let (tt, info) = exact_tt_cross(&dims, |idx| target.entry(idx), &opt).unwrap();
        assert!(info.ranks.iter().all(|&r| r <= 2), "cap respected");
        assert_ne!(
            tt.to_dense(),
            target.to_dense(),
            "capped cross cannot be exact"
        );
    }

    #[test]
    fn split_rank_certifies_non_contiguous_factorizations() {
        // f(x0,x1,x2,x3) = (x0 + x2 + 3)·(x1·x3 + 5): factors across the
        // NON-contiguous split {0,2} | {1,3} — invisible to contiguous bond
        // ranks, certified by the permuted cross. The contiguous split {0,1}
        // must NOT be rank 1.
        let dims = [4usize, 4, 4, 4];
        let f = |idx: &[usize]| {
            let g = Fp::new(idx[0] as i128 + idx[2] as i128 + 3, P);
            let h = Fp::new((idx[1] * idx[3]) as i128 + 5, P);
            g * h
        };
        let opt = ExactCrossOptions::default();
        let (r02, tt, info) = split_rank(&dims, f, &[0, 2], &opt).unwrap();
        assert_eq!(r02, 1, "factorization across {{0,2}}|{{1,3}} certified");
        assert!((info.evaluations as u128) < info.dense_grid);
        // Verify the certificate's witness tensor by probes (permuted order
        // is split ++ complement = [0,2,1,3]).
        for idx in [[0usize, 1, 2, 3], [3, 0, 1, 2], [2, 2, 2, 2]] {
            assert_eq!(
                tt.entry(&[idx[0], idx[2], idx[1], idx[3]]),
                f(&idx),
                "witness parity at {idx:?}"
            );
        }
        let (r01, _, _) = split_rank(&dims, f, &[0, 1], &opt).unwrap();
        assert!(r01 > 1, "contiguous {{0,1}} split is genuinely coupled");

        // Negative control: Pólya–Szegő xy+xz+yz factors across NO bipartition.
        let g = |idx: &[usize]| {
            let (x, y, z) = (idx[0] as i128, idx[1] as i128, idx[2] as i128);
            Fp::new(x * y + x * z + y * z, P)
        };
        for split in [
            vec![0usize],
            vec![1],
            vec![2],
            vec![0, 1],
            vec![0, 2],
            vec![1, 2],
        ] {
            let (r, _, _) = split_rank(&[3, 3, 3], g, &split, &opt).unwrap();
            assert!(r > 1, "xy+xz+yz must not factor across {split:?}");
        }

        // Degenerate splits are rejected.
        assert!(split_rank(&dims, f, &[], &opt).is_err());
        assert!(split_rank(&dims, f, &[0, 1, 2, 3], &opt).is_err());
        assert!(split_rank(&dims, f, &[0, 0], &opt).is_err());
        assert!(split_rank(&dims, f, &[9], &opt).is_err());
    }

    #[test]
    fn cross_poly_model_matches_black_box_off_grid() {
        // 6-var polynomial over 𝔽_p: f = ∏(x_l + l+1) + x_0·x_5 — value ranks 2.
        let n = 6;
        let lambda: Vec<Vec<Fp>> = (0..n)
            .map(|l| (0..3).map(|j| fp((10 * l + j) as i128)).collect())
            .collect();
        let f = |x: &[Fp]| {
            let mut prod = Fp::new(1, P);
            for (l, &xl) in x.iter().enumerate() {
                prod = prod * (xl + fp(l as i128 + 1));
            }
            prod + x[0] * x[5]
        };
        let (model, info) = cross_poly_model(lambda, f, &ExactCrossOptions::default()).unwrap();
        assert!((info.evaluations as u128) < info.dense_grid);
        for seed in [1u64, 2, 3] {
            let mut s = seed;
            let pt: Vec<Fp> = (0..n)
                .map(|_| {
                    s = s.wrapping_mul(6364136223846793005).wrapping_add(99);
                    fp((s >> 7) as i128)
                })
                .collect();
            assert_eq!(model.eval(&pt), Some(f(&pt)), "off-grid parity");
        }
    }
}
