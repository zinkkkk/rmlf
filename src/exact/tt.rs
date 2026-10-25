//! **Exact tensor-train (TT) compression over a [`Field`]** — the lossless,
//! tolerance-free analogue of the `f64` TT codec ([`crate::tt`]), and the
//! "exact TT over the field" extension flagged open in `docs/EXACT_FIELDS.md` §3.
//!
//! ## The mathematics
//!
//! The floating-point TT-SVD truncates singular values below a tolerance; over an
//! exact field there are no singular values — but there *is* exact rank, and that
//! is all a TT needs. Each unfolding `M` of the tensor is factored **exactly** as
//!
//! ```text
//! M = M[:, P] · R        (column-basis factorization)
//! ```
//!
//! where `P` are the pivot columns of the RREF of `M` and `R` its nonzero rows —
//! an identity over any field (each column of `M` is the RREF-coordinate
//! combination of the pivot columns). Sweeping left to right turns the dense
//! tensor into a chain of cores `G_l ∈ F^{r_l × k_l × r_{l+1}}` with
//! `r_l = rank` of the l-th unfolding — the **exact TT-ranks**, a theorem of the
//! data, not a threshold. Reconstruction is *bit-exact* (`to_dense` returns the
//! original tensor, entry for entry), so this is a lossless codec: storage
//! `Σ r_l·k_l·r_{l+1}` versus the dense `∏ k_l`, a win exactly when the
//! interaction structure is thin (separable ⇒ all ranks 1 ⇒ `Σ k_l`).
//!
//! ## Why this is especially strong for the exact decoupled models
//!
//! For a **polynomial** target (`d ≡ 1`) every slice's denominator weights are the
//! Lagrange coefficients `γ`, so the denominator tensor is *exactly* the Kronecker
//! product `C_den = γ_0 ⊗ … ⊗ γ_{n-1}` — **TT-rank 1, always** (storage `Σ k_l`
//! instead of `∏ k_l`; see `tt_of_polynomial_denominator_is_rank_one`). And since
//! `C_num = C_den ⊙ W` with `C_den` rank-1, the numerator's TT-ranks equal the
//! TT-ranks of the **value tensor** `W = H(λ_J)` itself — the model compresses
//! precisely as well as the target's intrinsic interaction structure allows
//! (a Hadamard product with a rank-1 tensor preserves TT-ranks).
//!
//! [`TtDecoupled`] packages the two compressed tensors with the support nodes and
//! evaluates **identically** to [`ExactDecoupled`] (same node-coincidence limits,
//! same `0/0` conventions) — in `O(Σ r² k_l)` field operations per query instead
//! of `O(∏ k_l)`.

use super::core::Field;
use super::decouple::{ExactDecoupled, cauchy_weights, full_node_index, lagrange_coeffs};
use super::linalg::Matrix;
use crate::error::{LoewnerError, Result};

/// An exact tensor-train: cores `G_l` of shape `(r_l, k_l, r_{l+1})` (row-major,
/// boundary ranks `r_0 = r_n = 1`) over a field `F`. Lossless: built by exact
/// rank factorization, [`Self::to_dense`] reproduces the input bit-for-bit.
#[derive(Clone, Debug)]
pub struct ExactTt<F: Field> {
    /// Per-axis mode sizes `k_l`.
    pub dims: Vec<usize>,
    /// Bond ranks `r_0, …, r_n` (`len = n+1`, `r_0 = r_n = 1`).
    pub ranks: Vec<usize>,
    /// Cores, `cores[l]` flat row-major `(r_l, k_l, r_{l+1})`.
    pub cores: Vec<Vec<F>>,
}

/// Exact column-basis factorization `M = B · T` of a `rows × cols` row-major
/// matrix: `B = M[:, P]` (the pivot columns, `rows × r`) and `T` the nonzero RREF
/// rows (`r × cols`). Exact over any field. A zero matrix factors with `r = 1`
/// (zero column × zero row) so the TT chain stays well-formed.
fn column_basis_factor<F: Field>(rows: usize, cols: usize, m: &[F]) -> (Vec<F>, Vec<F>, usize) {
    let mut mat = Matrix::from_rows(rows, cols, m.to_vec());
    let pivots = mat.rref();
    if pivots.is_empty() {
        return (vec![F::zero(); rows], vec![F::zero(); cols], 1);
    }
    let r = pivots.len();
    let mut b = Vec::with_capacity(rows * r);
    for i in 0..rows {
        for &p in &pivots {
            b.push(m[i * cols + p].clone());
        }
    }
    let t = mat.data[..r * cols].to_vec();
    (b, t, r)
}

impl<F: Field> ExactTt<F> {
    /// Exact TT decomposition of a dense row-major tensor. Lossless; the bond
    /// ranks are the exact unfolding ranks over `F`.
    pub fn from_dense(data: &[F], dims: &[usize]) -> Self {
        let n = dims.len();
        let total: usize = dims.iter().product();
        assert_eq!(data.len(), total, "tensor length must match dims product");
        assert!(n > 0, "need at least one axis");
        let mut cores = Vec::with_capacity(n);
        let mut ranks = vec![1usize];
        let mut cur = data.to_vec();
        let mut r_prev = 1usize;
        for &k in dims.iter().take(n - 1) {
            let rows = r_prev * k;
            let cols = cur.len() / rows;
            let (b, t, r) = column_basis_factor(rows, cols, &cur);
            cores.push(b);
            ranks.push(r);
            cur = t;
            r_prev = r;
        }
        // Final core: `cur` is r_{n-1} × k_{n-1}, i.e. (r_{n-1}, k_{n-1}, 1).
        cores.push(cur);
        ranks.push(1);
        Self {
            dims: dims.to_vec(),
            ranks,
            cores,
        }
    }

    /// Number of axes.
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    /// The interior bond ranks `r_1 … r_{n-1}` (the usual "TT-rank profile").
    pub fn bond_ranks(&self) -> &[usize] {
        &self.ranks[1..self.ranks.len() - 1]
    }

    /// Stored field elements `Σ r_l·k_l·r_{l+1}`.
    pub fn storage_len(&self) -> usize {
        self.cores.iter().map(Vec::len).sum()
    }

    /// Dense footprint `∏ k_l` for comparison.
    pub fn dense_len(&self) -> usize {
        self.dims.iter().product()
    }

    /// Exact dense reconstruction (row-major) — bit-for-bit the input of
    /// [`Self::from_dense`].
    pub fn to_dense(&self) -> Vec<F> {
        // Accumulate left-to-right: `left` is (rows_acc × r_l) row-major.
        let mut left = self.cores[0].clone();
        let mut rows_acc = self.dims[0];
        for l in 1..self.ndim() {
            let r_l = self.ranks[l];
            let r_next = self.ranks[l + 1];
            let kr = self.dims[l] * r_next;
            let mut out = vec![F::zero(); rows_acc * kr];
            for i in 0..rows_acc {
                for a in 0..r_l {
                    let lv = &left[i * r_l + a];
                    if lv.is_zero() {
                        continue;
                    }
                    let row = &self.cores[l][a * kr..(a + 1) * kr];
                    for (slot, g) in out[i * kr..(i + 1) * kr].iter_mut().zip(row) {
                        if !g.is_zero() {
                            *slot = slot.clone() + lv.clone() * g.clone();
                        }
                    }
                }
            }
            left = out;
            rows_acc *= self.dims[l];
        }
        left
    }

    /// Contract against per-axis weight vectors: `Σ_J T[J] ∏_l w_l[j_l]`, in
    /// `O(Σ r_l·k_l·r_{l+1})` field operations (the compressed analogue of
    /// the dense rank-1 contraction).
    pub fn contract(&self, weights: &[Vec<F>]) -> F {
        debug_assert_eq!(weights.len(), self.ndim());
        let mut v = vec![F::one()]; // r_0 = 1
        for (l, w) in weights.iter().enumerate() {
            let k = self.dims[l];
            let r_next = self.ranks[l + 1];
            let mut nv = vec![F::zero(); r_next];
            for (a, va) in v.iter().enumerate() {
                if va.is_zero() {
                    continue;
                }
                for (j, wj) in w.iter().enumerate() {
                    if wj.is_zero() {
                        continue;
                    }
                    let coef = va.clone() * wj.clone();
                    let row = &self.cores[l][(a * k + j) * r_next..(a * k + j + 1) * r_next];
                    for (slot, g) in nv.iter_mut().zip(row) {
                        if !g.is_zero() {
                            *slot = slot.clone() + coef.clone() * g.clone();
                        }
                    }
                }
            }
            v = nv;
        }
        v.into_iter().next().expect("r_n = 1")
    }

    /// **Parallel contraction** against per-axis weights for a *deep* TT: each core `l` is
    /// reduced to its transfer matrix `T_l[a,b] = Σ_j w_l[j]·G_l[a,j,b]` (`r_l × r_{l+1}`) in
    /// parallel, then the chain `∏_l T_l` is collapsed by an associative parallel reduction.
    /// Same exact result as [`Self::contract`]. The win is for many axes (large `n`) — for a
    /// shallow chain the serial sweep is faster (no thread overhead). Requires `F: Send+Sync`.
    pub fn contract_parallel(&self, weights: &[Vec<F>]) -> F
    where
        F: Send + Sync,
    {
        use rayon::prelude::*;
        debug_assert_eq!(weights.len(), self.ndim());
        // 1. Per-core transfer matrices, in parallel.
        let mats: Vec<(usize, usize, Vec<F>)> = (0..self.ndim())
            .into_par_iter()
            .map(|l| {
                let (r, s, k) = (self.ranks[l], self.ranks[l + 1], self.dims[l]);
                let w = &weights[l];
                let mut t = vec![F::zero(); r * s];
                for a in 0..r {
                    for (j, wj) in w.iter().enumerate() {
                        if wj.is_zero() {
                            continue;
                        }
                        let base = (a * k + j) * s;
                        for b in 0..s {
                            let g = &self.cores[l][base + b];
                            if !g.is_zero() {
                                let slot = &mut t[a * s + b];
                                *slot = slot.clone() + wj.clone() * g.clone();
                            }
                        }
                    }
                }
                (r, s, t)
            })
            .collect();
        // 2. Associative parallel reduction of the matrix chain (r_0 = r_n = 1).
        let (_, _, prod) = mats
            .into_par_iter()
            .reduce_with(|(ra, ca, a), (_rb, cb, bb)| (ra, cb, matmul(&a, ra, ca, &bb, cb)))
            .expect("non-empty chain");
        prod.into_iter().next().expect("1×1 result")
    }

    /// The exact entry at a multi-index (product of the per-axis core slices).
    pub fn entry(&self, idx: &[usize]) -> F {
        debug_assert_eq!(idx.len(), self.ndim());
        let mut v = vec![F::one()];
        for (l, &j) in idx.iter().enumerate() {
            let k = self.dims[l];
            let r_next = self.ranks[l + 1];
            let mut nv = vec![F::zero(); r_next];
            for (a, va) in v.iter().enumerate() {
                if va.is_zero() {
                    continue;
                }
                let row = &self.cores[l][(a * k + j) * r_next..(a * k + j + 1) * r_next];
                for (slot, g) in nv.iter_mut().zip(row) {
                    if !g.is_zero() {
                        *slot = slot.clone() + va.clone() * g.clone();
                    }
                }
            }
            v = nv;
        }
        v.into_iter().next().expect("r_n = 1")
    }

    /// The exact entry at a flat (row-major) index.
    pub fn entry_flat(&self, mut flat: usize) -> F {
        let mut idx = vec![0usize; self.ndim()];
        for l in (0..self.ndim()).rev() {
            idx[l] = flat % self.dims[l];
            flat /= self.dims[l];
        }
        self.entry(&idx)
    }

    /// Apply a linear operator along one axis: replace mode `l` by
    /// `G'[a, j', b] = Σ_j M[j', j]·G[a, j, b]` (`mat` is `k_out × k_l`,
    /// row-major). **Bond ranks are unchanged** — this is what makes per-axis
    /// transforms (e.g. a CRT-factored NTT) run on a
    /// compressed tensor in `O(k_out·k_l·r²)` instead of touching all `∏ k`
    /// entries.
    pub fn apply_axis_matrix(&self, l: usize, mat: &[F], k_out: usize) -> Self {
        let k = self.dims[l];
        assert_eq!(mat.len(), k_out * k, "matrix must be k_out × k_l");
        let (r, s) = (self.ranks[l], self.ranks[l + 1]);
        let mut core = vec![F::zero(); r * k_out * s];
        for a in 0..r {
            for jp in 0..k_out {
                for j in 0..k {
                    let m = &mat[jp * k + j];
                    if m.is_zero() {
                        continue;
                    }
                    for b in 0..s {
                        let g = &self.cores[l][(a * k + j) * s + b];
                        if !g.is_zero() {
                            let slot = &mut core[(a * k_out + jp) * s + b];
                            *slot = slot.clone() + m.clone() * g.clone();
                        }
                    }
                }
            }
        }
        let mut out = self.clone();
        out.dims[l] = k_out;
        out.cores[l] = core;
        out
    }

    /// Apply a linear operator along axis `l` given as a **map on fibers**
    /// (`f(v)` must return the operator applied to the length-`k_l` vector `v`,
    /// with output length `k_out`). Semantically identical to
    /// [`Self::apply_axis_matrix`] — bond ranks unchanged — but the operator is
    /// never materialized as a `k_out × k_l` matrix: with a fast transform
    /// (radix-2 NTT, Rader) each of the `r_l·r_{l+1}` fibers costs
    /// `O(k log k)` instead of `O(k²)`, and a permutation costs `O(k)`. This is
    /// what lets a CRT-torus transform carry axes of size
    /// 65537 — whose dense DFT matrix would hold 4.3·10⁹ entries.
    pub fn apply_axis_map(
        &self,
        l: usize,
        k_out: usize,
        mut f: impl FnMut(&[F]) -> Vec<F>,
    ) -> Self {
        let k = self.dims[l];
        let (r, s) = (self.ranks[l], self.ranks[l + 1]);
        let mut core = vec![F::zero(); r * k_out * s];
        let mut fiber = vec![F::zero(); k];
        for a in 0..r {
            for b in 0..s {
                for (j, slot) in fiber.iter_mut().enumerate() {
                    *slot = self.cores[l][(a * k + j) * s + b].clone();
                }
                let out = f(&fiber);
                debug_assert_eq!(out.len(), k_out, "fiber map must return k_out values");
                for (jp, v) in out.into_iter().enumerate() {
                    core[(a * k_out + jp) * s + b] = v;
                }
            }
        }
        let mut out = self.clone();
        out.dims[l] = k_out;
        out.cores[l] = core;
        out
    }

    /// The **rank-1** TT of a separable tensor `⊗_l factors[l]` — built directly,
    /// no factorization (storage `Σ k_l`).
    pub fn rank_one(factors: &[Vec<F>]) -> Self {
        assert!(!factors.is_empty(), "need at least one axis");
        Self {
            dims: factors.iter().map(Vec::len).collect(),
            ranks: vec![1; factors.len() + 1],
            cores: factors.to_vec(),
        }
    }

    /// Build a TT **directly from a finite automaton** that reads one symbol per axis,
    /// carrying a state in `0..states`. No sampling, no dense tensor — the cost and
    /// footprint are governed by the state count, *not* by `∏ k_l`, so this constructs
    /// a tensor over an astronomically large grid (e.g. a sieve mod `∏ p_l`) in
    /// `O(states²·Σ k_l)`.
    ///
    /// * `dims[l]` = alphabet size on axis `l` (the symbol is the index `0..dims[l]`).
    /// * `start` = initial state.
    /// * `step(l, s, sym) -> s'` = the transition reading symbol `sym` on axis `l` in
    ///   state `s`.
    /// * `accept(s) -> F` = the output weight for a run ending in state `s` (use `1`/`0`
    ///   for an indicator; any field value for a weighted automaton).
    ///
    /// The entry at `(sym_0,…,sym_{n-1})` equals `accept(δ*(start, syms))`. Each core is
    /// the transition tensor `G_l[s, sym, s'] = [δ_l(s,sym)=s']`, so every bond rank is
    /// `≤ states`. The result is generally *not* minimal — follow with
    /// [`Self::recompress`] to read the true unfolding ranks.
    pub fn from_automaton(
        dims: &[usize],
        states: usize,
        start: usize,
        mut step: impl FnMut(usize, usize, usize) -> usize,
        accept: impl Fn(usize) -> F,
    ) -> Self {
        let n = dims.len();
        assert!(n > 0, "need at least one axis");
        assert!(states > 0, "need at least one state");
        let mut cores = Vec::with_capacity(n);
        let mut ranks = Vec::with_capacity(n + 1);

        // First core: row bond is rank 1 (the fixed start state) → `states`.
        // Interior cores: `states × k × states`. Last core: `states × k × 1`
        // carrying the accept weight.
        for (l, &k) in dims.iter().enumerate() {
            let r_in = if l == 0 { 1 } else { states };
            let r_out = if l == n - 1 { 1 } else { states };
            ranks.push(r_in);
            let mut core = vec![F::zero(); r_in * k * r_out];
            for a in 0..r_in {
                let s = if l == 0 { start } else { a };
                for sym in 0..k {
                    let sp = step(l, s, sym);
                    debug_assert!(sp < states, "transition out of range");
                    if l == n - 1 {
                        let w = accept(sp);
                        if !w.is_zero() {
                            core[(a * k + sym) * r_out] = w;
                        }
                    } else {
                        core[(a * k + sym) * r_out + sp] = F::one();
                    }
                }
            }
            cores.push(core);
        }
        ranks.push(1);
        Self {
            dims: dims.to_vec(),
            ranks,
            cores,
        }
    }

    /// Exact elementwise (Hadamard) product `self ⊙ other` in TT form: bond ranks
    /// **multiply** (`r ← r·r'`), no densification. This is **compressed
    /// multiplication**: for value-on-grid tensors of two polynomials it computes
    /// the product's value tensor without ever forming `∏ k_l` entries. Follow
    /// with [`Self::recompress`] to restore minimal ranks.
    pub fn hadamard(&self, other: &Self) -> Self {
        assert_eq!(self.dims, other.dims, "Hadamard needs identical mode sizes");
        let n = self.ndim();
        let mut cores = Vec::with_capacity(n);
        let mut ranks = Vec::with_capacity(n + 1);
        ranks.push(1);
        for l in 0..n {
            let (r1, s1) = (self.ranks[l], self.ranks[l + 1]);
            let (r2, s2) = (other.ranks[l], other.ranks[l + 1]);
            let k = self.dims[l];
            let (nr, ns) = (r1 * r2, s1 * s2);
            let mut g = vec![F::zero(); nr * k * ns];
            for a1 in 0..r1 {
                for a2 in 0..r2 {
                    for j in 0..k {
                        for b1 in 0..s1 {
                            let v1 = &self.cores[l][(a1 * k + j) * s1 + b1];
                            if v1.is_zero() {
                                continue;
                            }
                            for b2 in 0..s2 {
                                let v2 = &other.cores[l][(a2 * k + j) * s2 + b2];
                                if v2.is_zero() {
                                    continue;
                                }
                                g[((a1 * r2 + a2) * k + j) * ns + (b1 * s2 + b2)] =
                                    v1.clone() * v2.clone();
                            }
                        }
                    }
                }
            }
            cores.push(g);
            ranks.push(ns);
        }
        Self {
            dims: self.dims.clone(),
            ranks,
            cores,
        }
    }

    /// Exact TT sum `self + other`: bond ranks **add** (block-diagonal cores,
    /// concatenated at the boundaries). Follow with [`Self::recompress`].
    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.dims, other.dims, "sum needs identical mode sizes");
        let n = self.ndim();
        let mut cores = Vec::with_capacity(n);
        let mut ranks = Vec::with_capacity(n + 1);
        ranks.push(1);
        for l in 0..n {
            let (r1, s1) = (self.ranks[l], self.ranks[l + 1]);
            let (r2, s2) = (other.ranks[l], other.ranks[l + 1]);
            let k = self.dims[l];
            // Boundary bonds stay rank 1 (concatenate); interior bonds add.
            let nr = if l == 0 { 1 } else { r1 + r2 };
            let ns = if l == n - 1 { 1 } else { s1 + s2 };
            let (row_off, col_off) = (if l == 0 { 0 } else { r1 }, if l == n - 1 { 0 } else { s1 });
            let mut g = vec![F::zero(); nr * k * ns];
            for a in 0..r1 {
                for j in 0..k {
                    for b in 0..s1 {
                        let v = &self.cores[l][(a * k + j) * s1 + b];
                        if !v.is_zero() {
                            let slot = &mut g[(a * k + j) * ns + b];
                            *slot = slot.clone() + v.clone();
                        }
                    }
                }
            }
            for a in 0..r2 {
                for j in 0..k {
                    for b in 0..s2 {
                        let v = &other.cores[l][(a * k + j) * s2 + b];
                        if !v.is_zero() {
                            let slot = &mut g[((a + row_off) * k + j) * ns + (b + col_off)];
                            *slot = slot.clone() + v.clone();
                        }
                    }
                }
            }
            cores.push(g);
            ranks.push(ns);
        }
        Self {
            dims: self.dims.clone(),
            ranks,
            cores,
        }
    }

    /// **Exact TT re-compression** to minimal bond ranks (the lossless analogue
    /// of TT rounding): a right-to-left exact *row-basis* sweep followed by a
    /// left-to-right exact *column-basis* sweep. After [`Self::hadamard`]
    /// (`r·r'`) or [`Self::add`] (`r+r'`) this restores the true unfolding ranks
    /// — `to_dense` is unchanged bit-for-bit.
    pub fn recompress(&self) -> Self {
        let mut t = self.clone();
        let n = t.ndim();
        // Right-to-left: replace each core by a row basis, absorb the mixing
        // factor into the left neighbour.
        for l in (1..n).rev() {
            let r = t.ranks[l];
            let cols = t.dims[l] * t.ranks[l + 1];
            // Transpose the (r × cols) unfolding, factor M^T = M^T[:,P]·T, so
            // M = T^T · M[P,:]: S = T^T is r × r', R = M[P,:] is r' × cols.
            let mt = transpose(&t.cores[l], r, cols);
            let (_, tr, rnew) = column_basis_factor_rows(cols, r, &mt, |p| {
                // rows of M picked by the pivot columns of M^T
                t.cores[l][p * cols..(p + 1) * cols].to_vec()
            });
            let (s, rsel) = tr;
            // s is rnew × r (the RREF rows of M^T); S = s^T is r × rnew.
            let s_t = transpose(&s, rnew, r);
            // Left neighbour: (r_{l-1}·k_{l-1} × r) · (r × rnew).
            let lrows = t.ranks[l - 1] * t.dims[l - 1];
            t.cores[l - 1] = matmul(&t.cores[l - 1], lrows, r, &s_t, rnew);
            t.cores[l] = rsel;
            t.ranks[l] = rnew;
        }
        // Left-to-right: the same column-basis sweep as `from_dense`, on cores.
        for l in 0..n - 1 {
            let rows = t.ranks[l] * t.dims[l];
            let cols = t.ranks[l + 1];
            let (b, tr, rnew) = column_basis_factor(rows, cols, &t.cores[l]);
            // Next core: (rnew × cols) · (cols-as-r_{l+1} unfolding).
            let next_cols = t.dims[l + 1] * t.ranks[l + 2];
            t.cores[l + 1] = matmul(&tr, rnew, cols, &t.cores[l + 1], next_cols);
            t.cores[l] = b;
            t.ranks[l + 1] = rnew;
        }
        t
    }
}

/// Row-major matrix product `(a: ra×ca) · (b: ca×cb)`.
pub(crate) fn matmul<F: Field>(a: &[F], ra: usize, ca: usize, b: &[F], cb: usize) -> Vec<F> {
    debug_assert_eq!(a.len(), ra * ca);
    debug_assert_eq!(b.len(), ca * cb);
    let mut out = vec![F::zero(); ra * cb];
    for i in 0..ra {
        for t in 0..ca {
            let av = &a[i * ca + t];
            if av.is_zero() {
                continue;
            }
            for j in 0..cb {
                let bv = &b[t * cb + j];
                if !bv.is_zero() {
                    let slot = &mut out[i * cb + j];
                    *slot = slot.clone() + av.clone() * bv.clone();
                }
            }
        }
    }
    out
}

/// Row-major transpose of a `rows × cols` matrix.
fn transpose<F: Field>(m: &[F], rows: usize, cols: usize) -> Vec<F> {
    let mut out = Vec::with_capacity(rows * cols);
    for j in 0..cols {
        for i in 0..rows {
            out.push(m[i * cols + j].clone());
        }
    }
    out
}

/// Like [`column_basis_factor`], but also hands back the pivot-selected *rows*
/// of the original (pre-transpose) matrix via `row_of` — the row-basis dual used
/// by the right-to-left recompression sweep. Returns `(B, (T, selected_rows), r)`.
#[allow(clippy::type_complexity)]
fn column_basis_factor_rows<F: Field>(
    rows: usize,
    cols: usize,
    m: &[F],
    row_of: impl Fn(usize) -> Vec<F>,
) -> (Vec<F>, (Vec<F>, Vec<F>), usize) {
    let mut mat = Matrix::from_rows(rows, cols, m.to_vec());
    let pivots = mat.rref();
    if pivots.is_empty() {
        // Zero matrix: rank-1 zero chain (mirrors `column_basis_factor`).
        let zero_row = vec![F::zero(); row_of(0).len()];
        return (vec![F::zero(); rows], (vec![F::zero(); cols], zero_row), 1);
    }
    let r = pivots.len();
    let mut b = Vec::with_capacity(rows * r);
    for i in 0..rows {
        for &p in &pivots {
            b.push(m[i * cols + p].clone());
        }
    }
    let t = mat.data[..r * cols].to_vec();
    let mut sel = Vec::new();
    for &p in &pivots {
        sel.extend(row_of(p));
    }
    (b, (t, sel), r)
}

/// A [`ExactDecoupled`] model with both coefficient tensors in **exact TT form**:
/// the lossless compressed exact model. Evaluates identically to the dense model
/// (same limits and `0/0` conventions), in `O(Σ r²·k_l)` per query.
#[derive(Clone, Debug)]
pub struct TtDecoupled<F: Field> {
    /// Column support nodes per axis.
    pub lambda: Vec<Vec<F>>,
    /// Numerator tensor in exact TT form.
    pub num: ExactTt<F>,
    /// Denominator tensor in exact TT form.
    pub den: ExactTt<F>,
}

impl<F: Field> TtDecoupled<F> {
    /// Losslessly compress a dense exact decoupled model.
    pub fn compress(model: &ExactDecoupled<F>) -> Self {
        let dims = model.sizes();
        Self {
            lambda: model.lambda.clone(),
            num: ExactTt::from_dense(&model.c_num, &dims),
            den: ExactTt::from_dense(&model.c_den, &dims),
        }
    }

    /// Exact dense re-expansion — the inverse of [`Self::compress`].
    pub fn decompress(&self) -> ExactDecoupled<F> {
        ExactDecoupled {
            lambda: self.lambda.clone(),
            c_num: self.num.to_dense(),
            c_den: self.den.to_dense(),
        }
    }

    /// Number of variables.
    pub fn ndim(&self) -> usize {
        self.lambda.len()
    }

    /// Stored field elements (both TT chains).
    pub fn storage_len(&self) -> usize {
        self.num.storage_len() + self.den.storage_len()
    }

    /// Dense footprint (`2·∏k_l`) for comparison.
    pub fn dense_len(&self) -> usize {
        self.num.dense_len() + self.den.dense_len()
    }

    /// Build a **polynomial** model directly from a TT of grid values, never
    /// densifying: `C_den = γ_0 ⊗ … ⊗ γ_{n-1}` (rank 1, the §10.2 structural
    /// theorem) and `C_num = C_den ⊙ values`. Exact for any target of per-axis
    /// degree `≤ k_l − 1`; this is the constructor that makes **compressed
    /// multivariate polynomial arithmetic** possible (see [`Self::mul_poly`]).
    pub fn poly_from_values(lambda: Vec<Vec<F>>, values: &ExactTt<F>) -> Self {
        assert_eq!(
            lambda.iter().map(Vec::len).collect::<Vec<_>>(),
            values.dims,
            "one node per value along each axis"
        );
        let gammas: Vec<Vec<F>> = lambda.iter().map(|l| lagrange_coeffs(l)).collect();
        let den = ExactTt::rank_one(&gammas);
        let num = den.hadamard(values); // rank-1 Hadamard preserves the value ranks
        Self { lambda, num, den }
    }

    /// The TT of the model's **grid values** `W[J] = C_num[J]/C_den[J]` for a
    /// polynomial model (rank-1 denominator): `num ⊙ (⊗ γ_l⁻¹)` — same bond
    /// ranks as the numerator, no densification.
    pub fn values_tt(&self) -> ExactTt<F> {
        let inv_g: Vec<Vec<F>> = self
            .lambda
            .iter()
            .map(|l| {
                lagrange_coeffs(l)
                    .into_iter()
                    .map(|g| g.inv().expect("Lagrange coefficients are nonzero"))
                    .collect()
            })
            .collect();
        self.num.hadamard(&ExactTt::rank_one(&inv_g))
    }

    /// **Compressed polynomial multiplication**: multiply two polynomial models
    /// living on the *same* node grid, entirely in TT form — value tensors
    /// Hadamard-multiplied (ranks multiply), recompressed to minimal exact ranks,
    /// reassembled with the shared rank-1 `⊗γ` denominator. `O(Σ k_l·(r r')²)`
    /// field ops; the dense equivalent is a `∏ k_l` convolution.
    ///
    /// **Aliasing semantics (the wraparound theorem):** the result is the product
    /// *modulo the per-axis node polynomial* `∏_j (x_l − λ_l(j))` — exact iff the
    /// grid has `k_l ≥ deg_f(l) + deg_g(l) + 1` nodes per axis, otherwise it
    /// aliases exactly as a too-short cyclic convolution does (NTT wraparound is
    /// the special case `nodes = roots of xⁿ−1`; see `exact::polymul`).
    pub fn mul_poly(&self, other: &Self) -> Result<Self> {
        if self.lambda != other.lambda {
            return Err(LoewnerError::Degenerate(
                "mul_poly: operands live on different node grids".into(),
            ));
        }
        let w = self.values_tt().hadamard(&other.values_tt()).recompress();
        Ok(Self::poly_from_values(self.lambda.clone(), &w))
    }

    /// **Compressed polynomial power** `self^m` by repeated squaring, every step
    /// exactly recompressed — the power algebra on top of [`Self::mul_poly`].
    /// `m = 0` returns the constant-1 model on the same grid. Same aliasing
    /// semantics as `mul_poly`: exact iff the grid carries `m·deg + 1` nodes
    /// per axis.
    pub fn pow_poly(&self, m: u32) -> Result<Self> {
        if m == 0 {
            let ones: Vec<Vec<F>> = self
                .lambda
                .iter()
                .map(|l| vec![F::one(); l.len()])
                .collect();
            return Ok(Self::poly_from_values(
                self.lambda.clone(),
                &ExactTt::rank_one(&ones),
            ));
        }
        let mut base = self.clone();
        let mut acc: Option<Self> = None;
        let mut e = m;
        while e > 0 {
            if e & 1 == 1 {
                acc = Some(match acc {
                    None => base.clone(),
                    Some(a) => a.mul_poly(&base)?,
                });
            }
            e >>= 1;
            if e > 0 {
                base = base.mul_poly(&base)?;
            }
        }
        Ok(acc.expect("m ≥ 1"))
    }

    /// The minimal exact bond ranks of the model's **value tensor** — the
    /// multiplicative-structure invariant: bond `s` has rank 1 **iff** the target
    /// factors as `g(x_0..x_s)·h(x_{s+1}..)` on the grid (a variable-disjoint
    /// factorization certificate; see `docs/TRANSFORMS.md` §2E).
    pub fn value_bond_ranks(&self) -> Vec<usize> {
        self.values_tt().recompress().bond_ranks().to_vec()
    }

    /// Evaluate `N(x)/D(x)` — exactly the [`ExactDecoupled::eval`] semantics
    /// (full-node fast path, partial-node selector limit, `0/0 → Some(0)`,
    /// genuine pole → `None`), via TT contractions.
    pub fn eval(&self, x: &[F]) -> Option<F> {
        debug_assert_eq!(x.len(), self.ndim());
        if let Some(j) = full_node_index(&self.lambda, x) {
            let cd = self.den.entry_flat(j);
            if !cd.is_zero() {
                return Some(self.num.entry_flat(j) / cd);
            }
        }
        let weights = cauchy_weights(&self.lambda, x);
        let n = self.num.contract(&weights);
        let d = self.den.contract(&weights);
        if d.is_zero() {
            return if n.is_zero() { Some(F::zero()) } else { None };
        }
        Some(n / d)
    }
}

#[cfg(test)]
mod tests {
    use super::super::decouple::{ExactGrid, decouple_from_closure};
    use super::super::field::Fp;
    use super::super::rational::Rational;
    use super::*;

    fn rgrid(lambda: Vec<Vec<i64>>, mu: Vec<Vec<i64>>) -> ExactGrid<Rational> {
        ExactGrid {
            lambda: lambda
                .into_iter()
                .map(|a| a.into_iter().map(Rational::integer).collect())
                .collect(),
            mu: mu
                .into_iter()
                .map(|a| a.into_iter().map(Rational::integer).collect())
                .collect(),
        }
    }

    #[test]
    fn roundtrip_is_bit_exact_over_q_and_fp() {
        // An arbitrary 2×3×2 tensor of rationals with no special structure.
        let dims = [2usize, 3, 2];
        let data: Vec<Rational> = (0..12)
            .map(|i| Rational::new(3 * i as i64 - 7, i as i64 + 2))
            .collect();
        let tt = ExactTt::from_dense(&data, &dims);
        assert_eq!(tt.to_dense(), data, "lossless over ℚ");

        let p = 101;
        let fdata: Vec<Fp> = (0..12).map(|i| Fp::new(5 * i + 3, p)).collect();
        let ftt = ExactTt::from_dense(&fdata, &dims);
        assert_eq!(ftt.to_dense(), fdata, "lossless over 𝔽_p");
    }

    #[test]
    fn separable_tensor_is_rank_one() {
        // u ⊗ v ⊗ w has every unfolding rank 1.
        let u = [1i64, 2];
        let v = [3i64, 5, 7];
        let w = [11i64, 13];
        let mut data = Vec::new();
        for a in u {
            for b in v {
                for c in w {
                    data.push(Rational::integer(a * b * c));
                }
            }
        }
        let tt = ExactTt::from_dense(&data, &[2, 3, 2]);
        assert_eq!(tt.bond_ranks(), &[1, 1]);
        assert!(tt.storage_len() < tt.dense_len());
        assert_eq!(tt.to_dense(), data);
    }

    /// The structural theorem: a **polynomial** target (denominator ≡ 1) has
    /// `C_den = γ_0 ⊗ … ⊗ γ_{n-1}` exactly — TT-rank 1 on every bond, storage
    /// `Σ k_l` instead of `∏ k_l`, *regardless* of how coupled the numerator is.
    #[test]
    fn tt_of_polynomial_denominator_is_rank_one() {
        // Pólya–Szegő xy + xz + yz — maximally coupled polynomial.
        let grid = rgrid(
            vec![vec![0, 1], vec![0, 1], vec![0, 1]],
            vec![vec![2, 3], vec![2, 3], vec![2, 3]],
        );
        let f = |x: &[Rational]| {
            x[0].clone() * x[1].clone() + x[0].clone() * x[2].clone() + x[1].clone() * x[2].clone()
        };
        let model = decouple_from_closure(&grid, f).unwrap();
        let tt = TtDecoupled::compress(&model);
        assert_eq!(tt.den.bond_ranks(), &[1, 1], "C_den is Kronecker rank-1");
        // The numerator's ranks are the value tensor's ranks (Hadamard with a
        // rank-1 tensor preserves TT-rank): xy+xz+yz has bond rank 2.
        assert_eq!(tt.num.bond_ranks(), &[2, 2]);
        // The rank-1 denominator stores Σ k_l < ∏ k_l. (The *total* model only
        // beats dense once n/k grow past the bond overhead — see the 5-var case.)
        assert!(tt.den.storage_len() < tt.den.dense_len());
    }

    /// Storage win at scale: a separable 5-var polynomial compresses both tensors
    /// to rank ≤ 2 chains, well under the 2·2⁵ dense footprint.
    #[test]
    fn five_var_separable_storage_win() {
        let grid = rgrid(
            (0..5).map(|l| vec![2 * l, 2 * l + 1]).collect(),
            (0..5).map(|l| vec![10 + 2 * l, 11 + 2 * l]).collect(),
        );
        let f = |x: &[Rational]| {
            let mut acc = Rational::integer(1);
            for (l, xl) in x.iter().enumerate() {
                acc = acc * (xl.clone() + Rational::integer(l as i64 + 1));
            }
            acc
        };
        let model = decouple_from_closure(&grid, f).unwrap();
        let tt = TtDecoupled::compress(&model);
        assert!(tt.den.bond_ranks().iter().all(|&r| r == 1));
        assert!(
            tt.num.bond_ranks().iter().all(|&r| r == 1),
            "separable ⇒ rank-1 numerator"
        );
        assert!(
            tt.storage_len() < tt.dense_len(),
            "TT {} should beat dense {}",
            tt.storage_len(),
            tt.dense_len()
        );
        // Exact off-grid parity.
        let pt: Vec<Rational> = (0..5).map(|l| Rational::integer(100 + 3 * l)).collect();
        assert_eq!(tt.eval(&pt), model.eval(&pt));
    }

    #[test]
    fn compressed_eval_matches_dense_exactly() {
        // Coupled rational (Appendix C): nontrivial num *and* den tensors.
        let grid = rgrid(
            vec![vec![0, 1], vec![0, 1, 2]],
            vec![vec![3, 4], vec![3, 4, 5]],
        );
        let h = |x: &[Rational]| {
            let (s, t) = (x[0].clone(), x[1].clone());
            let num = t.clone() * t.clone() + s.clone() - Rational::integer(2);
            let den = t.clone() * t.clone() + Rational::integer(2) * s + Rational::integer(1);
            num / den
        };
        let model = decouple_from_closure(&grid, h).unwrap();
        let tt = TtDecoupled::compress(&model);
        // Off-grid, on-node (full coincidence), and partial-node points.
        let pts: Vec<Vec<Rational>> = vec![
            vec![Rational::integer(7), Rational::integer(9)],
            vec![Rational::integer(0), Rational::integer(1)], // full node
            vec![Rational::integer(0), Rational::integer(8)], // partial node
            vec![Rational::new(1, 3), Rational::new(-5, 7)],  // genuinely rational
        ];
        for pt in &pts {
            assert_eq!(tt.eval(pt), model.eval(pt), "parity at {pt:?}");
        }
        // And the decompression is bit-exact.
        let back = tt.decompress();
        assert_eq!(back.c_num, model.c_num);
        assert_eq!(back.c_den, model.c_den);
    }

    #[test]
    fn zero_tensor_compresses_without_panicking() {
        let data = vec![Rational::zero(); 8];
        let tt = ExactTt::from_dense(&data, &[2, 2, 2]);
        assert_eq!(tt.to_dense(), data);
    }

    fn fp_tensor(seed: u64, dims: &[usize], p: u64) -> Vec<Fp> {
        let total: usize = dims.iter().product();
        let mut s = seed;
        (0..total)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                Fp::new((s >> 9) as i128, p)
            })
            .collect()
    }

    #[test]
    fn hadamard_and_add_match_dense() {
        let p = 101;
        let dims = [3usize, 4, 2];
        let a = fp_tensor(1, &dims, p);
        let b = fp_tensor(2, &dims, p);
        let ta = ExactTt::from_dense(&a, &dims);
        let tb = ExactTt::from_dense(&b, &dims);
        let prod: Vec<Fp> = a.iter().zip(&b).map(|(x, y)| *x * *y).collect();
        let sum: Vec<Fp> = a.iter().zip(&b).map(|(x, y)| *x + *y).collect();
        assert_eq!(ta.hadamard(&tb).to_dense(), prod);
        assert_eq!(ta.add(&tb).to_dense(), sum);
    }

    #[test]
    fn apply_axis_map_matches_apply_axis_matrix() {
        // A non-square operator (4 → 3) applied both ways must agree exactly,
        // and the map route must leave bond ranks unchanged.
        let p = 101;
        let dims = [3usize, 4, 2];
        let t = ExactTt::from_dense(&fp_tensor(11, &dims, p), &dims);
        let mat: Vec<Fp> = (0..3 * 4)
            .map(|i| Fp::new((2 * i + 1) as i128, p))
            .collect();
        let via_matrix = t.apply_axis_matrix(1, &mat, 3);
        let via_map = t.apply_axis_map(1, 3, |v| {
            (0..3)
                .map(|row| (0..4).fold(Fp::new(0, p), |acc, col| acc + mat[row * 4 + col] * v[col]))
                .collect()
        });
        assert_eq!(via_map.to_dense(), via_matrix.to_dense());
        assert_eq!(via_map.bond_ranks(), via_matrix.bond_ranks());
    }

    #[test]
    fn recompress_restores_minimal_ranks_losslessly() {
        let p = 101;
        let dims = [2usize, 3, 3, 2];
        // A genuinely low-rank tensor: sum of two separables (TT-rank ≤ 2).
        let u: Vec<Vec<Fp>> = (0..4)
            .map(|l| {
                (0..dims[l])
                    .map(|j| Fp::new((l + 2 * j + 1) as i128, p))
                    .collect()
            })
            .collect();
        let v: Vec<Vec<Fp>> = (0..4)
            .map(|l| {
                (0..dims[l])
                    .map(|j| Fp::new((3 * l + j + 5) as i128, p))
                    .collect()
            })
            .collect();
        let low = ExactTt::rank_one(&u).add(&ExactTt::rank_one(&v));
        // Hadamard square: ranks multiply to ≤ 4, but the true rank is ≤ 3
        // (square of a rank-2 sum has 3 distinct separable terms).
        let sq = low.hadamard(&low);
        let dense = sq.to_dense();
        let minimal = ExactTt::from_dense(&dense, &dims);
        let rec = sq.recompress();
        assert_eq!(rec.to_dense(), dense, "recompression is lossless");
        assert_eq!(
            rec.bond_ranks(),
            minimal.bond_ranks(),
            "minimal exact ranks restored"
        );
        assert!(rec.storage_len() <= sq.storage_len());
        // And over ℚ as well (different arithmetic path).
        let qdata: Vec<Rational> = (0..12)
            .map(|i| Rational::new(i as i64 % 5 - 2, 3))
            .collect();
        let qt = ExactTt::from_dense(&qdata, &[3, 4]);
        let qsq = qt.hadamard(&qt).recompress();
        let want: Vec<Rational> = qdata.iter().map(|x| x.clone() * x.clone()).collect();
        assert_eq!(qsq.to_dense(), want);
        assert_eq!(
            qsq.bond_ranks(),
            ExactTt::from_dense(&want, &[3, 4]).bond_ranks()
        );
    }

    #[test]
    fn poly_from_values_matches_decoupling() {
        // f(x,y) = (x+1)(y+2) + 3xy on a 3×3 grid: poly_from_values must agree
        // with the full decoupling pipeline at off-grid points.
        let p = 97;
        let lambda: Vec<Vec<Fp>> = vec![
            (0..3).map(|v| Fp::new(v, p)).collect(),
            (0..3).map(|v| Fp::new(v + 10, p)).collect(),
        ];
        let f = |x: &[Fp]| {
            (x[0] + Fp::new(1, p)) * (x[1] + Fp::new(2, p)) + Fp::new(3, p) * x[0] * x[1]
        };
        let dims = [3usize, 3];
        let mut vals = Vec::new();
        for i in 0..3 {
            for j in 0..3 {
                vals.push(f(&[lambda[0][i], lambda[1][j]]));
            }
        }
        let wtt = ExactTt::from_dense(&vals, &dims);
        let model = TtDecoupled::poly_from_values(lambda.clone(), &wtt);
        for &(a, b) in &[(50i128, 60), (7, 93), (0, 11)] {
            let pt = [Fp::new(a, p), Fp::new(b, p)];
            assert_eq!(model.eval(&pt), Some(f(&pt)), "at ({a},{b})");
        }
        // values_tt round-trips the value tensor.
        assert_eq!(model.values_tt().to_dense(), vals);
    }

    #[test]
    fn mul_poly_is_exact_on_a_big_enough_grid() {
        // f = x² + y, g = x − y² over ℚ on a 4-point/axis grid (product degree
        // (2,2) needs ≥ 3; use 4 to also exercise non-minimal grids).
        let lambda: Vec<Vec<Rational>> = (0..2)
            .map(|l| (0..4).map(|j| Rational::integer(5 * l + j)).collect())
            .collect();
        let f = |x: &[Rational]| x[0].clone() * x[0].clone() + x[1].clone();
        let g = |x: &[Rational]| x[0].clone() - x[1].clone() * x[1].clone();
        let dims = [4usize, 4];
        let sample = |h: &dyn Fn(&[Rational]) -> Rational| -> Vec<Rational> {
            let mut v = Vec::new();
            for i in 0..4 {
                for j in 0..4 {
                    v.push(h(&[lambda[0][i].clone(), lambda[1][j].clone()]));
                }
            }
            v
        };
        let mf =
            TtDecoupled::poly_from_values(lambda.clone(), &ExactTt::from_dense(&sample(&f), &dims));
        let mg =
            TtDecoupled::poly_from_values(lambda.clone(), &ExactTt::from_dense(&sample(&g), &dims));
        let mh = mf.mul_poly(&mg).unwrap();
        for &(a, b) in &[(20i64, -3), (-7, 13), (1, 1)] {
            let pt = [Rational::integer(a), Rational::integer(b)];
            let want = f(&pt) * g(&pt);
            assert_eq!(mh.eval(&pt), Some(want), "product at ({a},{b})");
        }
    }

    #[test]
    fn mul_poly_on_undersized_grid_aliases_mod_node_poly() {
        // x·x on a 2-node grid {0,1}: x² ≡ x (mod x(x−1)) — the wraparound
        // theorem. The product model must return the *aliased* value, exactly.
        let lambda = vec![vec![Rational::integer(0), Rational::integer(1)]];
        let vals = vec![Rational::integer(0), Rational::integer(1)]; // f(x) = x on grid
        let m = TtDecoupled::poly_from_values(lambda, &ExactTt::rank_one(&[vals]));
        let sq = m.mul_poly(&m).unwrap();
        let x = Rational::integer(5);
        // True x² = 25; aliased x² mod x(x−1) = x → 5.
        assert_eq!(
            sq.eval(std::slice::from_ref(&x)),
            Some(Rational::integer(5))
        );
    }

    #[test]
    fn contract_parallel_matches_serial() {
        let p = 101;
        // A deep-ish chain with nontrivial ranks (from an automaton), random weights.
        let dims = [3usize, 4, 5, 3, 4, 6, 3];
        let tt = ExactTt::from_automaton(
            &dims,
            4,
            0,
            |_l, s, sym| (s + sym) % 4,
            |s| Fp::new((s == 0) as i128, p),
        );
        let mut seed = 7u64;
        let weights: Vec<Vec<Fp>> = dims
            .iter()
            .map(|&k| {
                (0..k)
                    .map(|_| {
                        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                        Fp::new((seed >> 10) as i128, p)
                    })
                    .collect()
            })
            .collect();
        assert_eq!(tt.contract(&weights), tt.contract_parallel(&weights));
    }

    #[test]
    fn from_automaton_matches_dense_and_is_state_bounded() {
        // Automaton: accept iff Σ symbols ≡ 0 (mod 3) — a 3-state carried-sum DFA.
        let p = 101;
        let dims = [4usize, 5, 3, 4];
        let states = 3;
        let step = |_l: usize, s: usize, sym: usize| (s + sym) % 3;
        let accept = |s: usize| Fp::new((s == 0) as i128, p);
        let tt = ExactTt::from_automaton(&dims, states, 0, step, accept);
        // bond ranks ≤ states by construction.
        assert!(tt.bond_ranks().iter().all(|&r| r <= states));
        // Verify against the dense ground truth.
        let total: usize = dims.iter().product();
        let mut dense = vec![Fp::new(0, p); total];
        let mut idx = vec![0usize; dims.len()];
        for cell in dense.iter_mut() {
            let s: usize = idx.iter().sum::<usize>() % 3;
            *cell = Fp::new((s == 0) as i128, p);
            for l in (0..dims.len()).rev() {
                idx[l] += 1;
                if idx[l] < dims[l] {
                    break;
                }
                idx[l] = 0;
            }
        }
        assert_eq!(tt.to_dense(), dense, "automaton TT must match dense");
    }

    #[test]
    fn mul_poly_rejects_mismatched_grids() {
        let l1 = vec![vec![Rational::integer(0), Rational::integer(1)]];
        let l2 = vec![vec![Rational::integer(2), Rational::integer(3)]];
        let v = vec![Rational::integer(1), Rational::integer(1)];
        let a = TtDecoupled::poly_from_values(l1, &ExactTt::rank_one(std::slice::from_ref(&v)));
        let b = TtDecoupled::poly_from_values(l2, &ExactTt::rank_one(std::slice::from_ref(&v)));
        assert!(a.mul_poly(&b).is_err());
    }
}
