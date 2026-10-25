//! Exact, SVD-free variable decoupling over a [`Field`] — the integer / rational
//! / finite-field analogue of [`crate::decouple()`].
//!
//! The follow-up paper (*Variable decoupling and the KST for rational functions*,
//! arXiv:2605.07246) proves that for a rational target the barycentric weights are
//! *explicit
//! evaluations* — no SVD. That is exactly what makes the method port to an exact
//! field: the one analytic ingredient of the floating-point pipeline (the SVD
//! null space) is unnecessary. Where degrees are unknown and a slice is
//! over-resolved we fall back to an **exact RREF null space**
//! ([`super::linalg`]), the tolerance-free analogue of the SVD null space.
//!
//! The result is **bit-exact reconstruction**: an integer/rational multivariate
//! polynomial or rational function is recovered with *zero* error (the §12
//! floating-point conditioning / "magic nip" pathologies cannot occur), and over
//! 𝔽_p a low-degree table compresses to its exact barycentric coefficients.
//!
//! ## Grid and degree contract
//!
//! Like the `f64` decoupling this needs a **unisolvent Cartesian grid**: per axis,
//! `k_l = degree_l + 1` distinct *column* nodes `λ_l` and `q_l` distinct *row*
//! nodes `μ_l`, disjoint from the columns. Distinctness over the field is what
//! makes the node-gap divisions (`γ_j`, Loewner entries) legal. For 𝔽_p choose
//! `p` larger than the grid so distinct integers stay distinct mod `p`.

use super::core::Field;
use super::linalg::Matrix;
use crate::error::{LoewnerError, Result};

/// A built exact decoupled rational model over the field `F`.
///
/// Holds the per-axis column nodes and the two **independent** coefficient
/// tensors `C_num`, `C_den` (canonical row-major order, the same contract as
/// [`crate::decouple::DecoupledModel`]). Evaluation is `N(x)/D(x)` with the
/// Cauchy kernel `S_j = 1/(x_l − λ_l(j))` — all exact field arithmetic.
#[derive(Clone, Debug)]
pub struct ExactDecoupled<F: Field> {
    /// Column support nodes per axis.
    pub lambda: Vec<Vec<F>>,
    /// Numerator coefficient tensor (canonical order).
    pub c_num: Vec<F>,
    /// Denominator coefficient tensor (canonical order).
    pub c_den: Vec<F>,
}

impl<F: Field> ExactDecoupled<F> {
    /// Number of variables.
    pub fn ndim(&self) -> usize {
        self.lambda.len()
    }
    /// Per-axis node counts `k_l`.
    pub fn sizes(&self) -> Vec<usize> {
        self.lambda.iter().map(Vec::len).collect()
    }

    /// Number of nonzero entries in the denominator tensor — the natural exact
    /// "support size" (over a field, zero is decidable, so this is meaningful and
    /// the basis of lossless compression; see `docs/EXACT_FIELDS.md` §3).
    pub fn den_nnz(&self) -> usize {
        self.c_den.iter().filter(|c| !c.is_zero()).count()
    }
    /// Number of nonzero numerator entries.
    pub fn num_nnz(&self) -> usize {
        self.c_num.iter().filter(|c| !c.is_zero()).count()
    }
    /// Total stored coefficients (`2·∏k_l`) — the dense footprint.
    pub fn dense_len(&self) -> usize {
        self.c_num.len() + self.c_den.len()
    }

    /// Evaluate `N(x)/D(x)` at `x` (length `n`). `None` if `x` lands on a node and
    /// the denominator mass there is zero (a genuine `0/0`), otherwise the exact
    /// field value. On a full-node coincidence the sampled value `C_num/C_den` is
    /// returned directly (the removable-singularity limit, §6g).
    pub fn eval(&self, x: &[F]) -> Option<F> {
        debug_assert_eq!(x.len(), self.ndim());
        // Full-node fast path: x == λ on every axis → return C_num[J]/C_den[J].
        if let Some(j) = full_node_index(&self.lambda, x) {
            let (cn, cd) = (self.c_num[j].clone(), self.c_den[j].clone());
            if !cd.is_zero() {
                return Some(cn / cd);
            }
            // else fall through; the contraction's 0/0 is handled below.
        }
        let dims = self.sizes();
        let weights = cauchy_weights(&self.lambda, x);
        let mut num = self.c_num.clone();
        let mut den = self.c_den.clone();
        let n = contract_rank1(&mut num, &dims, &weights);
        let d = contract_rank1(&mut den, &dims, &weights);
        if d.is_zero() {
            return if n.is_zero() { Some(F::zero()) } else { None };
        }
        Some(n / d)
    }
}

impl ExactDecoupled<super::rational::Rational> {
    /// Total bit footprint of the stored coefficients: `Σ bits(numer) + bits(denom)`
    /// over the nonzero entries of both tensors. The honest size metric for a ℚ
    /// model (a `BigRational` is as big as its integers, not "one field element").
    pub fn bit_size(&self) -> u64 {
        self.c_num
            .iter()
            .chain(&self.c_den)
            .filter(|c| !c.is_zero())
            .map(|c| c.numer().bits() + c.denom().bits())
            .sum()
    }

    /// **Joint primitive-integer rescale** — the third exact compression lever
    /// (specific to ℚ). The model is the *pair* `(C_num, C_den)` and evaluation is
    /// the ratio `N(x)/D(x)`, so multiplying **both** tensors by the same nonzero
    /// scalar changes nothing. The cascade's Lagrange factors leave large shared
    /// denominators in every coefficient; this picks the canonical representative:
    /// scale by `lcm(all denominators) / gcd(all numerators)`, making every stored
    /// coefficient an **integer** with overall content 1 — typically a large
    /// `bit_size` reduction (and bit-exact, eval invariant).
    ///
    /// Returns the scale `s` that was applied (`C ← s·C`), `1` if the model was
    /// all-zero.
    pub fn canonicalize(&mut self) -> super::rational::Rational {
        use num_bigint::BigInt;
        use num_integer::Integer;
        use num_traits::{One, Zero};
        let mut den_lcm = BigInt::one();
        for c in self.c_num.iter().chain(&self.c_den) {
            if !c.is_zero() {
                den_lcm = den_lcm.lcm(c.denom());
            }
        }
        // gcd of the scaled (now integer) numerators: numer · (lcm / denom).
        let mut num_gcd = BigInt::zero();
        for c in self.c_num.iter().chain(&self.c_den) {
            if !c.is_zero() {
                num_gcd = num_gcd.gcd(&(c.numer() * (&den_lcm / c.denom())));
            }
        }
        if num_gcd.is_zero() {
            return super::rational::Rational::integer(1); // all-zero model
        }
        let s = super::rational::Rational::new(den_lcm, num_gcd);
        for c in self.c_num.iter_mut().chain(self.c_den.iter_mut()) {
            *c = c.clone() * s.clone();
        }
        s
    }
}

/// Per-axis Cauchy weights `S_l(j) = 1/(x_l − λ_l(j))`. When `x_l` hits a node on
/// an axis the weights for that axis collapse to the **selector** picking that
/// node — the removable-singularity limit (§6d/§6g), shared by every exact eval
/// path (dense, MIMO, TT-compressed, sparse).
pub(crate) fn cauchy_weights<F: Field>(lambda: &[Vec<F>], x: &[F]) -> Vec<Vec<F>> {
    let mut weights = Vec::with_capacity(lambda.len());
    for (l, lam) in lambda.iter().enumerate() {
        let mut wl = Vec::with_capacity(lam.len());
        let mut hit = None;
        for (j, lj) in lam.iter().enumerate() {
            let diff = x[l].clone() - lj.clone();
            if diff.is_zero() {
                hit = Some(j);
                break;
            }
            wl.push(diff.inv().expect("nonzero node gap"));
        }
        if let Some(j) = hit {
            // Limit: selector picking node j on this axis.
            wl = (0..lam.len())
                .map(|i| if i == j { F::one() } else { F::zero() })
                .collect();
        }
        weights.push(wl);
    }
    weights
}

/// If `x` coincides with a column node on **every** axis, the flat (row-major)
/// index of that node; else `None`.
pub(crate) fn full_node_index<F: Field>(lambda: &[Vec<F>], x: &[F]) -> Option<usize> {
    let mut flat = 0usize;
    for (lam, xl) in lambda.iter().zip(x) {
        let j = lam.iter().position(|v| v == xl)?;
        flat = flat * lam.len() + j;
    }
    Some(flat)
}

/// Contract a row-major tensor against the rank-1 weight `⊗_l weights[l]` to the
/// scalar `Σ_J buf[J] ∏_l weights[l][j_l]`. Generic over `F`.
fn contract_rank1<F: Field>(buf: &mut [F], dims: &[usize], weights: &[Vec<F>]) -> F {
    let mut block = buf.len();
    for (l, wl) in weights.iter().enumerate().rev() {
        let kl = dims[l];
        let outer = block / kl;
        for o in 0..outer {
            let base = o * kl;
            let mut s = F::zero();
            for (j, w) in wl.iter().enumerate() {
                s = s + buf[base + j].clone() * w.clone();
            }
            buf[o] = s;
        }
        block = outer;
    }
    buf[0].clone()
}

/// Lagrange coefficients `γ_j = 1/∏_{i≠j}(λ_j − λ_i)` over the field.
pub fn lagrange_coeffs<F: Field>(lambda: &[F]) -> Vec<F> {
    lambda
        .iter()
        .enumerate()
        .map(|(j, lj)| {
            let mut prod = F::one();
            for (i, li) in lambda.iter().enumerate() {
                if i != j {
                    prod = prod * (lj.clone() - li.clone());
                }
            }
            prod.inv().expect("distinct nodes ⇒ nonzero product")
        })
        .collect()
}

/// Per-axis grid for the exact decoupling: column nodes `λ` and disjoint row
/// nodes `μ`, one pair per axis.
#[derive(Clone, Debug)]
pub struct ExactGrid<F: Field> {
    /// Column nodes per axis (`k_l = degree_l + 1`).
    pub lambda: Vec<Vec<F>>,
    /// Row nodes per axis (disjoint from `lambda[l]`).
    pub mu: Vec<Vec<F>>,
}

impl<F: Field> ExactGrid<F> {
    /// Number of axes.
    pub fn ndim(&self) -> usize {
        self.lambda.len()
    }
    /// Combined `column ++ row` abscissae for axis `l` (the order [`sample`] uses).
    fn combined(&self, l: usize) -> Vec<F> {
        let mut v = self.lambda[l].clone();
        v.extend(self.mu[l].iter().cloned());
        v
    }
    /// Per-axis combined sizes `k_l + q_l`.
    fn combined_sizes(&self) -> Vec<usize> {
        (0..self.ndim())
            .map(|l| self.lambda[l].len() + self.mu[l].len())
            .collect()
    }
}

/// Sample `f` on the full combined Cartesian grid (`column ++ row` per axis), in
/// canonical row-major order. `f` takes the field-valued node coordinates.
pub fn sample<F: Field>(grid: &ExactGrid<F>, f: impl Fn(&[F]) -> F) -> Vec<F> {
    let dims = grid.combined_sizes();
    let axes: Vec<Vec<F>> = (0..grid.ndim()).map(|l| grid.combined(l)).collect();
    let total: usize = dims.iter().product();
    let n = grid.ndim();
    let mut out = Vec::with_capacity(total);
    let mut coord = vec![F::zero(); n];
    let mut idx = vec![0usize; n];
    for _ in 0..total {
        for l in 0..n {
            coord[l] = axes[l][idx[l]].clone();
        }
        out.push(f(&coord));
        // increment row-major (last axis fastest)
        for l in (0..n).rev() {
            idx[l] += 1;
            if idx[l] < dims[l] {
                break;
            }
            idx[l] = 0;
        }
    }
    out
}

/// Build an exact decoupled model from samples of `H` on the combined grid.
///
/// `tab` is `H` sampled by [`sample`] (combined `column ++ row` grid, canonical
/// order). Returns the model plus the per-axis degree used.
///
/// The construction is the data-driven Kronecker cascade of
/// [`crate::decouple()`], but every slice's denominator weights are the **exact**
/// 1-D Loewner null space (RREF over `F`, not SVD), and the numerator tensor is
/// the exact pointwise `C_num[J] = C_den[J] · H(λ_J)`.
pub fn decouple_from_tensor<F: Field>(grid: &ExactGrid<F>, tab: &[F]) -> Result<ExactDecoupled<F>> {
    let dims = grid.combined_sizes();
    let total: usize = dims.iter().product();
    if tab.len() != total {
        return Err(LoewnerError::Degenerate(format!(
            "sample tensor length {} != grid size {}",
            tab.len(),
            total
        )));
    }
    let k: Vec<usize> = grid.lambda.iter().map(Vec::len).collect();
    let ctotal: usize = k.iter().product();

    // Denominator tensor via the recursive cascade.
    let mut c_den = Vec::with_capacity(ctotal);
    recurse(grid, tab, &dims, &[], 0, false, &mut c_den)?;

    // Numerator tensor: C_num[J] = C_den[J] · H(λ_J). Read the column-block grid
    // values (every axis index in [0, k_l)) in canonical order.
    let col_vals = column_values(grid, tab, &dims);
    let c_num: Vec<F> = c_den
        .iter()
        .zip(&col_vals)
        .map(|(d, w)| d.clone() * w.clone())
        .collect();

    Ok(ExactDecoupled {
        lambda: grid.lambda.clone(),
        c_num,
        c_den,
    })
}

/// A **MIMO** exact decoupled model: several outputs sharing **one** denominator
/// tensor, each with its own numerator tensor.
///
/// This is the multi-output (`𝔽^n → 𝔽^m` or `ℚ^n → ℚ^m`) form, and the reason it
/// is *cheap* is exactly the denominator-only null-space result (Theorem 2.1): the
/// Loewner null space — hence the barycentric **denominator** weights `C_den` —
/// depends solely on the shared denominator `d`, never on the numerators. So the
/// (expensive) null-space cascade runs **once**; each output `i` then contributes
/// only a numerator tensor `C_num^(i)[J] = C_den[J] · H_i(λ_J)`, a pure evaluation.
/// No bit-slicing, no `m` independent solves, no `m` denominators — a single shared
/// denominator with `m` numerators, which is the matrix-Loewner / tangential
/// structure of the original framework specialized to a common pole set.
///
/// **Validity condition:** the outputs must genuinely share a denominator. When the
/// data is a vector rational `[N_1, …, N_m]/d` (one common `d`) this is exact. When
/// the outputs have *different* denominators it is an approximation that pins each
/// output exactly on the grid (the barycentric interpolant through that output's
/// samples) but uses the reference output's pole structure off-grid — fine for a
/// shared-pole family, lossy otherwise (use one [`ExactDecoupled`] per output then).
#[derive(Clone, Debug)]
pub struct MimoDecoupled<F: Field> {
    /// Column support nodes per input axis.
    pub lambda: Vec<Vec<F>>,
    /// The **shared** denominator coefficient tensor (built once).
    pub c_den: Vec<F>,
    /// One numerator coefficient tensor per output (`C_num^(i) = C_den ⊙ w^(i)`).
    pub c_num: Vec<Vec<F>>,
}

impl<F: Field> MimoDecoupled<F> {
    /// Number of input variables.
    pub fn ndim(&self) -> usize {
        self.lambda.len()
    }
    /// Number of outputs `m`.
    pub fn n_outputs(&self) -> usize {
        self.c_num.len()
    }
    /// Per-axis node counts.
    pub fn sizes(&self) -> Vec<usize> {
        self.lambda.iter().map(Vec::len).collect()
    }

    /// Total stored coefficients: **one** denominator plus `m` numerators. The
    /// saving versus `m` independent [`ExactDecoupled`]s is exactly the shared
    /// denominator — `(m+1)·∏k` instead of `2m·∏k`.
    pub fn dense_len(&self) -> usize {
        self.c_den.len() * (1 + self.n_outputs())
    }

    /// Evaluate all `m` outputs at `x`, sharing the single denominator contraction
    /// (computed once). Returns `None` for an output only if the shared denominator
    /// vanishes there with a nonzero numerator (a genuine pole).
    pub fn eval(&self, x: &[F]) -> Vec<Option<F>> {
        let dims = self.sizes();
        // Build the per-axis Cauchy weights once (shared across outputs).
        let weights = cauchy_weights(&self.lambda, x);
        // Shared denominator contraction (the expensive part, done once).
        let mut den_buf = self.c_den.clone();
        let d = contract_rank1(&mut den_buf, &dims, &weights);
        // Each output: its numerator contraction over the *same* weights.
        self.c_num
            .iter()
            .map(|cn| {
                let mut num_buf = cn.clone();
                let n = contract_rank1(&mut num_buf, &dims, &weights);
                if d.is_zero() {
                    if n.is_zero() { Some(F::zero()) } else { None }
                } else {
                    Some(n.clone() / d.clone())
                }
            })
            .collect()
    }

    /// Extract output `i` as a standalone [`ExactDecoupled`] (shared denominator +
    /// that output's numerator).
    pub fn output(&self, i: usize) -> ExactDecoupled<F> {
        ExactDecoupled {
            lambda: self.lambda.clone(),
            c_num: self.c_num[i].clone(),
            c_den: self.c_den.clone(),
        }
    }
}

/// Build a **MIMO** model with a shared denominator from `m` vector-valued samples.
///
/// `f` returns the full output vector `[H_1(x), …, H_m(x)]` at each grid point. The
/// shared denominator is built **once** from the reference output (index 0); every
/// output's numerator is `C_den ⊙ w^(i)`. This is the denominator-only-null-space
/// economy made concrete: `m` outputs cost one null-space cascade, not `m`.
///
/// Use when the outputs share a pole structure (a common denominator), e.g. a
/// vector rational `N(x)/d(x)`, a MIMO transfer function, or several coordinates of
/// a map sampled together. For independent denominators, build per-output instead.
pub fn decouple_mimo<F: Field>(
    grid: &ExactGrid<F>,
    n_outputs: usize,
    f: impl Fn(&[F]) -> Vec<F>,
) -> Result<MimoDecoupled<F>> {
    let dims = grid.combined_sizes();
    let total: usize = dims.iter().product();
    // Sample every output on the combined grid (canonical order).
    let axes: Vec<Vec<F>> = (0..grid.ndim()).map(|l| grid.combined(l)).collect();
    let n = grid.ndim();
    let mut tabs: Vec<Vec<F>> = vec![Vec::with_capacity(total); n_outputs];
    let mut coord = vec![F::zero(); n];
    let mut idx = vec![0usize; n];
    for _ in 0..total {
        for l in 0..n {
            coord[l] = axes[l][idx[l]].clone();
        }
        let out = f(&coord);
        if out.len() != n_outputs {
            return Err(LoewnerError::Degenerate(format!(
                "closure returned {} outputs, expected {n_outputs}",
                out.len()
            )));
        }
        for (i, v) in out.into_iter().enumerate() {
            tabs[i].push(v);
        }
        for l in (0..n).rev() {
            idx[l] += 1;
            if idx[l] < dims[l] {
                break;
            }
            idx[l] = 0;
        }
    }
    // Shared denominator from the reference output (Theorem 2.1: null space depends
    // only on the denominator, identical across outputs that share it).
    let k: Vec<usize> = grid.lambda.iter().map(Vec::len).collect();
    let ctotal: usize = k.iter().product();
    let mut c_den = Vec::with_capacity(ctotal);
    recurse(grid, &tabs[0], &dims, &[], 0, false, &mut c_den)?;
    // Each output's numerator = C_den ⊙ (its column-block grid values).
    let c_num: Vec<Vec<F>> = tabs
        .iter()
        .map(|tab| {
            let col_vals = column_values(grid, tab, &dims);
            c_den
                .iter()
                .zip(&col_vals)
                .map(|(d, w)| d.clone() * w.clone())
                .collect()
        })
        .collect();
    Ok(MimoDecoupled {
        lambda: grid.lambda.clone(),
        c_den,
        c_num,
    })
}

/// Build directly from a closure `f` over field coordinates.
pub fn decouple_from_closure<F: Field>(
    grid: &ExactGrid<F>,
    f: impl Fn(&[F]) -> F,
) -> Result<ExactDecoupled<F>> {
    let tab = sample(grid, f);
    decouple_from_tensor(grid, &tab)
}

/// **Pure by-inspection polynomial model** from values on the column grid:
/// `C_den[J] = ∏_l γ_l[j_l]` and `C_num = C_den ⊙ W` — Thm 2.1/Cor 2.1 with no
/// factorization, no null space, no row nodes, and **no division except the
/// Lagrange `γ`** (products of node-gap inverses).
///
/// Consequences worth stating: (1) the disjoint column/row requirement — hence
/// the `⌊p/2⌋−1` degree ceiling of the Loewner-matrix path — does not apply;
/// a degree-`d` axis needs only `d+1` distinct nodes. (2) The construction is
/// valid over any **commutative ring whose node gaps are units**, not just a
/// field — which is what would let it run over a CRT product ring ℤ/(p₁p₂…),
/// computing over all prime fields at once (an integer-exact path; not currently
/// built here).
pub fn poly_decouple_from_values<F: Field>(lambda: Vec<Vec<F>>, values: &[F]) -> ExactDecoupled<F> {
    let dims: Vec<usize> = lambda.iter().map(Vec::len).collect();
    let total: usize = dims.iter().product();
    assert_eq!(values.len(), total, "one value per column-grid point");
    let gammas: Vec<Vec<F>> = lambda.iter().map(|l| lagrange_coeffs(l)).collect();
    let mut c_den = Vec::with_capacity(total);
    let mut idx = vec![0usize; dims.len()];
    for _ in 0..total {
        let mut g = F::one();
        for (l, &j) in idx.iter().enumerate() {
            g = g * gammas[l][j].clone();
        }
        c_den.push(g);
        for l in (0..dims.len()).rev() {
            idx[l] += 1;
            if idx[l] < dims[l] {
                break;
            }
            idx[l] = 0;
        }
    }
    let c_num: Vec<F> = c_den
        .iter()
        .zip(values)
        .map(|(d, w)| d.clone() * w.clone())
        .collect();
    ExactDecoupled {
        lambda,
        c_num,
        c_den,
    }
}

/// Recurse over axes `axis..n`, appending denominator coefficients in canonical
/// order. `lead` holds the leading-axis column indices already fixed; `normalize`
/// is `false` for the top axis (un-normalized, Eq. 18) and `true` for nested axes
/// (normalize by last entry).
fn recurse<F: Field>(
    grid: &ExactGrid<F>,
    tab: &[F],
    dims: &[usize],
    lead: &[usize],
    axis: usize,
    normalize: bool,
    out: &mut Vec<F>,
) -> Result<()> {
    let n = grid.ndim();
    let k0 = grid.lambda[axis].len();

    // Representative slice along `axis`: leading axes fixed by `lead`, trailing
    // axes frozen at their last *column* index (k_l − 1).
    let (w, v) = slice_along(grid, tab, dims, lead, axis);
    let den = slice_den_weights(&grid.lambda[axis], &grid.mu[axis], &w, &v, normalize)?;

    if axis + 1 == n {
        out.extend(den);
        return Ok(());
    }
    let mut child_lead = lead.to_vec();
    child_lead.push(0);
    for (p, sden) in den.iter().enumerate().take(k0) {
        *child_lead.last_mut().unwrap() = p;
        let start = out.len();
        recurse(grid, tab, dims, &child_lead, axis + 1, true, out)?;
        for x in &mut out[start..] {
            *x = x.clone() * sden.clone();
        }
    }
    Ok(())
}

/// The 1-D slice of `tab` varying `axis` over its combined abscissae, with leading
/// axes pinned by `lead` (column indices) and trailing axes frozen at their last
/// column index. Returns `(w, v)` = (column-node values, row-node values).
fn slice_along<F: Field>(
    grid: &ExactGrid<F>,
    tab: &[F],
    dims: &[usize],
    lead: &[usize],
    axis: usize,
) -> (Vec<F>, Vec<F>) {
    let n = grid.ndim();
    let mut idx = vec![0usize; n];
    for (l, &li) in lead.iter().enumerate() {
        idx[l] = li;
    }
    for (l, slot) in idx.iter_mut().enumerate().take(n).skip(axis + 1) {
        *slot = grid.lambda[l].len() - 1; // last column index
    }
    let kl = grid.lambda[axis].len();
    let total = dims[axis];
    let mut line = Vec::with_capacity(total);
    for a in 0..total {
        idx[axis] = a;
        line.push(tab[flat_index(&idx, dims)].clone());
    }
    let v = line.split_off(kl);
    (line, v)
}

/// Denominator barycentric weights of a 1-D slice — the exact Loewner null space.
///
/// `lambda`/`mu` are the column/row abscissae; `w`/`v` the slice values there. The
/// `q×k` Loewner `L[i,j] = (v_i − w_j)/(μ_i − λ_j)` has 1-D null space `[γ_j d(λ_j)]`
/// (Thm 2.1). We compute it by **exact RREF** (no tolerance). For an over-resolved
/// slice (null dimension > 1) the first kernel basis vector is taken — over a field
/// that is deterministic, and post-selection to `order+1` nodes makes it unique.
fn slice_den_weights<F: Field>(
    lambda: &[F],
    mu: &[F],
    w: &[F],
    v: &[F],
    normalize: bool,
) -> Result<Vec<F>> {
    let k = lambda.len();
    let q = mu.len();
    let loewner = Matrix::from_fn(q, k, |i, j| {
        let denom = mu[i].clone() - lambda[j].clone();
        (v[i].clone() - w[j].clone()) * denom.inv().expect("disjoint column/row nodes")
    });
    let ns = loewner.null_space();
    // A *unique* (1-dimensional) null space is the well-posed case: `den = [γ_j d(λ_j)]`
    // by Theorem 2.1. A higher-dimensional null space means the slice is
    // **rank-deficient** — an over-resolved or identically-degenerate line (e.g. a
    // polynomial slice that vanishes for the frozen context, WORKPLAN §6c). There the
    // data is consistent with a *constant* denominator, so the weights are uniquely
    // the Lagrange coefficients `γ_j` (the `d ≡ const` instance of Thm 2.1). Picking
    // an arbitrary kernel basis vector (what an SVD does, by luck) would be wrong;
    // pinning to `γ` is exact and deterministic — a strict improvement on the f64 path.
    let mut den = if ns.len() == 1 {
        ns.into_iter().next().unwrap()
    } else {
        lagrange_coeffs(lambda)
    };
    if den.len() != k {
        return Err(LoewnerError::Degenerate(
            "null vector length mismatch".into(),
        ));
    }
    if normalize {
        normalize_last(&mut den);
    }
    Ok(den)
}

/// Normalize a weight vector by its last nonzero-or-not entry (paper convention).
fn normalize_last<F: Field>(v: &mut [F]) {
    if let Some(inv) = v.last().cloned().and_then(|last| last.inv()) {
        for x in v.iter_mut() {
            *x = x.clone() * inv.clone();
        }
    }
}

/// Grid values on the **column block** (every axis index in `[0, k_l)`), canonical
/// row-major order — i.e. `H(λ_J)` for every column multi-index `J`.
fn column_values<F: Field>(grid: &ExactGrid<F>, tab: &[F], dims: &[usize]) -> Vec<F> {
    let n = grid.ndim();
    let k: Vec<usize> = grid.lambda.iter().map(Vec::len).collect();
    let ctotal: usize = k.iter().product();
    let mut out = Vec::with_capacity(ctotal);
    let mut idx = vec![0usize; n];
    for _ in 0..ctotal {
        out.push(tab[flat_index(&idx, dims)].clone());
        for l in (0..n).rev() {
            idx[l] += 1;
            if idx[l] < k[l] {
                break;
            }
            idx[l] = 0;
        }
    }
    out
}

/// Row-major flat index of a multi-index into a tensor of shape `dims`.
fn flat_index(idx: &[usize], dims: &[usize]) -> usize {
    let mut flat = 0;
    for (i, d) in idx.iter().zip(dims) {
        flat = flat * d + i;
    }
    flat
}

#[cfg(test)]
mod tests {
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
    fn polya_szego_exact_from_8_samples_over_q() {
        // p(x,y,z) = xy + xz + yz, degree (1,1,1) ⇒ k=2 per axis. 8 column points.
        let grid = rgrid(
            vec![vec![0, 1], vec![0, 1], vec![0, 1]],
            vec![vec![2, 3], vec![2, 3], vec![2, 3]],
        );
        let f = |x: &[Rational]| {
            x[0].clone() * x[1].clone() + x[0].clone() * x[2].clone() + x[1].clone() * x[2].clone()
        };
        let model = decouple_from_closure(&grid, f).unwrap();
        // Exact at arbitrary off-grid rational points.
        for &(a, b, c) in &[(5, 7, 11), (-3, 4, 9), (10, -2, 6)] {
            let pt = [
                Rational::integer(a),
                Rational::integer(b),
                Rational::integer(c),
            ];
            let got = model.eval(&pt).unwrap();
            let want = f(&pt);
            assert_eq!(got, want, "exact recovery at {:?}", (a, b, c));
        }
    }

    #[test]
    fn coupled_rational_appendix_c_over_q() {
        // H(s,t) = (t² + s − 2)/(t² + 2s + 1), degrees (1,2) ⇒ k=(2,3).
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
        for &(s, t) in &[(7, 9), (-4, 5), (11, -3)] {
            let pt = [Rational::integer(s), Rational::integer(t)];
            assert_eq!(model.eval(&pt).unwrap(), h(&pt), "at {:?}", (s, t));
        }
    }

    #[test]
    fn polya_szego_exact_over_fp() {
        let p = 101;
        let e = |v: i128| Fp::new(v, p);
        let grid = ExactGrid {
            lambda: vec![vec![e(0), e(1)], vec![e(0), e(1)], vec![e(0), e(1)]],
            mu: vec![vec![e(2), e(3)], vec![e(2), e(3)], vec![e(2), e(3)]],
        };
        let f = |x: &[Fp]| x[0] * x[1] + x[0] * x[2] + x[1] * x[2];
        let model = decouple_from_closure(&grid, f).unwrap();
        for &(a, b, c) in &[(5, 7, 11), (40, 60, 80), (99, 1, 50)] {
            let pt = [e(a), e(b), e(c)];
            assert_eq!(model.eval(&pt).unwrap(), f(&pt), "Fp at {:?}", (a, b, c));
        }
    }

    /// Joint primitive-integer rescale: eval-invariant, all-integer coefficients,
    /// content 1, and a strictly smaller (or equal) bit footprint.
    #[test]
    fn canonicalize_shrinks_bits_and_preserves_eval() {
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
        let mut model = decouple_from_closure(&grid, h).unwrap();
        let before = model.bit_size();
        let reference: Vec<Option<Rational>> = [(7, 9), (-4, 5), (11, -3)]
            .iter()
            .map(|&(s, t)| model.eval(&[Rational::integer(s), Rational::integer(t)]))
            .collect();
        model.canonicalize();
        // All coefficients are now integers (denominators cleared).
        assert!(
            model
                .c_num
                .iter()
                .chain(&model.c_den)
                .all(|c| c.is_integer()),
            "canonical form is integral"
        );
        assert!(model.bit_size() <= before, "bit footprint must not grow");
        // Evaluation is unchanged (the pair scales jointly).
        for (i, &(s, t)) in [(7, 9), (-4, 5), (11, -3)].iter().enumerate() {
            let got = model.eval(&[Rational::integer(s), Rational::integer(t)]);
            assert_eq!(got, reference[i], "eval invariant at {:?}", (s, t));
        }
    }

    /// MIMO shared-denominator economy: a vector rational `[N_1, N_2]/d` with a
    /// **common** denominator recovers all outputs exactly from **one** null-space
    /// cascade, and stores one denominator + m numerators (not 2m tensors).
    #[test]
    fn mimo_shared_denominator_recovers_all_outputs() {
        // H(s,t) = [ (s+t)/(1+s+t²),  (s·t)/(1+s+t²) ] — two numerators, one
        // shared denominator d = 1 + s + t². Degrees: num deg (1,1)/(1,1), den (1,2).
        let grid = rgrid(
            vec![vec![0, 1], vec![0, 1, 2]],
            vec![vec![3, 4], vec![3, 4, 5]],
        );
        let d =
            |s: &Rational, t: &Rational| Rational::integer(1) + s.clone() + t.clone() * t.clone();
        let f = |x: &[Rational]| {
            let (s, t) = (x[0].clone(), x[1].clone());
            let den = d(&s, &t);
            vec![
                (s.clone() + t.clone()) / den.clone(),
                (s.clone() * t.clone()) / den,
            ]
        };
        let model = decouple_mimo(&grid, 2, f).unwrap();
        assert_eq!(model.n_outputs(), 2);
        // Shared denominator: stored len is (1 + 2)·∏k, not 2·2·∏k.
        let k: usize = 2 * 3;
        assert_eq!(model.dense_len(), 3 * k);
        for &(s, t) in &[(7, 9), (-4, 5), (11, -3)] {
            let pt = [Rational::integer(s), Rational::integer(t)];
            let got = model.eval(&pt);
            let want = f(&pt);
            assert_eq!(got[0].clone().unwrap(), want[0], "output 0 at {:?}", (s, t));
            assert_eq!(got[1].clone().unwrap(), want[1], "output 1 at {:?}", (s, t));
        }
    }
}
